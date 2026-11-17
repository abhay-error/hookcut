"""Local web app: a small JSON API plus a single static page.

Jobs run one at a time on a background thread (transcription and LLMs are
heavy; running two at once just makes both slow). Model downloads run on their
own threads so you can fetch a model while a job renders.
"""

from __future__ import annotations

import queue
import re
import shutil
import subprocess
import sys
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .. import catalog, config, fonts, llm, pipeline

STATIC = Path(__file__).parent / "static"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi"}


class NewJob(BaseModel):
    target: str


class PullRequest(BaseModel):
    name: str


class Worker:
    def __init__(self) -> None:
        self.queue: queue.Queue[str] = queue.Queue()
        self.cancels: dict[str, threading.Event] = {}
        threading.Thread(target=self._loop, daemon=True).start()

    def submit(self, job: pipeline.Job) -> None:
        self.cancels[job.id] = threading.Event()
        self.queue.put(job.id)

    def cancel(self, job_id: str) -> None:
        if job_id in self.cancels:
            self.cancels[job_id].set()

    def _loop(self) -> None:
        while True:
            job_id = self.queue.get()
            job = pipeline.load_job(job_id)
            cancel = self.cancels.get(job_id, threading.Event())
            if job is None:
                continue
            if cancel.is_set():
                job.status, job.message = "cancelled", "Cancelled."
                pipeline.save_job(job)
                continue
            pipeline.Runner(job, cancel=cancel).run()


class Pulls:
    """Model downloads in flight: name -> {status, progress, error, done}."""

    def __init__(self) -> None:
        self.state: dict[str, dict] = {}
        self.lock = threading.Lock()

    def start(self, settings: config.LLMSettings, name: str) -> None:
        with self.lock:
            current = self.state.get(name)
            if current and not current["done"]:
                return
            self.state[name] = {"status": "starting", "progress": 0.0, "error": "",
                                "done": False}
        threading.Thread(target=self._run, args=(settings, name), daemon=True).start()

    def _run(self, settings: config.LLMSettings, name: str) -> None:
        def report(status: str, fraction: float | None) -> None:
            with self.lock:
                entry = self.state[name]
                entry["status"] = status
                if fraction is not None:
                    entry["progress"] = fraction

        try:
            llm.pull_model(settings, name, report)
            report("success", 1.0)
        except Exception as exc:
            with self.lock:
                self.state[name]["error"] = str(exc)
        finally:
            with self.lock:
                self.state[name]["done"] = True

    def snapshot(self) -> dict[str, dict]:
        with self.lock:
            return {k: dict(v) for k, v in self.state.items()}


def create_app() -> FastAPI:
    app = FastAPI(title="hookcut", docs_url=None, redoc_url=None)
    worker = Worker()
    pulls = Pulls()

    # Jobs interrupted by a restart can't resume mid-stage; mark them clearly.
    for job in pipeline.list_jobs():
        if job.status in ("running", "queued"):
            job.status, job.message = "failed", "Interrupted (the app was stopped)."
            job.error = "Run it again; downloads and transcripts are reused."
            pipeline.save_job(job)

    # -- settings ---------------------------------------------------------------

    @app.get("/api/settings")
    def get_settings() -> dict:
        return {"settings": config.load().model_dump()}

    @app.put("/api/settings")
    def put_settings(payload: dict) -> dict:
        merged = config.load().model_dump()
        for section, values in payload.items():
            if section in merged and isinstance(values, dict) and isinstance(merged[section], dict):
                merged[section].update(values)
            elif section in merged:
                merged[section] = values
        try:
            settings = config.Settings.model_validate(merged)
        except Exception as exc:
            raise HTTPException(400, f"Invalid settings: {exc}") from exc
        if settings.render.caption_font not in fonts.FONTS:
            raise HTTPException(400, "Unknown font.")
        config.save(settings)
        return {"settings": settings.model_dump()}

    # -- models -----------------------------------------------------------------

    @app.get("/api/models")
    def models() -> dict:
        s = config.load()
        installed = llm.installed_models(s.llm)
        names = {m["name"] for m in installed}
        ram = catalog.total_ram_gb()
        for m in installed:
            m["reasoning"] = catalog.is_reasoning(m["name"])
        return {
            "backend": s.llm.backend,
            "server_ok": bool(installed) or bool(llm.list_models(s.llm)),
            "ram_gb": round(ram, 1),
            "installed": installed,
            "catalog": catalog.as_dicts(names, ram),
            "pulls": pulls.snapshot(),
        }

    @app.post("/api/models/pull", status_code=202)
    def pull(body: PullRequest) -> dict:
        name = body.name.strip().lower()
        if not llm.MODEL_NAME.match(name):
            raise HTTPException(400, "That doesn't look like a model name (e.g. qwen3:8b).")
        s = config.load()
        if s.llm.backend != "ollama":
            raise HTTPException(400, "Downloads work with Ollama only. Use your server's own UI.")
        pulls.start(s.llm, name)
        return {"name": name}

    @app.delete("/api/models/{name:path}", status_code=204)
    def remove_model(name: str) -> None:
        try:
            llm.delete_model(config.load().llm, name)
        except llm.LLMError as exc:
            raise HTTPException(400, str(exc)) from exc

    # -- fonts ------------------------------------------------------------------

    @app.get("/api/fonts")
    def font_list() -> list[dict]:
        return [{"id": f.id, "label": f.label, "family": f.family, "vibe": f.vibe}
                for f in fonts.FONTS.values()]

    @app.get("/api/fonts/{font_id}.ttf")
    def font_file(font_id: str) -> FileResponse:
        try:
            path = fonts.ensure(font_id)
        except fonts.FontError as exc:
            raise HTTPException(404, str(exc)) from exc
        return FileResponse(path, media_type="font/ttf",
                            headers={"Cache-Control": "public, max-age=31536000"})

    # -- uploads & jobs -----------------------------------------------------------

    @app.post("/api/upload", status_code=201)
    async def upload(file: UploadFile) -> dict:
        name = Path(file.filename or "video.mp4").name
        if Path(name).suffix.lower() not in VIDEO_EXTENSIONS:
            raise HTTPException(400, "Upload a video file (mp4, mov, mkv, webm).")
        safe = re.sub(r"[^A-Za-z0-9._-]+", "-", name)[-80:]
        dest = config.home() / "uploads" / f"{uuid.uuid4().hex[:8]}-{safe}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as out:
            while chunk := await file.read(8 * 1024 * 1024):
                out.write(chunk)
        return {"path": str(dest), "name": name}

    @app.get("/api/jobs")
    def jobs() -> list[dict]:
        return [_job_out(j) for j in pipeline.list_jobs()]

    @app.post("/api/jobs", status_code=201)
    def create_job(body: NewJob) -> dict:
        if not body.target.strip():
            raise HTTPException(400, "Paste a video link or upload a file.")
        job = pipeline.new_job(body.target, config.load())
        worker.submit(job)
        return _job_out(job)

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict:
        job = pipeline.load_job(job_id)
        if job is None:
            raise HTTPException(404, "No such job.")
        return _job_out(job)

    @app.post("/api/jobs/{job_id}/cancel", status_code=204)
    def cancel_job(job_id: str) -> None:
        worker.cancel(job_id)

    @app.delete("/api/jobs/{job_id}", status_code=204)
    def delete_job(job_id: str) -> None:
        job = pipeline.load_job(job_id)
        if job is None:
            raise HTTPException(404, "No such job.")
        if job.status in ("running", "queued"):
            raise HTTPException(409, "Cancel the job before deleting it.")
        shutil.rmtree(pipeline.job_dir(job_id), ignore_errors=True)

    @app.get("/api/jobs/{job_id}/clips/{index}")
    def clip_file(job_id: str, index: int, download: bool = False) -> FileResponse:
        job = pipeline.load_job(job_id)
        clip = next((c for c in (job.clips if job else []) if c.index == index), None)
        if clip is None:
            raise HTTPException(404, "No such clip.")
        root = pipeline.job_dir(job_id).resolve()
        path = (root / clip.file).resolve()
        if not path.is_relative_to(root) or not path.exists():
            raise HTTPException(404, "Clip file missing.")
        name = f"{_slug(clip.highlight.title) or 'clip'}-{index:02d}.mp4"
        return FileResponse(path, media_type="video/mp4", filename=name if download else None)

    @app.post("/api/jobs/{job_id}/open", status_code=204)
    def open_folder(job_id: str) -> None:
        folder = pipeline.job_dir(job_id) / "clips"
        if not folder.exists():
            raise HTTPException(404, "No clips yet.")
        if sys.platform == "darwin":
            subprocess.Popen(["open", str(folder)])
        elif sys.platform == "win32":
            subprocess.Popen(["explorer", str(folder)])
        else:
            subprocess.Popen(["xdg-open", str(folder)])

    app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
    return app


def _job_out(job: pipeline.Job) -> dict:
    data = job.model_dump(exclude={"settings"})
    data["model"] = job.settings.llm.model
    data["font"] = job.settings.render.caption_font
    if not data["title"] and not job.target.startswith("http"):
        data["title"] = Path(job.target).stem.split("-", 1)[-1]
    return data


def _slug(text: str) -> str:
    keep = "".join(c.lower() if c.isalnum() else "-" for c in text)
    return "-".join(p for p in keep.split("-") if p)[:50]
