"""Local web app: a tiny JSON API plus a single static page.

Jobs run one at a time on a background thread (transcription and LLMs are
heavy; running two at once just makes both slow).
"""

from __future__ import annotations

import queue
import shutil
import subprocess
import sys
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .. import config, llm, pipeline

STATIC = Path(__file__).parent / "static"


class NewJob(BaseModel):
    target: str
    max_clips: int | None = None
    model: str | None = None
    think: bool | None = None
    captions: bool | None = None


class Worker:
    def __init__(self) -> None:
        self.queue: queue.Queue[str] = queue.Queue()
        self.cancels: dict[str, threading.Event] = {}
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

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
                job.status = "cancelled"
                pipeline.save_job(job)
                continue
            pipeline.Runner(job, cancel=cancel).run()


def create_app() -> FastAPI:
    app = FastAPI(title="hookcut", docs_url=None, redoc_url=None)
    worker = Worker()

    # Jobs interrupted by a restart can't resume mid-stage; mark them clearly.
    for job in pipeline.list_jobs():
        if job.status in ("running", "queued"):
            job.status, job.message = "failed", "Interrupted (the app was stopped)."
            job.error = "Run it again; downloads and transcripts are reused."
            pipeline.save_job(job)

    @app.get("/api/settings")
    def get_settings() -> dict:
        s = config.load()
        return {"settings": s.model_dump(), "models": llm.list_models(s.llm)}

    @app.put("/api/settings")
    def put_settings(payload: dict) -> dict:
        merged = config.load().model_dump()
        for section, values in payload.items():
            if section in merged and isinstance(values, dict):
                merged[section].update(values)
            elif section in merged:
                merged[section] = values
        try:
            settings = config.Settings.model_validate(merged)
        except Exception as exc:
            raise HTTPException(400, f"Invalid settings: {exc}") from exc
        config.save(settings)
        return {"settings": settings.model_dump(), "models": llm.list_models(settings.llm)}

    @app.get("/api/jobs")
    def jobs() -> list[dict]:
        return [j.model_dump(exclude={"settings"}) for j in pipeline.list_jobs()]

    @app.post("/api/jobs", status_code=201)
    def create_job(body: NewJob) -> dict:
        if not body.target.strip():
            raise HTTPException(400, "Paste a video link or a file path.")
        s = config.load()
        if body.max_clips:
            s.clips.max_clips = max(1, min(20, body.max_clips))
        if body.model:
            s.llm.model = body.model
        if body.think is not None:
            s.llm.think = body.think
        if body.captions is not None:
            s.render.captions = body.captions
        job = pipeline.new_job(body.target, s)
        worker.submit(job)
        return job.model_dump(exclude={"settings"})

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict:
        job = pipeline.load_job(job_id)
        if job is None:
            raise HTTPException(404, "No such job.")
        return job.model_dump(exclude={"settings"})

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
        path = (pipeline.job_dir(job_id) / clip.file).resolve()
        if not path.is_relative_to(pipeline.job_dir(job_id).resolve()) or not path.exists():
            raise HTTPException(404, "Clip file missing.")
        name = f"{_slug(clip.highlight.title) or 'clip'}-{index:02d}.mp4"
        return FileResponse(path, media_type="video/mp4",
                            filename=name if download else None)

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


def _slug(text: str) -> str:
    keep = "".join(c.lower() if c.isalnum() else "-" for c in text)
    return "-".join(p for p in keep.split("-") if p)[:50]
