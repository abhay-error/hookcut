"""Run a job end to end: ingest, transcribe, highlights, reframe, captions, render.

Layout on disk (under ~/.hookcut, or HOOKCUT_HOME):

    sources/<hash>/   source video, audio.wav, transcript.json, title.txt
                      shared by every job on the same video, so re-running
                      with another model skips download and transcription
    jobs/<id>/        job.json, highlights.json, clips/01.mp4 ...
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from . import ffmpeg
from .config import Settings, home
from .models import ClipResult, Highlight, Transcript
from .stages import captions, highlights, ingest, reframe, render, transcribe

log = logging.getLogger(__name__)

Stage = Literal["queued", "download", "transcribe", "highlights", "render", "done"]
Status = Literal["queued", "running", "done", "failed", "cancelled"]

STAGE_WEIGHTS: dict[str, tuple[float, float]] = {
    # stage: (progress at start, share of the bar)
    "download": (0.0, 0.10),
    "transcribe": (0.10, 0.25),
    "highlights": (0.35, 0.35),
    "render": (0.70, 0.30),
}


class Job(BaseModel):
    id: str
    target: str
    title: str = ""
    status: Status = "queued"
    stage: Stage = "queued"
    progress: float = 0.0
    message: str = ""
    error: str = ""
    created: float = Field(default_factory=time.time)
    settings: Settings = Field(default_factory=Settings)
    clips: list[ClipResult] = Field(default_factory=list)


class Cancelled(Exception):
    pass


def jobs_root() -> Path:
    return home() / "jobs"


def job_dir(job_id: str) -> Path:
    return jobs_root() / job_id


def source_dir(target: str) -> Path:
    key = target.strip()
    if not ingest.is_url(key):
        key = str(Path(key).expanduser().resolve())
    return home() / "sources" / hashlib.sha1(key.encode()).hexdigest()[:16]


def new_job(target: str, settings: Settings) -> Job:
    job = Job(id=uuid.uuid4().hex[:12], target=target.strip(), settings=settings)
    save_job(job)
    return job


def save_job(job: Job) -> None:
    folder = job_dir(job.id)
    folder.mkdir(parents=True, exist_ok=True)
    tmp = folder / "job.json.tmp"
    tmp.write_text(job.model_dump_json(indent=2), encoding="utf-8")
    tmp.replace(folder / "job.json")


def load_job(job_id: str) -> Job | None:
    path = job_dir(job_id) / "job.json"
    if not path.exists():
        return None
    return Job.model_validate_json(path.read_text(encoding="utf-8"))


def list_jobs() -> list[Job]:
    root = jobs_root()
    if not root.exists():
        return []
    jobs = [j for p in root.iterdir() if (j := load_job(p.name)) is not None]
    return sorted(jobs, key=lambda j: j.created, reverse=True)


class Runner:
    """Runs one job. ``on_update`` fires whenever visible state changes."""

    def __init__(
        self,
        job: Job,
        *,
        on_update: Callable[[Job], None] | None = None,
        cancel: threading.Event | None = None,
    ) -> None:
        self.job = job
        self.on_update = on_update
        self.cancel = cancel or threading.Event()
        self._last_save = 0.0

    # -- state ---------------------------------------------------------------

    def _update(self, force: bool = False, **fields) -> None:
        for key, value in fields.items():
            setattr(self.job, key, value)
        now = time.monotonic()
        if force or now - self._last_save > 0.5:
            save_job(self.job)
            self._last_save = now
            if self.on_update:
                self.on_update(self.job)

    def _stage(self, stage: Stage, message: str) -> None:
        if self.cancel.is_set():
            raise Cancelled
        self._update(force=True, stage=stage, message=message,
                     progress=STAGE_WEIGHTS[stage][0])

    def _progress(self, stage: str) -> Callable[[float], None]:
        base, share = STAGE_WEIGHTS[stage]

        def report(fraction: float) -> None:
            if self.cancel.is_set():
                raise Cancelled
            self._update(progress=base + share * max(0.0, min(1.0, fraction)))

        return report

    # -- run -----------------------------------------------------------------

    def run(self) -> Job:
        self._update(force=True, status="running", error="")
        try:
            self._run()
        except Cancelled:
            self._update(force=True, status="cancelled", message="Cancelled.")
        except Exception as exc:
            log.exception("Job %s failed", self.job.id)
            self._update(force=True, status="failed", error=str(exc),
                         message=f"Failed during {self.job.stage}.")
        return self.job

    def _run(self) -> None:
        s = self.job.settings
        src_dir = source_dir(self.job.target)
        src_dir.mkdir(parents=True, exist_ok=True)
        out_dir = job_dir(self.job.id)

        self._stage("download", "Getting the video")
        video, title = ingest.fetch(
            self.job.target, src_dir,
            cookies_from_browser=s.cookies_from_browser,
            on_progress=self._progress("download"),
        )
        self._update(force=True, title=title)

        self._stage("transcribe", "Transcribing speech")
        transcript_file = src_dir / f"transcript-{s.whisper.model.replace('/', '_')}.json"
        if transcript_file.exists():
            transcript = Transcript.model_validate_json(transcript_file.read_text("utf-8"))
        else:
            audio = transcribe.extract_audio(video, src_dir / "audio.wav")
            transcript = transcribe.transcribe(
                audio, s.whisper, on_progress=self._progress("transcribe")
            )
            transcript_file.write_text(transcript.model_dump_json(), encoding="utf-8")

        self._stage("highlights", f"Finding the best moments with {s.llm.model}")
        picks_file = out_dir / "highlights.json"
        if picks_file.exists():
            picks = [Highlight.model_validate(h)
                     for h in json.loads(picks_file.read_text("utf-8"))]
        else:
            picks = highlights.find_highlights(
                transcript, s.llm, s.clips, on_progress=self._progress("highlights")
            )
            picks_file.write_text(
                json.dumps([p.model_dump() for p in picks], indent=2), encoding="utf-8"
            )
        if not picks:
            self._update(force=True, status="done", stage="done", progress=1.0,
                         message="The model found no clip-worthy moments in this video.")
            return

        self._stage("render", f"Rendering {len(picks)} clips")
        src_w, src_h, _ = ffmpeg.video_size(video)
        report = self._progress("render")
        clips_dir = out_dir / "clips"
        results: list[ClipResult] = []
        for n, pick in enumerate(picks, start=1):
            if self.cancel.is_set():
                raise Cancelled
            self._update(message=f"Rendering clip {n} of {len(picks)}")
            out_file = clips_dir / f"{n:02d}.mp4"
            if s.render.face_tracking:
                try:
                    shots = reframe.analyse(video, pick.start, pick.end, src_w, src_h,
                                            s.render.width, s.render.height)
                except Exception as exc:  # face tracking is best-effort
                    log.warning("Face tracking failed (%s); centre-cropping.", exc)
                    shots = reframe.centre_plan(pick.duration)
            else:
                shots = reframe.centre_plan(pick.duration)
            ass = None
            if s.render.captions:
                ass = captions.build_ass(
                    captions.clip_words(transcript.words, pick.start, pick.end),
                    width=s.render.width, height=s.render.height,
                    per_line=s.render.words_per_caption,
                    style=s.render.caption_style, font=s.render.font,
                )
            render.render_clip(video, pick.start, pick.end, shots, src_w, src_h,
                               out_file, s.render, ass)
            results.append(ClipResult(index=n, highlight=pick,
                                      file=str(out_file.relative_to(out_dir)), shots=shots))
            self._update(force=True, clips=list(results))
            report(n / len(picks))

        self._update(force=True, status="done", stage="done", progress=1.0,
                     message=f"{len(results)} clips ready.")
