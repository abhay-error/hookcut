"""Get the source video into the job folder: a YouTube (or any yt-dlp) URL, or a local file."""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

URL_RE = re.compile(r"^https?://", re.IGNORECASE)


class IngestError(RuntimeError):
    pass


def is_url(target: str) -> bool:
    return bool(URL_RE.match(target.strip()))


def fetch(
    target: str,
    job_dir: Path,
    *,
    cookies_from_browser: str = "",
    on_progress: Callable[[float], None] | None = None,
) -> tuple[Path, str]:
    """Return (path to source video, title)."""
    if not is_url(target):
        path = Path(target).expanduser().resolve()
        if not path.is_file():
            raise IngestError(f"File not found: {path}")
        return path, path.stem

    existing = next(job_dir.glob("source.*"), None)
    if existing and existing.suffix != ".part":
        title_file = job_dir / "title.txt"
        title = title_file.read_text(encoding="utf-8") if title_file.exists() else "video"
        return existing, title

    import yt_dlp

    def hook(d: dict) -> None:
        if on_progress and d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            if total:
                on_progress(min(0.99, d.get("downloaded_bytes", 0) / total))

    opts = {
        "format": "bv*[height<=1080]+ba/b[height<=1080]/b",
        "merge_output_format": "mp4",
        "outtmpl": str(job_dir / "source.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "progress_hooks": [hook],
        # Stalled connections are the most common failure; retry instead of hanging.
        "socket_timeout": 30,
        "retries": 10,
        "fragment_retries": 10,
    }
    if cookies_from_browser:
        opts["cookiesfrombrowser"] = (cookies_from_browser,)

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(target, download=True)
    except Exception as exc:  # yt-dlp raises many types
        message = str(exc)
        hint = ""
        if "Sign in" in message or "bot" in message.lower():
            hint = (
                "\nYouTube wants proof you're not a bot. Set cookies_from_browser "
                "(e.g. 'chrome') and close that browser, then retry."
            )
        raise IngestError(f"Download failed: {message}{hint}") from exc

    title = info.get("title") or "video"
    (job_dir / "title.txt").write_text(title, encoding="utf-8")
    path = next(p for p in job_dir.glob("source.*") if p.suffix != ".part")
    return path, title
