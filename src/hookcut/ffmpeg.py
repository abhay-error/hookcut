"""Finding and running ffmpeg/ffprobe.

Set HOOKCUT_FFMPEG to a full path (or to the folder holding ffmpeg and ffprobe)
when the ffmpeg on PATH lacks libass. Homebrew's default ffmpeg does; install
``ffmpeg-full`` instead.
"""

from __future__ import annotations

import functools
import json
import os
import shutil
import subprocess
from pathlib import Path


class FFmpegError(RuntimeError):
    pass


def _resolve(name: str) -> str:
    override = os.environ.get("HOOKCUT_FFMPEG")
    if override:
        p = Path(override)
        if p.is_dir():
            for candidate in (p / name, p / f"{name}.exe"):
                if candidate.exists():
                    return str(candidate)
        elif name == "ffmpeg":
            return str(p)
        else:
            sibling = p.with_name(name + p.suffix)
            if sibling.exists():
                return str(sibling)
    found = shutil.which(name)
    if not found:
        raise FFmpegError(
            f"{name} was not found. Install ffmpeg (https://ffmpeg.org/download.html) "
            "or set HOOKCUT_FFMPEG to its location."
        )
    return found


def ffmpeg() -> str:
    return _resolve("ffmpeg")


def ffprobe() -> str:
    return _resolve("ffprobe")


@functools.cache
def has_filter(name: str) -> bool:
    out = subprocess.run(
        [ffmpeg(), "-hide_banner", "-filters"], capture_output=True, text=True, check=False
    ).stdout
    return any(line.split()[1:2] == [name] for line in out.splitlines() if line.strip())


@functools.cache
def encoders() -> set[str]:
    out = subprocess.run(
        [ffmpeg(), "-hide_banner", "-encoders"], capture_output=True, text=True, check=False
    ).stdout
    names = set()
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].startswith("V"):
            names.add(parts[1])
    return names


def _encoder_works(name: str) -> bool:
    """Listed is not the same as usable (e.g. nvenc without an NVIDIA GPU)."""
    result = subprocess.run(
        [ffmpeg(), "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
         "color=c=black:s=256x256:d=0.1", "-c:v", name, "-f", "null", "-"],
        capture_output=True, check=False,
    )
    return result.returncode == 0


@functools.cache
def pick_encoder(preference: str = "auto") -> str:
    if preference != "auto":
        return preference
    available = encoders()
    for name in ("h264_nvenc", "h264_videotoolbox", "h264_qsv"):
        if name in available and _encoder_works(name):
            return name
    return "libx264"


def encoder_args(name: str, crf: int) -> list[str]:
    if name == "libx264":
        return ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf)]
    # Hardware encoders: ~6 Mbps is plenty for 1080x1920 talking heads and
    # keeps a one-minute clip around 45 MB, under most upload limits.
    rate = ["-b:v", "6M", "-maxrate", "8M", "-bufsize", "12M"]
    if name == "h264_videotoolbox":
        return ["-c:v", name, *rate, "-allow_sw", "1"]
    if name == "h264_nvenc":
        return ["-c:v", name, "-preset", "p5", "-rc", "vbr", "-cq", str(crf), *rate]
    return ["-c:v", name, *rate]


def probe(path: Path) -> dict:
    result = subprocess.run(
        [ffprobe(), "-v", "error", "-print_format", "json", "-show_streams", "-show_format",
         str(path)],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        raise FFmpegError(f"ffprobe could not read {path}: {result.stderr.strip()[-500:]}")
    return json.loads(result.stdout)


def video_size(path: Path) -> tuple[int, int, float]:
    """Return (width, height, duration_s) of the first video stream."""
    info = probe(path)
    for stream in info.get("streams", []):
        if stream.get("codec_type") == "video":
            w, h = int(stream["width"]), int(stream["height"])
            # Phone footage stores rotation as metadata; swap to displayed size.
            rotation = 0
            for side in stream.get("side_data_list", []) or []:
                rotation = int(side.get("rotation", 0) or 0)
            if abs(rotation) in (90, 270):
                w, h = h, w
            duration = float(stream.get("duration") or info["format"].get("duration") or 0)
            return w, h, duration
    raise FFmpegError(f"No video stream in {path}")


def run(args: list[str], cwd: Path | None = None) -> None:
    result = subprocess.run(
        [ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", *args],
        capture_output=True, text=True, check=False, cwd=cwd,
    )
    if result.returncode != 0:
        raise FFmpegError(result.stderr.strip()[-2000:] or "ffmpeg failed")
