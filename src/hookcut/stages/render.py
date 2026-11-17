"""Cut, crop, caption, and loudness-normalise one clip in a single ffmpeg pass."""

from __future__ import annotations

from pathlib import Path

from .. import ffmpeg
from ..config import RenderSettings
from ..models import Shot
from .reframe import crop_width

CAPTIONS_FILE = "captions.ass"


def build_filter(
    shots: list[Shot],
    src_w: int,
    src_h: int,
    settings: RenderSettings,
    *,
    captions: bool,
) -> str:
    """The filter graph: one trimmed+cropped segment per shot, joined, then captioned."""
    out_w, out_h = settings.width, settings.height
    cw = crop_width(src_w, src_h, out_w, out_h)
    ch = min(src_h, round(cw * out_h / out_w))
    ch -= ch % 2
    y = (src_h - ch) // 2

    parts: list[str] = []
    labels: list[str] = []
    for i, shot in enumerate(shots):
        x = shot.x if shot.x is not None else (src_w - cw) // 2
        x -= x % 2
        parts.append(
            f"[0:v]trim=start={shot.start:.3f}:end={shot.end:.3f},setpts=PTS-STARTPTS,"
            f"crop={cw}:{ch}:{x}:{y},scale={out_w}:{out_h}:flags=lanczos,setsar=1[v{i}]"
        )
        labels.append(f"[v{i}]")

    joined = "".join(labels) + f"concat=n={len(shots)}:v=1:a=0"
    if captions:
        parts.append(f"{joined}[vj]")
        parts.append(f"[vj]ass={CAPTIONS_FILE},format=yuv420p[vout]")
    else:
        parts.append(f"{joined},format=yuv420p[vout]")
    parts.append(f"[0:a]loudnorm=I={settings.loudness_lufs}:TP=-1.5:LRA=11,aresample=48000[aout]")
    return ";".join(parts)


def render_clip(
    source: Path,
    start: float,
    end: float,
    shots: list[Shot],
    src_w: int,
    src_h: int,
    out_file: Path,
    settings: RenderSettings,
    ass_text: str | None,
) -> Path:
    workdir = out_file.parent
    workdir.mkdir(parents=True, exist_ok=True)
    captions = bool(ass_text) and settings.captions
    if captions:
        if not ffmpeg.has_filter("ass"):
            raise ffmpeg.FFmpegError(
                "Your ffmpeg can't burn captions (no libass). Install a full build "
                "(macOS: `brew install ffmpeg-full`, then set HOOKCUT_FFMPEG to "
                "$(brew --prefix ffmpeg-full)/bin) or turn captions off."
            )
        # Captions are referenced by a relative name, with ffmpeg run from the
        # clip folder, which avoids Windows drive-letter escaping in filters.
        (workdir / CAPTIONS_FILE).write_text(ass_text, encoding="utf-8")

    graph = build_filter(shots, src_w, src_h, settings, captions=captions)
    encoder = ffmpeg.pick_encoder(settings.encoder)
    tmp = out_file.with_name(out_file.stem + ".tmp.mp4")
    ffmpeg.run(
        [
            "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}", "-i", str(source.resolve()),
            "-filter_complex", graph,
            "-map", "[vout]", "-map", "[aout]",
            *ffmpeg.encoder_args(encoder, settings.crf),
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            tmp.name,
        ],
        cwd=workdir,
    )
    tmp.replace(out_file)
    return out_file
