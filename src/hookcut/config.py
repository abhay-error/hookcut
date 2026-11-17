"""Settings. Defaults work out of the box; override per run or in config.json."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


def home() -> Path:
    """Where jobs and config live. Override with HOOKCUT_HOME."""
    return Path(os.environ.get("HOOKCUT_HOME", Path.home() / ".hookcut"))


class LLMSettings(BaseModel):
    #: "ollama" talks to Ollama's native API. "openai" talks to any
    #: OpenAI-compatible local server: LM Studio, llama.cpp, vLLM, Jan...
    backend: Literal["ollama", "openai"] = "ollama"
    base_url: str = "http://localhost:11434"
    model: str = "qwen3:14b"
    #: Tokens of context. Ollama's default is small and silently truncates.
    num_ctx: int = 16384
    #: Let reasoning models think first. Slower, often better picks.
    think: bool = False
    temperature: float = 0.3
    timeout_s: float = 900.0


class WhisperSettings(BaseModel):
    model: str = "large-v3-turbo"
    #: auto = mlx on Apple Silicon, faster-whisper (CUDA if present) elsewhere.
    backend: Literal["auto", "mlx", "faster-whisper"] = "auto"
    language: str = ""


class ClipSettings(BaseModel):
    max_clips: int = 5
    min_seconds: float = 15.0
    max_seconds: float = 75.0
    #: Minutes of transcript the model reads at once.
    window_minutes: float = 6.0


class RenderSettings(BaseModel):
    width: int = 1080
    height: int = 1920
    #: auto picks a hardware encoder when available, else libx264.
    encoder: str = "auto"
    crf: int = 20
    loudness_lufs: float = -14.0
    captions: bool = True
    #: highlight = spoken word lights up; box = text on a dark pill; plain = static.
    caption_style: Literal["highlight", "box", "plain"] = "highlight"
    #: Font id from hookcut.fonts.FONTS.
    caption_font: str = "poppins"
    #: Text height in pixels on a 1920-pixel-tall frame (scaled for other sizes).
    caption_size: int = Field(default=84, ge=32, le=200)
    #: Vertical centre of the caption, as a percentage from the top of the frame.
    caption_y: int = Field(default=72, ge=5, le=95)
    caption_color: str = Field(default="#FFFFFF", pattern=r"^#[0-9A-Fa-f]{6}$")
    highlight_color: str = Field(default="#FFE500", pattern=r"^#[0-9A-Fa-f]{6}$")
    uppercase: bool = True
    #: Words shown per caption line.
    words_per_caption: int = Field(default=3, ge=1, le=8)
    #: Track faces. Off = always centre-crop.
    face_tracking: bool = True


class Settings(BaseModel):
    llm: LLMSettings = Field(default_factory=LLMSettings)
    whisper: WhisperSettings = Field(default_factory=WhisperSettings)
    clips: ClipSettings = Field(default_factory=ClipSettings)
    render: RenderSettings = Field(default_factory=RenderSettings)
    #: yt-dlp: pull cookies from this browser to pass YouTube's bot check.
    cookies_from_browser: str = ""


def config_path() -> Path:
    return home() / "config.json"


def load() -> Settings:
    path = config_path()
    if path.exists():
        return Settings.model_validate_json(path.read_text(encoding="utf-8"))
    return Settings()


def save(settings: Settings) -> None:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings.model_dump(), indent=2), encoding="utf-8")
