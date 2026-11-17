"""Speech to words with timestamps, grouped into sentences.

Two engines behind one function: mlx-whisper on Apple Silicon (Metal GPU) and
faster-whisper everywhere else (CUDA when available, otherwise CPU int8).
"""

from __future__ import annotations

import logging
import platform
import sys
import wave
from collections.abc import Callable
from pathlib import Path

from .. import ffmpeg
from ..config import WhisperSettings
from ..models import Sentence, Transcript, Word

log = logging.getLogger(__name__)

#: A pause this long ends a sentence even without punctuation.
SENTENCE_GAP_S = 0.8
#: Unpunctuated speech still gets split so the LLM has fine-grained choices.
MAX_SENTENCE_WORDS = 40


class TranscribeError(RuntimeError):
    pass


def extract_audio(video: Path, out: Path) -> Path:
    if not out.exists():
        tmp = out.with_suffix(".tmp.wav")
        ffmpeg.run(["-i", str(video), "-vn", "-ac", "1", "-ar", "16000", str(tmp)])
        tmp.replace(out)
    return out


def pick_backend(settings: WhisperSettings) -> str:
    if settings.backend != "auto":
        return settings.backend
    if sys.platform == "darwin" and platform.machine() == "arm64":
        try:
            import mlx_whisper  # noqa: F401

            return "mlx"
        except ImportError:
            pass
    return "faster-whisper"


def mlx_repo(model: str) -> str:
    if "/" in model:
        return model
    if model in ("large-v3-turbo", "turbo"):
        return "mlx-community/whisper-large-v3-turbo"
    return f"mlx-community/whisper-{model}-mlx"


def load_wav(path: Path):
    """Read our 16 kHz mono 16-bit WAV as float32 samples in [-1, 1].

    Handing mlx-whisper samples (not a path) stops it from shelling out to
    whatever ffmpeg is on PATH, which may not be the one the user configured.
    """
    import numpy as np

    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2 or wav.getframerate() != 16000:
            raise TranscribeError(f"{path.name} must be 16 kHz mono 16-bit PCM")
        frames = wav.readframes(wav.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0


def _raw_words_mlx(audio: Path, settings: WhisperSettings) -> tuple[list[Word], str]:
    import mlx_whisper

    result = mlx_whisper.transcribe(
        load_wav(audio),
        path_or_hf_repo=mlx_repo(settings.model),
        language=settings.language or None,
        word_timestamps=True,
        condition_on_previous_text=False,
        verbose=None,
    )
    words = [
        Word(text=w["word"].strip(), start=float(w["start"]), end=float(w["end"]))
        for seg in result.get("segments", [])
        for w in seg.get("words", []) or []
        if w.get("word", "").strip()
    ]
    return words, result.get("language", "")


def _cuda_available() -> bool:
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def _raw_words_faster(
    audio: Path, settings: WhisperSettings, on_progress: Callable[[float], None] | None
) -> tuple[list[Word], str]:
    from faster_whisper import WhisperModel

    device = "cuda" if _cuda_available() else "cpu"
    compute = "float16" if device == "cuda" else "int8"
    log.info("faster-whisper on %s (%s)", device, compute)
    model = WhisperModel(settings.model, device=device, compute_type=compute)
    segments, info = model.transcribe(
        str(audio),
        language=settings.language or None,
        word_timestamps=True,
        vad_filter=True,
        condition_on_previous_text=False,
    )
    total = float(getattr(info, "duration", 0) or 0)
    words: list[Word] = []
    for seg in segments:
        for w in seg.words or []:
            if w.word.strip():
                words.append(Word(text=w.word.strip(), start=float(w.start), end=float(w.end)))
        if on_progress and total:
            on_progress(min(0.99, seg.end / total))
    return words, getattr(info, "language", "")


def group_sentences(words: list[Word]) -> list[Sentence]:
    """Split words into sentences at end punctuation, long pauses, or a length cap."""
    sentences: list[Sentence] = []
    start = 0
    for i, word in enumerate(words):
        last = i == len(words) - 1
        ends_punct = word.text.rstrip("\"')]").endswith((".", "?", "!"))
        gap = (words[i + 1].start - word.end) if not last else 0.0
        too_long = i - start + 1 >= MAX_SENTENCE_WORDS
        if last or ends_punct or gap >= SENTENCE_GAP_S or too_long:
            chunk = words[start : i + 1]
            sentences.append(
                Sentence(
                    text=" ".join(w.text for w in chunk),
                    start=chunk[0].start,
                    end=chunk[-1].end,
                    first_word=start,
                    last_word=i,
                )
            )
            start = i + 1
    return sentences


def transcribe(
    audio: Path,
    settings: WhisperSettings,
    *,
    on_progress: Callable[[float], None] | None = None,
) -> Transcript:
    backend = pick_backend(settings)
    log.info("Transcribing with %s (%s)", backend, settings.model)
    try:
        if backend == "mlx":
            words, language = _raw_words_mlx(audio, settings)
        else:
            words, language = _raw_words_faster(audio, settings, on_progress)
    except Exception as exc:
        raise TranscribeError(f"Transcription failed ({backend}): {exc}") from exc

    if not words:
        raise TranscribeError("No speech was found in this video.")
    if on_progress:
        on_progress(1.0)
    return Transcript(language=language, words=words, sentences=group_sentences(words))
