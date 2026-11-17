"""Plain data types shared by every stage.

Everything here round-trips through JSON so each stage can cache its output in
the job folder and a re-run can skip finished work.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Word(BaseModel):
    text: str
    start: float
    end: float


class Sentence(BaseModel):
    """A run of words ending at punctuation or a pause. The unit the LLM picks."""

    text: str
    start: float
    end: float
    first_word: int
    last_word: int


class Transcript(BaseModel):
    language: str = ""
    words: list[Word] = Field(default_factory=list)
    sentences: list[Sentence] = Field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.words[-1].end if self.words else 0.0


class Highlight(BaseModel):
    """A clip the model picked, resolved to real timestamps."""

    first_sentence: int
    last_sentence: int
    start: float
    end: float
    title: str = ""
    score: int = 50
    reason: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start


class Shot(BaseModel):
    """A stretch of a clip with one fixed crop. ``x`` is the crop's left edge in
    source pixels; ``None`` means no face was found and the crop is centred."""

    start: float
    end: float
    x: int | None = None


class ClipResult(BaseModel):
    index: int
    highlight: Highlight
    file: str
    shots: list[Shot] = Field(default_factory=list)
