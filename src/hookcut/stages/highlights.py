"""Ask a local LLM which stretches of the transcript would make strong short clips.

The model sees numbered sentences and answers with sentence numbers, never
timestamps. Real times are then looked up from Whisper's word timings, so a
model that is bad at arithmetic can't misplace a clip. Sentences (rather than
words) keep the prompt short, which matters a lot for small local models.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from .. import llm
from ..config import ClipSettings, LLMSettings
from ..models import Highlight, Sentence, Transcript

log = logging.getLogger(__name__)

PER_WINDOW = 3
OVERLAP_S = 60.0

SYSTEM_PROMPT = """\
You are a sharp short-form video editor. You read transcripts of long videos and \
find the moments that would work as standalone vertical clips on TikTok, Reels \
and YouTube Shorts.

A strong clip:
1. Opens with a hook. The first sentence grabs attention by itself: a bold claim, \
a question, a surprising fact, conflict, or emotion. Never start on filler \
("so", "um", "anyway", "like I said").
2. Makes sense with no context. A stranger scrolling past understands it. No \
dangling "he", "that thing", or references to earlier parts of the video.
3. Pays off. It ends on a punchline, insight, answer, or emotional beat, on a \
finished thought, not mid-story.
4. Is tight. Cut setup and rambling. Shorter and stronger beats longer.

Score each clip 0-100 honestly:
- 85-100: would likely go viral, a scroll-stopper
- 70-84: good, worth posting
- 50-69: fine but forgettable
- under 50: do not return it
Most moments are not clip-worthy. Returning zero clips is a valid answer. Do not \
give everything the same score.

Answer with JSON only."""

USER_TEMPLATE = """\
Transcript excerpt. Each line is a sentence: [number] (mm:ss) text

{lines}

Pick up to {per_window} clips. Each clip is a continuous run of sentences from \
"start" to "end" (inclusive sentence numbers from the list above), lasting \
between {min_s:.0f} and {max_s:.0f} seconds.

Return: {{"clips": [{{"start": <first sentence number>, "end": <last sentence \
number>, "title": "<catchy title, under 60 characters>", "score": <0-100>, \
"reason": "<one sentence: why it works>"}}]}}"""

SCHEMA = {
    "type": "object",
    "properties": {
        "clips": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start": {"type": "integer"},
                    "end": {"type": "integer"},
                    "title": {"type": "string"},
                    "score": {"type": "integer"},
                    "reason": {"type": "string"},
                },
                "required": ["start", "end", "title", "score", "reason"],
            },
        }
    },
    "required": ["clips"],
}


def windows(sentences: list[Sentence], minutes: float) -> list[tuple[int, int]]:
    """Split sentence indices into overlapping windows of about ``minutes``.

    Overlap stops a great moment from being cut in half at a window edge.
    """
    if not sentences:
        return []
    span = minutes * 60
    out: list[tuple[int, int]] = []
    first = 0
    while first < len(sentences):
        t0 = sentences[first].start
        last = first
        while last + 1 < len(sentences) and sentences[last + 1].end - t0 <= span:
            last += 1
        out.append((first, last))
        if last == len(sentences) - 1:
            break
        # Next window starts OVERLAP_S before this one ended.
        cutoff = sentences[last].end - OVERLAP_S
        nxt = last
        while nxt > first + 1 and sentences[nxt - 1].start >= cutoff:
            nxt -= 1
        first = max(first + 1, nxt)
    return out


def _mmss(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def render_lines(sentences: list[Sentence], first: int, last: int) -> str:
    return "\n".join(
        f"[{i}] ({_mmss(sentences[i].start)}) {sentences[i].text}" for i in range(first, last + 1)
    )


def fit_to_length(
    sentences: list[Sentence], start: int, end: int, clips: ClipSettings
) -> tuple[int, int] | None:
    """Trim or extend a sentence range to fit the length limits; None if impossible."""
    while end > start and sentences[end].end - sentences[start].start > clips.max_seconds:
        end -= 1
    while (
        sentences[end].end - sentences[start].start < clips.min_seconds
        and end + 1 < len(sentences)
        and sentences[end + 1].end - sentences[start].start <= clips.max_seconds
    ):
        end += 1
    duration = sentences[end].end - sentences[start].start
    if duration > clips.max_seconds or duration < clips.min_seconds * 0.8:
        return None
    return start, end


def validate(
    raw: object, sentences: list[Sentence], first: int, last: int, clips: ClipSettings
) -> list[Highlight]:
    if not isinstance(raw, dict) or not isinstance(raw.get("clips"), list):
        raise llm.LLMError('Expected {"clips": [...]}.')
    out: list[Highlight] = []
    for item in raw["clips"]:
        if not isinstance(item, dict):
            continue
        try:
            start, end = int(item["start"]), int(item["end"])
            score = int(float(item.get("score", 50)))
        except (KeyError, TypeError, ValueError):
            continue
        if start > end:
            start, end = end, start
        start, end = max(start, first), min(end, last)
        if start > end or score < 50:
            continue
        fitted = fit_to_length(sentences, start, end, clips)
        if fitted is None:
            continue
        start, end = fitted
        out.append(
            Highlight(
                first_sentence=start,
                last_sentence=end,
                start=sentences[start].start,
                end=sentences[end].end,
                title=str(item.get("title", ""))[:80].strip(),
                score=max(0, min(100, score)),
                reason=str(item.get("reason", "")).strip(),
            )
        )
    return out


def overlap_ratio(a: Highlight, b: Highlight) -> float:
    inter = min(a.end, b.end) - max(a.start, b.start)
    return max(0.0, inter) / min(a.duration, b.duration)


def pick_best(candidates: list[Highlight], max_clips: int) -> list[Highlight]:
    """Highest score first, skipping anything that mostly repeats a kept clip."""
    kept: list[Highlight] = []
    for cand in sorted(candidates, key=lambda h: (-h.score, h.start)):
        if all(overlap_ratio(cand, k) < 0.3 for k in kept):
            kept.append(cand)
        if len(kept) >= max_clips:
            break
    return kept


def find_highlights(
    transcript: Transcript,
    llm_settings: LLMSettings,
    clips: ClipSettings,
    *,
    on_progress: Callable[[float], None] | None = None,
) -> list[Highlight]:
    sentences = transcript.sentences
    spans = windows(sentences, clips.window_minutes)
    candidates: list[Highlight] = []
    errors: list[str] = []

    for n, (first, last) in enumerate(spans):
        user = USER_TEMPLATE.format(
            lines=render_lines(sentences, first, last),
            per_window=PER_WINDOW,
            min_s=clips.min_seconds,
            max_s=clips.max_seconds,
        )
        found: list[Highlight] | None = None
        for attempt in range(2):
            try:
                raw = llm.chat_json(llm_settings, SYSTEM_PROMPT, user, SCHEMA)
                found = validate(raw, sentences, first, last, clips)
                break
            except llm.LLMError as exc:
                if "Could not reach" in str(exc) or "has no model" in str(exc):
                    raise
                log.warning("Window %d attempt %d failed: %s", n, attempt + 1, exc)
                errors.append(str(exc))
                user += f"\n\nYour previous answer was rejected: {exc}. Reply with valid JSON only."
        candidates.extend(found or [])
        if on_progress:
            on_progress((n + 1) / len(spans))

    if not candidates and errors and len(errors) >= len(spans):
        raise llm.LLMError(f"The model never returned usable clips. Last error: {errors[-1]}")
    return pick_best(candidates, clips.max_clips)
