"""Burned-in captions as an ASS subtitle file.

Words are grouped into short lines (a few words, broken at pauses and sentence
ends). In the "highlight" style the word being spoken is coloured and slightly
enlarged, the look viewers expect from short-form video.
"""

from __future__ import annotations

from ..models import Word

#: ASS colours are &HAABBGGRR.
WHITE = "&H00FFFFFF"
BLACK = "&H00000000"
HIGHLIGHT = "&H0000E5FF"  # warm yellow
BREAK_GAP_S = 0.6
#: Bridge tiny gaps between lines so captions don't flicker.
HOLD_S = 0.35


def ass_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    cs = round(seconds * 100)
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    s, cs = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("{", "(").replace("}", ")").replace("\n", " ")


def clip_words(words: list[Word], start: float, end: float) -> list[Word]:
    """Words inside [start, end], re-timed so the clip starts at 0."""
    return [
        Word(text=w.text, start=max(0.0, w.start - start), end=min(end, w.end) - start)
        for w in words
        if w.end > start and w.start < end
    ]


def group(words: list[Word], per_line: int) -> list[list[Word]]:
    lines: list[list[Word]] = []
    current: list[Word] = []
    for i, word in enumerate(words):
        current.append(word)
        nxt = words[i + 1] if i + 1 < len(words) else None
        sentence_end = word.text.rstrip("\"')]").endswith((".", "?", "!", ","))
        pause = nxt is not None and nxt.start - word.end > BREAK_GAP_S
        if len(current) >= per_line or sentence_end or pause or nxt is None:
            lines.append(current)
            current = []
    return lines


def build_ass(
    words: list[Word],
    *,
    width: int = 1080,
    height: int = 1920,
    per_line: int = 3,
    style: str = "highlight",
    font: str = "Arial",
    uppercase: bool = True,
) -> str:
    size = round(height * 0.045)
    outline = max(2, round(size * 0.08))
    margin_v = round(height * 0.24)
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, \
Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, \
Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font},{size},{WHITE},{WHITE},{BLACK},&H80000000,-1,0,0,0,100,100,0,0,1,\
{outline},{max(1, outline // 2)},2,{round(width * 0.08)},{round(width * 0.08)},{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events: list[str] = []
    lines = group(words, per_line)
    for n, line in enumerate(lines):
        texts = [escape(w.text.upper() if uppercase else w.text) for w in line]
        line_end = line[-1].end
        if n + 1 < len(lines) and lines[n + 1][0].start - line_end < HOLD_S:
            line_end = lines[n + 1][0].start

        if style == "plain":
            events.append(_event(line[0].start, line_end, " ".join(texts)))
            continue

        for i, word in enumerate(line):
            w_start = line[0].start if i == 0 else word.start
            w_end = line[i + 1].start if i + 1 < len(line) else line_end
            if w_end <= w_start:
                continue
            parts = [
                f"{{\\c{HIGHLIGHT}&\\fscx110\\fscy110}}{t}{{\\r}}" if j == i else t
                for j, t in enumerate(texts)
            ]
            events.append(_event(w_start, w_end, " ".join(parts)))
    return header + "\n".join(events) + "\n"


def _event(start: float, end: float, text: str) -> str:
    return f"Dialogue: 0,{ass_time(start)},{ass_time(end)},Default,,0,0,0,,{text}"
