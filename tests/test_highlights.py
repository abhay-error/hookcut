import itertools

import pytest

from hookcut import llm
from hookcut.config import ClipSettings, LLMSettings
from hookcut.models import Highlight, Transcript
from hookcut.stages import highlights as hl

from .conftest import make_sentences


def test_windows_cover_everything_with_overlap() -> None:
    sentences = make_sentences(200)  # 1000 s
    spans = hl.windows(sentences, minutes=6)
    assert spans[0][0] == 0
    assert spans[-1][1] == 199
    for (a0, a1), (b0, _) in itertools.pairwise(spans):
        assert b0 <= a1, "windows must overlap"
        assert b0 > a0, "windows must advance"


def test_windows_empty_and_tiny() -> None:
    assert hl.windows([], 6) == []
    assert hl.windows(make_sentences(1), 6) == [(0, 0)]


def test_render_lines_numbers_and_times() -> None:
    lines = hl.render_lines(make_sentences(20), 12, 13)
    assert lines.splitlines() == ["[12] (01:00) Sentence 12.", "[13] (01:05) Sentence 13."]


CLIPS = ClipSettings(min_seconds=15, max_seconds=40)


def test_validate_resolves_times_and_fixes_order() -> None:
    sentences = make_sentences(50)
    raw = {"clips": [{"start": 8, "end": 4, "title": "T", "score": 80, "reason": "r"}]}
    [h] = hl.validate(raw, sentences, 0, 49, CLIPS)
    assert (h.first_sentence, h.last_sentence) == (4, 8)
    assert h.start == sentences[4].start and h.end == sentences[8].end


def test_validate_trims_too_long_and_extends_too_short() -> None:
    sentences = make_sentences(50)
    raw = {"clips": [
        {"start": 0, "end": 30, "title": "long", "score": 80, "reason": ""},
        {"start": 20, "end": 20, "title": "short", "score": 80, "reason": ""},
    ]}
    long_, short = hl.validate(raw, sentences, 0, 49, CLIPS)
    assert long_.duration <= CLIPS.max_seconds
    assert short.duration >= CLIPS.min_seconds * 0.8


def test_validate_drops_weak_bad_and_out_of_window() -> None:
    sentences = make_sentences(50)
    raw = {"clips": [
        {"start": 2, "end": 6, "score": 30},            # weak
        {"start": "x", "end": 6, "score": 90},          # junk
        {"start": 45, "end": 49, "score": 90},          # outside window 0..20
        "nonsense",
    ]}
    assert hl.validate(raw, sentences, 0, 20, CLIPS) == []


def test_validate_rejects_wrong_shape() -> None:
    with pytest.raises(llm.LLMError):
        hl.validate({"items": []}, make_sentences(5), 0, 4, CLIPS)


def _h(start: float, end: float, score: int) -> Highlight:
    return Highlight(first_sentence=0, last_sentence=0, start=start, end=end, score=score)


def test_pick_best_prefers_score_and_skips_duplicates() -> None:
    picks = hl.pick_best([_h(0, 30, 70), _h(5, 32, 90), _h(100, 130, 80), _h(200, 230, 60)], 2)
    assert [p.score for p in picks] == [90, 80]


def test_find_highlights_retries_bad_json(monkeypatch) -> None:
    sentences = make_sentences(30)
    transcript = Transcript(sentences=sentences)
    calls = []

    def fake_chat(settings, system, user, schema):
        calls.append(user)
        if len(calls) == 1:
            raise llm.LLMError("Model returned invalid JSON")
        return {"clips": [{"start": 2, "end": 6, "title": "Hook", "score": 88, "reason": "why"}]}

    monkeypatch.setattr(llm, "chat_json", fake_chat)
    picks = hl.find_highlights(transcript, LLMSettings(), ClipSettings(window_minutes=10))
    assert len(calls) == 2 and "rejected" in calls[1]
    assert picks[0].title == "Hook"


def test_find_highlights_stops_when_server_down(monkeypatch) -> None:
    def down(*a, **k):
        raise llm.LLMError("Could not reach the model server")

    monkeypatch.setattr(llm, "chat_json", down)
    with pytest.raises(llm.LLMError, match="Could not reach"):
        hl.find_highlights(Transcript(sentences=make_sentences(10)), LLMSettings(), ClipSettings())
