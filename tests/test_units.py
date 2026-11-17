"""Small pure functions: JSON parsing, sentences, captions, reframing, filters."""

import pytest

from hookcut import llm
from hookcut.config import RenderSettings, WhisperSettings
from hookcut.models import Shot
from hookcut.stages import captions, reframe, render, transcribe

from .conftest import make_words

# -- llm ---------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    '{"clips": []}',
    '<think>let me see {not json}</think>\n{"clips": []}',
    '```json\n{"clips": []}\n```',
    'Sure! Here you go: {"clips": []} Hope that helps.',
])
def test_parse_json_tolerates_wrappers(text) -> None:
    assert llm.parse_json(text) == {"clips": []}


def test_parse_json_rejects_garbage() -> None:
    with pytest.raises(llm.LLMError):
        llm.parse_json("no json here")


# -- transcribe --------------------------------------------------------------

def test_sentences_split_on_punctuation_and_pauses() -> None:
    words = make_words(["Hello", "there.", "How", "are", "you?", "Fine"])
    words[5].start, words[5].end = 10.0, 10.3  # long pause before "Fine"
    sentences = transcribe.group_sentences(words)
    assert [s.text for s in sentences] == ["Hello there.", "How are you?", "Fine"]
    assert sentences[1].first_word == 2 and sentences[1].last_word == 4


def test_sentences_capped_in_length() -> None:
    words = make_words(["word"] * 100, step=0.2)
    sentences = transcribe.group_sentences(words)
    assert max(s.last_word - s.first_word + 1 for s in sentences) <= transcribe.MAX_SENTENCE_WORDS
    assert sentences[-1].last_word == 99


def test_mlx_repo_names() -> None:
    assert transcribe.mlx_repo("large-v3-turbo") == "mlx-community/whisper-large-v3-turbo"
    assert transcribe.mlx_repo("small") == "mlx-community/whisper-small-mlx"
    assert transcribe.mlx_repo("org/model") == "org/model"


def test_load_wav_reads_16k_mono(tmp_path) -> None:
    import wave

    import numpy as np

    path = tmp_path / "a.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(np.array([0, 16384, -32768], dtype=np.int16).tobytes())
    samples = transcribe.load_wav(path)
    assert samples.dtype == np.float32
    assert list(samples) == [0.0, 0.5, -1.0]


def test_explicit_backend_is_respected() -> None:
    assert transcribe.pick_backend(WhisperSettings(backend="faster-whisper")) == "faster-whisper"


# -- captions ----------------------------------------------------------------

def test_ass_time_format() -> None:
    assert captions.ass_time(0) == "0:00:00.00"
    assert captions.ass_time(3723.456) == "1:02:03.46"


def test_clip_words_are_retimed_to_clip_start() -> None:
    words = make_words(["a", "b", "c", "d"], step=1.0)  # starts 0,1,2,3
    inside = captions.clip_words(words, 1.0, 3.0)
    assert [w.text for w in inside] == ["b", "c"]
    assert inside[0].start == 0.0


def test_grouping_respects_line_length_and_sentences() -> None:
    words = make_words(["one", "two", "three", "four", "end.", "next"])
    lines = captions.group(words, per_line=3)
    texts = [[w.text for w in line] for line in lines]
    assert texts == [["one", "two", "three"], ["four", "end."], ["next"]]


def test_highlight_style_makes_one_event_per_word() -> None:
    words = make_words(["hi", "{there}", "friend"])
    ass = captions.build_ass(words, per_line=3, style="highlight")
    events = [line for line in ass.splitlines() if line.startswith("Dialogue:")]
    assert len(events) == 3
    assert "(THERE)" in events[0], "braces must be escaped so they aren't read as tags"
    assert captions.HIGHLIGHT in events[1]


def test_plain_style_makes_one_event_per_line() -> None:
    ass = captions.build_ass(make_words(["a", "b", "c", "d"]), per_line=2, style="plain")
    assert sum(line.startswith("Dialogue:") for line in ass.splitlines()) == 2


# -- reframe -----------------------------------------------------------------

def test_crop_width_for_landscape_and_portrait() -> None:
    assert reframe.crop_width(1920, 1080, 1080, 1920) == 608
    assert reframe.crop_width(1080, 1920, 1080, 1920) == 1080


def test_crop_left_is_clamped_and_even() -> None:
    assert reframe.crop_left(10, 1920, 608) == 0
    assert reframe.crop_left(1910, 1920, 608) == 1312
    assert reframe.crop_left(961, 1920, 608) % 2 == 0


def test_short_shots_are_merged() -> None:
    shots = reframe.split_shots([5.0, 5.3, 12.0], 20.0)
    assert shots == [(0.0, 5.3), (5.3, 12.0), (12.0, 20.0)]


def test_plan_uses_median_face_and_falls_back_to_centre() -> None:
    faces = [(t / 4, 1500.0) for t in range(40)] + [(10 + t / 4, None) for t in range(40)]
    plan = reframe.plan_shots([(0.0, 10.0), (10.0, 20.0)], faces, 1920, 608)
    assert plan[0].x == reframe.crop_left(1500, 1920, 608)
    assert plan[1].x is None


def test_plan_merges_neighbours_with_same_framing() -> None:
    faces = [(t / 4, 900.0) for t in range(80)]
    plan = reframe.plan_shots([(0.0, 10.0), (10.0, 20.0)], faces, 1920, 608)
    assert len(plan) == 1 and plan[0].end == 20.0


# -- render ------------------------------------------------------------------

def test_filter_has_one_segment_per_shot() -> None:
    shots = [Shot(start=0, end=5, x=100), Shot(start=5, end=9, x=None)]
    graph = render.build_filter(shots, 1920, 1080, RenderSettings(), captions=True)
    assert graph.count("trim=") == 2
    assert "crop=608:1080:100:0" in graph
    assert "crop=608:1080:656:0" in graph  # centred
    assert "concat=n=2" in graph and "ass=captions.ass" in graph
    assert "loudnorm=I=-14.0" in graph
