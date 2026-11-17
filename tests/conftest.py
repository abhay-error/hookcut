import pytest

from hookcut.models import Sentence, Word


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """Every test gets its own HOOKCUT_HOME so nothing touches ~/.hookcut."""
    monkeypatch.setenv("HOOKCUT_HOME", str(tmp_path / "home"))
    return tmp_path / "home"


def make_sentences(n: int, seconds_each: float = 5.0) -> list[Sentence]:
    return [
        Sentence(text=f"Sentence {i}.", start=i * seconds_each, end=(i + 1) * seconds_each - 0.2,
                 first_word=i * 2, last_word=i * 2 + 1)
        for i in range(n)
    ]


def make_words(texts: list[str], step: float = 0.4) -> list[Word]:
    return [Word(text=t, start=i * step, end=i * step + step * 0.8) for i, t in enumerate(texts)]
