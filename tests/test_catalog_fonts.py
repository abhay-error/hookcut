from hookcut import catalog, fonts, llm


def test_catalog_tags_are_valid_and_local() -> None:
    tags = [m.tag for m in catalog.CATALOG]
    assert len(tags) == len(set(tags)), "no duplicates"
    for tag in tags:
        assert llm.MODEL_NAME.match(tag), tag
        assert "cloud" not in tag, "hookcut is local-only"


def test_catalog_covers_the_major_open_families() -> None:
    families = {m.family for m in catalog.CATALOG}
    assert {"Qwen", "Gemma", "Llama", "Nemotron", "GLM", "DeepSeek", "Mistral"} <= families


def test_fit_thresholds() -> None:
    m = catalog.BY_TAG["qwen3:14b"]  # needs ~10.8 GB
    assert catalog.fit(m, 16) == "good"
    assert catalog.fit(m, 12) == "tight"
    assert catalog.fit(m, 8) == "too-big"
    assert catalog.fit(m, 0) == "unknown"


def test_ram_is_detected() -> None:
    assert catalog.total_ram_gb() > 0.5


def test_reasoning_guess_for_unlisted_models() -> None:
    assert catalog.is_reasoning("deepseek-r1:70b")
    assert not catalog.is_reasoning("llama3.1:70b")


def test_fonts_are_unique_and_lookup_fails_clearly() -> None:
    assert len({f.family for f in fonts.FONTS.values()}) == len(fonts.FONTS)
    assert fonts.DEFAULT_FONT in fonts.FONTS
    try:
        fonts.get("comic-sans")
    except fonts.FontError as exc:
        assert "Choose from" in str(exc)
    else:
        raise AssertionError("expected FontError")
