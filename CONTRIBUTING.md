# Contributing to hookcut

Thanks for helping! hookcut is a small, readable Python codebase. Most changes touch one file.

## Setup

```bash
git clone https://github.com/abhay-error/hookcut && cd hookcut
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest          # fast: Whisper and the LLM are stubbed
.venv/bin/ruff check src tests
.venv/bin/hookcut serve   # run the app
```

You need ffmpeg with libass on your PATH (see the README).

## Where things live

| Path | What it does |
|---|---|
| `src/hookcut/pipeline.py` | Runs a job: ingest → transcribe → highlights → reframe → render |
| `src/hookcut/stages/highlights.py` | **The prompt** and clip selection. Highest-impact place to improve |
| `src/hookcut/stages/reframe.py` | Cut detection + face-centred crops |
| `src/hookcut/stages/captions.py` | ASS subtitle generation (styles, position, colours) |
| `src/hookcut/stages/render.py` | The single ffmpeg pass |
| `src/hookcut/catalog.py` | The list of open models shown in the app |
| `src/hookcut/fonts.py` | Caption fonts |
| `src/hookcut/web/static/index.html` | The whole web UI (no build step) |
| `src/hookcut/cli.py` | Command line |

## Good first issues

- **Add a caption font:** add one line to `fonts.py` (single-weight OFL font from
  google/fonts; `family` must match the name inside the file).
- **Add a model to the catalog:** add a line to `catalog.py`, after checking that the tag
  exists on ollama.com and isn't cloud-only.
- **Improve the prompt:** run the same video through `main` and your branch and share
  the clip lists in your PR.
- **New caption style:** extend `captions.build_ass` and the style picker in `index.html`.

## Pull requests

- Keep PRs focused; include a test for new behaviour.
- `pytest` and `ruff check` must pass (CI runs them on macOS, Windows and Linux).
- For UI changes, add a screenshot in light and dark mode.
