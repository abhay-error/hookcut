# hookcut

Turn long videos into short vertical clips — on your own computer, with open models.
No cloud, no API keys, no subscription.

Paste a YouTube link (or pick a video file). hookcut transcribes it, asks a local
LLM which moments would make strong Shorts / Reels / TikToks, then cuts each one
to 9:16 with the speaker kept in frame and captions burned in.

```
long video ──► transcribe ──► find highlights ──► reframe 9:16 ──► captions ──► clips/01.mp4 …
              (Whisper)       (Ollama / any       (face tracking,   (word-by-word
                               local LLM)          one crop per shot) highlight)
```

## Features

- **Fully local.** Whisper for speech, [Ollama](https://ollama.com) (or LM Studio,
  llama.cpp, vLLM — anything OpenAI-compatible) for picking clips.
- **Works with small models.** The model picks numbered sentences instead of
  timestamps, so it can't misplace a clip, and the prompt stays short.
- **Speaker stays in frame.** Detects camera cuts and centres each shot on the main
  face, with a steady crop per shot (no jittery panning).
- **Captions you control:** 13 open-licensed fonts (Poppins, Anton, Bebas Neue,
  Bangers, Luckiest Guy…), any size, any position, highlight / box / plain styles,
  custom colours, with a live preview. Normalised audio (−14 LUFS), 1080×1920 H.264.
- **29 open models, one click.** Qwen, Gemma, Llama, Nemotron, GLM, DeepSeek,
  Mistral, Phi, gpt-oss, Granite and OLMo. hookcut checks your memory and tells you
  which ones run well, then downloads them for you.
- **Fast on any machine:** Apple Silicon GPU (mlx-whisper), NVIDIA GPU (CUDA), or CPU.
- **Web app + CLI.** Re-running a video with a different model reuses the download
  and transcript.

## Install

You need **Python 3.10–3.12**, **ffmpeg with libass** (for captions), and **Ollama**.

**1. ffmpeg**

| OS | Command |
|---|---|
| macOS | `brew install ffmpeg-full` (Homebrew's plain `ffmpeg` has no libass; hookcut finds `ffmpeg-full` automatically) |
| Windows | `winget install Gyan.FFmpeg` (the "full" build) |
| Ubuntu/Debian | `sudo apt install ffmpeg` |

**2. Ollama and a model**

Install from [ollama.com](https://ollama.com), then pull a model that fits your memory:

| RAM / VRAM | Good choices |
|---|---|
| 8 GB | `qwen3:4b`, `nemotron-3-nano:4b`, `llama3.2:3b`, `phi4-mini:3.8b` |
| 16 GB | **`qwen3:14b`** (default), `gemma4:12b`, `deepseek-r1:14b`, `phi4:14b`, `glm4:9b` |
| 24 GB+ | `qwen3:30b`, `nemotron-3-nano:30b`, `gemma4:26b-a4b`, `gpt-oss:20b`, `mistral-small3.2:24b` |

Or skip this step and download models from the **Models** tab in the app.

```bash
ollama pull qwen3:14b
```

**3. hookcut**

```bash
pip install git+https://github.com/abhay-error/hookcut
hookcut doctor        # checks ffmpeg, captions, model server
```

## Use

**Web app**

```bash
hookcut serve
```

Opens <http://localhost:8765>:

- **Create**: paste a link or drop a file, choose the model, number and length of
  clips, and style the captions (font, size, position, colours) with a live preview.
- **Library**: follow progress and watch, download or open your clips.
- **Models**: see which open models fit your computer and download them.

**Command line**

```bash
hookcut clip "https://www.youtube.com/watch?v=..." -n 5
hookcut clip interview.mp4 --model qwen3:8b --think
hookcut clip talk.mp4 --no-captions --centre
hookcut clip talk.mp4 --font bangers --size 110 --position top --style box
hookcut models     # which open models fit this computer
hookcut fonts      # caption fonts
```

Clips land in `~/.hookcut/jobs/<id>/clips/`.

**Settings**

```bash
hookcut config                                  # show everything
hookcut config --set llm.model=qwen3:8b
hookcut config --set llm.backend=openai --set llm.base_url=http://localhost:1234/v1   # LM Studio
hookcut config --set cookies_from_browser=chrome   # fixes YouTube "confirm you're not a bot"
```

## How it picks clips

The transcript is split into ~6-minute overlapping windows. For each, the model
sees numbered sentences and returns the sentence ranges that would work as
standalone clips, scored 0–100 against a rubric: a hook in the first line, makes
sense without context, a real payoff, and tight. Clips are snapped to sentence
boundaries, fitted to your length limits, de-duplicated, and the top scores are
rendered. The prompt is in [`stages/highlights.py`](src/hookcut/stages/highlights.py) —
improvements welcome.

## Development

```bash
git clone https://github.com/abhay-error/hookcut && cd hookcut
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
```

The tests render real video with ffmpeg but stub out Whisper and the LLM, so
they run in seconds with no models downloaded.

## Please clip responsibly

Only clip videos you own or have permission to use. Downloading may be against
a site's terms; respect creators' rights.

## Credits

Inspired by [AutoClip](https://github.com/artbyjazi/autoclip). Built on
[Whisper](https://github.com/openai/whisper),
[faster-whisper](https://github.com/SYSTRAN/faster-whisper),
[mlx-whisper](https://github.com/ml-explore/mlx-examples),
[MediaPipe](https://github.com/google-ai-edge/mediapipe),
[yt-dlp](https://github.com/yt-dlp/yt-dlp), [ffmpeg](https://ffmpeg.org) and
[Ollama](https://ollama.com).

## License

MIT — see [LICENSE](LICENSE).
