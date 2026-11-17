"""Real ffmpeg + real pipeline on a generated video, with transcription and the
LLM stubbed out (so the test needs no models and runs in seconds)."""

import json
import shutil

import pytest
from fastapi.testclient import TestClient

from hookcut import ffmpeg, llm, pipeline
from hookcut.config import Settings
from hookcut.models import Transcript
from hookcut.stages import transcribe

from .conftest import make_words

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None and not __import__("os").environ.get(
    "HOOKCUT_FFMPEG"), reason="ffmpeg not installed")


@pytest.fixture
def sample_video(tmp_path):
    path = tmp_path / "talk.mp4"
    ffmpeg.run([
        "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30:duration=40",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=40",
        "-shortest", "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", str(path),
    ])
    return path


@pytest.fixture
def stub_models(monkeypatch):
    texts = [f"word{i}." if i % 6 == 5 else f"word{i}" for i in range(90)]
    words = make_words(texts, step=0.4)  # 36 s of speech

    def fake_transcribe(audio, settings, on_progress=None):
        return Transcript(language="en", words=words, sentences=transcribe.group_sentences(words))

    def fake_chat(settings, system, user, schema):
        return {"clips": [{"start": 1, "end": 8, "title": "Big moment", "score": 91,
                           "reason": "Strong hook."}]}

    monkeypatch.setattr(transcribe, "transcribe", fake_transcribe)
    monkeypatch.setattr(llm, "chat_json", fake_chat)


def test_pipeline_renders_a_vertical_clip(sample_video, stub_models) -> None:
    settings = Settings()
    settings.clips.min_seconds = 5
    settings.render.encoder = "libx264"
    settings.render.captions = ffmpeg.has_filter("ass")
    settings.render.face_tracking = False

    job = pipeline.Runner(pipeline.new_job(str(sample_video), settings)).run()

    assert job.status == "done", job.error
    [clip] = job.clips
    out = pipeline.job_dir(job.id) / clip.file
    w, h, duration = ffmpeg.video_size(out)
    assert (w, h) == (1080, 1920)
    assert abs(duration - clip.highlight.duration) < 0.5

    # Re-running reuses the cached transcript (stub would not be needed).
    src = pipeline.source_dir(str(sample_video))
    assert any(p.name.startswith("transcript-") for p in src.iterdir())


def _frame(path, t: float) -> bytes:
    import subprocess

    return subprocess.run(
        [ffmpeg.ffmpeg(), "-v", "error", "-ss", str(t), "-i", str(path), "-frames:v", "1",
         "-vf", "scale=108:192", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        capture_output=True, check=True).stdout


@pytest.mark.skipif(not ffmpeg.has_filter("ass") if shutil.which("ffmpeg") or
                    __import__("os").environ.get("HOOKCUT_FFMPEG") else True,
                    reason="ffmpeg without libass")
def test_font_and_position_change_the_rendered_captions(sample_video, stub_models) -> None:
    from hookcut import fonts

    def render(font: str, y: int):
        settings = Settings()
        settings.clips.min_seconds = 5
        settings.render.encoder = "libx264"
        settings.render.face_tracking = False
        settings.render.caption_font, settings.render.caption_y = font, y
        settings.render.caption_size = 140
        try:
            fonts.ensure(font)
        except fonts.FontError:
            pytest.skip("font download unavailable")
        job = pipeline.Runner(pipeline.new_job(str(sample_video), settings)).run()
        assert job.status == "done", job.error
        clips_dir = pipeline.job_dir(job.id) / "clips"
        assert [p.name for p in clips_dir.iterdir()] == ["01.mp4"], "no scratch files left"
        return _frame(clips_dir / "01.mp4", 1.0)

    base = render("poppins", 72)
    assert render("bangers", 72) != base, "a different font must render differently"
    assert render("poppins", 20) != base, "a different position must render differently"


def test_face_tracking_runs_on_footage_without_faces(sample_video, stub_models) -> None:
    pytest.importorskip("mediapipe")
    from hookcut.stages import reframe

    try:
        shots = reframe.analyse(sample_video, 2, 12, 1280, 720, 1080, 1920)
    except OSError:
        pytest.skip("face model download unavailable")
    assert shots and shots[0].start == 0 and abs(shots[-1].end - 10) < 1e-6
    assert all(s.x is None for s in shots), "a test pattern has no faces"


def test_web_api_runs_a_job(sample_video, stub_models) -> None:
    from hookcut.web.app import create_app

    with TestClient(create_app()) as client:
        assert client.get("/").status_code == 200
        assert "settings" in client.get("/api/settings").json()
        client.put("/api/settings", json={"render": {"face_tracking": False, "encoder": "libx264",
                                                     "captions": False},
                                          "clips": {"min_seconds": 5}})
        assert client.post("/api/jobs", json={"target": " "}).status_code == 400

        job = client.post("/api/jobs", json={"target": str(sample_video)}).json()
        import time
        for _ in range(120):
            state = client.get(f"/api/jobs/{job['id']}").json()
            if state["status"] in ("done", "failed"):
                break
            time.sleep(0.25)
        assert state["status"] == "done", json.dumps(state)
        video = client.get(f"/api/jobs/{job['id']}/clips/1?download=true")
        assert video.status_code == 200 and video.headers["content-type"] == "video/mp4"
        assert "big-moment-01.mp4" in video.headers["content-disposition"]
        assert client.get(f"/api/jobs/{job['id']}/clips/9").status_code == 404
        assert client.delete(f"/api/jobs/{job['id']}").status_code == 204


def test_web_api_models_fonts_and_uploads(sample_video, monkeypatch) -> None:
    from hookcut.web.app import create_app

    monkeypatch.setattr(llm, "installed_models", lambda s: [{"name": "qwen3:14b", "size_gb": 9.3},
                                                            {"name": "my-model:1b", "size_gb": 1}])
    monkeypatch.setattr(llm, "list_models", lambda s: ["qwen3:14b", "my-model:1b"])
    with TestClient(create_app()) as client:
        data = client.get("/api/models").json()
        assert data["server_ok"] and data["ram_gb"] > 0
        row = next(m for m in data["catalog"] if m["tag"] == "qwen3:14b")
        assert row["installed"] and row["fit"] in ("good", "tight", "too-big")
        assert client.post("/api/models/pull", json={"name": "rm -rf /"}).status_code == 400

        fonts_list = client.get("/api/fonts").json()
        assert any(f["id"] == "anton" for f in fonts_list)
        assert client.get("/api/fonts/nope.ttf").status_code == 404

        bad = client.put("/api/settings", json={"render": {"caption_font": "nope"}})
        assert bad.status_code == 400
        ok = client.put("/api/settings", json={"render": {"caption_size": 120, "caption_y": 30}})
        assert ok.json()["settings"]["render"]["caption_size"] == 120

        with open(sample_video, "rb") as f:
            up = client.post("/api/upload", files={"file": ("my talk.mp4", f, "video/mp4")})
        assert up.status_code == 201
        from pathlib import Path
        assert Path(up.json()["path"]).stat().st_size == sample_video.stat().st_size
        txt = client.post("/api/upload", files={"file": ("notes.txt", b"hi", "text/plain")})
        assert txt.status_code == 400
