"""Decide where to crop each part of a clip so the speaker stays in frame.

Approach: detect camera cuts (frame-accurate, from colour histograms), then give
each shot ONE fixed crop centred on the main face. A steady frame per shot is
what a human editor would do; a crop that constantly drifts looks amateur.
Shots with no face fall back to a centre crop.
"""

from __future__ import annotations

import contextlib
import itertools
import logging
import os
import statistics
import sys
import urllib.request
from pathlib import Path

from ..config import home
from ..models import Shot

log = logging.getLogger(__name__)

FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)
#: Faces are checked this many times per second (cuts are checked every frame).
FACE_SAMPLES_PER_S = 4.0
#: Histogram correlation below this between consecutive frames = a camera cut.
CUT_THRESHOLD = 0.55
#: Shots shorter than this are merged into a neighbour (flashes, fast b-roll).
MIN_SHOT_S = 0.8
#: A shot gets a face-centred crop only if faces appear in this share of samples.
MIN_FACE_SHARE = 0.3


@contextlib.contextmanager
def quiet_stderr():
    """Mute native-code stderr (MediaPipe logs every model load from C++)."""
    try:
        fd = sys.stderr.fileno()
    except (AttributeError, OSError, ValueError):
        yield
        return
    sys.stderr.flush()
    saved = os.dup(fd)
    with open(os.devnull, "w") as devnull:
        os.dup2(devnull.fileno(), fd)
        try:
            yield
        finally:
            os.dup2(saved, fd)
            os.close(saved)


def face_model_path() -> Path:
    path = home() / "models" / "blaze_face_short_range.tflite"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        log.info("Downloading face detection model")
        tmp = path.with_suffix(".tmp")
        urllib.request.urlretrieve(FACE_MODEL_URL, tmp)
        tmp.replace(path)
    return path


def crop_width(src_w: int, src_h: int, out_w: int, out_h: int) -> int:
    """Width of the source region that matches the output aspect ratio."""
    width = round(src_h * out_w / out_h)
    width -= width % 2
    return min(width, src_w - src_w % 2)


def crop_left(face_x: float, src_w: int, width: int) -> int:
    left = round(face_x - width / 2)
    left = max(0, min(left, src_w - width))
    return left - left % 2


def split_shots(cut_times: list[float], duration: float) -> list[tuple[float, float]]:
    """Turn cut times into (start, end) shots, merging ones that are too short."""
    bounds = [0.0, *sorted(t for t in cut_times if 0 < t < duration), duration]
    shots: list[list[float]] = []
    for a, b in itertools.pairwise(bounds):
        if shots and (b - a < MIN_SHOT_S or shots[-1][1] - shots[-1][0] < MIN_SHOT_S):
            shots[-1][1] = b
        else:
            shots.append([a, b])
    return [(a, b) for a, b in shots]


def plan_shots(
    shots: list[tuple[float, float]],
    faces: list[tuple[float, float | None]],
    src_w: int,
    width: int,
) -> list[Shot]:
    """Assign each shot a crop from the face samples that fall inside it."""
    planned: list[Shot] = []
    for start, end in shots:
        inside = [x for t, x in faces if start <= t < end]
        seen = [x for x in inside if x is not None]
        x = None
        if inside and len(seen) / len(inside) >= MIN_FACE_SHARE:
            x = crop_left(statistics.median(seen), src_w, width)
        planned.append(Shot(start=start, end=end, x=x))

    # Neighbouring shots with nearly the same framing render as one.
    merged: list[Shot] = []
    tolerance = src_w * 0.02
    for shot in planned:
        prev = merged[-1] if merged else None
        same = prev is not None and (
            (prev.x is None and shot.x is None)
            or (prev.x is not None and shot.x is not None and abs(prev.x - shot.x) <= tolerance)
        )
        if same:
            prev.end = shot.end
        else:
            merged.append(shot.model_copy())
    return merged


def analyse(
    video: Path, start: float, end: float, src_w: int, src_h: int, out_w: int, out_h: int
) -> list[Shot]:
    """Scan the clip's frames and return the crop plan (times relative to ``start``)."""
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import FaceDetector, FaceDetectorOptions

    width = crop_width(src_w, src_h, out_w, out_h)
    duration = end - start
    if width >= src_w - 2:
        return [Shot(start=0, end=duration, x=0)]  # already narrow enough

    options = FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=str(face_model_path())),
        min_detection_confidence=0.5,
    )
    with quiet_stderr():
        detector = FaceDetector.create_from_options(options)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)

    step = max(1, round(fps / FACE_SAMPLES_PER_S))
    cuts: list[float] = []
    faces: list[tuple[float, float | None]] = []
    prev_hist = None
    frame_no = 0
    try:
        while True:
            t = frame_no / fps
            if t >= duration:
                break
            ok, frame = cap.read()
            if not ok:
                break
            small = cv2.resize(frame, (96, 54))
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
            cv2.normalize(hist, hist)
            if (prev_hist is not None
                    and cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL) < CUT_THRESHOLD):
                cuts.append(t)
            prev_hist = hist

            if frame_no % step == 0:
                scale = 640 / frame.shape[1]
                view = cv2.resize(frame, (640, round(frame.shape[0] * scale)))
                rgb = cv2.cvtColor(view, cv2.COLOR_BGR2RGB)
                result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
                best = max(
                    result.detections,
                    key=lambda d: d.bounding_box.width * d.bounding_box.height,
                    default=None,
                )
                if best is None:
                    faces.append((t, None))
                else:
                    box = best.bounding_box
                    faces.append((t, (box.origin_x + box.width / 2) / scale))
            frame_no += 1
    finally:
        cap.release()
        with quiet_stderr():
            detector.close()

    return plan_shots(split_shots(cuts, duration), faces, src_w, width)


def centre_plan(duration: float) -> list[Shot]:
    return [Shot(start=0, end=duration, x=None)]
