"""Test parsing sumber + baca video kecil (SYNTHETIC — bukan untuk evaluasi akurasi)."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from loaflens.capture import FrameSource, SourceError, SourceSpec, parse_source


def test_parse_source_video_and_camera() -> None:
    assert parse_source("video:samples/demo.mp4") == SourceSpec("video", "samples/demo.mp4")
    assert parse_source("video:C:/x/y.mp4").target == "C:/x/y.mp4"  # ':' berikutnya bagian path
    cam = parse_source("camera:0")
    assert cam == SourceSpec("camera", 0) and cam.is_camera


@pytest.mark.parametrize(
    "spec", ["samples/demo.mp4", "video:", "camera:abc", "rtsp:x", "camera:-1"]
)
def test_parse_source_rejects_bad_specs(spec: str) -> None:
    with pytest.raises(SourceError):
        parse_source(spec)


def test_missing_video_file_has_clear_error(tmp_path: Path) -> None:
    with pytest.raises(SourceError, match="tidak ditemukan"):
        FrameSource(f"video:{tmp_path / 'nope.mp4'}")


def test_reads_tiny_video(tmp_path: Path) -> None:
    path = tmp_path / "tiny.avi"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 12.0, (64, 48))
    for i in range(5):
        w.write(np.full((48, 64, 3), i * 40, np.uint8))
    w.release()
    with FrameSource(f"video:{path}") as src:
        frames = list(src)
        assert src.fps == pytest.approx(12.0)
    assert len(frames) == 5 and frames[0].shape == (48, 64, 3)
    assert frames[3].mean() == pytest.approx(120, abs=6)
