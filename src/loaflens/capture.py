"""Sumber frame: ``video:<path>`` atau ``camera:<id>`` (hanya I/O, tanpa logika skor)."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Self

import cv2
import numpy as np

log = logging.getLogger(__name__)

DEFAULT_FPS = 30.0  # bila sumber tidak melaporkan fps (umum pada kamera)


class SourceError(ValueError):
    """Spesifikasi sumber salah atau sumber tidak bisa dibuka."""


@dataclass(frozen=True)
class SourceSpec:
    kind: str  # "video" | "camera"
    target: str | int  # path file atau indeks kamera

    @property
    def is_camera(self) -> bool:
        return self.kind == "camera"


def parse_source(spec: str) -> SourceSpec:
    """``video:<path>`` → SourceSpec('video', path); ``camera:<id>`` → SourceSpec('camera', id)."""
    kind, sep, target = spec.partition(":")
    if not sep or not target:
        raise SourceError(f"sumber {spec!r} tidak valid; pakai 'video:<path>' atau 'camera:<id>'")
    if kind == "video":
        return SourceSpec("video", target)
    if kind == "camera":
        if not target.isdigit():
            raise SourceError(f"id kamera harus angka >= 0, dapat {target!r}")
        return SourceSpec("camera", int(target))
    raise SourceError(f"jenis sumber {kind!r} tidak dikenal; pakai 'video' atau 'camera'")


class FrameSource:
    """Pembungkus ``cv2.VideoCapture``: iterasi frame BGR, ``fps``, ``release`` bersih."""

    def __init__(self, spec: str | SourceSpec) -> None:
        self.spec = parse_source(spec) if isinstance(spec, str) else spec
        if self.spec.kind == "video" and not Path(str(self.spec.target)).is_file():
            raise SourceError(f"file video tidak ditemukan: {self.spec.target}")
        self.cap = cv2.VideoCapture(self.spec.target)
        if not self.cap.isOpened():
            self.cap.release()
            raise SourceError(f"sumber tidak bisa dibuka: {self.spec.kind}:{self.spec.target}")
        fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.fps = fps if 0 < fps < 1000 else DEFAULT_FPS
        log.info("Sumber %s:%s dibuka (fps %.1f)", self.spec.kind, self.spec.target, self.fps)

    def __iter__(self) -> Iterator[np.ndarray]:
        """Frame berikutnya sampai video habis / kamera berhenti."""
        while True:
            ok, frame = self.cap.read()
            if not ok:
                return
            yield frame

    def release(self) -> None:
        self.cap.release()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()
