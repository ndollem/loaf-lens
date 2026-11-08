"""Gambar roti sintetis untuk test (SYNTHETIC — bukan untuk evaluasi akurasi)."""

import cv2
import numpy as np

BG = (200, 200, 200)  # latar abu, saturasi ~0
LOAF = (40, 120, 200)  # BGR oranye keemasan


def synth_loaf(
    size: tuple[int, int] = (320, 240),
    center: tuple[int, int] = (160, 125),
    axes: tuple[int, int] = (100, 70),
    wrinkles: int = 0,
    seed: int = 0,
) -> np.ndarray:
    """Elips oranye di latar abu; ``wrinkles`` = jumlah garis gelap (meniru kerutan/lipatan)."""
    w, h = size
    img = np.full((h, w, 3), BG, np.uint8)
    cv2.ellipse(img, center, axes, 0, 0, 360, LOAF, thickness=cv2.FILLED)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 3, img.shape)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    cx, cy = center
    for _ in range(wrinkles):
        x0 = int(cx + rng.uniform(-0.6, 0.6) * axes[0])
        y0 = int(cy + rng.uniform(-0.6, 0.6) * axes[1])
        dx, dy = rng.uniform(-25, 25, 2).astype(int)
        cv2.line(img, (x0, y0), (x0 + dx, y0 + dy), (20, 70, 130), 2)
    return img
