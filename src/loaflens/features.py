"""Fitur ringkas dari crop roti + mask: tekstur permukaan, warna/kilap, bentuk (bebas skala).

Tidak ada fitur ukuran gambar asli (lebar/tinggi/aspek gambar) — input sudah di-crop & di-resize
oleh ``segment.crop`` sehingga fitur tidak bergantung pada resolusi foto.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

LBP_BINS = 10  # LBP uniform rotation-invariant P=8: kode 0..8 + 1 bin non-uniform

FEATURE_NAMES: list[str] = [
    "tex_lap_var",
    "tex_grad_mean",
    "tex_edge_density",
    "tex_crease",
    *[f"tex_lbp_{i}" for i in range(LBP_BINS)],
    "col_L_mean",
    "col_L_std",
    "col_a_mean",
    "col_a_std",
    "col_b_mean",
    "col_b_std",
    "col_specular_frac",
    "shape_solidity",
    "shape_extent",
    "shape_circularity",
    "shape_aspect",
]


def lbp_hist(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Histogram LBP uniform rotation-invariant (P=8, R=1) di dalam mask, ternormalisasi."""
    g = gray.astype(np.int16)
    center = g[1:-1, 1:-1]
    offsets = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0)]  # searah jarum jam
    bits = np.stack(
        [g[dy : dy + center.shape[0], dx : dx + center.shape[1]] >= center for dy, dx in offsets]
    ).astype(np.int8)
    ones = bits.sum(axis=0)
    transitions = np.abs(bits - np.roll(bits, 1, axis=0)).sum(axis=0)
    codes = np.where(transitions <= 2, ones, LBP_BINS - 1)
    sel = codes[mask[1:-1, 1:-1] > 0]
    hist = np.bincount(sel, minlength=LBP_BINS).astype(float)
    return hist / max(hist.sum(), 1.0)


def texture_features(
    gray: np.ndarray, inner: np.ndarray, canny: tuple[int, int], crease_px: int
) -> list[float]:
    """Variansi Laplacian, rata-rata gradien, kepadatan tepi Canny, dan energi lipatan.

    Lipatan = black-hat morfologi (alur gelap lebih sempit dari kernel), skala kerutan kasar.
    """
    sel = inner > 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * crease_px + 1, 2 * crease_px + 1))
    crease = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad = np.hypot(gx, gy)
    edges = cv2.Canny(gray, canny[0], canny[1]) > 0
    return [
        float(lap[sel].var()),
        float(grad[sel].mean()),
        float(edges[sel].mean()),
        float(crease[sel].mean()),
    ]


def color_features(image: np.ndarray, mask: np.ndarray, spec_v: int, spec_s: int) -> list[float]:
    """Mean/std L*, a*, b* di mask + fraksi piksel kilap (sangat terang, saturasi rendah)."""
    sel = mask > 0
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(float)[sel]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[sel]
    spec = (hsv[:, 2] >= spec_v) & (hsv[:, 1] <= spec_s)
    out: list[float] = []
    for c in range(3):
        out += [float(lab[:, c].mean()), float(lab[:, c].std())]
    return out + [float(spec.mean())]


def shape_features(mask: np.ndarray) -> list[float]:
    """Solidity, extent, circularity, aspek bbox roti (tinggi/lebar) — semuanya bebas skala."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    hull = cv2.contourArea(cv2.convexHull(c))
    _, _, w, h = cv2.boundingRect(c)
    perim = cv2.arcLength(c, closed=True)
    return [
        area / max(hull, 1.0),
        area / max(w * h, 1),
        4 * np.pi * area / max(perim**2, 1.0),
        h / max(w, 1),
    ]


def extract(crop_img: np.ndarray, crop_mask: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    """Vektor fitur (urutan = FEATURE_NAMES) dari crop BGR + mask (0/255) ukuran tetap."""
    erode = int(params.get("erode_px", 9))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode + 1, 2 * erode + 1))
    inner = cv2.erode(crop_mask, kernel)
    if not inner.any():  # mask terlalu kecil untuk dierosi
        inner = crop_mask
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    sigma = float(params.get("texture_blur_sigma", 0.0))
    if sigma > 0:  # redam noise piksel/resampling → tekstur skala kerutan, kurang peka resolusi
        gray = cv2.GaussianBlur(gray, (0, 0), sigma)
    canny = tuple(params.get("canny", (50, 150)))
    vec = (
        texture_features(gray, inner, canny, int(params.get("crease_px", 6)))
        + lbp_hist(gray, inner).tolist()
        + color_features(
            crop_img, inner, params.get("specular_v", 230), params.get("specular_s", 60)
        )
        + shape_features(crop_mask)
    )
    return np.asarray(vec, dtype=float)
