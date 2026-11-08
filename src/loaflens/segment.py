"""Segmentasi roti tunggal dari latar polos: saturasi HSV → threshold → morfologi → kontur terbesar.

Asumsi POC: satu roti per gambar, latar abu/putih/krem bersaturasi rendah, roti oranye/keemasan.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from loaflens.config import SegmentConfig


@dataclass
class Segmentation:
    """Hasil segmentasi: mask uint8 (0/255) seukuran gambar, bbox (x, y, w, h), kontur."""

    mask: np.ndarray
    bbox: tuple[int, int, int, int]
    contour: np.ndarray


def segment(image: np.ndarray, cfg: SegmentConfig) -> Segmentation | None:
    """Isolasi roti terbesar di gambar BGR; None bila tidak ada objek yang cukup besar."""
    sat = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[..., 1]
    sat = cv2.GaussianBlur(sat, (cfg.blur, cfg.blur), 0)
    if cfg.sat_threshold is None:
        _, binary = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(sat, cfg.sat_threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.morph_kernel, cfg.morph_kernel))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < cfg.min_area_frac * image.shape[0] * image.shape[1]:
        return None
    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)  # isi lubang
    x, y, w, h = cv2.boundingRect(contour)
    return Segmentation(mask=mask, bbox=(x, y, w, h), contour=contour)


def full_frame(image: np.ndarray) -> Segmentation:
    """Fallback bila segmentasi gagal: seluruh gambar dianggap roti."""
    h, w = image.shape[:2]
    mask = np.full((h, w), 255, np.uint8)
    contour = np.array([[[0, 0]], [[w - 1, 0]], [[w - 1, h - 1]], [[0, h - 1]]], np.int32)
    return Segmentation(mask=mask, bbox=(0, 0, w, h), contour=contour)


def crop(
    image: np.ndarray, seg: Segmentation, size: int, margin: float
) -> tuple[np.ndarray, np.ndarray]:
    """Crop bbox (+margin), skala sisi terpanjang → size, pad ke size x size (aspek dijaga).

    Hasil tidak bergantung pada ukuran gambar asli. Pad = hitam di gambar dan 0 di mask.
    """
    x, y, w, h = seg.bbox
    mx, my = round(w * margin), round(h * margin)
    ih, iw = image.shape[:2]
    x0, y0 = max(x - mx, 0), max(y - my, 0)
    x1, y1 = min(x + w + mx, iw), min(y + h + my, ih)
    img_c, mask_c = image[y0:y1, x0:x1], seg.mask[y0:y1, x0:x1]

    scale = size / max(img_c.shape[:2])
    nw = max(1, round(img_c.shape[1] * scale))
    nh = max(1, round(img_c.shape[0] * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    img_r = cv2.resize(img_c, (nw, nh), interpolation=interp)
    mask_r = cv2.resize(mask_c, (nw, nh), interpolation=cv2.INTER_NEAREST)

    out_img = np.zeros((size, size, 3), np.uint8)
    out_mask = np.zeros((size, size), np.uint8)
    oy, ox = (size - nh) // 2, (size - nw) // 2
    out_img[oy : oy + nh, ox : ox + nw] = img_r
    out_mask[oy : oy + nh, ox : ox + nw] = mask_r
    return out_img, out_mask


def overlay(image: np.ndarray, seg: Segmentation | None) -> np.ndarray:
    """Gambar debug: kontur hijau + bbox merah (atau tanda GAGAL)."""
    out = image.copy()
    if seg is None:
        cv2.putText(out, "GAGAL", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        return out
    tint = out.copy()
    tint[seg.mask > 0] = (0, 200, 0)
    out = cv2.addWeighted(out, 0.75, tint, 0.25, 0)
    cv2.drawContours(out, [seg.contour], -1, (0, 255, 0), 2)
    x, y, w, h = seg.bbox
    cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return out
