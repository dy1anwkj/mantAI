"""
Find tiles in an image region.

Tile faces are bright and low in colour saturation compared with the table,
so the region is thresholded on that, and every blob of roughly tile shape is
kept. Tiles standing side by side in a hand often merge into one long blob;
such a blob is split into equal tile-width pieces using the expected aspect
ratio (height / width) of a tile.

Tune DetectParams for your camera if tiles are missed or over-split. The
`crops` command saves a debug image with the detected boxes drawn on it.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from .frames import crop_roi


@dataclass
class DetectParams:
    tile_aspect: float = 1.35     # tile height / width as seen by the camera
    min_value: int = 150          # brightness (HSV V) a tile face must reach
    max_saturation: int = 90      # tile faces are pale; the table is not
    min_height_px: int = 12       # ignore blobs shorter than this (noise, edges)
    max_height_frac: float = 0.95 # ignore blobs nearly as tall as the region (table glare)
    close_kernel: int = 0         # >0 closes gaps; only needed if tile edges break up
    open_kernel: int = 3          # removes specks of noise

    @classmethod
    def from_dict(cls, d):
        d = d or {}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def tile_mask(bgr, p):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = ((hsv[..., 2] >= p.min_value) & (hsv[..., 1] <= p.max_saturation)).astype(np.uint8) * 255
    if p.close_kernel:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((p.close_kernel, p.close_kernel), np.uint8))
    if p.open_kernel:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((p.open_kernel, p.open_kernel), np.uint8))
    return mask


def split_points(gray_strip, n, tile_w):
    """Cut positions for a blob of n touching tiles: start from equal widths,
    then move each cut to the darkest column nearby (the seam between tiles)."""
    w = gray_strip.shape[1]
    profile = gray_strip.mean(axis=0)
    cuts = [0]
    for i in range(1, n):
        guess = int(round(i * w / n))
        r = max(2, int(tile_w * 0.25))
        lo, hi = max(cuts[-1] + 4, guess - r), min(w - 4, guess + r)
        cuts.append(lo + int(np.argmin(profile[lo:hi])) if hi > lo else guess)
    cuts.append(w)
    return cuts


def detect_tiles(frame, roi=None, params=None):
    """Boxes (x, y, w, h) in frame pixels, in reading order (left to right)."""
    p = params or DetectParams()
    region, (ox, oy) = crop_roi(frame, roi)
    if region.size == 0:
        return []
    mask = tile_mask(region, p)
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_h = max(p.min_height_px + 1, p.max_height_frac * region.shape[0])
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if h < p.min_height_px or h > max_h or w < 4:
            continue
        tile_w = h / p.tile_aspect
        n = int(round(w / tile_w))
        if n < 1:
            continue
        # Touching tiles form one blob: split it at the seams between them.
        cuts = split_points(gray[y:y + h, x:x + w], n, tile_w) if n > 1 else [0, w]
        for a, b in zip(cuts, cuts[1:]):
            boxes.append((x + a + ox, y + oy, b - a, h))
    # Reading order: rows top to bottom, then left to right.
    boxes.sort(key=lambda b: (round(b[1] / max(1, b[3])), b[0]))
    return boxes


def cut(frame, boxes, size=(48, 64)):
    """Crop each box and resize to (width, height) for the classifier."""
    out = []
    for x, y, w, h in boxes:
        crop = frame[y:y + h, x:x + w]
        if crop.size:
            out.append(cv2.resize(crop, size, interpolation=cv2.INTER_AREA))
    return out


def draw_boxes(frame, boxes, labels=None):
    img = frame.copy()
    for i, (x, y, w, h) in enumerate(boxes):
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        if labels:
            cv2.putText(img, str(labels[i]), (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.35, (0, 0, 255), 1)
    return img
