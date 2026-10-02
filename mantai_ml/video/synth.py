"""
Synthetic tiles and gameplay videos.

These are NOT realistic tile pictures: each tile is drawn as text on a pale
face (e.g. "5 BAM"). They exist so the whole video pipeline can be run and
tested end to end without real footage. A classifier trained on them will not
recognise real tiles; for real videos, label real crops (see crops.py).
"""

import json
import os
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..tiles import ALL_TILES, tile_name

ASPECT = 1.35
SUIT_STYLE = {
    "bamboo": ("BAM", (20, 120, 40)), "dot": ("DOT", (30, 60, 170)), "wan": ("WAN", (170, 30, 30)),
    "wind": ("WIND", (30, 30, 30)), "dragon": ("DRGN", (30, 30, 30)),
    "flower": ("FLWR", (200, 40, 130)), "season": ("SEAS", (150, 60, 200)), "animal": ("ANML", (120, 80, 20)),
}
DRAGON_COLOUR = {"red": (200, 20, 20), "green": (20, 140, 40), "white": (60, 60, 200)}


def _font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def render_tile(tile, width=48, rng=None):
    """BGR image of one tile face, width x round(width * ASPECT)."""
    rng = rng or random.Random(0)
    height = int(round(width * ASPECT))
    scale = 4                                   # draw large, then shrink (anti-aliasing)
    W, H = width * scale, height * scale
    tint = tuple(int(c) for c in (245 + rng.randint(-10, 8), 240 + rng.randint(-10, 10), 225 + rng.randint(-10, 10)))
    img = Image.new("RGB", (W, H), tint)
    d = ImageDraw.Draw(img)
    d.rectangle([2, 2, W - 3, H - 3], outline=(150, 150, 140), width=scale * 2)
    suit, value = tile
    word, colour = SUIT_STYLE[suit]
    if suit == "dragon":
        colour = DRAGON_COLOUR[value]
        big = value[0].upper()
    elif suit == "wind":
        big = value[0].upper()
    elif suit == "animal":
        big = value[:3].upper()
    else:
        big = str(value)
    big_font = _font(int(H * (0.42 if len(big) == 1 else 0.24)))
    small_font = _font(int(H * 0.16))
    d.text((W / 2, H * 0.40), big, fill=colour, font=big_font, anchor="mm")
    d.text((W / 2, H * 0.78), word, fill=colour, font=small_font, anchor="mm")
    arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return cv2.resize(arr, (width, height), interpolation=cv2.INTER_AREA)


def make_tile_dataset(out_dir, per_class=60, seed=0, log=print):
    """Folder dataset of synthetic crops: out_dir/<tile_name>/*.png

    Each tile is placed on the table, compressed like a video frame and cut
    out with a slightly-off box, so the crops look like what the detector
    produces from real footage of these synthetic tiles."""
    rng = random.Random(seed)
    for tile in ALL_TILES:
        folder = os.path.join(out_dir, tile_name(tile))
        os.makedirs(folder, exist_ok=True)
        for i in range(per_class):
            tw = rng.choice([28, 30, 34, 40, 44, 46, 48, 56])
            face = render_tile(tile, width=tw, rng=rng)
            th = face.shape[0]
            canvas = np.full((th + 16, tw + 16, 3), TABLE, np.uint8)
            canvas[8:8 + th, 8:8 + tw] = face
            ok, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(45, 95)])
            canvas = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            j = lambda: rng.randint(-2, 2)
            x0, y0 = 8 + j(), 8 + j()
            crop = canvas[max(0, y0):8 + th + j(), max(0, x0):8 + tw + j()]
            im = cv2.resize(crop, (48, 64), interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(folder, f"synth_{i:03d}.png"), im)
    log(f"Wrote {per_class} synthetic images for each of {len(ALL_TILES)} tiles to {out_dir}")


# --------------------------------------------------------------------------
# Synthetic video of Player 1's view of a simulated game
# --------------------------------------------------------------------------
FRAME_W, FRAME_H = 960, 540
LAYOUT = {                                     # fractions of the frame
    "hand_roi": [0.02, 0.84, 0.96, 0.15],
    "meld_roi": [0.02, 0.68, 0.70, 0.13],
    "bonus_roi": [0.76, 0.68, 0.22, 0.13],
    "visible_rois": [[0.02, 0.03, 0.96, 0.60]],
}
TABLE = (40, 110, 45)                          # BGR felt green


def _px(roi):
    x, y, w, h = roi
    return int(x * FRAME_W), int(y * FRAME_H), int(w * FRAME_W), int(h * FRAME_H)


def _place_row(frame, tiles, roi, tile_w, gap, rng, group_gap=0, groups=None):
    x0, y0, w, h = _px(roi)
    tile_h = int(round(tile_w * ASPECT))
    one_row = groups is not None or len(tiles) * (tile_w + gap) <= w
    x = x0 + 4
    y = y0 + (h - tile_h) // 2 if one_row else y0 + 4
    seq = groups if groups is not None else [tiles]
    for group in seq:
        for tile in group:
            if x + tile_w > x0 + w:                    # wrap to the next row
                x, y = x0 + 4, y + tile_h + 8
            if y + tile_h > y0 + h:
                return
            frame[y:y + tile_h, x:x + tile_w] = render_tile(tile, tile_w, rng)
            x += tile_w + gap
        x += group_gap


def render_frame(snap, rng):
    frame = np.full((FRAME_H, FRAME_W, 3), TABLE, np.uint8)
    _place_row(frame, snap["hand"], LAYOUT["hand_roi"], 46, 2, rng)
    _place_row(frame, [], LAYOUT["meld_roi"], 40, 2, rng, group_gap=16, groups=snap["melds"])
    _place_row(frame, snap["bonus"], LAYOUT["bonus_roi"], 40, 6, rng)
    _place_row(frame, snap["visible"], LAYOUT["visible_rois"][0], 30, 3, rng)
    return frame


def simulated_snapshots(seed=0):
    """Play one engine game and return Player 1's view before and after every
    decision, plus the true discards (ground truth for tests)."""
    import mahjong_simulator as sim
    from ..features import state_from_game

    snaps, truth = [], []

    class Snap:
        def on_decision(self, game, player, kind, tile, call_type=None, taken=None):
            if player != 0:
                return
            s = state_from_game(game, 0)
            snaps.append({k: s[k] for k in ("hand", "melds", "bonus", "visible")})
            if kind == "discard":
                truth.append((tuple(tile), sorted(s["hand"], key=str)))
                after = dict(snaps[-1])
                after["hand"] = list(s["hand"])
                after["hand"].remove(tile)
                after["visible"] = s["visible"] + [tile]
                snaps.append(after)

        def on_game_end(self, game):
            pass

    def setup(game):
        game.recorder = Snap()

    # Seed the simulator for this game only, then give back its own RNG.
    saved_rng, sim.rng = sim.rng, random.Random(seed)
    try:
        sim.play_game(player1="engine", delay=0, verbose=False, setup=setup)
    finally:
        sim.rng = saved_rng
    return snaps, truth


def make_game_video(out_path, seed=0, fps=8, hold_seconds=1.0, log=print):
    """Write a synthetic video plus <video>.json config (regions) next to it."""
    snaps, truth = simulated_snapshots(seed)
    rng = random.Random(seed)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (FRAME_W, FRAME_H))
    for snap in snaps:
        frame = render_frame(snap, rng)
        for _ in range(int(fps * hold_seconds)):
            writer.write(frame)
    writer.release()
    config = dict(LAYOUT, seat=1, prevailing="east", sample_fps=4, min_confidence=0.6,
                  stable_frames=2, detect={"tile_aspect": ASPECT})
    config_path = os.path.splitext(str(out_path))[0] + ".json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    log(f"Wrote {out_path} ({len(snaps)} scenes, {len(truth)} true discards) and {config_path}")
    return config_path, truth
