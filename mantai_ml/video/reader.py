"""
Turn gameplay video into decision records.

The camera must show the filmed player's own hand. Regions of the frame are
given as fractions of the frame, [x, y, width, height], in a JSON config:

    {
      "hand_roi":     [0.10, 0.82, 0.80, 0.16],   // required: your concealed hand
      "meld_roi":     [0.10, 0.70, 0.80, 0.10],   // optional: your exposed melds
      "visible_rois": [[0.10, 0.20, 0.80, 0.45]], // optional: discards / others' melds
      "bonus_roi":    [0.00, 0.70, 0.10, 0.10],   // optional: your flowers etc.
      "seat": 1, "prevailing": "east",
      "sample_fps": 2, "min_confidence": 0.8, "stable_frames": 2,
      "detect": {"tile_aspect": 1.35}
    }

How moves are recognised: the hand is read on every sampled frame. A reading
only counts once it has been the same for `stable_frames` samples in a row and
every tile was classified with at least `min_confidence`. When the stable hand
changes, the difference tells what happened:

    one tile gone (maybe after a draw)          -> discard
    tiles gone and a new meld appears           -> Pong / Chi / Kong call
    four tiles gone and a new Kong appears      -> concealed Kong (self_kong)
    one tile gone and a Pong becomes a Kong     -> promoted Kong (self_kong)
    one tile added                              -> draw (no record)

If the meld region is configured but cannot be read confidently, the whole
frame is skipped, so a misread meld is never mistaken for a move.
"""

import json
import os
from collections import Counter
from dataclasses import dataclass, field

import cv2

from ..records import tile_list, write_records
from ..tiles import TILE_INDEX, parse_tile, sg
from .detect import DetectParams, cut, detect_tiles, draw_boxes
from .frames import iter_frames


@dataclass
class VideoConfig:
    hand_roi: list = None
    meld_roi: list = None
    visible_rois: list = field(default_factory=list)
    bonus_roi: list = None
    seat: int = 1
    prevailing: str = "east"
    sample_fps: float = 2.0
    min_confidence: float = 0.8
    stable_frames: int = 2
    detect: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        cfg = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        if cfg.hand_roi is None:
            raise ValueError("The video config needs a hand_roi.")
        return cfg


def read_region(frame, roi, clf, cfg, params):
    """Tiles in a region, or None if any tile is below min_confidence."""
    if roi is None:
        return [], []
    boxes = detect_tiles(frame, roi, params)
    preds = clf.predict(cut(frame, boxes))
    if any(conf < cfg.min_confidence for _, conf in preds):
        return None, boxes
    return [parse_tile(name) for name, _ in preds], boxes


def group_melds(tiles):
    """Split exposed-meld tiles (reading order) into Kongs of 4 and sets of 3."""
    melds, i = [], 0
    while i + 3 <= len(tiles):
        if i + 4 <= len(tiles) and len(set(tiles[i:i + 4])) == 1:
            melds.append(tiles[i:i + 4])
            i += 4
        else:
            melds.append(tiles[i:i + 3])
            i += 3
    return melds


def _meld_type(meld):
    if len(meld) == 4:
        return "kong"
    return "pong" if len(set(meld)) == 1 else "chi"


class VideoReader:
    def __init__(self, clf, cfg, source="video", game_id="video", outcome=None):
        self.clf, self.cfg = clf, cfg
        self.params = DetectParams.from_dict(cfg.detect)
        self.source, self.game_id, self.outcome = source, game_id, outcome
        self.records = []
        self.snapshots = []
        self.skipped = Counter()

    def read_frame(self, frame):
        hand, hand_boxes = read_region(frame, self.cfg.hand_roi, self.clf, self.cfg, self.params)
        if hand is None:
            self.skipped["low confidence"] += 1
            return None, hand_boxes
        melds, _ = read_region(frame, self.cfg.meld_roi, self.clf, self.cfg, self.params)
        if melds is None:
            self.skipped["low confidence melds"] += 1
            return None, hand_boxes
        bonus, _ = read_region(frame, self.cfg.bonus_roi, self.clf, self.cfg, self.params)
        visible = []
        for roi in self.cfg.visible_rois or []:
            v, _ = read_region(frame, roi, self.clf, self.cfg, self.params)
            visible += [t for t in (v or []) if not sg.is_bonus(t)]
        # Bonus tiles are revealed at once; ignore any caught in the hand.
        hand = [t for t in hand if not sg.is_bonus(t)]
        return {
            "hand": hand,
            "melds": group_melds([t for t in (melds or []) if not sg.is_bonus(t)]),
            "bonus": [t for t in (bonus or []) if sg.is_bonus(t)],
            "visible": visible,
        }, hand_boxes

    def _record(self, snap, kind, tile, hand, melds, **extra):
        r = {
            "source": self.source, "game_id": self.game_id, "player": 0,
            "controller": "video", "kind": kind, "tile": list(tile),
            "hand": tile_list(hand), "melds": [tile_list(m) for m in melds],
            "visible": tile_list(snap["visible"]), "bonus": tile_list(snap["bonus"]),
            "seat": self.cfg.seat, "prevailing": self.cfg.prevailing,
            "wall_left": None, "time": round(snap["time"], 2),
        }
        r.update(extra)
        if self.outcome is not None:
            r["outcome"] = self.outcome
        self.records.append(r)

    def on_change(self, prev, cur):
        """Work out what happened between two stable readings."""
        hand = list(prev["hand"])
        melds = [list(m) for m in prev["melds"]]
        removed = Counter(prev["hand"]) - Counter(cur["hand"])
        added = Counter(cur["hand"]) - Counter(prev["hand"])

        # A Pong that grew into a Kong: one copy moved from the hand.
        for i, (old, new) in enumerate(zip(prev["melds"], cur["melds"])):
            if (len(old) == 3 and len(new) == 4 and len(set(old)) == 1
                    and set(new) == set(old) and removed[old[0]] >= 1):
                tile = old[0]
                self._record(prev, "call", tile, hand, melds, call_type="self_kong", taken=True)
                hand.remove(tile)
                removed[tile] -= 1
                removed += Counter()          # drop zero counts
                melds[i] = list(new)

        # New exposed melds: calls on someone else's discard, or a Kong
        # declared entirely from the hand.
        for meld in cur["melds"][len(prev["melds"]):]:
            from_hand = Counter(meld) & removed
            if len(meld) == 4 and sum(from_hand.values()) == 4:
                self._record(prev, "call", meld[0], hand, melds, call_type="self_kong", taken=True)
                for t in from_hand.elements():
                    hand.remove(t)
                removed -= from_hand
                melds.append(list(meld))
                continue
            if sum(from_hand.values()) != len(meld) - 1:
                self.skipped["unreadable call"] += 1
                return
            claimed = next(iter(Counter(meld) - from_hand))
            self._record(prev, "call", claimed, hand, melds,
                         call_type=_meld_type(meld), taken=True)
            for t in from_hand.elements():
                hand.remove(t)
            removed -= from_hand
            melds.append(list(meld))

        n_removed = sum(removed.values())
        if n_removed == 0:
            return                                   # a draw, or nothing
        if n_removed == 1 and sum(added.values()) <= 1:
            tile = next(iter(removed))
            before = hand + list(added.elements())   # the drawn tile, if we missed it
            if len(before) + 3 * len(melds) == 14 and tile in TILE_INDEX:
                self._record(prev, "discard", tile, before, melds)
                return
        self.skipped["unreadable change"] += 1

    def process(self, video_path, debug_dir=None):
        candidate, count, stable = None, 0, None
        for t, frame in iter_frames(video_path, self.cfg.sample_fps):
            snap, boxes = self.read_frame(frame)
            if snap is None:
                continue
            key = (tuple(sorted(snap["hand"], key=str)), tuple(tuple(sorted(m, key=str)) for m in snap["melds"]))
            count = count + 1 if key == candidate else 1
            candidate = key
            if count != self.cfg.stable_frames:
                continue
            snap["time"] = t
            if stable is not None and key == stable[0]:
                stable = (key, snap)          # same hand; refresh visible tiles
                continue
            if stable is not None:
                self.on_change(stable[1], snap)
            stable = (key, snap)
            self.snapshots.append(snap)
            if debug_dir:
                os.makedirs(debug_dir, exist_ok=True)
                names = [f"{a[0][0]}{a[1]}" for a in snap["hand"]]
                cv2.imwrite(os.path.join(debug_dir, f"t{t:08.2f}.jpg"), draw_boxes(frame, boxes, names))
        return self.records


def video_to_records(video_path, tiles_model, config_path, out_path, outcome=None,
                     debug_dir=None, append=True, log=print):
    from .classifier import TileClassifier
    cfg = VideoConfig.load(config_path)
    name = os.path.basename(str(video_path))
    reader = VideoReader(TileClassifier(tiles_model), cfg, source=f"video:{name}",
                         game_id=name, outcome=outcome)
    records = reader.process(video_path, debug_dir)
    write_records(out_path, records, append=append)
    n_disc = sum(r["kind"] == "discard" for r in records)
    log(f"{name}: {len(reader.snapshots)} stable hand readings, {n_disc} discards, "
        f"{len(records) - n_disc} calls -> {out_path}")
    if reader.skipped:
        log(f"  skipped: {dict(reader.skipped)}")
    return records
