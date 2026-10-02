"""Cut tile images out of gameplay video so they can be labelled.

Crops go to <out>/unlabeled/. Sort them into <dataset>/<tile_name>/ folders
(bamboo_1 ... dragon_white, flower_1, animal_cat, ...) and train with
`python -m mantai_ml train-tiles`. With --tiles-model, crops are pre-sorted
into <out>/suggested/<tile_name>/ so you only fix the mistakes.
"""

import os

import cv2
import numpy as np

from .detect import DetectParams, cut, detect_tiles, draw_boxes
from .frames import iter_frames


def extract_crops(video_path, out_dir, config_path=None, tiles_model=None,
                  sample_fps=1.0, max_crops=5000, log=print):
    from .reader import VideoConfig
    rois, params = [None], DetectParams()
    if config_path:
        cfg = VideoConfig.load(config_path)
        rois = [r for r in [cfg.hand_roi, cfg.meld_roi, cfg.bonus_roi] + list(cfg.visible_rois or []) if r]
        params = DetectParams.from_dict(cfg.detect)
    clf = None
    if tiles_model:
        from .classifier import TileClassifier
        clf = TileClassifier(tiles_model)

    os.makedirs(out_dir, exist_ok=True)
    seen = []
    saved = 0
    stem = os.path.splitext(os.path.basename(str(video_path)))[0]
    for t, frame in iter_frames(video_path, sample_fps):
        boxes = [b for roi in rois for b in detect_tiles(frame, roi, params)]
        if saved == 0 and boxes:
            cv2.imwrite(os.path.join(out_dir, f"{stem}_detected_boxes.jpg"), draw_boxes(frame, boxes))
        crops = cut(frame, boxes)
        preds = clf.predict(crops) if clf else [(None, 0.0)] * len(crops)
        for crop, (label, conf) in zip(crops, preds):
            thumb = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (12, 16)).astype(np.float32)
            if any(np.abs(thumb - s).mean() < 6 for s in seen[-400:]):
                continue                     # near-duplicate of a recent crop
            seen.append(thumb)
            folder = os.path.join(out_dir, "suggested", label) if label else os.path.join(out_dir, "unlabeled")
            os.makedirs(folder, exist_ok=True)
            cv2.imwrite(os.path.join(folder, f"{stem}_{t:08.2f}_{saved:05d}.png"), crop)
            saved += 1
            if saved >= max_crops:
                log(f"Saved {saved} crops to {out_dir} (limit reached)")
                return saved
    log(f"Saved {saved} tile crops to {out_dir}")
    return saved
