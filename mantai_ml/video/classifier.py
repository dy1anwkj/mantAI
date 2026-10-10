"""
Tile image classifier: a small CNN that names a 48x64 tile crop.

Training data layout (one folder per tile, names from tiles.TILE_NAMES):

    dataset/
        bamboo_1/  img001.png img002.png ...
        dot_5/     ...
        wind_east/ ...
        dragon_red/ ...
        flower_1/  season_3/  animal_cat/ ...

Folders may be missing or empty; the model still has all 46 outputs.
"""

import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..model import pick_device
from ..tiles import TILE_NAMES

IMG_W, IMG_H = 48, 64
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


class TileNet(nn.Module):
    def __init__(self, n_classes=len(TILE_NAMES)):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout),
                                 nn.ReLU(), nn.MaxPool2d(2))
        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128), block(128, 128))
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3),
                                  nn.Linear(128 * (IMG_H // 16) * (IMG_W // 16), 256), nn.ReLU(),
                                  nn.Linear(256, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


def to_tensor(bgr_images):
    arr = np.stack([cv2.resize(im, (IMG_W, IMG_H)) for im in bgr_images]).astype(np.float32) / 255.0
    arr = arr[..., ::-1].copy()                       # BGR -> RGB
    return torch.from_numpy(arr).permute(0, 3, 1, 2)  # N, 3, H, W


def load_folder(dataset_dir):
    images, labels = [], []
    for label_index, name in enumerate(TILE_NAMES):
        folder = os.path.join(dataset_dir, name)
        if not os.path.isdir(folder):
            continue
        for f in sorted(os.listdir(folder)):
            if f.lower().endswith(IMG_EXTS):
                im = cv2.imread(os.path.join(folder, f))
                if im is not None:
                    images.append(im)
                    labels.append(label_index)
    unknown = [d for d in os.listdir(dataset_dir)
               if os.path.isdir(os.path.join(dataset_dir, d)) and d not in TILE_NAMES]
    return images, labels, unknown


def augment(x):
    """Random lighting, blur-free jitter, small shifts and scale changes."""
    n = x.shape[0]
    bright = 1 + (torch.rand(n, 1, 1, 1, device=x.device) - 0.5) * 0.5
    shift = (torch.rand(n, 3, 1, 1, device=x.device) - 0.5) * 0.12
    x = (x * bright + shift).clamp(0, 1)
    theta = torch.zeros(n, 2, 3, device=x.device)
    scale = 1 + (torch.rand(n, device=x.device) - 0.5) * 0.15
    angle = (torch.rand(n, device=x.device) - 0.5) * 0.15
    theta[:, 0, 0] = scale * torch.cos(angle)
    theta[:, 0, 1] = -scale * torch.sin(angle)
    theta[:, 1, 0] = scale * torch.sin(angle)
    theta[:, 1, 1] = scale * torch.cos(angle)
    theta[:, :, 2] = (torch.rand(n, 2, device=x.device) - 0.5) * 0.12
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, padding_mode="border", align_corners=False)
    return (x + torch.randn_like(x) * 0.02).clamp(0, 1)


def train_tile_classifier(dataset_dir, out_path, epochs=25, batch_size=64, lr=2e-3,
                          val_fraction=0.15, seed=0, device=None, log=print):
    if epochs < 1 or batch_size < 1 or not 0 <= val_fraction < 1:
        raise ValueError("epochs and batch_size must be positive; val_fraction must be in [0, 1)")
    random.seed(seed)
    torch.manual_seed(seed)
    device = pick_device(device)
    images, labels, unknown = load_folder(dataset_dir)
    if unknown:
        log(f"Ignoring folders that are not tile names: {unknown}")
    if not images:
        raise ValueError(f"No labelled tile images found in {dataset_dir}")
    order = list(range(len(images)))
    random.shuffle(order)
    n_val = int(len(order) * val_fraction)
    val_idx, train_idx = order[:n_val], order[n_val:]
    X = to_tensor(images).to(device)
    Y = torch.tensor(labels, device=device)
    log(f"{len(train_idx)} training / {len(val_idx)} validation images, "
        f"{len(set(labels))} tile types (device: {device})")

    net = TileNet().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    tr = torch.tensor(train_idx, device=device)
    acc = float("nan")
    for epoch in range(1, epochs + 1):
        net.train()
        perm = tr[torch.randperm(len(tr), device=device)]
        for i in range(0, len(perm), batch_size):
            idx = perm[i:i + batch_size]
            loss = F.cross_entropy(net(augment(X[idx])), Y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        if val_idx:
            net.eval()
            with torch.no_grad():
                v = torch.tensor(val_idx, device=device)
                acc = (net(X[v]).argmax(-1) == Y[v]).float().mean().item()
        log(f"epoch {epoch:2d}  loss {loss.item():.4f}  val acc {acc:.3f}")
    torch.save({"state_dict": net.cpu().state_dict(), "labels": TILE_NAMES, "val_acc": acc}, out_path)
    log(f"Saved tile classifier to {out_path}")
    return acc


class TileClassifier:
    def __init__(self, path, device="cpu"):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        if ckpt["labels"] != TILE_NAMES:
            raise ValueError("Tile classifier labels do not match this version of mantai_ml.")
        self.net = TileNet()
        self.net.load_state_dict(ckpt["state_dict"])
        self.net.to(device).eval()
        self.device = device
        self.labels = ckpt["labels"]

    @torch.no_grad()
    def predict(self, crops):
        """[(tile_name, confidence), ...] for a list of BGR crops."""
        if not crops:
            return []
        probs = torch.softmax(self.net(to_tensor(crops).to(self.device)), -1)
        conf, idx = probs.max(-1)
        return [(self.labels[i], c) for i, c in zip(idx.tolist(), conf.tolist())]
