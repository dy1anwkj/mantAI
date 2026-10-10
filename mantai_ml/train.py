"""
Train the policy network on decision records (imitation learning).

The network learns to copy the decisions in the records. Records from games
the player went on to win can be weighted up (win_weight), so the model leans
towards decisions that led to wins. Games are split into train / validation
by game_id, so the validation score is on games the model never saw.
"""

import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from .features import encode_call_record, encode_record, legal_mask, CALL_TYPES
from .model import PolicyNet, load_model, pick_device, save_model
from .records import read_records
from .tiles import TILE_INDEX


def load_dataset(paths, controllers=None, winners_only=False, win_weight=2.0):
    discards = defaultdict(list)   # game_id -> [(x, mask, y, w)]
    calls = defaultdict(list)      # game_id -> [(x, y, w)]
    for r in read_records(paths):
        if controllers and r.get("controller") not in controllers:
            continue
        outcome = r.get("outcome") or {}
        won = outcome.get("won", False)
        if winners_only and not won:
            continue
        weight = win_weight if won else 1.0
        gid = f"{r.get('source')}:{r.get('game_id')}"
        state = encode_record(r)
        if r["kind"] == "discard" and r["tile"] in TILE_INDEX:
            discards[gid].append((state, legal_mask(r["hand"]), TILE_INDEX[r["tile"]], weight))
        elif r["kind"] == "call" and r.get("call_type") in CALL_TYPES:
            calls[gid].append((encode_call_record(state, r), float(r["taken"]), weight))
    return discards, calls


def _split(groups, val_fraction, rng):
    ids = sorted(groups)
    rng.shuffle(ids)
    n_val = max(1, int(len(ids) * val_fraction)) if len(ids) > 1 else 0
    val, train = ids[:n_val], ids[n_val:]
    return [x for g in train for x in groups[g]], [x for g in val for x in groups[g]]


def _tensors(rows, device):
    cols = list(zip(*rows))
    return [torch.tensor(np.array(c), dtype=torch.float32, device=device) for c in cols]


def _split_decisions(discards, calls, val_fraction, rng):
    """Keep every decision from a game on the same side of the split."""
    if not 0 <= val_fraction < 1:
        raise ValueError("val_fraction must be between 0 (inclusive) and 1 (exclusive)")
    ids = sorted(set(discards) | set(calls))
    rng.shuffle(ids)
    n_val = max(1, int(len(ids) * val_fraction)) if len(ids) > 1 and val_fraction > 0 else 0
    validation = set(ids[:n_val])
    def partition(groups):
        return ([row for gid, rows in groups.items() if gid not in validation for row in rows],
                [row for gid, rows in groups.items() if gid in validation for row in rows])
    return (*partition(discards), *partition(calls))


def train(paths, out_path, epochs=15, batch_size=256, lr=1e-3, hidden=256,
          controllers=None, winners_only=False, win_weight=2.0,
          val_fraction=0.1, seed=0, device=None, log=print, init=None):
    rng = random.Random(seed)
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    torch.manual_seed(seed)
    device = pick_device(device)

    discards, calls = load_dataset(paths, controllers, winners_only, win_weight)
    d_train, d_val, c_train, c_val = _split_decisions(discards, calls, val_fraction, rng)
    if not d_train:
        raise ValueError("No discard records to train on. Check the file and filters.")
    log(f"Training on {len(d_train)} discards / {len(c_train)} calls, "
        f"validating on {len(d_val)} / {len(c_val)} (device: {device})")

    net = load_model(init, device)[0] if init else PolicyNet(hidden=hidden).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)

    Xd, Md, Yd, Wd = _tensors(d_train, device)
    Yd = Yd.long()
    Xc = Yc = Wc = None
    if c_train:
        Xc, Yc, Wc = _tensors(c_train, device)

    metrics = {}
    for epoch in range(1, epochs + 1):
        net.train()
        perm = torch.randperm(len(Xd), device=device)
        total = 0.0
        for i in range(0, len(perm), batch_size):
            idx = perm[i:i + batch_size]
            loss = (F.cross_entropy(net.discard_logits(Xd[idx], Md[idx]), Yd[idx], reduction="none")
                    * Wd[idx]).mean()
            if Xc is not None:
                cidx = torch.randint(0, len(Xc), (min(batch_size, len(Xc)),), device=device)
                loss = loss + (F.binary_cross_entropy_with_logits(
                    net.call_logit(Xc[cidx]), Yc[cidx], reduction="none") * Wc[cidx]).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(idx)
        metrics = evaluate_accuracy(net, d_val, c_val, device)
        metrics["train_loss"] = round(total / len(Xd), 4)
        log(f"epoch {epoch:2d}  loss {metrics['train_loss']:.4f}  "
            f"discard acc {metrics.get('discard_acc', float('nan')):.3f}  "
            f"call acc {metrics.get('call_acc', float('nan')):.3f}")

    metrics.update(n_discards=len(d_train), n_calls=len(c_train))
    save_model(out_path, net.cpu(), metrics)
    log(f"Saved model to {out_path}")
    return metrics


@torch.no_grad()
def evaluate_accuracy(net, d_rows, c_rows, device):
    net.eval()
    out = {}
    if d_rows:
        X, M, Y, _ = _tensors(d_rows, device)
        pred = net.discard_logits(X, M).argmax(-1)
        out["discard_acc"] = round((pred == Y.long()).float().mean().item(), 4)
    if c_rows:
        X, Y, _ = _tensors(c_rows, device)
        pred = (net.call_logit(X) > 0).float()
        out["call_acc"] = round((pred == Y).float().mean().item(), 4)
    return out
