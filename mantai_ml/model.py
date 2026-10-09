"""The policy network: which tile to discard, and whether to take a call."""

import torch
from torch import nn

from .features import CALL_SIZE, N, STATE_SIZE, book_prior, call_prior

FEATURE_VERSION = 3


def pick_device(prefer=None):
    if prefer:
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")       # Apple Silicon GPU
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class PolicyNet(nn.Module):
    """prior: weight of features.book_prior (the study library in docs/)
    added to the discard scores, so a new network already plays the way the
    books say and only learns what to change on top of that (0 = off)."""

    def __init__(self, hidden=256, dropout=0.1, prior=0.0):
        super().__init__()
        self.hidden = hidden
        self.prior = prior
        self.engine_calls = False
        self.discard = nn.Sequential(
            nn.Linear(STATE_SIZE, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, N),
        )
        self.call = nn.Sequential(
            nn.Linear(CALL_SIZE, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def discard_logits(self, state, mask):
        logits = self.discard(state)
        if self.prior:
            logits = logits + self.prior * book_prior(state)
        # Tiles not in the hand can never be discarded.
        return logits.masked_fill(mask == 0, -1e9)

    def call_logit(self, call_input):
        logit = self.call(call_input).squeeze(-1)
        if self.prior:
            logit = logit + call_prior(call_input)
        return logit


def save_model(path, net, metrics=None):
    torch.save({
        "feature_version": FEATURE_VERSION,
        "hidden": net.hidden,
        "prior": getattr(net, "prior", 0.0),
        "engine_calls": getattr(net, "engine_calls", False),
        "state_dict": net.state_dict(),
        "metrics": metrics or {},
    }, path)


def load_model(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if ckpt.get("feature_version") != FEATURE_VERSION:
        raise ValueError(f"{path} was trained with different features; retrain it.")
    net = PolicyNet(hidden=ckpt["hidden"], prior=ckpt.get("prior", 0.0))
    net.engine_calls = ckpt.get("engine_calls", False)
    net.load_state_dict(ckpt["state_dict"])
    net.to(device).eval()
    return net, ckpt.get("metrics", {})
