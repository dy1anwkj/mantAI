"""Let a trained policy play Player 1 in the simulator."""

import numpy as np
import torch

from .features import call_analysis, encode_call, encode_state, legal_mask, state_from_game
from .model import load_model
from .tiles import PLAY_TILES


class MLPolicy:
    def __init__(self, model_path, device="cpu", call_threshold=0.5):
        self.net, self.metrics = load_model(model_path, device)
        self.device = device
        self.call_threshold = call_threshold

    def _state(self, game, player):
        s = state_from_game(game, player)
        return s, encode_state(**s)

    @torch.no_grad()
    def choose_discard(self, game, player):
        s, x = self._state(game, player)
        mask = legal_mask(s["hand"])
        logits = self.net.discard_logits(
            torch.tensor(x[None], device=self.device),
            torch.tensor(mask[None], device=self.device))
        return PLAY_TILES[int(logits.argmax(-1).item())]

    @torch.no_grad()
    def should_call(self, game, player, tile, call_type):
        if self.net.engine_calls:
            from Decision_Strategy_engine import engine_should_call
            engine = game.strategy if player == 0 else game.opponent_engines[player]
            return engine_should_call(game, player, tile, call_type, engine)
        s, x = self._state(game, player)
        own = [t for m in s["melds"] for t in m]
        a = call_analysis(s["hand"], s["melds"], s["visible"] + own, s["bonus"],
                          s["seat"], s["prevailing"], tile, call_type)
        logit = self.net.call_logit(torch.tensor(encode_call(x, tile, call_type, a)[None], device=self.device))
        return torch.sigmoid(logit).item() >= self.call_threshold

    def chi_sequence(self, game, player, tile):
        # Which run to form is left to the strategy engine; the model decides
        # only whether to Chi at all.
        from Decision_Strategy_engine import engine_chi_sequence
        engine = game.strategy if player == 0 else game.opponent_engines[player]
        seq = engine_chi_sequence(game, player, tile, engine)
        if seq is None:
            options = game.chi_options(player, tile)
            seq = options[0] if options else None
        return seq


def evaluate_policy(model_path, n_games=500, seed=None, progress=True):
    """Play n_games with the model as Player 1; returns simulator stats."""
    import mahjong_simulator as sim

    policy = MLPolicy(model_path)

    def setup(game):
        game.ml_policy = policy

    return sim.simulate(n_games, seed=seed, check=True, progress=progress,
                        player1="ml", setup=setup)
