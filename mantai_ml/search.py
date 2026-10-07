"""
Choosing a discard the way AlphaGo chose a move.

    policy network : the trained network (models/alpha_best.pt) - the study
                     library's opinion (book_prior, docs/) corrected by
                     self-play - gives each legal discard a prior probability.
                     The most likely ones become candidates, and the strategy
                     engine's own choice is always added as one more.
    search         : each candidate is played out on many guessed deals of the
                     hidden tiles (the same deals for every candidate, so the
                     differences come from the discard, not the luck of the
                     deal). Hidden information is why the deals are guessed:
                     Mahjong, unlike Go, does not show the opponents' hands.
    rollouts       : the rest of each guessed hand is played fast by the
                     strategy engine - the part of AlphaGo's fast rollout
                     policy - with Players 2-4 keeping their own styles.
    value network  : the self-play critic (models/alpha_best_critic.pt)
                     estimates how many units the position is worth when the
                     player is next to discard. Each simulation is scored
                     (1 - mix) * value estimate + mix * units the rollout
                     actually won, as AlphaGo mixed its value network with
                     rollout results.
    allocation     : simulations go to candidates by PUCT, AlphaGo's rule:
                     the best average so far plus a bonus for a high prior and
                     few visits. The network's favourite is played unless
                     another candidate won clearly more on the same deals.

Calls (Pong, Chi, Kong) use the network's call head, which starts from the
book's call rules (features.call_prior) and is trained by the same self-play.
"""

import math
import os
import random

import numpy as np

from .features import encode_state, legal_mask, state_from_game
from .ml_player import MLPolicy
from .tiles import PLAY_TILES

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL = os.path.join(_ROOT, "models", "alpha_best.pt")


def critic_path(model_path):
    """Self-play saves the value network beside the policy: <out>_critic.pt,
    and the best test's policy as <out>_best.pt with the latest critic."""
    stem = os.path.splitext(model_path)[0]
    for path in (stem + "_critic.pt", stem.removesuffix("_best") + "_critic.pt"):
        if os.path.exists(path):
            return path
    return None


def _load_critic(path):
    """The critic's layers as numpy (W, b) pairs, or None."""
    if path is None:
        return None
    import torch
    state = torch.load(path, map_location="cpu")
    keys = sorted({k.rsplit(".", 1)[0] for k in state}, key=lambda k: int(k))
    return [(state[k + ".weight"].numpy().T.copy(), state[k + ".bias"].numpy().copy()) for k in keys]


def _mlp(x, layers):
    for i, (w, b) in enumerate(layers):
        x = x @ w + b
        if i < len(layers) - 1:
            x = np.maximum(x, 0)
    return x


def _simulate(snap, tile, seeds, critic, mix):
    """One scored simulation per seed of discarding `tile`: [(score, units, won)].
    Runs in worker processes, so everything it needs comes in as plain data."""
    import mahjong_simulator as sim

    player = snap["player"]
    before = snap["instant"].get(player, 0)
    out = []
    for s in seeds:
        g = sim._playout_game(snap, random.Random(s))
        session = sim.GameSession(game=g, start=False)
        session._first_turn = False
        session._offer_turn(player, claimed=True)          # discard only
        session.apply(sim.Action("discard", tile))
        value = None
        while not session.is_over:
            d = session.decision
            if (value is None and critic is not None and d.player == player
                    and d.kind == "turn" and any(a.kind == "discard" for a in d.actions)):
                x = encode_state(**state_from_game(g, player))
                value = g.instant[player] - before + float(_mlp(x, critic)[0])
            session.apply(sim.choose_action(g, d))
        units = session.result()["total"][player] - before
        score = units if value is None else (1 - mix) * value + mix * units
        out.append((score, units, session.result()["winner"] == player))
    return out


class AlphaSearch:
    """A player for any seat. simulations: total playouts per discard."""

    def __init__(self, model_path=DEFAULT_MODEL, simulations=96, c_puct=1.5, mix=0.5,
                 max_candidates=5, coverage=0.95, batch=8, confidence=1.5, parallel=True, seed=None):
        self.policy = MLPolicy(model_path)
        self.model_path = model_path
        self.critic = _load_critic(critic_path(model_path))
        self.simulations = simulations
        self.c_puct = c_puct
        self.mix = mix
        self.max_candidates = max_candidates
        self.coverage = coverage
        self.batch = batch
        self.confidence = confidence
        self.parallel = parallel
        self.rand = random.Random(seed)
        self.last = None            # the analysis behind the last discard

    # ---- the policy network's opinion --------------------------------------
    def priors(self, game, player):
        """{tile: probability} over the legal discards."""
        import torch
        s = state_from_game(game, player)
        x = encode_state(**s)
        mask = legal_mask(s["hand"])
        with torch.no_grad():
            logits = self.policy.net.discard_logits(torch.tensor(x[None]), torch.tensor(mask[None]))[0]
        p = torch.softmax(logits, -1).numpy()
        return {PLAY_TILES[i]: float(p[i]) for i in np.flatnonzero(mask)}

    def candidates(self, game, player, priors):
        from Decision_Strategy_engine import engine_discard_decision
        ranked = sorted(priors, key=priors.get, reverse=True)
        chosen, mass = [], 0.0
        for tile in ranked:
            if len(chosen) >= self.max_candidates or mass >= self.coverage:
                break
            chosen.append(tile)
            mass += priors[tile]
        engine = game.strategy if player == 0 else game.opponent_engines[player]
        pick = engine_discard_decision(game, player, engine).tile
        if pick not in chosen:
            chosen.append(pick)
        return chosen, pick

    # ---- search --------------------------------------------------------------
    def _run(self, snap, jobs):
        """Results of [(tile, seeds)], in order."""
        if self.parallel:
            try:
                import mahjong_simulator as sim
                pool = sim._playout_pool()
                futures = [[pool.submit(_simulate, snap, tile, seeds[i::2], self.critic, self.mix)
                            for i in range(min(2, len(seeds)))] for tile, seeds in jobs]
                return [[r for f in fs for r in f.result()] for fs in futures]
            except Exception:
                pass        # no worker processes here: run them in this one
        return [_simulate(snap, tile, seeds, self.critic, self.mix) for tile, seeds in jobs]

    def analyse(self, game, player, simulations=None):
        """Search the discard. Returns {"tile", "candidates": [{tile, prior,
        visits, value, units, win} ...] best first, "simulations", "engine"}."""
        import mahjong_simulator as sim
        budget = simulations or self.simulations
        priors = self.priors(game, player)
        tiles, engine_pick = self.candidates(game, player, priors)
        stats = {t: {"scores": [], "units": [], "wins": 0} for t in tiles}
        if len(tiles) > 1:
            snap = sim._table_snapshot(game, player)
            worlds = [self.rand.random() for _ in range(budget)]
            # Every candidate first gets one batch on the same deals.
            jobs = [(t, worlds[:self.batch]) for t in tiles]
            spent = 0
            while jobs:
                for (tile, _), results in zip(jobs, self._run(snap, jobs)):
                    for score, units, won in results:
                        stats[tile]["scores"].append(score)
                        stats[tile]["units"].append(units)
                        stats[tile]["wins"] += won
                spent += sum(len(seeds) for _, seeds in jobs)
                if spent >= budget:
                    break
                every = [x for t in tiles for x in stats[t]["scores"]]
                scale = max(1.0, float(np.std(every)))
                total = sum(len(stats[t]["scores"]) for t in tiles)

                def puct(t):
                    n = len(stats[t]["scores"])
                    q = sum(stats[t]["scores"]) / n / scale
                    return q + self.c_puct * priors[t] * math.sqrt(total) / (1 + n)
                ahead = sorted(tiles, key=puct, reverse=True)[:2]
                jobs = []
                for t in ahead:
                    n = len(stats[t]["scores"])
                    seeds = worlds[n:n + self.batch] if n < len(worlds) else [self.rand.random() for _ in range(self.batch)]
                    jobs.append((t, seeds))
        rows = []
        for t in tiles:
            n = len(stats[t]["scores"])
            rows.append({"tile": t, "prior": priors[t], "visits": n,
                         "value": sum(stats[t]["scores"]) / n if n else None,
                         "units": sum(stats[t]["units"]) / n if n else None,
                         "win": 100.0 * stats[t]["wins"] / n if n else None})
        pick = self._decide(tiles, priors, stats)
        rows.sort(key=lambda r: (r["tile"] == pick, r["value"] if r["value"] is not None else -1e9,
                                 r["prior"]), reverse=True)
        self.last = {"tile": pick, "candidates": rows, "engine": engine_pick,
                     "simulations": sum(r["visits"] for r in rows)}
        return self.last

    def _decide(self, tiles, priors, stats):
        """The network's favourite stands unless another candidate scored more
        on the same guessed deals by more than `confidence` standard errors.
        With thousands of simulations AlphaGo could simply play the most
        visited move; with a few dozen per tile the luck of the deals is too
        loud for that."""
        favourite = max(tiles, key=priors.get)
        base = stats[favourite]["scores"]
        best, best_margin = favourite, 0.0
        for t in tiles:
            other = stats[t]["scores"]
            n = min(len(base), len(other))       # the j-th score of each used deal j
            if t == favourite or n < 2 * self.batch:
                continue
            diffs = [a - b for a, b in zip(other[:n], base[:n])]
            mean = sum(diffs) / n
            spread = (sum((d - mean) ** 2 for d in diffs) / (n - 1)) ** 0.5
            margin = mean - self.confidence * spread / n ** 0.5
            if margin > best_margin:
                best, best_margin = t, margin
        return best

    # ---- the MLPolicy interface (simulator Player 1, any seat in combined play)
    def choose_discard(self, game, player):
        return self.analyse(game, player)["tile"]

    def should_call(self, game, player, tile, call_type):
        return self.policy.should_call(game, player, tile, call_type)

    def chi_sequence(self, game, player, tile):
        """Which run to make is the seat's engine's choice; the network
        decided whether to Chi at all."""
        from Decision_Strategy_engine import engine_chi_sequence
        engine = game.strategy if player == 0 else game.opponent_engines[player]
        seq = engine_chi_sequence(game, player, tile, engine)
        if seq is None:
            options = game.chi_options(player, tile)
            seq = options[0] if options else None
        return seq


def decide(policy, game, decision):
    """The Action `policy` takes for a GameSession decision of any seat.
    Winning is a rule, not a choice: a legal Hu is always declared."""
    from mahjong_simulator import Action
    player = decision.player
    by_kind = {}
    for a in decision.actions:
        by_kind.setdefault(a.kind, []).append(a)
    if "hu" in by_kind:
        return by_kind["hu"][0]
    if decision.kind == "turn":
        for a in by_kind.get("kong", []):
            if policy.should_call(game, player, a.tile, "self_kong"):
                return a
        return Action("discard", policy.choose_discard(game, player))
    for kind in ("kong", "pong"):
        if kind in by_kind and policy.should_call(game, player, decision.tile, kind):
            return by_kind[kind][0]
    if "chi" in by_kind and policy.should_call(game, player, decision.tile, "chi"):
        seq = policy.chi_sequence(game, player, decision.tile)
        for a in by_kind["chi"]:
            if seq is not None and sorted(a.chi) == sorted(seq):
                return a
    return Action("pass")


def evaluate_search(model_path, n_games=200, simulations=64, seed=None, progress=True):
    """Play n_games with the search as Player 1; returns simulator stats."""
    import mahjong_simulator as sim

    player = AlphaSearch(model_path, simulations=simulations, seed=seed)

    def setup(game):
        game.ml_policy = player

    return sim.simulate(n_games, seed=seed, check=True, progress=progress, player1="ml", setup=setup)
