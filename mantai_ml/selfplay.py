"""
Self-play reinforcement learning: the policy network teaches itself to win
units under the Singapore rules in the simulator (SPGG 2024, see docs/),
without copying any player's decisions.

Like AlphaGo's reinforcement-learning stage, the model plays games, then
makes the decisions that led to more units than expected more likely, and
the ones that led to fewer less likely (actor-critic policy gradient). A
value head learns how many units a position is worth, which is what
"expected" means here.

A new network starts from the book prior (features.book_prior, the study
library in docs/ read by knowledge.py): it already
prefers discards that keep the hand close to a legal win with enough tai,
and self-play learns what to change on top of that (defence, calls, reading
the table). With --prior 0 it starts from nothing instead; such a model
almost never wins, so at first every game looks equally bad and nothing is
learned. Early on it also gets a progress
reward taken from the rules ("complete shape first, then at least one tai",
docs/07-08): a step closer to a winning shape, or reaching a ready hand
that scores the minimum tai, is rewarded. That reward fades to nothing over
`shaping_iters` iterations, after which only units won count.

Player 1 is the learning model. Players 2-4 are the simulator's opponents
(each with its own style), standing in for the humans at the table.

    python -m mantai_ml selfplay --iterations 150 --games 400 --out models/selfplay.pt
    python -m mantai_ml evaluate --model models/selfplay.pt --games 1000 --compare

Games are played in worker processes with a numpy copy of the network, so
the workers need no torch.
"""

import random

import numpy as np

from .features import (ANALYSIS_AT, N, STATE_SIZE, book_prior, call_analysis, call_prior, encode_call,
                       encode_state, legal_mask, state_from_game)
from .tiles import PLAY_TILES

# Where features.discard_analysis sits in the state vector: shanten / 8,
# useful tiles / 60 and ready-with-tai, one value per tile thrown.
_ANALYSIS = ANALYSIS_AT
SHAPING_DISCOUNT = 0.8   # per own discard: progress counts most for the discards just before it


def progress(x, a):
    """How close the hand is to a legal win after throwing tile index `a`:
    minus its shanten, plus a little for useful tiles and for being ready
    with a hand that reaches the minimum tai."""
    sh = x[_ANALYSIS + a] * 8
    uk = x[_ANALYSIS + N + a]
    ready = x[_ANALYSIS + 2 * N + a]
    return -sh + 0.5 * uk + 0.5 * ready


# --------------------------------------------------------------------------
# Playing (numpy, in worker processes)
# --------------------------------------------------------------------------
def _mlp(x, layers):
    for i, (w, b) in enumerate(layers):
        x = x @ w + b
        if i < len(layers) - 1:
            x = np.maximum(x, 0)
    return x


class SamplingPolicy:
    """Plays Player 1 by sampling from the network, and remembers every
    choice with the units Player 1 had been paid before it."""

    def __init__(self, weights, rand, greedy=False):
        self.discard_layers, self.call_layers, self.prior = weights[:3]
        self.engine_calls = weights[3] if len(weights) > 3 else False
        self.rand = rand
        self.greedy = greedy
        self.discards = []      # (state, mask, action, instant before, progress after, best progress)
        self.calls = []         # (call input, taken, instant before, discards made before it)

    def choose_discard(self, game, player):
        s = state_from_game(game, player)
        x = encode_state(**s)
        mask = legal_mask(s["hand"])
        logits = _mlp(x, self.discard_layers)
        if self.prior:
            logits = logits + self.prior * book_prior(x)
        logits = np.where(mask > 0, logits, -1e9)
        if self.greedy:
            a = int(np.argmax(logits))
        else:
            p = np.exp(logits - logits.max())
            p /= p.sum()
            a = int(np.searchsorted(np.cumsum(p), self.rand.random() * p.sum()))
            a = min(a, N - 1)
            if mask[a] == 0:            # rounding at the very end of the range
                a = int(np.argmax(logits))
        best = max(progress(x, i) for i in np.flatnonzero(mask))
        self.discards.append((x, mask, a, game.instant[player], progress(x, a), best))
        return PLAY_TILES[a]

    def should_call(self, game, player, tile, call_type):
        if self.engine_calls:
            from Decision_Strategy_engine import engine_should_call
            engine = game.strategy if player == 0 else game.opponent_engines[player]
            return engine_should_call(game, player, tile, call_type, engine)
        s = state_from_game(game, player)
        x = encode_state(**s)
        own = [t for m in s["melds"] for t in m]
        a = call_analysis(s["hand"], s["melds"], s["visible"] + own, s["bonus"],
                          s["seat"], s["prevailing"], tile, call_type)
        xc = encode_call(x, tile, call_type, a)
        logit = float(_mlp(xc, self.call_layers)[0])
        if self.prior:
            logit += float(call_prior(xc))
        p = 1.0 / (1.0 + np.exp(-logit))
        taken = p >= 0.5 if self.greedy else self.rand.random() < p
        self.calls.append((xc, float(taken), game.instant[player], len(self.discards)))
        return bool(taken)

    def chi_sequence(self, game, player, tile):
        # The model decides whether to Chi; the run kept is the one that
        # leaves the hand closest to ready.
        from Decision_Strategy_engine import engine_chi_sequence
        engine = game.strategy if player == 0 else game.opponent_engines[player]
        seq = engine_chi_sequence(game, player, tile, engine)
        if seq is None:
            options = game.chi_options(player, tile)
            seq = options[0] if options else None
        return seq


def play_games(weights, seeds, greedy=False, opponent_weights=None):
    """Play one game per seed with Player 1 sampling from `weights`.
    Returns (discard rows, call rows, game summaries). Each row carries the
    units Player 1 won from that decision to the end of the game, and its
    progress return: the discounted gains in progress from there on."""
    import mahjong_simulator as sim

    discards, calls, games = [], [], []
    for seed in seeds:
        rand = random.Random(seed)
        g = sim.MahjongGame(rand=rand)
        g.player1, g.verbose = "ml", False
        policy = SamplingPolicy(weights, rand, greedy)
        g.ml_policy = policy
        g.deal()
        seat_policies = {}
        if opponent_weights:
            from .search import decide
            opponent_rand = random.Random(seed ^ 0x5E1F)
            for player in (1, 2, 3):
                if opponent_rand.random() < 0.5:
                    seat_policies[player] = SamplingPolicy(
                        opponent_rand.choice(opponent_weights), opponent_rand, greedy=True)
        session = sim.GameSession(game=g, check=True)
        while not session.is_over:
            d = session.decision
            action = (decide(seat_policies[d.player], g, d) if d.player in seat_policies
                      else sim.choose_action(g, d))
            session.apply(action)
        r = session.result()
        total = r["total"][0]
        steps = policy.discards
        # Progress gained by each discard since the last one; the first is
        # measured from the best the opening hand could do.
        gains = [phi - (steps[k - 1][4] if k else steps[0][5])
                 for k, (_, _, _, _, phi, _) in enumerate(steps)]
        shaped, acc = [0.0] * (len(steps) + 1), 0.0
        for k in range(len(steps) - 1, -1, -1):
            acc = gains[k] + SHAPING_DISCOUNT * acc
            shaped[k] = acc
        for k, (x, mask, a, before, _, _) in enumerate(steps):
            discards.append((x, mask, a, total - before, shaped[k]))
        for xc, taken, before, k in policy.calls:
            calls.append((xc, taken, total - before, shaped[k]))
        games.append({"units": total, "won": r["winner"] == 0, "discards": len(steps),
                      "best_progress": sum(phi >= best - 1e-6 for _, _, _, _, phi, best in steps),
                      "shooter": r["shooter"] == 0, "draw": r["winner"] is None})
    return discards, calls, games


# --------------------------------------------------------------------------
# Learning (torch, in the main process)
# --------------------------------------------------------------------------
def _numpy_weights(net):
    """The policy's layers as numpy (W, b) pairs, plus its rules prior weight,
    for the workers."""
    from torch import nn

    def layers(seq):
        return [(m.weight.detach().cpu().numpy().T.copy(), m.bias.detach().cpu().numpy().copy())
                for m in seq if isinstance(m, nn.Linear)]
    return (layers(net.discard), layers(net.call), getattr(net, "prior", 0.0),
            getattr(net, "engine_calls", False))


def _ppo_update(net, critic, opt, X, M, A, R, XC, TK, RC, entropy, value_coef,
                epochs=4, batch_size=512, clip=0.2):
    """A few passes of PPO over one batch of games: make the chosen actions
    more likely in proportion to how much better they did than the critic
    expected, without moving too far from the policy that played them.
    Returns the discard policy's entropy (how undecided it still is)."""
    import torch
    import torch.nn.functional as F
    from torch import nn

    def call_logp(xc, taken):
        return -F.binary_cross_entropy_with_logits(net.call_logit(xc), taken, reduction="none")

    def advantages(v, r):
        a = r - v
        return (a - a.mean()) / (a.std(unbiased=False) + 1e-6)

    with torch.no_grad():
        old = F.log_softmax(net.discard_logits(X, M), -1).gather(1, A[:, None]).squeeze(1)
        adv = advantages(critic(X).squeeze(-1), R)
        if XC is not None:
            old_c = call_logp(XC, TK)
            adv_c = advantages(critic(XC[:, :STATE_SIZE]).squeeze(-1), RC)

    def clipped(logp, old_logp, a):
        ratio = (logp - old_logp).exp()
        return -torch.min(ratio * a, ratio.clamp(1 - clip, 1 + clip) * a).mean()

    params = list(net.parameters()) + list(critic.parameters())
    net.train()
    ent_total, steps = 0.0, 0
    for _ in range(epochs):
        perm = torch.randperm(len(X))
        n_batches = max(1, (len(X) + batch_size - 1) // batch_size)
        for b in range(n_batches):
            idx = perm[b * batch_size:(b + 1) * batch_size]
            logp_all = F.log_softmax(net.discard_logits(X[idx], M[idx]), -1)
            logp = logp_all.gather(1, A[idx][:, None]).squeeze(1)
            ent = -(logp_all.exp() * logp_all.masked_fill(M[idx] == 0, 0)).sum(-1).mean()
            v = critic(X[idx]).squeeze(-1)
            loss = clipped(logp, old[idx], adv[idx]) + value_coef * F.mse_loss(v, R[idx]) - entropy * ent
            if XC is not None:
                cidx = torch.randint(0, len(XC), (min(batch_size, len(XC)),))
                vc = critic(XC[cidx, :STATE_SIZE]).squeeze(-1)
                loss = (loss + clipped(call_logp(XC[cidx], TK[cidx]), old_c[cidx], adv_c[cidx])
                        + value_coef * F.mse_loss(vc, RC[cidx]))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            ent_total += float(ent.detach())
            steps += 1
    net.eval()
    return ent_total / max(1, steps)


def selfplay(out_path, iterations=150, games_per_iter=400, hidden=256, lr=3e-4,
             entropy=0.01, value_coef=0.5, init=None, seed=0, workers=None,
             eval_every=25, ppo_epochs=4, batch_size=512, shaping=1.0, shaping_iters=600,
             start_iter=1, eval_games=1000, prior=2.5, log=print, opponent_models=(), engine_calls=None):
    """Train the policy by self-play and save it (loadable by MLPolicy /
    `evaluate`), plus the best greedy-test version as <out>_best.pt.

    shaping       : weight of the progress reward at the start, in units
    shaping_iters : iterations over which it fades to zero
    prior         : weight of the rules prior for a new network (see PolicyNet);
                    a continued network keeps its own
    init          : continue from a saved policy (and its critic, if saved)
    start_iter    : iteration number to continue from (for the fade)"""
    import os
    from concurrent.futures import ProcessPoolExecutor

    import torch
    from torch import nn

    from .model import PolicyNet, load_model, save_model

    if min(iterations, games_per_iter, eval_every, eval_games, ppo_epochs, batch_size, start_iter) < 1:
        raise ValueError("iterations, games, evaluation intervals, epochs and batch sizes must be positive")
    if workers is not None and workers < 1:
        raise ValueError("workers must be positive")

    torch.manual_seed(seed)
    if init:
        net, _ = load_model(init)
        hidden = net.hidden              # the critic must match the saved model's size
        for m in net.modules():          # self-play runs without dropout
            if isinstance(m, nn.Dropout):
                m.p = 0.0
    else:
        net = PolicyNet(hidden=hidden, dropout=0.0, prior=prior)
    if engine_calls is not None:
        net.engine_calls = engine_calls
    # Critic: how many units a position is worth to Player 1 from here on.
    critic = nn.Sequential(nn.Linear(STATE_SIZE, hidden), nn.ReLU(),
                           nn.Linear(hidden, hidden // 2), nn.ReLU(),
                           nn.Linear(hidden // 2, 1))
    critic_path = os.path.splitext(out_path)[0] + "_critic.pt"
    if init:
        from .search import critic_path as find_critic
        initial_critic = find_critic(init)
        if initial_critic is not None:
            critic.load_state_dict(torch.load(initial_critic, map_location="cpu"))
    opt = torch.optim.Adam(list(net.parameters()) + list(critic.parameters()), lr=lr)
    best_path = os.path.splitext(out_path)[0] + "_best.pt"
    best_critic_path = os.path.splitext(best_path)[0] + "_critic.pt"
    best_units = None

    workers = workers or os.cpu_count() or 1
    master = random.Random(seed)
    t = lambda a: torch.tensor(np.array(a), dtype=torch.float32)
    history = []
    frozen_opponents = [_numpy_weights(load_model(path)[0]) for path in opponent_models]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        def evaluate_current():
            eval_seeds = [10 ** 9 + i for i in range(eval_games)]
            ev = []
            for _, _, rows in pool.map(play_games, [_numpy_weights(net)] * workers,
                                      [eval_seeds[i::workers] for i in range(workers)],
                                      [True] * workers):
                ev += rows
            return ev

        # The starting policy is a candidate too. A continuation run must
        # not label a weaker learned checkpoint "best" just because it is new.
        net.eval()
        initial = evaluate_current()
        best_units = sum(g["units"] for g in initial) / len(initial)
        initial_win = 100.0 * sum(g["won"] for g in initial) / len(initial)
        save_model(best_path, net, {"selfplay_iterations": start_iter - 1,
                                   "test_units": best_units, "test_win": initial_win,
                                   "history": [], "selection": "starting policy"})
        torch.save(critic.state_dict(), best_critic_path)
        log(f"Starting policy on {len(initial)} selection games: {best_units:+.2f} units/game, "
            f"win {initial_win:.1f}%")
        for it in range(start_iter, start_iter + iterations):
            beta = shaping * max(0.0, 1.0 - (it - 1) / shaping_iters) if shaping_iters else 0.0
            net.eval()
            weights = _numpy_weights(net)
            seeds = [master.randrange(2 ** 62) for _ in range(games_per_iter)]
            chunks = [seeds[i::workers] for i in range(workers)]
            D, C, G = [], [], []
            opponents = frozen_opponents + [weights] if frozen_opponents else None
            for d, c, g in pool.map(play_games, [weights] * workers, chunks,
                                    [False] * workers, [opponents] * workers):
                D += d
                C += c
                G += g

            X, M = t([r[0] for r in D]), t([r[1] for r in D])
            A = torch.tensor([r[2] for r in D])
            R = t([r[3] + beta * r[4] for r in D])
            if C:
                XC, TK = t([r[0] for r in C]), t([r[1] for r in C])
                RC = t([r[2] + beta * r[3] for r in C])
            else:
                XC = TK = RC = None
            ent = _ppo_update(net, critic, opt, X, M, A, R, XC, TK, RC,
                              entropy, value_coef, epochs=ppo_epochs, batch_size=batch_size)

            n = len(G)
            stats = {"iteration": it, "units": sum(g["units"] for g in G) / n,
                     "win": 100.0 * sum(g["won"] for g in G) / n,
                     "shooter": 100.0 * sum(g["shooter"] for g in G) / n,
                     "entropy": ent, "shaping": beta,
                     "best_progress": 100.0 * sum(g["best_progress"] for g in G)
                     / max(1, sum(g["discards"] for g in G))}
            history.append(stats)
            log(f"iter {it:3d}  {stats['units']:+.2f} units/game  win {stats['win']:4.1f}%  "
                f"shooter {stats['shooter']:4.1f}%  best-progress discards {stats['best_progress']:4.1f}%  "
                f"entropy {stats['entropy']:.2f}  progress weight {beta:.2f}")

            if it % eval_every == 0 or it == start_iter + iterations - 1:
                # Greedy play on the same fresh deals every time: how strong
                # the model is right now, comparable from one test to the next.
                ev = evaluate_current()
                units = sum(g["units"] for g in ev) / len(ev)
                win = 100.0 * sum(g["won"] for g in ev) / len(ev)
                meta = {"selfplay_iterations": it, "test_units": units, "test_win": win,
                        "history": history[-eval_every:]}
                save_model(out_path, net, meta)
                torch.save(critic.state_dict(), critic_path)
                note = ""
                if best_units is None or units > best_units:
                    best_units = units
                    save_model(best_path, net, meta)
                    torch.save(critic.state_dict(), best_critic_path)
                    note = f", new best -> {best_path}"
                log(f"  greedy test on {len(ev)} games: {units:+.2f} units/game, win {win:.1f}%  "
                    f"(saved to {out_path}{note})")
    return history
