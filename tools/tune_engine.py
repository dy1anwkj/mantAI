"""
Tune Player 1's strategy-engine weights by simulation.

Each candidate change to the weights is played on the same deals as the
current weights (every game is fully decided by its seed), so the
difference between them comes from the decisions, not from luck of the
deal. A change is kept only when it wins more units per game by more than
`z` standard errors of the paired difference. Every round uses fresh deals,
and the result is finally checked against the starting weights on deals the
search never saw.

    python -m tools.tune_engine --games 4000 --rounds 2 --check 20000

Prints each step and writes the best weights to data/tuned_weights.json.
Players 2-4 keep their styles (mahjong_simulator.OPPONENT_STYLES).
"""

import argparse
import json
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor

import mahjong_simulator as sim
from Decision_Strategy_engine import DEFAULT_WEIGHTS, StrategyEngine


def play(weights, seeds):
    """Units Player 1 won in each game, Player 1 played by the engine with
    `weights`."""
    units = []
    for seed in seeds:
        g = sim.MahjongGame(rand=random.Random(seed))
        g.player1, g.verbose = "engine", False
        g.strategy = StrategyEngine(weights)
        g.deal()
        session = sim.GameSession(game=g)
        while not session.is_over:
            session.apply(sim.choose_action(g, session.decision))
        units.append(session.result()["total"][0])
    return units


def run(pool, weights, seeds, workers):
    chunks = [seeds[i::workers] for i in range(workers)]
    out = [None] * len(seeds)
    for i, part in enumerate(pool.map(play, [weights] * workers, chunks)):
        out[i::workers] = part
    return out


def paired(a, b):
    """Mean of b - a per game and its standard error."""
    d = [y - x for x, y in zip(a, b)]
    n = len(d)
    mean = sum(d) / n
    var = sum((x - mean) ** 2 for x in d) / (n - 1)
    return mean, (var / n) ** 0.5


def candidates(w):
    """Changes to try from weights w: one setting at a time, except the
    late-fold pair, which only works together."""
    out = []
    for key in ("ukeire", "structure", "danger", "draw_probability", "honor_pong_bonus",
                "tai_potential", "no_tai_ready"):
        for f in (0.5, 2.0):
            out.append({key: round(w[key] * f, 3)})
    for d in (-2, -1, 1, 2):
        v = w["call_widen"] + d
        if 0 <= v <= 8:
            out.append({"call_widen": v})
    for key in ("fresh_danger", "flush_danger"):
        if w[key] == 0:
            out += [{key: v} for v in (50.0, 150.0, 400.0)]
        else:
            out += [{key: round(w[key] * f, 1)} for f in (0.5, 2.0)] + [{key: 0.0}]
    out += [{"fresh_wall": v} for v in (5, 7, 12, 20) if v != w["fresh_wall"] and w["fresh_danger"]]
    out += [{"flush_melds": v} for v in (2, 3) if v != w["flush_melds"] and w["flush_danger"]]
    if w["fold"] == 0:
        out += [{"fold_wall": fw, "fold": f} for fw in (10, 20, 30) for f in (0.6, 1.0)]
    else:
        out += [{"fold_wall": max(0, w["fold_wall"] + d)} for d in (-10, 10)]
        out += [{"fold": f} for f in (0.3, 0.6, 1.0) if f != w["fold"]] + [{"fold": 0.0}]
    return out


def tune(start, games=4000, rounds=2, check=20000, z=2.0, workers=None, seed=0, log=print):
    workers = workers or os.cpu_count() or 1
    rng = random.Random(seed)
    w = dict(DEFAULT_WEIGHTS)
    w.update(start)
    begin = dict(w)
    steps = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for rnd in range(1, rounds + 1):
            seeds = [rng.randrange(2 ** 62) for _ in range(games)]
            t0 = time.time()
            base = run(pool, w, seeds, workers)
            log(f"\nRound {rnd}: current weights {sum(base) / games:+.3f} units/game "
                f"on {games} deals ({time.time() - t0:.0f}s)")
            for change in candidates(w):
                trial = dict(w, **change)
                got = run(pool, trial, seeds, workers)
                mean, se = paired(base, got)
                kept = mean > z * se
                log(f"  {json.dumps(change):42s} {mean:+.3f} +/- {se:.3f}" + ("  KEEP" if kept else ""))
                if kept:
                    w, base = trial, got
                    steps.append({"round": rnd, "change": change, "gain": round(mean, 3)})

        log(f"\nChecking on {check} new deals: starting weights vs tuned weights")
        seeds = [rng.randrange(2 ** 62) for _ in range(check)]
        a, b = run(pool, begin, seeds, workers), run(pool, w, seeds, workers)
        mean, se = paired(a, b)
        log(f"  starting {sum(a) / check:+.3f}, tuned {sum(b) / check:+.3f} units/game: "
            f"{mean:+.3f} +/- {se:.3f}")
    changed = {k: v for k, v in w.items() if v != DEFAULT_WEIGHTS[k]}
    return {"weights": changed, "steps": steps, "check_games": check,
            "start_units": sum(a) / check, "tuned_units": sum(b) / check,
            "gain": mean, "gain_se": se}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--games", type=int, default=4000, help="deals per comparison")
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--check", type=int, default=20000, help="deals for the final check")
    p.add_argument("--z", type=float, default=2.0, help="standard errors a change must win by")
    p.add_argument("--workers", type=int)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="data/tuned_weights.json")
    a = p.parse_args()
    result = tune(sim.PLAYER1_WEIGHTS, a.games, a.rounds, a.check, a.z, a.workers, a.seed)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nTuned weights: {result['weights']}  (written to {a.out})")
