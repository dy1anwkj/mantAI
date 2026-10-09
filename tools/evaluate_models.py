"""Evaluate policies on identical seeded deals, with rule checks and paired intervals.

Example: python tools/evaluate_models.py --models models/alpha.pt models/improved_rl_best.pt --games 5000
This measures greedy learned policies without search against engine opponents.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
from pathlib import Path
import random
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def evaluate_batch(model, indices, seed):
    import mahjong_simulator as sim
    from mantai_ml.model import load_model
    from mantai_ml.selfplay import SamplingPolicy, _numpy_weights
    weights = _numpy_weights(load_model(model)[0]) if model else None
    rows = []
    for index in indices:
        try:
            game = sim.MahjongGame(rand=random.Random(seed + index))
            game.player1, game.verbose = ('ml' if model else 'engine'), False
            if model:
                game.ml_policy = SamplingPolicy(weights, random.Random(seed + index), greedy=True)
            game.deal()
            session = sim.GameSession(game=game, check=True)
            while not session.is_over:
                session.apply(sim.choose_action(game, session.decision))
            result = session.result()
            rows.append(dict(index=index, units=result['total'][0], won=result['winner'] == 0,
                             deal_in=result['winner'] is not None and result['winner'] != 0 and result['shooter'] == 0))
        except Exception:
            rows.append(dict(index=index, error=traceback.format_exc()))
    return model or 'engine', rows


def interval(values):
    n = len(values)
    if not n:
        return None
    mean = sum(values) / n
    half = (1.96 * math.sqrt(sum((x - mean)**2 for x in values) / (n - 1) / n)
            if n > 1 else None)
    return dict(mean=mean, approximate_95_percent_ci=[mean-half, mean+half] if half is not None else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', required=True)
    parser.add_argument('--games', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=20261012000)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--no-engine', action='store_true', help='omit the extra strategy-engine benchmark')
    parser.add_argument('--out', default='data/model_comparison.json')
    args = parser.parse_args()
    if args.games < 2 or args.workers < 1:
        parser.error('games must be at least 2 and workers must be positive')
    models = list(dict.fromkeys(str(Path(p).resolve()) for p in args.models))
    policies = models + ([] if args.no_engine else [None])
    results = {model or 'engine': [] for model in policies}
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(evaluate_batch, model, range(start, min(start+100, args.games)), args.seed)
                for model in policies for start in range(0, args.games, 100)]
        for job in as_completed(jobs):
            model, rows = job.result()
            results[model].extend(rows)
            if len(results[model]) % 500 == 0 or len(results[model]) == args.games:
                print(f'{Path(model).name}: {len(results[model])}/{args.games}', flush=True)
    summary, paired = {}, {}
    baseline = {r['index']: r for r in results[models[0]] if 'error' not in r}
    for model, rows in results.items():
        rows.sort(key=lambda r: r['index'])
        good = [r for r in rows if 'error' not in r]
        summary[model] = dict(games=len(rows), errors=len(rows)-len(good),
                             points_per_game=interval([r['units'] for r in good]),
                             win_percent=100*sum(r['won'] for r in good)/len(good) if good else None,
                             deal_in_percent=100*sum(r['deal_in'] for r in good)/len(good) if good else None)
        paired[model] = interval([r['units'] - baseline[r['index']]['units'] for r in good if r['index'] in baseline])
    report = dict(seed=args.seed, games_per_policy=args.games, baseline=models[0], seconds=time.time()-started,
                  summary=summary, paired_difference_from_baseline=paired, hands=results)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(summary=summary, paired_difference_from_baseline=paired), indent=2))
    if any(item['errors'] for item in summary.values()):
        raise SystemExit('Evaluation had errors; see the report before using its results.')


if __name__ == '__main__':
    main()
