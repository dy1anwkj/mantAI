"""
mantAI machine learning command line. Run from the project folder:

  Gameplay
    python -m mantai_ml record   --games 2000 --out data/sim.jsonl
    python -m mantai_ml train    --data data/sim.jsonl --out models/policy.pt
    python -m mantai_ml evaluate --model models/policy.pt --games 500

  Self-play (AlphaGo-style: starts from the study library in docs/, then
  teaches itself under the rules; no copying)
    python -m mantai_ml selfplay --iterations 1200 --games 400 --shaping 0 --out models/alpha.pt
    python -m mantai_ml evaluate --model models/alpha_best.pt --games 1000 --compare
    python -m mantai_ml evaluate --model models/alpha_best.pt --games 200 --search 64 --compare

  Video
    python -m mantai_ml crops       --video game.mp4 --out crops/ [--config roi.json]
    python -m mantai_ml train-tiles --dataset data/tiles --out models/tiles.pt
    python -m mantai_ml video       --video game.mp4 --config roi.json \\
                                    --tiles-model models/tiles.pt --out data/video.jsonl

  Then train on both:
    python -m mantai_ml train --data data/sim.jsonl data/video.jsonl --out models/policy.pt

  Try everything on synthetic data first:
    python -m mantai_ml demo
"""

import argparse
import os
import sys


def _ensure_dir(path):
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)


def cmd_record(a):
    from .recorder import record_games
    _ensure_dir(a.out)
    n, stats = record_games(a.games, a.out, player1=a.player1, seed=a.seed)
    print(f"Wrote {n} decisions from {a.games} games to {a.out} "
          f"(Player 1 won {stats['p1_wins']} games)")


def cmd_train(a):
    from .train import train
    _ensure_dir(a.out)
    train(a.data, a.out, epochs=a.epochs, hidden=a.hidden, lr=a.lr,
          controllers=set(a.controllers) if a.controllers else None,
          winners_only=a.winners_only, win_weight=a.win_weight, seed=a.seed, device=a.device, init=a.init)


def cmd_evaluate(a):
    import mahjong_simulator as sim
    if a.search:
        from .search import evaluate_search
        stats = evaluate_search(a.model, a.games, a.search, seed=a.seed, reference=a.search_reference)
        sim.print_report(stats, label=f"Player 1 = search ({a.search} playouts a discard)")
    else:
        from .ml_player import evaluate_policy
        stats = evaluate_policy(a.model, a.games, seed=a.seed)
        sim.print_report(stats, label="Player 1 = ML model")
    if a.compare:
        sim.print_report(sim.simulate(a.games, seed=a.seed, progress=False), label="Player 1 = strategy engine")


def cmd_train_value(a):
    from .value_train import train_value
    train_value(a.model, a.out, games=a.games, val_games=a.val_games,
                epochs=a.epochs, workers=a.workers, seed=a.seed)


def cmd_selfplay(a):
    from .selfplay import selfplay
    _ensure_dir(a.out)
    selfplay(a.out, iterations=a.iterations, games_per_iter=a.games, hidden=a.hidden, lr=a.lr,
             entropy=a.entropy, init=a.init, seed=a.seed, workers=a.workers,
             eval_every=a.eval_every, shaping=a.shaping, shaping_iters=a.shaping_iters,
             start_iter=a.start_iter, eval_games=a.eval_games, prior=a.prior,
             opponent_models=a.opponent_models, engine_calls=a.engine_calls)


def cmd_crops(a):
    from .video.crops import extract_crops
    extract_crops(a.video, a.out, a.config, a.tiles_model, sample_fps=a.fps)


def cmd_train_tiles(a):
    from .video.classifier import train_tile_classifier
    _ensure_dir(a.out)
    train_tile_classifier(a.dataset, a.out, epochs=a.epochs, device=a.device)


def cmd_video(a):
    from .video.reader import video_to_records
    _ensure_dir(a.out)
    outcome = None
    if a.result:
        outcome = {"won": a.result == "won", "draw": a.result == "draw", "self_drawn": False, "tai": 0}
    for v in a.video:
        video_to_records(v, a.tiles_model, a.config, a.out, outcome=outcome, debug_dir=a.debug)


def cmd_demo(a):
    """End-to-end run on synthetic data, written under demo/."""
    from .recorder import record_games
    from .train import train
    from .ml_player import evaluate_policy
    from .video.synth import make_tile_dataset, make_game_video
    from .video.classifier import train_tile_classifier
    from .video.reader import video_to_records
    import mahjong_simulator as sim

    root = a.out
    os.makedirs(root, exist_ok=True)
    j = lambda *p: os.path.join(root, *p)
    print("\n[1/5] Recording simulator gameplay")
    n, _ = record_games(a.games, j("sim.jsonl"), seed=1, progress=False)
    print(f"      {n} decisions")
    print("\n[2/5] Building synthetic tile images and training the tile classifier")
    make_tile_dataset(j("tiles"))
    train_tile_classifier(j("tiles"), j("tiles.pt"), epochs=15)
    print("\n[3/5] Making a synthetic gameplay video and reading it back")
    config, truth = make_game_video(j("game.mp4"), seed=3)
    records = video_to_records(j("game.mp4"), j("tiles.pt"), config, j("video.jsonl"), append=False)
    got = [tuple(r["tile"]) for r in records if r["kind"] == "discard"]
    want = [t for t, _ in truth]
    print(f"      video discards recovered: {sum(g == w for g, w in zip(got, want))} of {len(want)} in order")
    print("\n[4/5] Training the policy on gameplay + video records")
    train([j("sim.jsonl"), j("video.jsonl")], j("policy.pt"), epochs=a.epochs)
    print("\n[5/5] Playing the trained model as Player 1")
    sim.print_report(evaluate_policy(j("policy.pt"), a.eval_games, seed=5, progress=False),
                     label="Player 1 = ML model")


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m mantai_ml", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("record", help="log simulator games as training records")
    s.add_argument("--games", type=int, default=1000)
    s.add_argument("--out", default="data/sim.jsonl")
    s.add_argument("--player1", default="engine", choices=["engine", "ml"])
    s.add_argument("--seed", type=int)
    s.set_defaults(func=cmd_record)

    s = sub.add_parser("train", help="train the policy network on records")
    s.add_argument("--init", help="fine-tune an existing policy on decision records")
    s.add_argument("--data", nargs="+", required=True, help="one or more .jsonl files")
    s.add_argument("--out", default="models/policy.pt")
    s.add_argument("--epochs", type=int, default=15)
    s.add_argument("--hidden", type=int, default=256)
    s.add_argument("--lr", type=float, default=1e-3)
    s.add_argument("--controllers", nargs="*", help="only learn from these, e.g. engine video")
    s.add_argument("--winners-only", action="store_true", help="only learn from games that were won")
    s.add_argument("--win-weight", type=float, default=2.0, help="extra weight on winning games")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--device", help="cpu / mps / cuda (default: best available)")
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("train-value", help="calibrate the search value model on engine-rollout outcomes")
    s.add_argument("--model", required=True)
    s.add_argument("--out")
    s.add_argument("--games", type=int, default=2000)
    s.add_argument("--val-games", type=int, default=500)
    s.add_argument("--epochs", type=int, default=20)
    s.add_argument("--workers", type=int, default=4)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_train_value)

    s = sub.add_parser("evaluate", help="play the model as Player 1 in the simulator")
    s.add_argument("--search-reference", choices=["network", "engine"], default="network",
                   help="which choice search keeps unless a challenger scores clearly better")
    s.add_argument("--model", default="models/policy.pt")
    s.add_argument("--games", type=int, default=500)
    s.add_argument("--seed", type=int)
    s.add_argument("--compare", action="store_true", help="also run the strategy engine for comparison")
    s.add_argument("--search", type=int, default=0,
                   help="play with AlphaGo-style search, this many playouts per discard (0 = network alone)")
    s.set_defaults(func=cmd_evaluate)

    s = sub.add_parser("selfplay", help="train the policy by self-play (reinforcement learning)")
    s.add_argument("--engine-calls", action="store_true", default=None,
                   help="learn discards while the strategy engine chooses calls")
    s.add_argument("--opponent-models", nargs="*", default=[],
                   help="mix frozen learned opponents and the current policy with engine opponents")
    s.add_argument("--iterations", type=int, default=150)
    s.add_argument("--games", type=int, default=400, help="games per iteration")
    s.add_argument("--out", default="models/selfplay.pt")
    s.add_argument("--hidden", type=int, default=256)
    s.add_argument("--lr", type=float, default=3e-4)
    s.add_argument("--entropy", type=float, default=0.01, help="bonus for keeping options open")
    s.add_argument("--init", help="start from a saved policy instead of random weights")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--workers", type=int, help="game-playing processes (default: all cores)")
    s.add_argument("--eval-every", type=int, default=25)
    s.add_argument("--eval-games", type=int, default=1000, help="greedy test games at each check")
    s.add_argument("--shaping", type=float, default=1.0,
                   help="starting weight of the progress reward (0 = units only)")
    s.add_argument("--shaping-iters", type=int, default=600,
                   help="iterations over which the progress reward fades out")
    s.add_argument("--start-iter", type=int, default=1, help="continue numbering from here (with --init)")
    s.add_argument("--prior", type=float, default=2.5,
                   help="start from the study library's opinion (features.book_prior); 0 = blank start")
    s.set_defaults(func=cmd_selfplay)

    s = sub.add_parser("crops", help="cut tile images from a video for labelling")
    s.add_argument("--video", required=True)
    s.add_argument("--out", default="crops")
    s.add_argument("--config", help="region config JSON (default: whole frame)")
    s.add_argument("--tiles-model", help="pre-sort crops with an existing classifier")
    s.add_argument("--fps", type=float, default=1.0)
    s.set_defaults(func=cmd_crops)

    s = sub.add_parser("train-tiles", help="train the tile image classifier")
    s.add_argument("--dataset", required=True, help="folder with one sub-folder per tile")
    s.add_argument("--out", default="models/tiles.pt")
    s.add_argument("--epochs", type=int, default=25)
    s.add_argument("--device")
    s.set_defaults(func=cmd_train_tiles)

    s = sub.add_parser("video", help="turn gameplay video into training records")
    s.add_argument("--video", nargs="+", required=True)
    s.add_argument("--config", required=True, help="region config JSON (see mantai_ml/video/reader.py)")
    s.add_argument("--tiles-model", required=True)
    s.add_argument("--out", default="data/video.jsonl")
    s.add_argument("--result", choices=["won", "lost", "draw"], help="how the filmed game ended")
    s.add_argument("--debug", help="folder for annotated frames of every recognised hand")
    s.set_defaults(func=cmd_video)

    s = sub.add_parser("demo", help="run the whole pipeline on synthetic data")
    s.add_argument("--out", default="demo")
    s.add_argument("--games", type=int, default=1500)
    s.add_argument("--epochs", type=int, default=15)
    s.add_argument("--eval-games", type=int, default=300)
    s.set_defaults(func=cmd_demo)

    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    sys.exit(main())
