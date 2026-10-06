"""Tests for the mantai_ml package. Needs torch, opencv, numpy and pillow,
which the project's mantAI/.venv has (plain python3 skips these tests):
    mantAI/.venv/bin/python -m pytest test_mantai_ml.py
Elsewhere, install them first with: pip install -r requirements-ml.txt
"""
import random
from collections import Counter

import pytest

torch = pytest.importorskip("torch")
cv2 = pytest.importorskip("cv2")

import mahjong_simulator as sim
from mantai_ml.features import CALL_SIZE, STATE_SIZE, encode_call_record, encode_record
from mantai_ml.recorder import record_games
from mantai_ml.records import read_records
from mantai_ml.tiles import ALL_TILES, parse_tile, tile_name
from mantai_ml.video.detect import DetectParams, detect_tiles
from mantai_ml.video.reader import VideoConfig, VideoReader, group_melds
from mantai_ml.video import synth


def B(*r):
    return [('bamboo', x) for x in r]


def test_tile_names_round_trip():
    assert len(ALL_TILES) == 46
    for t in ALL_TILES:
        assert parse_tile(tile_name(t)) == t


@pytest.fixture(scope="module")
def sim_records(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "sim.jsonl"
    n, stats = record_games(25, str(path), seed=4, progress=False)
    assert n > 0 and stats["errors"] == []
    return str(path)


def test_recorded_gameplay_is_complete_and_encodable(sim_records):
    records = list(read_records(sim_records))
    assert {r["kind"] for r in records} == {"discard", "call"}
    for r in records[:300]:
        assert r["player"] == 0 and r["controller"] == "engine"
        assert "outcome" in r
        x = encode_record(r)
        assert x.shape == (STATE_SIZE,)
        if r["kind"] == "discard":
            assert r["tile"] in r["hand"]
            assert len(r["hand"]) + 3 * len(r["melds"]) == 14
        else:
            assert encode_call_record(x, r).shape == (CALL_SIZE,)
    games = Counter(r["game_id"] for r in records)
    assert len(games) == 25


def test_trained_policy_plays_legal_games(sim_records, tmp_path):
    from mantai_ml.train import train
    from mantai_ml.ml_player import evaluate_policy
    model = tmp_path / "policy.pt"
    metrics = train([sim_records], str(model), epochs=3, log=lambda *_: None, device="cpu")
    assert 0 <= metrics["discard_acc"] <= 1
    # check=True inside simulate asserts tile conservation, hand sizes and legal wins
    stats = evaluate_policy(str(model), 20, seed=1, progress=False)
    assert stats["errors"] == []


def test_touching_tiles_are_split_at_seams():
    rng = random.Random(0)
    frame = synth.render_frame({"hand": B(1, 2, 3, 4, 5, 6, 7, 8, 9) + B(1, 2, 3, 4, 5),
                                "melds": [], "bonus": [], "visible": []}, rng)
    boxes = detect_tiles(frame, synth.LAYOUT["hand_roi"], DetectParams(tile_aspect=synth.ASPECT))
    assert len(boxes) == 14
    widths = [w for _, _, w, _ in boxes]
    assert max(widths) - min(widths) <= 4


def test_group_melds_reads_kongs_and_sets():
    tiles = B(5, 5, 5, 5) + B(1, 2, 3) + [('dragon', 'red')] * 3
    assert group_melds(tiles) == [B(5, 5, 5, 5), B(1, 2, 3), [('dragon', 'red')] * 3]


class _NoClf:
    def predict(self, crops):
        return []


def test_reader_turns_pong_then_discard_into_records():
    reader = VideoReader(_NoClf(), VideoConfig(hand_roi=[0, 0, 1, 1]))
    prev = {"hand": B(1, 1, 2, 3, 4, 5, 6, 7, 8, 9) + [('dot', 1), ('dot', 2), ('dot', 5)],
            "melds": [], "bonus": [], "visible": [('bamboo', 1)], "time": 1.0}
    cur = dict(prev, hand=B(2, 3, 4, 5, 6, 7, 8, 9) + [('dot', 1), ('dot', 2)],
               melds=[B(1, 1, 1)], time=2.0)
    reader.on_change(prev, cur)
    kinds = [(r["kind"], tuple(r["tile"]), r.get("call_type")) for r in reader.records]
    assert kinds == [("call", ('bamboo', 1), "pong"), ("discard", ('dot', 5), None)]
    assert len(reader.records[1]["hand"]) + 3 * len(reader.records[1]["melds"]) == 14


def test_video_pipeline_recovers_every_discard(tmp_path):
    from mantai_ml.video.classifier import train_tile_classifier
    from mantai_ml.video.reader import video_to_records
    synth.make_tile_dataset(str(tmp_path / "tiles"), per_class=40, log=lambda *_: None)
    train_tile_classifier(str(tmp_path / "tiles"), str(tmp_path / "tiles.pt"), epochs=10,
                          log=lambda *_: None)
    config, truth = synth.make_game_video(str(tmp_path / "game.mp4"), seed=3, log=lambda *_: None)
    records = video_to_records(str(tmp_path / "game.mp4"), str(tmp_path / "tiles.pt"), config,
                               str(tmp_path / "video.jsonl"), log=lambda *_: None)
    got = [(tuple(r["tile"]), sorted(map(tuple, r["hand"]), key=str))
           for r in records if r["kind"] == "discard"]
    assert got == truth


def _change(*snaps):
    reader = VideoReader(_NoClf(), VideoConfig(hand_roi=[0, 0, 1, 1]))
    for prev, cur in zip(snaps, snaps[1:]):
        reader.on_change(prev, cur)
    return [(r["kind"], tuple(r["tile"]), r.get("call_type")) for r in reader.records]


def test_reader_records_concealed_kong_then_discard():
    base = {"bonus": [], "visible": [], "time": 0.0}
    before = dict(base, hand=B(1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9) + [('dot', 1), ('dot', 2)], melds=[])
    kong = dict(base, hand=B(2, 3, 4, 5, 6, 7, 8, 9) + [('dot', 1), ('dot', 2)], melds=[B(1, 1, 1, 1)])
    after = dict(kong, hand=B(2, 3, 4, 5, 6, 7, 8, 9) + [('dot', 2), ('dot', 3)])
    assert _change(before, kong, after) == [("call", ('bamboo', 1), "self_kong"),
                                            ("discard", ('dot', 1), None)]


def test_reader_does_not_mistake_promoted_kong_for_discard():
    base = {"bonus": [], "visible": [], "time": 0.0}
    seven = ('dot', 7)
    before = dict(base, hand=B(1, 2, 3, 4, 5, 6, 7, 8, 9) + [('dot', 1), seven], melds=[[seven] * 3])
    kong = dict(before, hand=B(1, 2, 3, 4, 5, 6, 7, 8, 9) + [('dot', 1)], melds=[[seven] * 4])
    assert _change(before, kong) == [("call", seven, "self_kong")]


def test_unreadable_meld_region_skips_the_frame(monkeypatch):
    from mantai_ml.video import reader as reader_mod
    cfg = VideoConfig(hand_roi=[0, 0, 1, 0.5], meld_roi=[0, 0.5, 1, 0.5])
    reader = VideoReader(_NoClf(), cfg)
    monkeypatch.setattr(reader_mod, "read_region",
                        lambda frame, roi, *a: (B(1, 2, 3), []) if roi is cfg.hand_roi else (None, []))
    snap, _ = reader.read_frame(None)
    assert snap is None and reader.skipped["low confidence melds"] == 1


def test_synthetic_video_leaves_simulator_rng_alone():
    before = sim.rng
    synth.simulated_snapshots(seed=2)
    assert sim.rng is before


def test_selfplay_games_are_legal_and_carry_returns():
    import torch
    from mantai_ml.model import PolicyNet
    from mantai_ml.selfplay import SHAPING_DISCOUNT, _numpy_weights, play_games
    torch.manual_seed(0)
    weights = _numpy_weights(PolicyNet(hidden=32, dropout=0.0).eval())
    discards, calls, games = play_games(weights, [1, 2, 3])
    assert len(games) == 3 and discards
    for x, mask, a, units, shaped in discards:
        assert x.shape == (STATE_SIZE,) and mask[a] == 1
    for g in games:
        assert 0 <= g["best_progress"] <= g["discards"]
    # The last discard of a game earns only its own progress gain.
    assert 0 < SHAPING_DISCOUNT < 1


def test_numpy_copy_matches_torch_policy():
    import numpy as np
    from mantai_ml.model import PolicyNet
    from mantai_ml.selfplay import _mlp, _numpy_weights
    net = PolicyNet(hidden=32, dropout=0.0).eval()
    x = torch.rand(3, STATE_SIZE)
    want = net.discard(x).detach().numpy()
    got = _mlp(x.numpy(), _numpy_weights(net)[0])
    assert np.allclose(want, got, atol=1e-5)


def test_selfplay_saves_a_model_evaluate_can_play(tmp_path):
    from mantai_ml.ml_player import evaluate_policy
    from mantai_ml.selfplay import selfplay
    out = tmp_path / "sp.pt"
    selfplay(str(out), iterations=2, games_per_iter=8, hidden=32, workers=2,
             eval_every=2, eval_games=4, log=lambda *_: None)
    assert out.exists() and (tmp_path / "sp_best.pt").exists() and (tmp_path / "sp_critic.pt").exists()
    # check=True inside simulate asserts tile conservation, hand sizes and legal wins
    stats = evaluate_policy(str(out), 10, seed=1, progress=False)
    assert stats["errors"] == []
    # Continuing from the saved model picks up its critic too.
    selfplay(str(out), iterations=1, games_per_iter=4, init=str(out), start_iter=3,
             workers=1, eval_every=1, eval_games=2, log=lambda *_: None)
