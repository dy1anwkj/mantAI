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

import numpy as np

import mahjong_simulator as sim
import singapore_rules as sg
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


def test_rules_prior_prefers_discards_closer_to_a_win(tmp_path):
    from mantai_ml.features import encode_state, legal_mask, rules_prior
    from mantai_ml.model import PolicyNet, load_model, save_model
    from mantai_ml.tiles import TILE_INDEX
    # 1-9 bamboo, 2-3-4 dot, a lone 5 dot and a lone North wind: throwing
    # North leaves a ready hand, throwing 5 bamboo breaks a run.
    hand = B(1, 2, 3, 4, 5, 6, 7, 8, 9) + [('dot', 2), ('dot', 3), ('dot', 4),
                                           ('dot', 5), ('wind', 'north')]
    assert not rules_prior(encode_state(hand[:-1], [], [], [], 1, "east", 80)).any()  # 13 tiles: nothing to throw
    x = encode_state(hand, [], [], [], 1, "east", 80)
    p = rules_prior(x)
    assert p[TILE_INDEX[('wind', 'north')]] > p[TILE_INDEX[('bamboo', 5)]]
    net = PolicyNet(hidden=16, dropout=0.0, prior=2.5).eval()
    logits = net.discard_logits(torch.tensor(x[None]), torch.tensor(legal_mask(hand)[None]))
    assert logits.argmax().item() == TILE_INDEX[('wind', 'north')]
    save_model(tmp_path / "p.pt", net)
    assert load_model(tmp_path / "p.pt")[0].prior == 2.5



# ---------------------------------------------------------------------------
# The study library (docs/) as the network reads it: knowledge.py, book_prior
# ---------------------------------------------------------------------------
def W(*r):
    return [('wan', x) for x in r]


def D(*r):
    return [('dot', x) for x in r]


def _book_top(hand, visible=(), wall=100, opponents=None):
    from mantai_ml.features import book_prior, encode_state, legal_mask
    from mantai_ml.tiles import PLAY_TILES
    opponents = opponents or [{"discards": [], "melds": []}] * 3
    x = encode_state(hand, [], list(visible), [], 1, "east", wall, opponents)
    scores = book_prior(x)
    return PLAY_TILES[int(max(range(len(scores)), key=lambda i: scores[i] if legal_mask(hand)[i] else -1e9))], x


def test_docs_17_exercise_2_counts_unseen_copies_once():
    import Decision_Strategy_engine as de
    # 6-7 dots waits on 5 or 8; two 5 dots discarded, one 8 dot in a meld.
    unseen = de.unseen_counts(D(6, 7), D(5, 5, 8))
    assert unseen[('dot', 5)] + unseen[('dot', 8)] == 5


def test_docs_17_exercise_1_book_keeps_the_two_sided_shapes():
    # Two two-sided shapes (3-4 bamboo, 7-8 dots) and an edge (8-9 bamboo):
    # the edge accepts 4 copies, each two-sided shape 8, so the edge goes.
    hand = W(1, 2, 3) + W(4, 5, 6) + W(9, 9) + B(3, 4) + B(8, 9) + D(7, 8)
    tile, _ = _book_top(hand)
    assert tile in B(8, 9)


def test_docs_00_and_17_exercise_3_isolated_honour_goes_before_a_flexible_tile():
    # 3-4-5-6 wan is two two-sided shapes, not a run plus a spare 6; the lone
    # North and South winds (not East's seat or round wind) score nothing.
    hand = W(3, 4, 5, 6) + B(1, 2, 3) + D(4, 5, 6) + D(9, 9) + [('wind', 'north'), ('wind', 'south')]
    tile, _ = _book_top(hand)
    assert tile in [('wind', 'north'), ('wind', 'south')]


def test_docs_07_08_complete_shape_first_then_at_least_one_tai():
    red, west = [('dragon', 'red')] * 3, [('wind', 'west')] * 3
    hand = W(1, 2, 3) + W(1, 2, 3) + D(7, 8, 9) + D(9, 9)
    assert sg.score_hand(hand + red, [], [], 1, "east")["tai"] == 1      # Red Dragon Pong: Hu
    assert sg.score_hand(hand + west, [], [], 1, "east")["tai"] < sg.MIN_TAI   # West: no Hu
    # ... and the book keeps that scoring pair rather than a spare tile.
    tile, _ = _book_top(W(1, 2, 3) + W(4, 5, 6) + D(7, 8, 9) + B(5) + [('dragon', 'red')] * 2 + D(1, 1))
    assert tile != ('dragon', 'red')


def test_docs_00_02_12_book_defends_against_a_visible_flush_late():
    from mantai_ml.features import BOOK_AT, N
    from mantai_ml.tiles import TILE_INDEX
    hand = W(1, 2, 4, 7, 9) + D(2, 3, 5, 8, 9) + B(5) + [('wind', 'north')] * 2 + [('dragon', 'white')]
    assert _book_top(hand)[0] == ('bamboo', 5)          # nobody threatens: the lone 5 bamboo goes
    flush = {"discards": [('dragon', 'white'), ('wan', 9)], "melds": [B(1, 2, 3), B(6, 7, 8), B(2, 2, 2)]}
    visible = B(1, 2, 3) + B(6, 7, 8) + B(2, 2, 2) + [('dragon', 'white'), ('wan', 9)]
    quiet = {"discards": [], "melds": []}
    tile, x = _book_top(hand, visible, wall=30, opponents=[flush, quiet, quiet])
    assert x[BOOK_AT + 4 * N + 1] > 0.5                 # slow hand under threat: fold
    danger = lambda t: x[BOOK_AT + 2 * N + TILE_INDEX[t]]
    assert danger(('bamboo', 5)) > danger(('wan', 9))    # their suit vs a tile they threw
    assert tile == ('wan', 9)


def test_docs_12_suji_and_kabe_lower_danger():
    from collections import Counter
    from mantai_ml.knowledge import tile_danger
    plain = tile_danger(('dot', 4), set(), Counter())
    assert tile_danger(('dot', 4), {('dot', 1), ('dot', 7)}, Counter()) < plain      # suji
    assert tile_danger(('dot', 4), set(), Counter({('dot', 3): 4, ('dot', 5): 4})) < plain   # kabe
    assert tile_danger(('wind', 'north'), set(), Counter({('wind', 'north'): 3})) < 0.05
    assert tile_danger(('dot', 1), set(), Counter()) < plain                            # terminal


def test_book_call_prior_follows_docs_00_and_17():
    from mantai_ml.features import call_prior, encode_call, encode_state
    hand = W(1, 2, 3) + W(5, 6) + D(3, 4) + [('dragon', 'red')] * 2 + B(2, 5, 8) + [('wind', 'north')]
    x = encode_state(hand, [], [], [], 1, "east", 100)
    useful = encode_call(x, ('dragon', 'red'), "pong", np.array([3 / 8, 0.3, 2 / 8, 0.3, 1], dtype=np.float32))
    pointless = encode_call(x, ('bamboo', 8), "pong", np.array([3 / 8, 0.3, 3 / 8, 0.25, 1], dtype=np.float32))
    assert float(call_prior(useful)) > 0 > float(call_prior(pointless))


# ---------------------------------------------------------------------------
# AlphaGo-style search (search.py)
# ---------------------------------------------------------------------------
def _tiny_model(tmp_path):
    from torch import nn
    from mantai_ml.features import STATE_SIZE
    from mantai_ml.model import PolicyNet, save_model
    torch.manual_seed(0)
    save_model(str(tmp_path / "alpha_best.pt"), PolicyNet(hidden=16, dropout=0.0, prior=2.5))
    critic = nn.Sequential(nn.Linear(STATE_SIZE, 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 1))
    torch.save(critic.state_dict(), tmp_path / "alpha_critic.pt")
    return str(tmp_path / "alpha_best.pt")


def test_search_finds_the_critic_and_plays_legal_discards_for_every_seat(tmp_path):
    from mantai_ml.search import AlphaSearch, critic_path, decide
    path = _tiny_model(tmp_path)
    assert critic_path(path) == str(tmp_path / "alpha_critic.pt")
    search = AlphaSearch(path, simulations=8, batch=4, max_candidates=2, parallel=False, seed=1)
    assert search.critic is not None
    game = sim.MahjongGame(rand=random.Random(3))
    game.player1, game.verbose = "engine", False
    game.deal()
    session = sim.GameSession(game=game, check=True)
    searched = set()
    while not session.is_over and len(searched) < 4:
        d = session.decision
        action = decide(search, game, d)
        assert action in session.legal_actions()
        if action.kind == "discard":
            found = search.last
            assert found["tile"] == action.tile
            assert {c["tile"] for c in found["candidates"]} >= {found["engine"], action.tile}
            searched.add(d.player)
        session.apply(action)


def test_search_keeps_the_favourite_unless_another_tile_clearly_wins(tmp_path):
    from mantai_ml.search import AlphaSearch
    search = AlphaSearch(_tiny_model(tmp_path), batch=4, parallel=False)
    a, b = ('dot', 1), ('dot', 2)
    priors = {a: 0.7, b: 0.3}
    noisy = {a: {"scores": [1, -1, 2, 0, 1, -2, 0, 1]}, b: {"scores": [2, -2, 2, 1, 0, -2, 1, 0]}}
    assert search._decide([a, b], priors, noisy) == a
    clear = {a: {"scores": [0, -1, 1, 0, 0, -1, 1, 0]}, b: {"scores": [5, 4, 6, 5, 5, 4, 6, 5]}}
    assert search._decide([a, b], priors, clear) == b
