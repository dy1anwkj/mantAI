"""game_log: saved games replay exactly and can be reviewed."""
import pytest

import game_log
import mahjong_simulator as sim


def _play_and_load(tmp_path, seed, player1="engine"):
    game = sim.play_game(player1=player1, delay=0, verbose=False, log_dir=str(tmp_path), seed=seed)
    return game, game_log.load(game_log.latest_log(str(tmp_path)))


@pytest.mark.parametrize("seed", range(6))
def test_saved_game_replays_to_the_same_end(tmp_path, seed, capsys):
    game, data = _play_and_load(tmp_path, seed)
    session = game_log.replay(data)
    replayed = session.game
    assert replayed.hands == game.hands and replayed.melds == game.melds
    assert replayed.discards == game.discards and replayed.wall == game.wall
    assert session.result()["total"] == {int(p): u for p, u in data["result"]["total"].items()}


def test_same_seed_deals_the_same_game(tmp_path, capsys):
    _, first = _play_and_load(tmp_path / "a", 42)
    _, second = _play_and_load(tmp_path / "b", 42)
    assert first["wall"] == second["wall"] and first["actions"] == second["actions"]


def test_engine_game_review_finds_no_differences(tmp_path, capsys):
    _, data = _play_and_load(tmp_path, 3)
    text = game_log.review(data)
    assert "DIFFERENT FROM THE ENGINE: 0 of" in text
    assert "VERDICT" in text and "HANDS AT THE END" in text


def test_review_flags_a_move_the_engine_would_not_make(tmp_path, capsys):
    _, data = _play_and_load(tmp_path, 3)
    # Player 1 passes on a call the engine took: the review must point at it.
    i = next(i for i, e in enumerate(data["actions"])
             if e["player"] == 0 and e["decision"] == "claim" and e["kind"] != "pass")
    data["actions"] = data["actions"][:i] + [dict(data["actions"][i], kind="pass", tile=None, chi=None)]
    with pytest.raises(ValueError):          # the rest of the log no longer fits
        game_log.replay(data)
    differences = []
    try:
        game_log.replay(data, lambda g, s, d, a: differences.append(game_log.engine_action(g, d) != a))
    except ValueError:
        pass
    assert differences[-1]


def test_log_that_does_not_match_is_rejected(tmp_path, capsys):
    _, data = _play_and_load(tmp_path, 1)
    data["actions"] = data["actions"][:-1]
    with pytest.raises(ValueError):
        game_log.replay(data)
