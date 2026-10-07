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


# ---- sharing games through the games branch --------------------------------
import subprocess


def _run(*args, cwd):
    return subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def remote(tmp_path, monkeypatch):
    """A bare repo standing in for GitHub, and a helper to make players' clones."""
    monkeypatch.delenv("MANTAI_NO_UPLOAD", raising=False)
    origin = tmp_path / "origin.git"
    _run("git", "init", "-q", "--bare", str(origin), cwd=tmp_path)
    seed = tmp_path / "seed"
    _run("git", "init", "-q", "-b", "main", str(seed), cwd=tmp_path)
    _run("git", "-c", "user.name=x", "-c", "user.email=x@x", "commit", "-q", "--allow-empty", "-m", "start", cwd=seed)
    _run("git", "push", "-q", str(origin), "main", cwd=seed)

    def player(name):
        clone = tmp_path / name
        _run("git", "clone", "-q", str(origin), str(clone), cwd=tmp_path)
        _run("git", "config", "user.name", name, cwd=clone)
        _run("git", "config", "user.email", f"{name}@x", cwd=clone)
        return clone

    def use(clone):
        monkeypatch.setattr(game_log, "REPO", str(clone))
        return str(clone / "data" / "games")
    return origin, player, use


def _branch_games(origin):
    out = _run("git", "ls-tree", "--name-only", "games", "games/", cwd=origin)
    return {line.split("/")[-1] for line in out.splitlines()}


def test_games_are_pushed_to_the_games_branch_only(remote, capsys):
    origin, player, use = remote
    alice = player("alice")
    games = use(alice)
    sim.play_game(player1="engine", delay=0, verbose=False, log_dir=games, seed=1)
    assert game_log.upload(games) == 1
    sim.play_game(player1="engine", delay=0, verbose=False, log_dir=games, seed=2)
    assert game_log.upload(games) == 1          # only the new one
    assert game_log.upload(games) == 0
    assert len(_branch_games(origin)) == 2
    assert _run("git", "log", "--format=%s", "main", cwd=origin) == "start"   # main untouched
    assert _run("git", "status", "--porcelain", "-uno", cwd=alice) == ""             # working tree untouched
    assert "alice" in _run("git", "log", "-1", "--format=%an %s", "games", cwd=origin)


def test_friends_games_are_merged_and_fetched(remote, capsys):
    origin, player, use = remote
    alice, bob, dylan = player("alice"), player("bob"), player("dylan")
    for clone, seed in ((alice, 1), (bob, 2)):
        games = use(clone)
        sim.play_game(player1="engine", delay=0, verbose=False, log_dir=games, seed=seed)
        game_log.upload(games)                  # bob's push builds on alice's
    assert len(_branch_games(origin)) == 2
    games = use(dylan)
    added = game_log.fetch_games(games)
    assert len(added) == 2
    assert {game_log.load(p)["played_by"] for p in added} == {"alice", "bob"}
    assert game_log.fetch_games(games) == []
    assert "DIFFERENT FROM THE ENGINE" in game_log.review(game_log.load(added[0]))


def test_failed_upload_does_not_stop_the_game(remote, monkeypatch, capsys):
    origin, player, use = remote
    games = use(player("alice"))
    _run("git", "remote", "set-url", "origin", "/nonexistent/repo.git", cwd=game_log.REPO)
    sim.play_game(player1="engine", delay=0, verbose=False, log_dir=games, seed=1)
    with pytest.raises(game_log.UploadError):
        game_log.upload(games)
    game_log.upload_and_report(games)
    assert "failed" in capsys.readouterr().out


def test_no_upload_setting(remote, monkeypatch, capsys):
    origin, player, use = remote
    games = use(player("alice"))
    sim.play_game(player1="engine", delay=0, verbose=False, log_dir=games, seed=1)
    monkeypatch.setenv("MANTAI_NO_UPLOAD", "1")
    assert game_log.upload(games) == 0
    assert _run("git", "branch", "--list", "games", cwd=origin) == ""
