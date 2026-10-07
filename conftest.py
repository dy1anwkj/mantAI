import pytest

import game_log


@pytest.fixture(autouse=True)
def _keep_games_out_of_the_repo(tmp_path, monkeypatch):
    """Tests that play a terminal game must not save it to data/games or push it to GitHub."""
    monkeypatch.setattr(game_log, "DEFAULT_DIR", str(tmp_path / "games"))
    monkeypatch.setenv("MANTAI_NO_UPLOAD", "1")
