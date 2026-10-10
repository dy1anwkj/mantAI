"""Record simulator gameplay as decision records (see records.py)."""

import uuid

from .features import state_from_game
from .records import tile_list, write_records


class GameRecorder:
    """Attach to a game (game.recorder = GameRecorder(...)) to log decisions.

    players : which seats to record. Default is Player 1, whose tuned engine
              is the teacher when recording with player1="engine".
    """

    def __init__(self, players=(0,), source="sim", teacher=False):
        self.players = set(players)
        self.source = source
        self.records = []
        self._current = []
        self.game_id = None
        self.teacher = teacher

    def on_decision(self, game, player, kind, tile, call_type=None, taken=None):
        if player not in self.players:
            return
        if self.game_id is None:
            self.game_id = uuid.uuid4().hex[:12]
        from mahjong_simulator import controller_name
        state = state_from_game(game, player)
        r = {
            "source": self.source, "game_id": self.game_id, "player": player,
            "controller": controller_name(game, player), "kind": kind,
            "tile": list(tile) if tile is not None else None,
            "hand": tile_list(state["hand"]),
            "melds": [tile_list(m) for m in state["melds"]],
            "visible": tile_list(state["visible"]),
            "bonus": tile_list(state["bonus"]),
            "seat": state["seat"], "prevailing": state["prevailing"],
            "wall_left": state["wall_left"],
            "opponents": state["opponents"],
        }
        if kind == "call":
            r["call_type"] = call_type
            r["taken"] = taken
        if self.teacher:
            from Decision_Strategy_engine import engine_discard_decision, engine_should_call
            engine = game.strategy if player == 0 else game.opponent_engines[player]
            r["behavior_controller"] = r["controller"]
            r["controller"] = "engine_teacher"
            if kind == "discard":
                r["behavior_tile"] = r["tile"]
                r["tile"] = list(engine_discard_decision(game, player, engine).tile)
            elif call_type in ("pong", "chi", "kong", "self_kong"):
                r["behavior_taken"] = taken
                r["taken"] = engine_should_call(game, player, tile, call_type, engine)
        self._current.append(r)

    def on_game_end(self, game):
        result = game.result
        for r in self._current:
            won = bool(result) and result["winner"] == r["player"]
            r["outcome"] = {
                "won": won,
                "draw": result is None,
                "self_drawn": bool(won and result["self_drawn"]),
                "tai": result["score"]["tai"] if won else 0,
            }
        self.records.extend(self._current)
        self._current = []
        self.game_id = None


def record_games(n_games, out_path, player1="engine", players=(0,), seed=None, progress=True,
                 model_path=None, teacher=False):
    """Play n_games in the simulator and write every decision to out_path."""
    import mahjong_simulator as sim

    if player1 == "ml" and model_path is None:
        raise ValueError("A model_path is required when recording an ML player")
    policy = None
    if player1 == "ml":
        from .ml_player import MLPolicy
        policy = MLPolicy(model_path)
    recorder = GameRecorder(players, teacher=teacher)

    def setup(game):
        game.recorder = recorder
        if policy is not None:
            game.ml_policy = policy

    stats = sim.simulate(n_games, seed=seed, check=True, progress=progress,
                         player1=player1, setup=setup)
    if stats["errors"]:
        raise RuntimeError("Recording failed; refusing to write incomplete training data: " + stats["errors"][0])
    write_records(out_path, recorder.records, append=False)
    return len(recorder.records), stats
