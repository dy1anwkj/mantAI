"""Local browser UI for the existing Singapore Mahjong GameSession.

Run: python web_game.py, then open http://127.0.0.1:8000.
One shared table per server process; intended for local single-player use.
Players 2-4 play like people: each is the strategy engine with its own style
(aggressive, balanced or cautious), the stand-ins for the human players the
project aims to beat. Your recommended discard is chosen the way AlphaGo chose
moves (mantai_ml/search.py): the self-play network trained from the study
library in docs/ proposes discards, and a search over guessed deals, scored by
its value network and by rollouts, decides. --ai-opponents lets that search
play Players 2-4 as well; --engine-only plays the strategy engine alone.
"""
import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import secrets
import random
from urllib.parse import urlsplit

from wall_setup import deal_three_dice
from mahjong_simulator import PLAYER1_WEIGHTS, MahjongGame, GameSession, choose_action, engine_discard_decision, rate_discards
import singapore_rules as sg
from Decision_Strategy_engine import StrategyEngine, engine_chi_sequence, engine_should_call
from R_opponent_analysis import OpponentAnalyzer, load_danger_model
try:
    from mantai_ml.search import DEFAULT_MODEL, AlphaSearch, decide
except ImportError:          # the trained models need torch; without it the engine plays alone
    AlphaSearch = None

STATIC = Path(__file__).parent / "web"
SAVE_DIR = Path(__file__).parent / "data" / "web_hands"   # gitignored
HINT_SIMULATIONS = 160       # searched playouts behind your recommended discard
OPPONENT_SIMULATIONS = 64    # ... and behind each AI discard


class Table:
    def __init__(self, brain=None, ai_opponents=False, save_dir=None):
        """brain: an AlphaSearch that recommends your discard; None leaves it to
        the strategy engine. ai_opponents: the brain also plays Players 2-4
        instead of the human-style engine opponents. save_dir: where each hand
        is saved for review (see save_record); None saves nothing."""
        self.brain = brain
        self.ai_opponents = ai_opponents and brain is not None
        self.save_dir = Path(save_dir) if save_dir else None
        self.record = None
        self.session = None
        self.revision = 0
        self.events = []
        self.playback = []
        self.last_discard = None
        self.hint_cache = None
        self.balances = [1000] * 4

    def points(self):
        """One existing payout unit is one point; reading never settles twice."""
        if self.session is None:
            return list(self.balances)
        delta = (self.session.result()["total"] if self.session.is_over
                 else self.session.game.instant)
        return [self.balances[p] + delta.get(p, 0) for p in range(4)]

    def next_dealer(self):
        if self.session is None:
            return 0
        dealer = self.session.game.dealer
        if self.session.is_over:
            winner = self.session.result()["winner"]
            if winner is not None and winner != dealer:
                return (dealer + 1) % 4
        return dealer

    def new(self, seed=None):
        balances = self.points()
        if self.session is not None:
            self.save_record()      # a hand left unfinished is kept too
        dealer = self.next_dealer()
        wind_round = self.session.game.wind_round if self.session else 0
        if self.session and dealer == 0 and self.session.game.dealer == 3:
            wind_round = (wind_round + 1) % 4
        if seed is None:            # recorded, so a saved hand can be dealt again
            seed = secrets.randbelow(2 ** 31)
        game = MahjongGame(dealer=dealer, wind_round=wind_round, rand=random.Random(seed))
        game.verbose = False
        deal_three_dice(game)
        self.session = GameSession(game=game, check=True)
        self.balances = balances
        if load_danger_model() is not None:
            # Your seat's engine reads the opponents with the learned danger model.
            game.strategy = StrategyEngine(PLAYER1_WEIGHTS, opponent_model=OpponentAnalyzer())
        self.record = {
            "started": datetime.now().strftime("%Y%m%dT%H%M%S"), "seed": seed,
            "dealer": dealer, "wind_round": wind_round, "styles": dict(game.styles),
            "opponents": "alphago-search" if self.ai_opponents else "human-style",
            "brain": "alphago-search" if self.brain is not None else "engine",
            "start_hands": [game.sort_tiles(h) for h in game.hands],
            "start_bonus": [list(b) for b in game.bonus],
            "moves": [], "log": [],
            "setup": deepcopy(game.wall_setup), "start_points": list(self.balances),
        }
        self.hint_cache = None
        self.events = ["A new hand begins. You are Player 1."]
        self.playback = []
        self.last_discard = None
        self.revision += 1
        self.advance()

    def play(self, action):
        if hasattr(self.session.game.wall, "draws"):
            self.session.game.wall.draws = []
        player = self.session.current_player
        name = "You" if player == 0 else f"Player {player + 1}"
        if action.kind != "pass":
            tile = self.session.game.format_tile(action.tile) if action.tile else ""
            self.events.append(f"{name}: {action.kind.upper()} {tile}".strip())
        if self.record is not None:
            self.record["log"].append(dict(asdict(action), player=player))
        self.session.apply(action, player=player)
        self.events = self.events[-60:]
        if action.kind == "discard":
            self.last_discard = {"player": player, "tile": action.tile}
        elif action.kind in ("pong", "chi", "kong", "hu"):
            self.last_discard = None
        draws = getattr(self.session.game.wall, "draws", [])
        if action.kind != "pass" or draws:
            snapshot = deepcopy(self.state())
            snapshot["actions"] = []
            visible_action = asdict(action) if action.kind != "pass" else {"kind": "draw", "tile": None, "chi": None}
            self.playback.append({"player": draws[0]["player"] if action.kind == "pass" else player,
                                  "action": visible_action, "state": snapshot})

    def advance(self):
        while not self.session.is_over:
            decision = self.session.decision
            if decision.player == 0:
                # Don't stop the browser for a claim where passing is the only choice.
                if len(decision.actions) != 1 or decision.actions[0].kind != "pass":
                    break
                self.play(decision.actions[0])
            elif self.ai_opponents:
                self.play(decide(self.brain, self.session.game, decision))
            else:
                self.play(choose_action(self.session.game, decision))
        if self.session.is_over:
            self.save_record()

    def act(self, index, revision):
        if self.session is None or revision != self.revision:
            raise ValueError("The table changed. Refresh and choose again.")
        if self.session.is_over or self.session.current_player != 0:
            raise ValueError("It is not your turn.")
        actions = self.session.legal_actions()
        if type(index) is not int or not 0 <= index < len(actions):
            raise ValueError("Choose a legal action.")
        self.playback = []
        if self.record is not None:
            self.record["moves"].append(self.move_record(actions[index]))
        self.play(actions[index])
        self.hint_cache = None
        self.revision += 1
        self.advance()

    def move_record(self, action):
        """One of your moves for the saved hand: your tiles at the time, what
        the box recommended (None if it had not loaded yet) and what you chose."""
        s, game, rec = self.session, self.session.game, self.hint_cache
        shown = None
        if rec is not None and "action" in rec:
            shown = {k: rec["action"][k] for k in ("kind", "tile", "chi")}
        elif rec is not None:
            shown = {"kind": "discard", "tile": rec["tile"], "chi": None, "win_percent": rec["win_percent"],
                     "scores": [{"tile": c["tile"], "score": c["value"], "playouts": c["visits"]}
                                for c in rec.get("candidates", []) if c.get("visits")]}
        chosen = asdict(action)
        return {
            "turn": game.turns, "decision": s.decision.kind,
            "claim_tile": s.decision.tile if s.decision.kind != "turn" else None,
            "hand": game.sort_tiles(game.hands[0]), "melds": [list(m) for m in game.melds[0]],
            "recommended": shown, "chosen": chosen,
            "followed": None if shown is None else all(shown[k] == chosen[k] for k in ("kind", "tile", "chi")),
        }

    def save_record(self):
        """Write the hand to save_dir as JSON, once: when it ends, or when a new
        hand replaces it after you have moved. Nothing is written without a
        save_dir, so tests and simulations leave no files."""
        if self.save_dir is None or self.record is None or self.record.get("saved"):
            return
        s, game = self.session, self.session.game
        if not s.is_over and not self.record["moves"]:
            return
        out = dict(self.record, finished=s.is_over, result=s.result() if s.is_over else None,
                   final_hands=[game.sort_tiles(h) for h in game.hands],
                   melds=game.melds, discards=game.discards, bonus=game.bonus)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        path = self.save_dir / f"hand_{self.record['started']}_seed{self.record['seed']}.json"
        path.write_text(json.dumps(out, indent=1, default=str))
        self.record["saved"] = str(path)

    def recommended_call(self):
        """Index of your recommended Hu, Kong, Pong, Chi or Pass among the legal
        actions, or None when the answer is a discard. A legal Hu is always
        taken. The calls are your seat's strategy engine's: on 4000 deals
        against the human-style opponents, engine calls won +1.40 units a hand
        and the network's call head +0.79, with the engine discarding in both."""
        s = self.session
        game, decision, actions = s.game, s.decision, s.legal_actions()
        kinds = [a.kind for a in actions]
        if "hu" in kinds:
            return kinds.index("hu")
        call = lambda tile, kind: engine_should_call(game, 0, tile, kind, game.strategy)
        if decision.kind == "turn":
            return next((i for i, a in enumerate(actions) if a.kind == "kong" and call(a.tile, "self_kong")), None)
        for kind in ("kong", "pong"):
            if kind in kinds and call(decision.tile, kind):
                return kinds.index(kind)
        if "chi" in kinds:
            seq = engine_chi_sequence(game, 0, decision.tile, game.strategy)
            if seq is not None:
                for i, a in enumerate(actions):
                    if a.kind == "chi" and sorted(a.chi) == sorted(seq):
                        return i
        return kinds.index("pass") if "pass" in kinds else None

    def hint(self, revision):
        s = self.session
        if s is None or revision != self.revision:
            raise ValueError("The table changed. Refresh the recommendation.")
        if s.is_over or s.current_player != 0:
            return {"revision": self.revision, "recommendation": None}
        if self.hint_cache is None:
            index = self.recommended_call()
            if index is not None:
                self.hint_cache = {"action": dict(asdict(s.legal_actions()[index]), id=index)}
        if self.hint_cache is not None and "action" in self.hint_cache:
            return {"revision": self.revision, "recommendation": self.hint_cache}
        if self.hint_cache is None and self.brain is not None:
            found = self.brain.analyse(s.game, 0, simulations=HINT_SIMULATIONS)
            best = found["candidates"][0]
            if not best["visits"]:      # one obvious discard: only rate it
                rating = rate_discards(s.game, 0, [best["tile"]], playouts=64, seed=self.revision,
                                       parallel=False)[best["tile"]]
                best = dict(best, win=rating["win"], visits=rating["playouts"])
            self.hint_cache = {
                "tile": found["tile"], "win_percent": best["win"], "samples": found["simulations"] or best["visits"],
                "prior": best["prior"], "engine": found["engine"],
                # The network's own first choice, which the search keeps unless
                # another tile scored clearly more on the same guessed deals.
                "favourite": max(found["candidates"], key=lambda c: c["prior"])["tile"],
                "candidates": [{"tile": c["tile"], "prior": c["prior"], "units": c["units"], "win": c["win"],
                                "visits": c["visits"], "value": c["value"]}
                               for c in [best] + found["candidates"][1:]],
            }
        elif self.hint_cache is None:
            tile = engine_discard_decision(s.game, 0, s.game.strategy).details[0]["tile"]
            rating = rate_discards(s.game, 0, [tile], playouts=64, seed=self.revision, parallel=False)[tile]
            self.hint_cache = {"tile": tile, "win_percent": rating["win"], "samples": rating["playouts"]}
        if "deal_in" not in self.hint_cache:
            # The learned danger model's chance that each tile pays an opponent.
            reader = OpponentAnalyzer().update_from_game(s.game, 0)
            if reader.model is not None:
                self.hint_cache["deal_in"] = reader.deal_in_chance(self.hint_cache["tile"])
                for c in self.hint_cache.get("candidates", []):
                    c["deal_in"] = reader.deal_in_chance(c["tile"])
        return {"revision": self.revision, "recommendation": self.hint_cache}

    def state(self):
        if self.session is None:
            return {"started": False, "revision": self.revision, "points": self.points()}
        session, game = self.session, self.session.game
        # Explicit allowlist: never send AI concealed tiles or the remaining wall.
        result = session.result() if session.is_over else None
        decision = session.decision
        setup = getattr(game, "wall_setup", None)
        wall_stacks = [[0] * n for n in setup["stacks"]] if setup else []
        for player, stack, _ in getattr(game.wall, "slots", []):
            wall_stacks[player][stack - 1] += 1
        return {
            "started": True, "revision": self.revision, "over": session.is_over,
            "points": self.points(), "hand_start_points": list(self.balances),
            "setup": setup, "wall_stacks": wall_stacks,
            "wall_ends": ({"live": game.wall.slots[-1], "replacement": game.wall.slots[0]}
                          if getattr(game.wall, "slots", []) else None),
            "wall_draws": getattr(game.wall, "draws", []),
            "hand": game.sort_tiles(game.hands[0]),
            "counts": [len(h) for h in game.hands],
            "discards": game.discards, "melds": game.melds, "bonus": game.bonus,
            "dealer": game.dealer, "wind": game.prevailing_wind,
            "next_dealer": self.next_dealer(),
            "seats": [game.seat(p) for p in range(4)],
            "wall": max(0, len(game.wall) - sg.DEAD_WALL), "turn": game.turns,
            "decision": decision.kind if decision else None,
            "pending": decision.tile if decision else None,
            "source": decision.source if decision else None,
            "actions": [dict(asdict(a), id=i) for i, a in enumerate(session.legal_actions())],
            "events": self.events, "result": result,
            "last_discard": self.last_discard,
            "brain": "alphago-search" if self.brain is not None else "engine",
            "opponents": "alphago-search" if self.ai_opponents else "human-style",
            "winning_hand": ({
                "player": result["winner"],
                "hand": game.sort_tiles(game.hands[result["winner"]]),
                "melds": game.melds[result["winner"]],
                "bonus": game.bonus[result["winner"]],
            } if result is not None and result["winner"] is not None else None),
        }


class Handler(BaseHTTPRequestHandler):
    def respond(self, status, data, content_type="application/json"):
        body = json.dumps(data).encode() if content_type == "application/json" else data
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(body)

    def valid_host(self):
        return self.headers.get("Host") in {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}

    def do_GET(self):
        if not self.valid_host():
            return self.respond(403, {"error": "Local connections only."})
        path = urlsplit(self.path).path
        if path == "/api/state":
            return self.respond(200, dict(self.server.table.state(), token=self.server.token))
        files = {"/": ("index.html", "text/html; charset=utf-8"),
                 "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                 "/table_scene.js": ("table_scene.js", "text/javascript; charset=utf-8"),
                 "/style.css": ("style.css", "text/css; charset=utf-8"),
                 "/tiles.css": ("tiles.css", "text/css; charset=utf-8"),
                 "/arcade.css": ("arcade.css", "text/css; charset=utf-8"),
                 "/tiles.webp": ("tiles.webp", "image/webp")}
        if path not in files:
            return self.respond(404, {"error": "Not found"})
        name, mime = files[path]
        self.respond(200, (STATIC / name).read_bytes(), mime)

    def do_POST(self):
        if not self.valid_host() or self.headers.get("X-Table-Token") != self.server.token:
            return self.respond(403, {"error": "Refresh the page before playing."})
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 4096:
                raise ValueError("Invalid request size")
            data = json.loads(self.rfile.read(size))
            if not isinstance(data, dict):
                raise ValueError("Expected an object")
            if self.path == "/api/new":
                self.server.table.new()
            elif self.path == "/api/hint":
                return self.respond(200, self.server.table.hint(data.get("revision")))
            elif self.path == "/api/action":
                self.server.table.act(data.get("id"), data.get("revision"))
            else:
                return self.respond(404, {"error": "Not found"})
            self.respond(200, dict(self.server.table.state(), playback=self.server.table.playback))
        except (ValueError, TypeError) as exc:
            self.respond(400, {"error": str(exc)})


def create_server(port=8000, brain=None, ai_opponents=False, save_dir=None):
    server = HTTPServer(("127.0.0.1", port), Handler)
    server.table = Table(brain, ai_opponents, save_dir)
    server.token = secrets.token_urlsafe(32)
    return server


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--engine-only", action="store_true",
                        help="play the strategy engine alone, without the trained models")
    parser.add_argument("--ai-opponents", action="store_true",
                        help="the trained search plays Players 2-4 instead of the human-style engine")
    parser.add_argument("--no-save", action="store_true",
                        help=f"don't save each hand for review (default: {SAVE_DIR.relative_to(Path(__file__).parent)})")
    parser.add_argument("--model", help="trained network (default models/alpha_best.pt)")
    parser.add_argument("--simulations", type=int, default=OPPONENT_SIMULATIONS,
                        help="searched playouts behind each AI discard")
    args = parser.parse_args()
    brain = None
    if not args.engine_only:
        model = args.model or (DEFAULT_MODEL if AlphaSearch else None)
        if AlphaSearch is None:
            print("torch is not installed, so the trained network is off.", flush=True)
        elif not Path(model).exists():
            print(f"{model} not found: train it with python -m mantai_ml selfplay --out models/alpha.pt",
                  flush=True)
        else:
            brain = AlphaSearch(model, simulations=args.simulations)
    with create_server(args.port, brain, args.ai_opponents, None if args.no_save else SAVE_DIR) as server:
        print(f"Play Mahjong at http://127.0.0.1:{server.server_port} (Ctrl+C to stop)", flush=True)
        print("Opponents: " + (f"search ({args.simulations} playouts per discard)"
                                if args.ai_opponents and brain is not None
                                else "human-style engine (aggressive / balanced / cautious)"), flush=True)
        print("Your recommendation: " + ("strategy engine" if brain is None else
              f"search ({HINT_SIMULATIONS} playouts) over {brain.model_path}"), flush=True)
        print("Hands are " + ("not saved" if args.no_save else f"saved to {SAVE_DIR}"), flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
