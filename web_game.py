"""Local browser UI for the existing Singapore Mahjong GameSession.

Run: python web_game.py, then open http://127.0.0.1:8000.
One shared table per server process; intended for local single-player use.
By default Players 2-4 and your recommended discard are chosen the way
AlphaGo chose moves (mantai_ml/search.py): the self-play network trained from
the study library in docs/ proposes discards, and a search over guessed deals,
scored by its value network and by rollouts, decides. --engine-only plays the
strategy engine alone.
"""
import argparse
from copy import deepcopy
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import secrets
from urllib.parse import urlsplit

from mahjong_simulator import PLAYER1_WEIGHTS, GameSession, choose_action, engine_discard_decision, rate_discards
import singapore_rules as sg
from Decision_Strategy_engine import StrategyEngine
from R_opponent_analysis import OpponentAnalyzer, load_danger_model
try:
    from mantai_ml.search import DEFAULT_MODEL, AlphaSearch, decide
except ImportError:          # the trained models need torch; without it the engine plays alone
    AlphaSearch = None

STATIC = Path(__file__).parent / "web"
HINT_SIMULATIONS = 160       # searched playouts behind your recommended discard
OPPONENT_SIMULATIONS = 64    # ... and behind each AI discard


class Table:
    def __init__(self, brain=None):
        """brain: an AlphaSearch that plays Players 2-4 and recommends your
        discard; None leaves both to the strategy engine."""
        self.brain = brain
        self.session = None
        self.revision = 0
        self.events = []
        self.playback = []
        self.last_discard = None
        self.hint_cache = None

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
        dealer = self.next_dealer()
        wind_round = self.session.game.wind_round if self.session else 0
        if self.session and dealer == 0 and self.session.game.dealer == 3:
            wind_round = (wind_round + 1) % 4
        self.session = GameSession(seed=seed, dealer=dealer, wind_round=wind_round, check=True)
        if load_danger_model() is not None:
            # Your seat's engine reads the opponents with the learned danger model.
            self.session.game.strategy = StrategyEngine(PLAYER1_WEIGHTS, opponent_model=OpponentAnalyzer())
        self.hint_cache = None
        self.events = ["A new hand begins. You are Player 1."]
        self.playback = []
        self.last_discard = None
        self.revision += 1
        self.advance()

    def play(self, action):
        player = self.session.current_player
        name = "You" if player == 0 else f"Player {player + 1}"
        if action.kind != "pass":
            tile = self.session.game.format_tile(action.tile) if action.tile else ""
            self.events.append(f"{name}: {action.kind.upper()} {tile}".strip())
        self.session.apply(action, player=player)
        self.events = self.events[-60:]
        if action.kind == "discard":
            self.last_discard = {"player": player, "tile": action.tile}
        elif action.kind in ("pong", "chi", "kong", "hu"):
            self.last_discard = None
        if action.kind != "pass":
            snapshot = deepcopy(self.state())
            snapshot["actions"] = []
            self.playback.append({"player": player, "action": asdict(action), "state": snapshot})

    def advance(self):
        while not self.session.is_over:
            decision = self.session.decision
            if decision.player == 0:
                # Don't stop the browser for a claim where passing is the only choice.
                if len(decision.actions) != 1 or decision.actions[0].kind != "pass":
                    break
                self.play(decision.actions[0])
            elif self.brain is not None:
                self.play(decide(self.brain, self.session.game, decision))
            else:
                self.play(choose_action(self.session.game, decision))

    def act(self, index, revision):
        if self.session is None or revision != self.revision:
            raise ValueError("The table changed. Refresh and choose again.")
        if self.session.is_over or self.session.current_player != 0:
            raise ValueError("It is not your turn.")
        actions = self.session.legal_actions()
        if type(index) is not int or not 0 <= index < len(actions):
            raise ValueError("Choose a legal action.")
        self.playback = []
        self.play(actions[index])
        self.hint_cache = None
        self.revision += 1
        self.advance()

    def hint(self, revision):
        s = self.session
        if s is None or revision != self.revision:
            raise ValueError("The table changed. Refresh the recommendation.")
        if s.is_over or s.current_player != 0 or s.decision.kind != "turn":
            return {"revision": self.revision, "recommendation": None}
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
                "candidates": [{"tile": c["tile"], "prior": c["prior"], "units": c["units"], "win": c["win"]}
                               for c in found["candidates"]],
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
            return {"started": False, "revision": self.revision}
        session, game = self.session, self.session.game
        # Explicit allowlist: never send AI concealed tiles or the remaining wall.
        result = session.result() if session.is_over else None
        decision = session.decision
        return {
            "started": True, "revision": self.revision, "over": session.is_over,
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
                 "/style.css": ("style.css", "text/css; charset=utf-8"),
                 "/tiles.css": ("tiles.css", "text/css; charset=utf-8"),
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


def create_server(port=8000, brain=None):
    server = HTTPServer(("127.0.0.1", port), Handler)
    server.table = Table(brain)
    server.token = secrets.token_urlsafe(32)
    return server


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--engine-only", action="store_true",
                        help="play the strategy engine alone, without the trained models")
    args = parser.parse_args()
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
    with create_server(args.port, brain) as server:
        print(f"Play Mahjong at http://127.0.0.1:{server.server_port} (Ctrl+C to stop)", flush=True)
        print("Decisions: " + ("strategy engine only" if brain is None else
              f"search ({args.simulations} playouts per AI discard, {HINT_SIMULATIONS} for yours) "
              f"over {brain.model_path}"), flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
