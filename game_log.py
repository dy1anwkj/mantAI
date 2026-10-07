"""
Save every game played at the terminal and review it afterwards.

play_game(log_dir=...) writes one JSON file per game: the shuffled wall,
dealer, round wind, opponent styles and every action in order. That is all
it takes to replay the game exactly, even if the random seeding changes later.

    python3 mahjong_simulator.py --review               # the latest game
    python3 mahjong_simulator.py --review FILE         # a given game
    python3 mahjong_simulator.py --review FILE --replay   # print it move by move

The review replays the game and, at every Player 1 decision, asks the tuned
strategy engine what it would have done in the same position. It shows:
  - how the game ended and what Player 1 won or lost
  - every Player 1 decision the engine would have made differently
  - when each player first got ready (one tile from winning), so a lost game
    can be told apart from bad luck: the winner was ready early, or Player 1
    never got close
  - the shooting discard, if Player 1 fed the win, and whether the engine
    would have thrown the same tile
  - every player's hand at the end
"""
import datetime
import glob
import json
import os
import random

DEFAULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "games")
VERSION = 1


def _tile(t):
    return None if t is None else tuple(t)


class GameLog:
    """Collects one game's actions; save() writes them to a JSON file."""

    def __init__(self, game, seed):
        self.data = {
            "version": VERSION,
            "started": datetime.datetime.now().isoformat(timespec="seconds"),
            "seed": seed,
            "player1": game.player1,
            "opponents": game.opponents,
            "dealer": game.dealer,
            "wind_round": game.wind_round,
            "styles": {str(p): s for p, s in game.styles.items()},
            "wall": [list(t) for t in game.wall],   # shuffled, before the deal
            "actions": [],
            "result": None,
        }

    def add(self, decision, action):
        self.data["actions"].append({
            "player": decision.player, "decision": decision.kind, "kind": action.kind,
            "tile": None if action.tile is None else list(action.tile),
            "chi": None if action.chi is None else [list(t) for t in action.chi],
        })

    def save(self, session, log_dir=DEFAULT_DIR):
        r = session.result()
        self.data["result"] = {
            "winner": r["winner"], "self_drawn": r["self_drawn"], "shooter": r["shooter"],
            "tai": r["tai"], "turns": r["turns"],
            "total": {str(p): u for p, u in r["total"].items()},
        }
        os.makedirs(log_dir, exist_ok=True)
        stamp = self.data["started"].replace(":", "").replace("-", "")
        path = os.path.join(log_dir, f"game_{stamp}_seed{self.data['seed']}.json")
        with open(path, "w") as f:
            json.dump(self.data, f)
        return path


def latest_log(log_dir=DEFAULT_DIR):
    files = sorted(glob.glob(os.path.join(log_dir, "game_*.json")), key=os.path.getmtime)
    return files[-1] if files else None


def load(path):
    with open(path) as f:
        return json.load(f)


def new_game(data, verbose=False):
    """A dealt MahjongGame set up exactly as the logged game started."""
    import mahjong_simulator as ms
    from Decision_Strategy_engine import StrategyEngine
    game = ms.MahjongGame(data["dealer"], data["wind_round"], rand=random.Random(data["seed"]))
    game.wall = [_tile(t) for t in data["wall"]]
    game.styles = {int(p): s for p, s in data["styles"].items()}
    game.opponent_engines = {p: StrategyEngine(ms.OPPONENT_STYLES[s]) for p, s in game.styles.items()}
    game.player1 = data["player1"]
    game.opponents = data["opponents"]
    game.verbose = verbose
    game.deal()
    return game


def logged_action(entry):
    import mahjong_simulator as ms
    chi = None if entry["chi"] is None else tuple(_tile(t) for t in entry["chi"])
    return ms.Action(entry["kind"], _tile(entry["tile"]), chi)


def replay(data, on_decision=None, verbose=False):
    """Replay a logged game. on_decision(game, session, decision, action) is
    called before each action is applied. Returns the finished session."""
    import mahjong_simulator as ms
    game = new_game(data, verbose)
    session = ms.GameSession(game=game)
    for i, entry in enumerate(data["actions"]):
        decision = session.decision
        if decision is None or decision.player != entry["player"]:
            raise ValueError(f"log does not match the game at action {i + 1}")
        action = logged_action(entry)
        if on_decision is not None:
            on_decision(game, session, decision, action)
        session.apply(action)
    if not session.is_over:
        raise ValueError("log ends before the game does")
    return session


def engine_action(game, decision):
    """What the tuned strategy engine would do as Player 1 in this position."""
    import mahjong_simulator as ms
    saved = game.player1, game.verbose, getattr(game, "recorder", None)
    game.player1, game.verbose, game.recorder = "engine", False, None
    try:
        return ms.choose_action(game, decision)
    finally:
        game.player1, game.verbose, game.recorder = saved


def describe(game, action, source=None):
    fmt = game.format_tile
    if action.kind == "discard":
        return f"discard {fmt(action.tile)}"
    if action.kind == "pass":
        return "pass"
    if action.kind == "chi":
        run = game.sort_tiles(list(action.chi) + [action.tile])
        return "Chi " + ", ".join(fmt(t) for t in run)
    if action.kind == "hu":
        return "declare Hu (win)"
    return f"{action.kind.capitalize()} {fmt(action.tile)}"


def review(data):
    """A plain-text review of a logged game, from Player 1's side."""
    from Decision_Strategy_engine import shanten
    lines = []
    out = lines.append
    differences = []
    ready_turn = {}            # player -> first turn they were one tile from winning
    last_move = []             # Player 1's latest (turn, action, engine's action)

    def on_decision(game, session, decision, action):
        p = decision.player
        if p == 0:
            best = engine_action(game, decision)
            if decision.kind == "turn":
                last_move[:] = [game.turns, action, best]
            if best != action:
                hand = ", ".join(game.format_tile(t) for t in game.sort_tiles(game.hands[0]))
                what = ("own turn" if decision.kind == "turn" else
                        f"Player {decision.source + 1} discarded {game.format_tile(decision.tile)}")
                differences.append(
                    f"  Turn {game.turns:>2} ({what})\n"
                    f"      hand  : {hand}\n"
                    f"      played: {describe(game, action)}\n"
                    f"      engine: {describe(game, best)}")
        if decision.kind == "turn" and action.kind == "discard":
            hand = list(game.hands[p])
            hand.remove(action.tile)
            if p not in ready_turn and shanten(hand, game.melds[p]) <= 0:
                ready_turn[p] = game.turns

    session = replay(data, on_decision)
    game = session.game
    r = session.result()
    fmt = game.format_tile
    who = lambda p: "Player 1 (you)" if p == 0 and data["player1"] == "human" else f"Player {p + 1}"

    styles = ", ".join(f"Player {int(p) + 1} {s}" for p, s in sorted(data["styles"].items()))
    out(f"Game of {data['started']}  (seed {data['seed']})")
    out(f"Player 1: {data['player1']}.  Opponents: {styles}.")
    out(f"Dealer: Player {data['dealer'] + 1}.  Prevailing wind: {game.prevailing_wind.capitalize()}.  "
        f"Turns played: {r['turns']}.")
    out("")

    out("RESULT")
    if r["winner"] is None:
        out("  Draw: the wall ran out.")
    else:
        how = "by self-draw" if r["self_drawn"] else f"on {who(r['shooter'])}'s discard"
        out(f"  {who(r['winner'])} won {how} with {r['tai']} tai.")
        for desc, tai in r["score"]["items"]:
            out(f"      +{tai}  {desc}")
    out("  Units: " + ", ".join(f"Player {p + 1} {u:+d}" for p, u in r["total"].items()))
    out("")

    out("WHO GOT READY WHEN (one tile from winning)")
    for p in range(4):
        turn = ready_turn.get(p)
        out(f"  {who(p):<15} " + (f"turn {turn}" if turn else "never"))
    out("")

    if r["shooter"] == 0 and last_move:
        turn, action, best = last_move
        played = (f"threw {fmt(action.tile)}" if action.kind == "discard"
                  else f"added {fmt(action.tile)} to a Pong (robbed)")
        out("THE SHOOTING DISCARD")
        out(f"  Player 1 {played} on turn {turn} and {who(r['winner'])} won on it.")
        if best == action:
            out("  The engine would have played the same: it was not an obvious danger.")
        else:
            out(f"  The engine would have played: {describe(game, best)}.")
        out("")

    total = sum(1 for e in data["actions"] if e["player"] == 0)
    out(f"PLAYER 1 DECISIONS DIFFERENT FROM THE ENGINE: {len(differences)} of {total}")
    lines.extend(differences)
    out("")

    out("VERDICT")
    out("  " + verdict(r, ready_turn, differences, data))
    out("")

    out("HANDS AT THE END")
    for p in range(4):
        hand = ", ".join(fmt(t) for t in game.sort_tiles(game.hands[p]))
        melds = "  melds: " + " | ".join(", ".join(fmt(t) for t in game.sort_tiles(m))
                                        for m in game.melds[p]) if game.melds[p] else ""
        out(f"  {who(p):<15} {hand}{melds}")
    return "\n".join(lines)


def verdict(r, ready_turn, differences, data):
    winner = r["winner"]
    if winner == 0:
        return "Player 1 won this game."
    mine, theirs = ready_turn.get(0), ready_turn.get(winner) if winner is not None else None
    if differences and data["player1"] != "engine":
        note = (f"Player 1 made {len(differences)} decision(s) the engine would not have. "
                "Look at those first.")
    else:
        note = "Player 1 played as the engine would have."
    if winner is None:
        return note + " Nobody won; the wall ran out."
    if mine is None:
        luck = f"Player 1 never got ready, while Player {winner + 1} was ready by turn {theirs}."
    elif theirs is not None and theirs < mine:
        luck = f"Player {winner + 1} got ready on turn {theirs}, before Player 1 (turn {mine})."
    else:
        luck = (f"Player 1 was ready on turn {mine}, but Player {winner + 1} "
                "drew or was fed the winning tile first.")
    return note + " " + luck
