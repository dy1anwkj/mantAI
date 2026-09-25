"""
Decision / Strategy Engine  (Part 3 of mantAI)

Decides what the AI player should DO:
    - which tile to discard
    - whether to call Pong / Chi / Kong on another player's discard
    - whether to declare Hu

How it thinks
-------------
1. Shanten  = how many tiles away the hand is from "ready" (tenpai).
              0 = ready (waiting for one tile), -1 = already a winning hand.
2. Ukeire   = how many unseen tiles would improve the hand right now.
              More ukeire = more ways to get closer to winning.
3. Danger   = how risky a tile is to throw when an opponent looks close
              to winning (simple built-in guess; Part 4 can replace it).

Discard score = attack value (low shanten, high ukeire, good shape)
                - defence penalty (danger x how threatening opponents are)

Tiles use the same format as the simulator / hand_eval:
    ('bamboo', 5), ('dot', 1), ('wan', 9), ('wind', 'east'), ('dragon', 'red')

Quick use with the simulator:
    from Decision_Strategy_engine import engine_choose_discard, engine_should_call
    tile = engine_choose_discard(game, player_index)
    if engine_should_call(game, player_index, discard_tile, "pong"): ...
"""

from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache

from hand_eval import (
    ALL_TILE_TYPES,
    NUMBER_SUITS,
    format_tile,
    hand_structure_score,
    is_winning_hand,
    tile_sort_key,
)

# --------------------------------------------------------------------------
# Tunable weights (Part 5 - AI training - can adjust these)
# --------------------------------------------------------------------------
DEFAULT_WEIGHTS = {
    "shanten": 100.0,    # each step closer to ready is worth a lot
    "ukeire": 3.0,       # each extra useful unseen tile
    "structure": 0.5,    # hand_eval's shape score (tie-breaker)
    "danger": 150.0,     # penalty for risky discards when defending
    "honor_pong_bonus": 15.0,  # dragons are worth points, pong them more readily
}

# --------------------------------------------------------------------------
# Tile <-> index helpers (34 tile types, same order as hand_eval)
# --------------------------------------------------------------------------
TILE_INDEX = {tile: i for i, tile in enumerate(ALL_TILE_TYPES)}
NUM_TYPES = len(ALL_TILE_TYPES)  # 34


def to_counts(tiles):
    """Turn a list of tiles into a tuple of 34 counts."""
    counts = [0] * NUM_TYPES
    for t in tiles:
        counts[TILE_INDEX[t]] += 1
    return tuple(counts)


# --------------------------------------------------------------------------
# Shanten calculation
# --------------------------------------------------------------------------
@lru_cache(maxsize=200_000)
def _shanten_from_counts(counts, exposed_melds):
    """
    Standard-hand shanten (4 sets + 1 pair).
    exposed_melds = number of melds already shown (pong/chi/kong).
    """
    counts = list(counts)
    best = [8]

    def dfs(i, sets, partials, has_pair):
        while i < NUM_TYPES and counts[i] == 0:
            i += 1
        if i >= NUM_TYPES:
            usable_partials = max(0, min(partials, 4 - sets))
            s = 8 - 2 * sets - usable_partials - (1 if has_pair else 0)
            s = max(s, -1)  # never below "winning", even for oversized hands
            if s < best[0]:
                best[0] = s
            return

        c = counts[i]
        suited = i < 27
        pos = i % 9  # 0..8 inside a suit

        # complete sets
        if c >= 3:
            counts[i] -= 3
            dfs(i, sets + 1, partials, has_pair)
            counts[i] += 3
        if suited and pos <= 6 and counts[i + 1] and counts[i + 2]:
            counts[i] -= 1; counts[i + 1] -= 1; counts[i + 2] -= 1
            dfs(i, sets + 1, partials, has_pair)
            counts[i] += 1; counts[i + 1] += 1; counts[i + 2] += 1

        # pair (as the hand's head, or as a partial set waiting for pong)
        if c >= 2:
            counts[i] -= 2
            if not has_pair:
                dfs(i, sets, partials, True)
            dfs(i, sets, partials + 1, has_pair)
            counts[i] += 2

        # partial sequences: 4-5 (open/edge) and 4-6 (middle gap)
        if suited and pos <= 7 and counts[i + 1]:
            counts[i] -= 1; counts[i + 1] -= 1
            dfs(i, sets, partials + 1, has_pair)
            counts[i] += 1; counts[i + 1] += 1
        if suited and pos <= 6 and counts[i + 2]:
            counts[i] -= 1; counts[i + 2] -= 1
            dfs(i, sets, partials + 1, has_pair)
            counts[i] += 1; counts[i + 2] += 1

        # treat the rest of this tile as isolated
        saved = counts[i]
        counts[i] = 0
        dfs(i + 1, sets, partials, has_pair)
        counts[i] = saved

    dfs(0, exposed_melds, 0, False)
    return best[0]


def shanten(hand, melds=None):
    """-1 = winning hand, 0 = ready, 1 = one tile from ready, ..."""
    melds = melds or []
    return _shanten_from_counts(to_counts(hand), len(melds))


def _candidate_draws(hand):
    """Only tiles near something in the hand can possibly improve it."""
    cands = set()
    for t in hand:
        i = TILE_INDEX[t]
        if i < 27:
            base = i - i % 9
            for d in (-2, -1, 0, 1, 2):
                j = i + d
                if base <= j < base + 9:
                    cands.add(j)
        else:
            cands.add(i)
    return cands


def unseen_counts(hand, visible_tiles):
    """How many copies of each tile we have NOT seen (could still be drawn)."""
    seen = Counter(hand)
    seen.update(visible_tiles)
    return {t: max(0, 4 - seen[t]) for t in ALL_TILE_TYPES}


def ukeire(hand, melds, visible_tiles, current_shanten=None):
    """
    Tiles that would lower shanten, and how many unseen copies remain.
    `hand` is a hand waiting to draw (e.g. 13 tiles with no melds).
    """
    melds = melds or []
    if current_shanten is None:
        current_shanten = shanten(hand, melds)
    unseen = unseen_counts(hand, visible_tiles)
    base = list(to_counts(hand))
    n_melds = len(melds)

    improving = {}
    for i in _candidate_draws(hand):
        tile = ALL_TILE_TYPES[i]
        if unseen[tile] == 0:
            continue
        base[i] += 1
        s = _shanten_from_counts(tuple(base), n_melds)
        base[i] -= 1
        if s < current_shanten:
            improving[tile] = unseen[tile]
    return sum(improving.values()), improving


# --------------------------------------------------------------------------
# Defence: danger of a tile + threat from opponents
# (Part 4 - opponent analysis - can plug in something smarter)
# --------------------------------------------------------------------------
def default_tile_danger(tile, hand, visible_tiles):
    """0.0 = very safe, 1.0 = very dangerous. Simple heuristic."""
    visible = Counter(visible_tiles)[tile]
    unseen = 4 - visible - hand.count(tile)
    suit, value = tile

    if suit not in NUMBER_SUITS:
        # honours can only be waited on as a pair or pong
        if unseen <= 0:
            return 0.0
        if unseen == 1:
            return 0.1
        return 0.3 if unseen == 2 else 0.4

    if value in (1, 9):
        danger = 0.35
    elif value in (2, 8):
        danger = 0.55
    else:
        danger = 0.75

    # tiles others have already thrown are usually safer
    if visible >= 3:
        danger *= 0.3
    elif visible >= 1:
        danger *= 0.7
    return round(danger, 3)


def default_threat_level(melds_by_player, me, wall_remaining=None):
    """0.0 = nobody close to winning, 1.0 = someone very likely ready."""
    threat = 0.0
    for p, melds in enumerate(melds_by_player):
        if p == me:
            continue
        threat = max(threat, 0.25 * len(melds))
    if wall_remaining is not None and wall_remaining < 30:
        threat += 0.2
    return min(1.0, threat)


# --------------------------------------------------------------------------
# Decision result (useful for logging / training in Part 5)
# --------------------------------------------------------------------------
@dataclass
class Decision:
    action: str                 # "discard", "pong", "chi", "kong", "hu", "pass"
    tile: tuple = None
    score: float = 0.0
    reason: str = ""
    details: list = field(default_factory=list)


# --------------------------------------------------------------------------
# The engine
# --------------------------------------------------------------------------
class StrategyEngine:
    def __init__(self, weights=None, danger_fn=None):
        """
        weights   : dict overriding DEFAULT_WEIGHTS
        danger_fn : optional function(tile, hand, visible_tiles) -> 0..1
                    (lets Part 4 plug in a better opponent model)
        """
        self.w = dict(DEFAULT_WEIGHTS)
        if weights:
            self.w.update(weights)
        self.danger_fn = danger_fn or default_tile_danger

    # ---------------- Hu ----------------
    def should_declare_hu(self, hand, melds=None, tile=None):
        """
        Always take a win if the hand is complete.
        Pass `tile` to check winning on someone else's discard.
        """
        if tile is not None:
            hand = list(hand) + [tile]
        # use Part 2's official win check so both parts always agree
        return is_winning_hand(list(hand), melds or [])

    # ---------------- Discard ----------------
    def evaluate_discards(self, hand, melds=None, visible_tiles=None, threat=0.0):
        """Score every possible discard. Hand should have 14 - 3*melds tiles."""
        melds = melds or []
        visible_tiles = visible_tiles or []
        results = []

        for tile in sorted(set(hand), key=tile_sort_key):
            remaining = list(hand)
            remaining.remove(tile)          # remove ONE copy only

            s = shanten(remaining, melds)
            uk, improving = ukeire(remaining, melds, visible_tiles, s)
            structure = hand_structure_score(remaining)
            danger = self.danger_fn(tile, hand, visible_tiles)

            # defend harder when we are far from ready and opponents look close
            defence = threat * min(1.0, max(0, s) / 3.0)

            score = (
                -self.w["shanten"] * s
                + self.w["ukeire"] * uk
                + self.w["structure"] * structure
                - self.w["danger"] * defence * danger
            )
            results.append({
                "tile": tile,
                "score": round(score, 2),
                "shanten": s,
                "ukeire": uk,
                "improving_tiles": improving,
                "structure": structure,
                "danger": danger,
            })

        results.sort(key=lambda r: r["score"], reverse=True)
        return results

    def choose_discard(self, hand, melds=None, visible_tiles=None, threat=0.0):
        ranked = self.evaluate_discards(hand, melds, visible_tiles, threat)
        best = ranked[0]
        state = "ready" if best["shanten"] == 0 else f"{best['shanten']} away from ready"
        reason = (
            f"Discard {format_tile(best['tile'])}: leaves hand {state}, "
            f"{best['ukeire']} useful tiles left to draw"
        )
        if threat >= 0.5 and best["shanten"] >= 2:
            reason += f" (defending, danger {best['danger']})"
        return Decision("discard", best["tile"], best["score"], reason, ranked)

    # ---------------- Calls (Pong / Chi / Kong) ----------------
    def _best_after_call(self, hand_after, melds_after, visible_tiles, threat):
        """After a Pong/Chi we must discard: find the best result."""
        ranked = self.evaluate_discards(hand_after, melds_after, visible_tiles, threat)
        return ranked[0]

    def _current_value(self, hand, melds, visible_tiles):
        s = shanten(hand, melds)
        uk, _ = ukeire(hand, melds, visible_tiles, s)
        return s, uk

    def should_pong(self, hand, melds, tile, visible_tiles=None, threat=0.0):
        melds = melds or []
        visible_tiles = visible_tiles or []
        if hand.count(tile) < 2:
            return Decision("pass", tile, reason="Not enough tiles to Pong")

        cur_s, cur_uk = self._current_value(hand, melds, visible_tiles)
        after = list(hand)
        after.remove(tile); after.remove(tile)
        new_melds = melds + [[tile] * 3]
        best = self._best_after_call(after, new_melds, visible_tiles, threat)

        bonus = self.w["honor_pong_bonus"] if tile[0] == "dragon" else 0.0
        gain = self.w["shanten"] * (cur_s - best["shanten"]) + self.w["ukeire"] * (best["ukeire"] - cur_uk) + bonus

        # Only pong for free (same shanten) on dragons, and never break a ready hand
        if best["shanten"] < cur_s or (best["shanten"] == cur_s and cur_s > 0 and gain > 0 and bonus):
            return Decision("pong", tile, gain,
                            f"Pong {format_tile(tile)}: shanten {cur_s} -> {best['shanten']}, "
                            f"then discard {format_tile(best['tile'])}")
        return Decision("pass", tile, gain,
                        f"Skip Pong {format_tile(tile)}: doesn't bring hand closer to ready")

    def chi_options(self, hand, tile):
        """All ways to Chi `tile` with tiles from hand."""
        suit, rank = tile
        if suit not in NUMBER_SUITS:
            return []
        options = []
        for pair in ([(suit, rank - 2), (suit, rank - 1)],
                     [(suit, rank - 1), (suit, rank + 1)],
                     [(suit, rank + 1), (suit, rank + 2)]):
            if all(1 <= t[1] <= 9 for t in pair) and all(t in hand for t in pair):
                options.append(pair)
        return options

    def should_chi(self, hand, melds, tile, visible_tiles=None, threat=0.0):
        """Returns Decision; if action == 'chi', details[0] is the 2 tiles to use."""
        melds = melds or []
        visible_tiles = visible_tiles or []
        options = self.chi_options(hand, tile)
        if not options:
            return Decision("pass", tile, reason="No Chi possible")

        cur_s, cur_uk = self._current_value(hand, melds, visible_tiles)
        best_option, best_result = None, None
        for pair in options:
            after = list(hand)
            for t in pair:
                after.remove(t)
            new_melds = melds + [sorted([tile] + pair, key=tile_sort_key)]
            result = self._best_after_call(after, new_melds, visible_tiles, threat)
            if best_result is None or result["score"] > best_result["score"]:
                best_option, best_result = pair, result

        if best_result["shanten"] < cur_s:
            return Decision("chi", tile, best_result["score"],
                            f"Chi {format_tile(tile)} with {[format_tile(t) for t in best_option]}: "
                            f"shanten {cur_s} -> {best_result['shanten']}",
                            [best_option])
        return Decision("pass", tile, 0.0,
                        f"Skip Chi {format_tile(tile)}: doesn't bring hand closer to ready")

    def should_kong(self, hand, melds, tile, visible_tiles=None, threat=0.0):
        melds = melds or []
        if hand.count(tile) != 3:
            return Decision("pass", tile, reason="Need exactly 3 in hand to Kong a discard")
        cur_s = shanten(hand, melds)
        after = [t for t in hand if t != tile]
        new_s = shanten(after, melds + [[tile] * 4])
        if new_s <= cur_s:
            return Decision("kong", tile, 0.0,
                            f"Kong {format_tile(tile)}: keeps hand at shanten {new_s}")
        return Decision("pass", tile, 0.0,
                        f"Skip Kong {format_tile(tile)}: would break up the hand")


# --------------------------------------------------------------------------
# Easy hooks for mahjong_simulator.py
# --------------------------------------------------------------------------
_engine = StrategyEngine()


def collect_visible_tiles(game):
    """Every tile that is face-up on the table (all discards + all melds)."""
    visible = []
    for pile in game.discards:
        visible.extend(pile)
    for player_melds in game.melds:
        for meld in player_melds:
            visible.extend(meld)
    return visible


def engine_choose_discard(game, player_index, engine=None, verbose=False):
    """Drop-in replacement for ai_choose_discard(game, player_index)."""
    engine = engine or _engine
    hand = game.hands[player_index]
    if not hand:
        return None
    threat = default_threat_level(game.melds, player_index, len(game.wall))
    decision = engine.choose_discard(hand, game.melds[player_index],
                                     collect_visible_tiles(game), threat)
    if verbose:
        print(f"[Strategy] Player {player_index + 1}: {decision.reason}")
    return decision.tile


def engine_should_call(game, player_index, tile, call_type, engine=None, verbose=False):
    """call_type: 'pong', 'chi', 'kong' or 'hu'. Returns True/False."""
    engine = engine or _engine
    hand = game.hands[player_index]
    melds = game.melds[player_index]
    visible = collect_visible_tiles(game)
    threat = default_threat_level(game.melds, player_index, len(game.wall))

    if call_type == "hu":
        # on your own turn the drawn tile is already in hand; on a discard it isn't
        on_discard = len(hand) + 3 * len(melds) == 13
        return engine.should_declare_hu(hand, melds, tile if on_discard else None)
    if call_type == "pong":
        decision = engine.should_pong(hand, melds, tile, visible, threat)
    elif call_type == "chi":
        decision = engine.should_chi(hand, melds, tile, visible, threat)
    elif call_type == "kong":
        decision = engine.should_kong(hand, melds, tile, visible, threat)
    else:
        raise ValueError(f"Unknown call type: {call_type}")

    if verbose:
        print(f"[Strategy] Player {player_index + 1}: {decision.reason}")
    return decision.action == call_type


# --------------------------------------------------------------------------
# Safe call handling (works around simulator bugs without editing it)
# --------------------------------------------------------------------------
def engine_can_win(game, player_index, tile=None):
    """
    Correct Hu check that also works with Kongs and exposed melds.
    (The simulator's can_hu() adds meld tiles to the hand and needs exactly
    14 tiles, so it fails once a player has a Kong.)
    Pass `tile` to check winning on another player's discard.
    """
    return _engine.should_declare_hu(game.hands[player_index],
                                     game.melds[player_index], tile)


def engine_perform_call(game, player_index, discarder_index, tile, call_type,
                        engine=None, decide=True, verbose=False):
    """
    Decide on AND carry out a Pong / Chi / Kong correctly.
    Use this instead of game.perform_pong / perform_chi / perform_kong.

    Fixes compared with the simulator's versions:
      - removes exactly the right tiles from the hand (no extra tile lost)
      - Chi uses the sequence the engine picked, not just the first one found
      - sets game.skip_next_draw = True after Pong/Chi, because the player
        should discard WITHOUT drawing (the simulator loop must check this).
        After a Kong it stays False: the player draws a replacement tile.

    decide=False skips the AI decision (e.g. the human already said yes).
    Returns True if the call was made.
    """
    engine = engine or _engine
    call_type = call_type.lower()
    hand = game.hands[player_index]
    melds = game.melds[player_index]

    # the tile must be the last tile the discarder threw
    if not game.discards[discarder_index] or game.discards[discarder_index][-1] != tile:
        return False

    visible = collect_visible_tiles(game)
    threat = default_threat_level(game.melds, player_index, len(game.wall))

    if call_type == "pong":
        if hand.count(tile) < 2:
            return False
        decision = engine.should_pong(hand, melds, tile, visible, threat) if decide else None
        if decide and decision.action != "pong":
            if verbose:
                print(f"[Strategy] Player {player_index + 1}: {decision.reason}")
            return False
        from_hand = [tile, tile]
        meld = [tile] * 3

    elif call_type == "kong":
        if hand.count(tile) != 3:
            return False
        decision = engine.should_kong(hand, melds, tile, visible, threat) if decide else None
        if decide and decision.action != "kong":
            if verbose:
                print(f"[Strategy] Player {player_index + 1}: {decision.reason}")
            return False
        from_hand = [tile, tile, tile]
        meld = [tile] * 4

    elif call_type == "chi":
        # only the player right after the discarder can Chi
        if player_index != (discarder_index + 1) % 4:
            return False
        decision = engine.should_chi(hand, melds, tile, visible, threat)
        if decide and decision.action != "chi":
            if verbose:
                print(f"[Strategy] Player {player_index + 1}: {decision.reason}")
            return False
        if decision.action == "chi":
            from_hand = decision.details[0]
        else:  # decide=False: use the best available option anyway
            options = engine.chi_options(hand, tile)
            if not options:
                return False
            from_hand = options[0]
        meld = sorted([tile] + list(from_hand), key=tile_sort_key)

    else:
        raise ValueError(f"Unknown call type: {call_type}")

    # carry out the call
    for t in from_hand:
        hand.remove(t)
    game.discards[discarder_index].pop()
    melds.append(meld)
    game.current_player = player_index
    game.skip_next_draw = call_type != "kong"

    if verbose and decision is not None:
        print(f"[Strategy] Player {player_index + 1}: {decision.reason}")
    print(f"Player {player_index + 1} calls {call_type.capitalize()} on {format_tile(tile)} "
          f"-> meld {[format_tile(t) for t in meld]}, {len(hand)} tiles left in hand")
    return True


# --------------------------------------------------------------------------
# Demo: run this file directly to see the engine think
# --------------------------------------------------------------------------
if __name__ == "__main__":
    engine = StrategyEngine()

    hand = [
        ("bamboo", 1), ("bamboo", 2), ("bamboo", 3),
        ("dot", 4), ("dot", 5), ("dot", 7),
        ("wan", 2), ("wan", 3), ("wan", 6), ("wan", 7),
        ("dragon", "red"), ("dragon", "red"),
        ("wind", "north"), ("dot", 9),
    ]
    print("Hand:", [format_tile(t) for t in sorted(hand, key=tile_sort_key)])
    print("Shanten (before discard):", shanten(hand))

    decision = engine.choose_discard(hand)
    print("\n" + decision.reason)
    print("\nTop 5 discards:")
    for r in decision.details[:5]:
        print(f"  {format_tile(r['tile']):<14} score={r['score']:>7}  "
              f"shanten={r['shanten']}  useful tiles={r['ukeire']}  danger={r['danger']}")

    thirteen = [t for t in hand]
    thirteen.remove(decision.tile)
    print("\nSomeone discards Red Dragon:")
    print(" ", engine.should_pong(thirteen, [], ("dragon", "red")).reason)
    print("Previous player discards 5 wan:")
    print(" ", engine.should_chi(thirteen, [], ("wan", 5)).reason)
