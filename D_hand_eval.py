"""Pure Mahjong hand evaluation utilities.

This module is intentionally limited to analysis only:
- inspect a player's current hand and exposed melds
- estimate how close the hand is to winning
- highlight useful tiles, dead tiles, and improvement paths
- score discard options for downstream AI decision logic

It does not manage the game loop, turns, draws, discards, or player actions.
That belongs in the simulator layer.
"""

from collections import Counter
from functools import lru_cache

NUMBER_SUITS = ("bamboo", "dot", "wan")
WINDS = ("east", "south", "west", "north")
DRAGONS = ("red", "green", "white")

ALL_TILE_TYPES = (
    [("bamboo", n) for n in range(1, 10)]
    + [("dot", n) for n in range(1, 10)]
    + [("wan", n) for n in range(1, 10)]
    + [("wind", w) for w in WINDS]
    + [("dragon", d) for d in DRAGONS]
)
ALL_TILE_SET = frozenset(ALL_TILE_TYPES)
# Thirteen Wonders: one of every terminal and honour, plus one more of any of them.
THIRTEEN_WONDERS = frozenset(
    [(s, n) for s in NUMBER_SUITS for n in (1, 9)]
    + [("wind", w) for w in WINDS]
    + [("dragon", d) for d in DRAGONS]
)


def _compute_sort_key(tile):
    suit, value = tile
    suit_order = {
        "bamboo": 0,
        "dot": 1,
        "wan": 2,
        "wind": 3,
        "dragon": 4,
    }

    if suit in NUMBER_SUITS:
        value_order = int(value)
    elif suit == "wind":
        value_order = WINDS.index(value) + 1
    elif suit == "dragon":
        value_order = DRAGONS.index(value) + 1
    else:
        value_order = 99

    return suit_order.get(suit, 99), value_order


_SORT_KEYS = {tile: _compute_sort_key(tile) for tile in ALL_TILE_TYPES}


def tile_sort_key(tile):
    key = _SORT_KEYS.get(tile)
    return key if key is not None else _compute_sort_key(tile)


def format_tile(tile):
    suit, value = tile
    if suit == "bamboo":
        return f"{value} bamboo"
    if suit == "dot":
        return f"{value} dots"
    if suit == "wan":
        return f"{value} wan"
    return f"{str(value).capitalize()} {suit.capitalize()}"


def validate_hand(hand):
    counts = Counter(hand)
    for tile, count in counts.items():
        if tile not in ALL_TILE_SET:
            return False, f"Invalid tile: {tile}"
        if count > 4:
            return False, f"Too many copies of {format_tile(tile)}: {count}"
    return True, ""


def is_sequence(three_tiles):
    if len(three_tiles) != 3:
        return False
    ordered = sorted(three_tiles, key=tile_sort_key)
    suits = {tile[0] for tile in ordered}
    if len(suits) != 1:
        return False
    suit = ordered[0][0]
    if suit not in NUMBER_SUITS:
        return False
    ranks = [tile[1] for tile in ordered]
    return ranks[1] == ranks[0] + 1 and ranks[2] == ranks[1] + 1


def is_triplet(three_tiles):
    return len(three_tiles) == 3 and len(set(three_tiles)) == 1


def is_pair(two_tiles):
    return len(two_tiles) == 2 and two_tiles[0] == two_tiles[1]


def _normalise_hand(hand):
    return tuple(sorted(hand, key=tile_sort_key))


# Long-running search workers see many distinct hands. Eviction limits retained
# memory; these pure checks recompute the same answer when an entry is evicted.
@lru_cache(maxsize=100_000)
def _can_form_sets_cached(hand_tuple, sets_needed):
    """Exact recursive partition check: can this sorted hand be split into
    `sets_needed` valid melds without ignoring any tiles?

    Only the lowest tile needs to be tried: in any valid split it must start
    either a triplet or a sequence, since nothing lower can precede it."""
    if sets_needed < 0:
        return False
    if sets_needed == 0:
        return len(hand_tuple) == 0
    if len(hand_tuple) != sets_needed * 3:
        return False

    hand = list(hand_tuple)
    first = hand[0]

    if hand.count(first) >= 3:
        if _can_form_sets_cached(tuple(hand[3:]), sets_needed - 1):
            return True

    suit, rank = first
    if suit in NUMBER_SUITS and rank <= 7:
        second = (suit, rank + 1)
        third = (suit, rank + 2)
        if second in hand and third in hand:
            hand.remove(first)
            hand.remove(second)
            hand.remove(third)
            # Removing tiles from a sorted list keeps it sorted.
            if _can_form_sets_cached(tuple(hand), sets_needed - 1):
                return True
    return False


def is_winning_hand(hand, exposed_melds=None):
    if exposed_melds is None:
        exposed_melds = []

    if len(exposed_melds) > 4:
        return False

    sets_needed = 4 - len(exposed_melds)
    required_concealed_tiles = sets_needed * 3 + 2
    if len(hand) != required_concealed_tiles:
        return False

    valid, reason = validate_hand(hand)
    if not valid:
        return False
    if sets_needed == 4 and is_thirteen_wonders(hand):
        return True
    return _is_winning_cached(_normalise_hand(hand), sets_needed)


def is_thirteen_wonders(hand):
    """All 13 terminals and honours plus a pair of one of them (concealed only)."""
    return len(hand) == 14 and set(hand) == THIRTEEN_WONDERS


@lru_cache(maxsize=100_000)
def _is_winning_cached(hand, sets_needed):
    counts = Counter(hand)
    pair_candidates = [pair_tile for pair_tile, count in counts.items() if count >= 2]
    for pair_tile in pair_candidates:
        remaining = list(hand)
        for _ in range(2):
            remaining.remove(pair_tile)
        if _can_form_sets_cached(_normalise_hand(remaining), sets_needed):
            return True
    return False


def winning_tiles(hand, exposed_melds=None):
    if exposed_melds is None:
        exposed_melds = []

    return list(_winning_tiles_cached(_normalise_hand(hand), len(exposed_melds)))


@lru_cache(maxsize=50_000)
def _winning_tiles_cached(hand, meld_count):
    # Only a hand one tile short of complete can be waiting on anything.
    if meld_count > 4 or len(hand) != (4 - meld_count) * 3 + 1:
        return ()
    placeholder_melds = [None] * meld_count
    counts = Counter(hand)
    waits = []
    for tile in ALL_TILE_TYPES:
        if counts[tile] >= 4:
            continue
        if is_winning_hand(list(hand) + [tile], placeholder_melds):
            waits.append(tile)
    return tuple(waits)


def hand_structure_score(hand):
    counts = Counter(hand)
    score = 0.0

    for tile in counts:
        count = counts[tile]
        suit, rank = tile

        if suit not in NUMBER_SUITS:
            if count >= 3:
                score += 6.0
            elif count == 2:
                score += 3.0
            continue

        if count >= 3:
            score += 6.0
        elif count == 2:
            score += 3.0

        for diff in (-1, 1):
            neighbor = (suit, rank + diff)
            if neighbor in counts:
                score += 2.0 * min(counts[neighbor], 2)

        for diff in (-2, 2):
            neighbor = (suit, rank + diff)
            if neighbor in counts:
                score += 1.0 * min(counts[neighbor], 2)

        if 3 <= rank <= 7:
            score += 0.5
        elif rank in (2, 8):
            score += 0.25

    for suit in NUMBER_SUITS:
        for rank in range(1, 8):
            tile1 = (suit, rank)
            tile2 = (suit, rank + 1)
            tile3 = (suit, rank + 2)
            sequence_count = min(counts.get(tile1, 0), counts.get(tile2, 0), counts.get(tile3, 0))
            score += sequence_count * 4.0

    for count in counts.values():
        if count >= 2:
            score += 1.5

    return score


def evaluate_hand_strength(hand, exposed_melds=None):
    if exposed_melds is None:
        exposed_melds = []

    valid, reason = validate_hand(hand)
    if not valid:
        raise ValueError(reason)

    wait_list = winning_tiles(hand, exposed_melds)
    structure = hand_structure_score(hand)
    score = structure

    if wait_list:
        score += 1000
        score += 100 * len(wait_list)

    return {
        "score": score,
        "structure": structure,
        "waits": wait_list,
        "wait_count": len(wait_list),
        "pair_count": sum(1 for count in Counter(hand).values() if count >= 2),
        "triplet_count": sum(1 for count in Counter(hand).values() if count >= 3),
        "tile_count": len(hand),
    }


def useful_tiles(hand, exposed_melds=None, visible_tiles=None):
    """Rank the tiles in the hand by how much they help it.

    Shape is judged from the player's own concealed tiles only. Copies that
    are already exposed in melds or discarded can never be drawn, so a pair
    whose other copies are all visible loses its value as a future pong."""
    if exposed_melds is None:
        exposed_melds = []
    if visible_tiles is None:
        visible_tiles = []

    own = Counter(hand)
    seen = Counter(visible_tiles)
    for meld in exposed_melds:
        seen.update(meld)

    ranked = []
    for tile in sorted(set(hand), key=tile_sort_key):
        suit, rank = tile
        count = own[tile]
        live = max(0, 4 - count - seen[tile])
        value = 0.0

        if count >= 3:
            value += 6.0
        elif count == 2:
            # Still usable as the pair; only a live copy can make it a pong.
            value += 3.0 if live else 1.5

        if suit in NUMBER_SUITS:
            for diff in (-1, 1, -2, 2):
                neighbor = (suit, rank + diff)
                if neighbor in own:
                    value += 2.0 if abs(diff) == 1 else 1.0
        elif count == 2 and live:
            value += 3.0

        ranked.append({"tile": tile, "usefulness": value})

    return sorted(ranked, key=lambda item: item["usefulness"], reverse=True)


def improvement_candidates(hand, exposed_melds=None):
    if exposed_melds is None:
        exposed_melds = []

    candidates = []
    counts = Counter(hand)
    for tile in ALL_TILE_TYPES:
        if counts[tile] >= 4:
            continue
        score = evaluate_hand_strength(list(hand) + [tile], exposed_melds)["score"]
        candidates.append({"tile": tile, "score": score})
    return sorted(candidates, key=lambda item: item["score"], reverse=True)


def evaluate_discard_quality(hand, discard, exposed_melds=None, visible_tiles=None):
    if exposed_melds is None:
        exposed_melds = []
    if visible_tiles is None:
        visible_tiles = []

    if discard not in hand:
        raise ValueError(f"Discard {format_tile(discard)} is not in the current hand.")

    before = evaluate_hand_strength(hand, exposed_melds)
    remaining = list(hand)
    remaining.remove(discard)
    after = evaluate_hand_strength(remaining, exposed_melds)

    return {
        "discard": discard,
        "before_score": before["score"],
        "after_score": after["score"],
        "delta": after["score"] - before["score"],
        "waits_before": before["waits"],
        "waits_after": after["waits"],
        "quality": "good" if after["score"] >= before["score"] else "bad",
    }


def analyze_hand(hand, exposed_melds=None, visible_tiles=None):
    if exposed_melds is None:
        exposed_melds = []
    if visible_tiles is None:
        visible_tiles = []

    valid, reason = validate_hand(hand)
    if not valid:
        raise ValueError(reason)

    strength = evaluate_hand_strength(hand, exposed_melds)
    waits = winning_tiles(hand, exposed_melds)
    useful = useful_tiles(hand, exposed_melds, visible_tiles)
    improvements = improvement_candidates(hand, exposed_melds)[:10]

    return {
        "strength": strength,
        "waits": waits,
        "useful_tiles": useful,
        "improvements": improvements,
        "is_winning": is_winning_hand(hand, exposed_melds),
    }


