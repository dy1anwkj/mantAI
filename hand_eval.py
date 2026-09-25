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


def tile_sort_key(tile):
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
        if tile not in ALL_TILE_TYPES:
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


@lru_cache(maxsize=None)
def _can_form_sets_cached(hand_tuple, sets_needed):
    hand = list(hand_tuple)
    if sets_needed == 0:
        return len(hand) == 0
    if len(hand) != sets_needed * 3:
        return False
    first = hand[0]

    if hand.count(first) >= 3:
        temp_hand = hand.copy()
        temp_hand.remove(first)
        temp_hand.remove(first)
        temp_hand.remove(first)
        if _can_form_sets_cached(_normalise_hand(temp_hand), sets_needed - 1):
            return True

    suit, rank = first
    if suit in NUMBER_SUITS and rank <= 7:
        second = (suit, rank + 1)
        third = (suit, rank + 2)
        if second in hand and third in hand:
            temp_hand = hand.copy()
            temp_hand.remove(first)
            temp_hand.remove(second)
            temp_hand.remove(third)
            if _can_form_sets_cached(_normalise_hand(temp_hand), sets_needed - 1):
                return True
    return False


def is_winning_hand(hand, exposed_melds=None):
    if exposed_melds is None:
        exposed_melds = []

    valid, reason = validate_hand(hand)
    if not valid:
        return False
    if len(exposed_melds) > 4:
        return False

    sets_needed = 4 - len(exposed_melds)
    required_concealed_tiles = sets_needed * 3 + 2
    if len(hand) != required_concealed_tiles:
        return False

    counts = Counter(hand)
    for pair_tile, count in counts.items():
        if count < 2:
            continue
        remaining = list(hand)
        remaining.remove(pair_tile)
        remaining.remove(pair_tile)
        if _can_form_sets_cached(_normalise_hand(remaining), sets_needed):
            return True
    return False


def winning_tiles(hand, exposed_melds=None):
    if exposed_melds is None:
        exposed_melds = []

    waits = []
    counts = Counter(hand)
    for tile in ALL_TILE_TYPES:
        if counts[tile] >= 4:
            continue
        test_hand = list(hand) + [tile]
        if is_winning_hand(test_hand, exposed_melds):
            waits.append(tile)
    return sorted(waits, key=tile_sort_key)


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
    if exposed_melds is None:
        exposed_melds = []
    if visible_tiles is None:
        visible_tiles = []

    counts = Counter(hand)
    for meld in exposed_melds:
        counts.update(meld)
    counts.update(visible_tiles)

    ranked = []
    for tile in sorted(set(hand), key=tile_sort_key):
        suit, rank = tile
        value = 0.0

        if counts[tile] >= 3:
            value += 6.0
        elif counts[tile] == 2:
            value += 3.0

        if suit in NUMBER_SUITS:
            for diff in (-1, 1, -2, 2):
                neighbor = (suit, rank + diff)
                if neighbor in counts:
                    value += 2.0 if abs(diff) == 1 else 1.0

        if suit not in NUMBER_SUITS:
            if counts[tile] >= 2:
                value += 3.0

        ranked.append({"tile": tile, "usefulness": value})

    return sorted(ranked, key=lambda item: item["usefulness"], reverse=True)


def improvement_candidates(hand, exposed_melds=None):
    if exposed_melds is None:
        exposed_melds = []

    candidates = []
    for tile in ALL_TILE_TYPES:
        if Counter(hand)[tile] >= 4:
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
    remaining = [tile for tile in hand if tile != discard]
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


if __name__ == "__main__":
    sample_hand = [
        ("bamboo", 1), ("bamboo", 2), ("bamboo", 3),
        ("dot", 4), ("dot", 5), ("dot", 6),
        ("wan", 5), ("wan", 6), ("wan", 7),
        ("wan", 2), ("wan", 3),
        ("dragon", "red"), ("dragon", "red"),
    ]

    result = analyze_hand(sample_hand)
    print("Hand analysis:")
    print("- strength score:", result["strength"]["score"])
    print("- winning wait tiles:", [format_tile(t) for t in result["waits"]])
    print("- top useful tiles:")
    for item in result["useful_tiles"][:5]:
        print("  *", format_tile(item["tile"]), "->", round(item["usefulness"], 2))
    print("- top improvement tiles:")
    for item in result["improvements"][:5]:
        print("  *", format_tile(item["tile"]), "-> score", round(item["score"], 2))

    discard_result = evaluate_discard_quality(sample_hand, ("dragon", "red"))
    print("- discard quality:", discard_result)
