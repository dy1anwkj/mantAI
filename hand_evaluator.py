from collections import Counter
from functools import lru_cache


# ============================================================
# MantAI - Hand Evaluator
# ============================================================
#
# Tile format is the SAME as mahjong_simulator.py:
#
# ('bamboo', 1)
# ('bamboo', 2)
# ...
# ('character', 1)
# ...
# ('wind', 'east')
# ('dragon', 'red')
#
# IMPORTANT:
#
# The "score" used in this file is NOT Mahjong points.
#
# It is only an INTERNAL AI VALUE that lets MantAI compare:
#
# "If I discard tile A, is my remaining hand better or worse
# than if I discard tile B?"
#
# This is the beginning of the decision-making system.
# ============================================================


# ------------------------------------------------------------
# TILE DEFINITIONS
# ------------------------------------------------------------

NUMBER_SUITS = ("bamboo", "character")

WINDS = (
    "east",
    "south",
    "west",
    "north",
)

DRAGONS = (
    "red",
    "green",
    "white",
)


ALL_TILE_TYPES = (
    [("bamboo", n) for n in range(1, 10)]
    + [("character", n) for n in range(1, 10)]
    + [("wind", w) for w in WINDS]
    + [("dragon", d) for d in DRAGONS]
)


# ------------------------------------------------------------
# BASIC TILE FUNCTIONS
# ------------------------------------------------------------

def tile_sort_key(tile):
    """
    Gives tiles a consistent sorting order.

    Example:
    1 bamboo
    2 bamboo
    ...
    1 character
    ...
    East Wind
    ...
    Red Dragon
    """

    suit, value = tile

    suit_order = {
        "bamboo": 0,
        "character": 1,
        "wind": 2,
        "dragon": 3,
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
    """
    Converts a tile tuple into readable text.

    Example:

    ('bamboo', 5)
    becomes
    "5 bamboo"

    ('wind', 'east')
    becomes
    "East Wind"
    """

    suit, value = tile

    if suit in NUMBER_SUITS:
        return f"{value} {suit}"

    return f"{str(value).capitalize()} {suit.capitalize()}"


def validate_hand(hand):
    """
    Checks whether all tiles in the hand are valid.

    Also checks that there are no more than 4 copies
    of the same tile.

    Returns:

    (True, "")

    if valid.

    Otherwise:

    (False, reason)
    """

    counts = Counter(hand)

    for tile, count in counts.items():

        if tile not in ALL_TILE_TYPES:
            return False, f"Invalid tile: {tile}"

        if count > 4:
            return False, (
                f"Too many copies of "
                f"{format_tile(tile)}: {count}"
            )

    return True, ""


# ------------------------------------------------------------
# MAHJONG SET CHECKS
# ------------------------------------------------------------

def is_sequence(three_tiles):
    """
    Checks whether 3 tiles form a Chi.

    Example:

    3 bamboo
    4 bamboo
    5 bamboo
    """

    if len(three_tiles) != 3:
        return False

    ordered = sorted(
        three_tiles,
        key=tile_sort_key
    )

    suits = {
        tile[0]
        for tile in ordered
    }

    # All tiles must be the same suit
    if len(suits) != 1:
        return False

    suit = ordered[0][0]

    # Winds and dragons cannot form sequences
    if suit not in NUMBER_SUITS:
        return False

    ranks = [
        tile[1]
        for tile in ordered
    ]

    return (
        ranks[1] == ranks[0] + 1
        and
        ranks[2] == ranks[1] + 1
    )


def is_triplet(three_tiles):
    """
    Checks whether 3 tiles form a Pong.
    """

    return (
        len(three_tiles) == 3
        and
        len(set(three_tiles)) == 1
    )


def is_pair(two_tiles):
    """
    Checks whether 2 tiles are identical.
    """

    return (
        len(two_tiles) == 2
        and
        two_tiles[0] == two_tiles[1]
    )


# ------------------------------------------------------------
# WINNING HAND CHECK
# ------------------------------------------------------------

def _normalise_hand(hand):
    """
    Converts hand into a sorted tuple.

    This allows Python to cache previous calculations,
    making recursive checks much faster.
    """

    return tuple(
        sorted(
            hand,
            key=tile_sort_key
        )
    )


@lru_cache(maxsize=None)
def _can_form_sets_cached(
    hand_tuple,
    sets_needed
):
    """
    Recursive helper.

    Checks whether ALL remaining tiles can be separated
    into valid Mahjong sets.

    A set can be:

    Pong:
    AAA

    or Chi:
    ABC
    """

    hand = list(hand_tuple)

    # If we need zero more sets,
    # there should be zero tiles remaining.
    if sets_needed == 0:
        return len(hand) == 0

    # Every set requires exactly 3 concealed tiles.
    if len(hand) != sets_needed * 3:
        return False

    # Always begin with first tile.
    # This reduces unnecessary possibilities.
    first = hand[0]


    # --------------------------------------------------------
    # POSSIBILITY 1:
    # FIRST TILE IS PART OF A PONG
    # --------------------------------------------------------

    if hand.count(first) >= 3:

        temp_hand = hand.copy()

        temp_hand.remove(first)
        temp_hand.remove(first)
        temp_hand.remove(first)

        if _can_form_sets_cached(
            _normalise_hand(temp_hand),
            sets_needed - 1
        ):
            return True


    # --------------------------------------------------------
    # POSSIBILITY 2:
    # FIRST TILE IS PART OF A CHI
    # --------------------------------------------------------

    suit, rank = first

    if suit in NUMBER_SUITS:

        # Sequences cannot start above 7.
        if rank <= 7:

            second = (
                suit,
                rank + 1
            )

            third = (
                suit,
                rank + 2
            )

            if (
                second in hand
                and
                third in hand
            ):

                temp_hand = hand.copy()

                temp_hand.remove(first)
                temp_hand.remove(second)
                temp_hand.remove(third)

                if _can_form_sets_cached(
                    _normalise_hand(temp_hand),
                    sets_needed - 1
                ):
                    return True


    # No valid combination found.
    return False


def is_winning_hand(
    hand,
    exposed_melds=None
):
    """
    Checks whether the hand is a winning Mahjong hand.

    Standard winning shape:

    4 sets
    +
    1 pair

    Example:

    1-2-3 bamboo
    4-5-6 bamboo
    2-3-4 character
    7-8-9 character
    Red Dragon + Red Dragon


    exposed_melds can be supplied using:

    game.melds[player_index]

    Example:

    [
        [
            ('bamboo', 3),
            ('bamboo', 3),
            ('bamboo', 3)
        ]
    ]

    A Kong counts as ONE completed set,
    even though it contains 4 tiles.
    """

    if exposed_melds is None:
        exposed_melds = []


    # --------------------------------------------------------
    # CHECK THAT HAND IS VALID
    # --------------------------------------------------------

    valid, reason = validate_hand(hand)

    if not valid:
        return False


    # Cannot have more than 4 sets.
    if len(exposed_melds) > 4:
        return False


    # --------------------------------------------------------
    # CALCULATE HOW MANY SETS STILL NEED TO BE MADE
    # --------------------------------------------------------

    sets_needed = (
        4
        -
        len(exposed_melds)
    )


    # Example:
    #
    # No exposed melds:
    # 4 sets needed
    #
    # 4 * 3 + 2
    # = 14 tiles
    #
    #
    # One exposed Pong:
    #
    # 3 sets needed
    #
    # 3 * 3 + 2
    # = 11 concealed tiles

    required_concealed_tiles = (
        sets_needed * 3
        +
        2
    )


    if (
        len(hand)
        !=
        required_concealed_tiles
    ):
        return False


    counts = Counter(hand)


    # --------------------------------------------------------
    # TRY EVERY POSSIBLE PAIR
    # --------------------------------------------------------

    for pair_tile, count in counts.items():

        if count < 2:
            continue


        remaining = list(hand)

        remaining.remove(pair_tile)
        remaining.remove(pair_tile)


        # Everything else must form sets.
        if _can_form_sets_cached(
            _normalise_hand(remaining),
            sets_needed
        ):
            return True


    return False


# ------------------------------------------------------------
# WINNING TILE / WAIT CHECKER
# ------------------------------------------------------------

def winning_tiles(
    hand,
    exposed_melds=None
):
    """
    Finds all tiles that would immediately complete the hand.

    Usually used when the player has 13 tiles.

    Example:

    If player has:

    1-2-3
    4-5-6
    2-3-4
    7-8-9
    Red Dragon

    then:

    Red Dragon

    would be returned as a winning tile.
    """

    if exposed_melds is None:
        exposed_melds = []


    counts = Counter(hand)

    waits = []


    # Try adding every possible Mahjong tile.
    for tile in ALL_TILE_TYPES:

        # Mahjong only has 4 copies of each tile.
        if counts[tile] >= 4:
            continue


        test_hand = (
            list(hand)
            +
            [tile]
        )


        if is_winning_hand(
            test_hand,
            exposed_melds
        ):
            waits.append(tile)


    return sorted(
        waits,
        key=tile_sort_key
    )


# ------------------------------------------------------------
# COUNT REMAINING WINNING TILES
# ------------------------------------------------------------

def count_immediate_winning_copies(
    hand,
    exposed_melds=None,
    visible_tiles=None
):
    """
    Counts approximately how many winning tile copies
    are still available.

    Example:

    You are waiting for Red Dragon.

    Mahjong has 4 Red Dragons.

    If:
        1 is in your hand
        1 has already been discarded

    then:

        2 copies remain.

    visible_tiles can contain things such as:

    - discarded tiles
    - exposed tiles
    - tiles already seen on the table
    """

    if exposed_melds is None:
        exposed_melds = []

    if visible_tiles is None:
        visible_tiles = []


    known_tiles = Counter(hand)

    known_tiles.update(
        visible_tiles
    )


    # Add exposed meld tiles.
    for meld in exposed_melds:
        known_tiles.update(meld)


    total_remaining = 0


    waits = winning_tiles(
        hand,
        exposed_melds
    )


    for tile in waits:

        remaining_copies = (
            4
            -
            known_tiles[tile]
        )


        if remaining_copies < 0:
            remaining_copies = 0


        total_remaining += (
            remaining_copies
        )


    return total_remaining


# ------------------------------------------------------------
# TILE CONNECTION VALUE
# ------------------------------------------------------------

def _tile_connection_value(
    tile,
    counts
):
    """
    Gives one tile an INTERNAL usefulness value.

    Again:

    THIS IS NOT MAHJONG POINT SCORING.

    It simply helps MantAI decide whether a tile is
    structurally useful.

    Example:

    5 bamboo

    is more useful if you already have:

    4 bamboo
    6 bamboo

    than if 5 bamboo is completely isolated.
    """

    suit, rank = tile


    # --------------------------------------------------------
    # HONOR TILES
    # --------------------------------------------------------

    if suit not in NUMBER_SUITS:

        count = counts[tile]


        # Triplet
        if count >= 3:
            return 6.0


        # Pair
        if count == 2:
            return 3.0


        # Single isolated honor
        return 0.0


    # --------------------------------------------------------
    # NUMBER TILE
    # --------------------------------------------------------

    value = 0.0

    count = counts[tile]


    # Duplicate usefulness
    if count >= 3:

        value += 6.0

    elif count == 2:

        value += 3.0


    # --------------------------------------------------------
    # DIRECT NEIGHBOURS
    #
    # Example:
    #
    # 4 and 5
    # 5 and 6
    # --------------------------------------------------------

    for difference in (-1, 1):

        neighbour = (
            suit,
            rank + difference
        )


        if neighbour in counts:

            value += (
                2.0
                *
                min(
                    counts[neighbour],
                    2
                )
            )


    # --------------------------------------------------------
    # GAP-ONE CONNECTIONS
    #
    # Example:
    #
    # 3 and 5
    #
    # waiting for 4
    # --------------------------------------------------------

    for difference in (-2, 2):

        neighbour = (
            suit,
            rank + difference
        )


        if neighbour in counts:

            value += (
                1.0
                *
                min(
                    counts[neighbour],
                    2
                )
            )


    # --------------------------------------------------------
    # CENTRAL TILE BONUS
    #
    # Tiles such as 4,5,6 have more possible sequences.
    # --------------------------------------------------------

    if 3 <= rank <= 7:

        value += 0.5


    elif rank in (2, 8):

        value += 0.25


    return value


# ------------------------------------------------------------
# WHOLE HAND STRUCTURE VALUE
# ------------------------------------------------------------

def hand_structure_score(hand):
    """
    Gives the entire hand an INTERNAL AI VALUE.

    Higher number =
    generally better connected hand.

    THIS IS NOT MAHJONG SCORING.

    It is purely for decision comparison.

    For example:

    MantAI can compare:

    discard 1 bamboo -> remaining hand value 45

    discard Red Dragon -> remaining hand value 63

    MantAI would currently prefer Red Dragon.
    """

    counts = Counter(hand)

    score = 0.0


    # --------------------------------------------------------
    # TILE CONNECTIONS
    # --------------------------------------------------------

    for tile in counts:

        score += _tile_connection_value(
            tile,
            counts
        )


    # --------------------------------------------------------
    # COMPLETED TRIPLETS
    # --------------------------------------------------------

    for tile, count in counts.items():

        completed_triplets = (
            count // 3
        )

        score += (
            completed_triplets
            *
            5.0
        )


    # --------------------------------------------------------
    # COMPLETED SEQUENCES
    # --------------------------------------------------------

    for suit in NUMBER_SUITS:

        for rank in range(1, 8):

            tile1 = (
                suit,
                rank
            )

            tile2 = (
                suit,
                rank + 1
            )

            tile3 = (
                suit,
                rank + 2
            )


            sequence_count = min(
                counts[tile1],
                counts[tile2],
                counts[tile3],
            )


            score += (
                sequence_count
                *
                4.0
            )


    # --------------------------------------------------------
    # PAIR BONUS
    # --------------------------------------------------------

    for count in counts.values():

        if count >= 2:

            score += 1.5


    return score


# ------------------------------------------------------------
# EVALUATE EVERY POSSIBLE DISCARD
# ------------------------------------------------------------

def evaluate_discards(
    hand,
    exposed_melds=None,
    visible_tiles=None
):
    """
    Tests every UNIQUE tile that could be discarded.

    Example:

    If your hand has:

    1 bamboo
    1 bamboo
    2 bamboo
    3 bamboo
    ...

    MantAI evaluates:

    "What happens if I discard 1 bamboo?"

    then:

    "What happens if I discard 2 bamboo?"

    etc.


    Returns a list from:

    BEST DISCARD
    ->
    WORST DISCARD


    Each result contains:

    {
        "discard": tile,

        "ai_value": number,

        "winning_tiles": [...],

        "winning_tile_types": number,

        "winning_copies_remaining": number,

        "structure_score": number
    }
    """

    if exposed_melds is None:
        exposed_melds = []

    if visible_tiles is None:
        visible_tiles = []


    # Make sure hand is legal.
    valid, reason = validate_hand(hand)

    if not valid:
        raise ValueError(reason)


    results = []


    # --------------------------------------------------------
    # TRY DISCARDING EVERY UNIQUE TILE
    # --------------------------------------------------------

    unique_tiles = sorted(
        set(hand),
        key=tile_sort_key
    )


    for discard in unique_tiles:

        remaining_hand = list(hand)

        remaining_hand.remove(
            discard
        )


        # ----------------------------------------------------
        # DOES THIS DISCARD PUT US INTO TENPAI?
        # ----------------------------------------------------

        waits = winning_tiles(
            remaining_hand,
            exposed_melds
        )


        # ----------------------------------------------------
        # HOW MANY PHYSICAL WINNING TILES MAY REMAIN?
        # ----------------------------------------------------

        winning_copies = (
            count_immediate_winning_copies(
                remaining_hand,
                exposed_melds=exposed_melds,
                visible_tiles=visible_tiles,
            )
        )


        # ----------------------------------------------------
        # GENERAL HAND STRUCTURE
        # ----------------------------------------------------

        structure = (
            hand_structure_score(
                remaining_hand
            )
        )


        # ----------------------------------------------------
        # BUILD FINAL AI COMPARISON VALUE
        # ----------------------------------------------------

        ai_value = structure


        # Being one tile from winning is FAR more important
        # than small structural differences.
        if waits:

            ai_value += 1000.0


            # More different winning tile types is better.
            ai_value += (
                100.0
                *
                len(waits)
            )


            # More remaining physical copies is better.
            ai_value += (
                10.0
                *
                winning_copies
            )


        # Save result.
        results.append(
            {
                "discard": discard,

                "ai_value": ai_value,

                "winning_tiles": waits,

                "winning_tile_types": len(
                    waits
                ),

                "winning_copies_remaining":
                    winning_copies,

                "structure_score":
                    structure,
            }
        )


    # --------------------------------------------------------
    # SORT BEST -> WORST
    # --------------------------------------------------------

    results.sort(
        key=lambda item: (
            item["ai_value"],
            item[
                "winning_copies_remaining"
            ],
            item[
                "winning_tile_types"
            ],
            item[
                "structure_score"
            ],
        ),
        reverse=True,
    )


    return results


# ------------------------------------------------------------
# BEST DISCARD
# ------------------------------------------------------------

def best_discard(
    hand,
    exposed_melds=None,
    visible_tiles=None
):
    """
    Returns only MantAI's currently preferred discard.

    Example:

    tile = best_discard(hand)

    print(tile)

    might return:

    ('wind', 'east')
    """

    if not hand:
        return None


    evaluations = evaluate_discards(
        hand,
        exposed_melds=exposed_melds,
        visible_tiles=visible_tiles,
    )


    if not evaluations:
        return None


    return evaluations[0]["discard"]


# ------------------------------------------------------------
# EXPLANATION FUNCTION
# ------------------------------------------------------------

def explain_best_discard(
    hand,
    exposed_melds=None,
    visible_tiles=None
):
    """
    Gives a readable explanation of MantAI's decision.

    Useful while developing and debugging.
    """

    evaluations = evaluate_discards(
        hand,
        exposed_melds=exposed_melds,
        visible_tiles=visible_tiles,
    )


    if not evaluations:

        return (
            "No discard available."
        )


    best = evaluations[0]


    discard_text = format_tile(
        best["discard"]
    )


    # --------------------------------------------------------
    # CASE 1:
    # DISCARD PUTS PLAYER INTO WINNING WAIT
    # --------------------------------------------------------

    if best["winning_tiles"]:

        wait_names = [
            format_tile(tile)
            for tile
            in best["winning_tiles"]
        ]


        waits_text = ", ".join(
            wait_names
        )


        return (

            f"Discard {discard_text}. "

            f"This leaves an immediate winning "
            f"wait on: {waits_text}. "

            f"Estimated remaining winning copies: "
            f"{best['winning_copies_remaining']}."

        )


    # --------------------------------------------------------
    # CASE 2:
    # NOT IN TENPAI YET
    # --------------------------------------------------------

    return (

        f"Discard {discard_text}. "

        f"No immediate winning wait is available, "

        f"so MantAI currently prefers the discard "

        f"that keeps the strongest tile connections."

    )


# ============================================================
# TEST CODE
# ============================================================
#
# This section ONLY runs if you run:
#
# python hand_evaluator.py
#
# It will NOT run when another Python file imports this file.
#
# ============================================================

if __name__ == "__main__":


    test_hand = [

        # Bamboo sequence
        ("bamboo", 1),
        ("bamboo", 2),
        ("bamboo", 3),

        # Bamboo sequence
        ("bamboo", 4),
        ("bamboo", 5),
        ("bamboo", 6),

        # Character sequence
        ("character", 2),
        ("character", 3),
        ("character", 4),

        # Character sequence
        ("character", 7),
        ("character", 8),
        ("character", 9),

        # Two unrelated honor tiles
        ("dragon", "red"),
        ("wind", "east"),

    ]


    print()
    print("=" * 60)

    print(
        "MantAI Hand Evaluator Test"
    )

    print("=" * 60)


    print()
    print("HAND:")
    print()


    sorted_hand = sorted(
        test_hand,
        key=tile_sort_key
    )


    for tile in sorted_hand:

        print(
            " -",
            format_tile(tile)
        )


    print()
    print("=" * 60)

    print(
        "MANTAI DECISION:"
    )

    print("=" * 60)

    print()


    print(
        explain_best_discard(
            test_hand
        )
    )


    print()
    print("=" * 60)

    print(
        "TOP DISCARD OPTIONS:"
    )

    print("=" * 60)

    print()


    results = evaluate_discards(
        test_hand
    )


    for result in results[:5]:

        discard_name = format_tile(
            result["discard"]
        )


        wait_names = [

            format_tile(tile)

            for tile
            in result[
                "winning_tiles"
            ]

        ]


        print(
            f"Discard: {discard_name}"
        )

        print(
            f"AI value: "
            f"{result['ai_value']:.2f}"
        )

        print(
            f"Winning tiles: "
            f"{wait_names}"
        )

        print(
            f"Winning copies remaining: "
            f"{result['winning_copies_remaining']}"
        )

        print(
            f"Structure value: "
            f"{result['structure_score']:.2f}"
        )

        print("-" * 40)