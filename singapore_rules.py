"""Singapore-style Mahjong rules.

- 148 tiles: the 136 standard tiles plus 12 bonus tiles
  (4 flowers, 4 seasons, 4 animals).
- Bonus tiles are never kept in the hand. When one is dealt or drawn it is
  revealed and replaced with a tile from the back of the wall.
- A player can win on their own draw (self-drawn) or on another player's
  discard. The player whose discard completes the winning hand is the "shooter".
- Hands are scored in tai. A hand needs at least MIN_TAI to win, and tai is
  capped at MAX_TAI.
- Payouts double with every tai. A self-drawn win is paid double by all three
  opponents. On a discard win the shooter pays more (see SHOOTER_MODE).
"""

from collections import Counter

from hand_eval import NUMBER_SUITS, WINDS, _normalise_hand, tile_sort_key

FLOWERS = [('flower', n) for n in range(1, 5)]   # red set, numbered 1-4
SEASONS = [('season', n) for n in range(1, 5)]   # blue set, numbered 1-4
ANIMALS = [('animal', a) for a in ('cat', 'mouse', 'rooster', 'centipede')]
BONUS_TILES = FLOWERS + SEASONS + ANIMALS
BONUS_SUITS = ('flower', 'season', 'animal')

FLOWER_NAMES = {1: 'Plum', 2: 'Orchid', 3: 'Chrysanthemum', 4: 'Bamboo'}
SEASON_NAMES = {1: 'Spring', 2: 'Summer', 3: 'Autumn', 4: 'Winter'}

MIN_TAI = 1
MAX_TAI = 5
# "half": shooter pays double, the other two pay single.
# "full": shooter alone pays for the whole table (4x).
SHOOTER_MODE = "half"


def is_bonus(tile):
    return tile[0] in BONUS_SUITS


def format_bonus(tile):
    suit, value = tile
    if suit == 'flower':
        return f"Flower {value} ({FLOWER_NAMES[value]})"
    if suit == 'season':
        return f"Season {value} ({SEASON_NAMES[value]})"
    return f"{value.capitalize()} Animal"


def seat_number(player_index, dealer):
    """Seat 1 is East (the dealer), then 2 South, 3 West, 4 North."""
    return (player_index - dealer) % 4 + 1


def seat_wind(player_index, dealer):
    return WINDS[seat_number(player_index, dealer) - 1]


# --------------------------------------------------------------------------
# Tai scoring
# --------------------------------------------------------------------------
def bonus_tai(bonus_tiles, seat):
    """Tai from revealed flowers, seasons and animals."""
    items = []
    for suit, label in (('flower', 'flowers'), ('season', 'seasons')):
        owned = {value for s, value in bonus_tiles if s == suit}
        if len(owned) == 4:
            items.append((f"Complete set of {label}", 2))
        elif seat in owned:
            items.append((f"Own seat {format_bonus((suit, seat))}", 1))
    for tile in bonus_tiles:
        if tile[0] == 'animal':
            items.append((format_bonus(tile), 1))
    return items


def _set_splits(hand, sets_needed):
    """Yield every way to split a sorted tile tuple into pongs and chows."""
    if sets_needed == 0:
        if not hand:
            yield []
        return
    if len(hand) != sets_needed * 3:
        return
    first = hand[0]
    if hand.count(first) >= 3:
        for rest in _set_splits(hand[3:], sets_needed - 1):
            yield [(first,) * 3] + rest
    suit, rank = first
    if suit in NUMBER_SUITS and rank <= 7:
        second, third = (suit, rank + 1), (suit, rank + 2)
        if second in hand and third in hand:
            remaining = list(hand)
            for tile in (first, second, third):
                remaining.remove(tile)
            for rest in _set_splits(tuple(remaining), sets_needed - 1):
                yield [(first, second, third)] + rest


def decompositions(concealed, sets_needed):
    """Yield (pair_tile, sets) for every way the concealed tiles form a win."""
    hand = _normalise_hand(concealed)
    for pair in sorted(set(hand), key=tile_sort_key):
        if hand.count(pair) < 2:
            continue
        remaining = list(hand)
        remaining.remove(pair)
        remaining.remove(pair)
        for sets in _set_splits(tuple(remaining), sets_needed):
            yield pair, sets


def _is_pong(meld):
    return len(meld) >= 3 and len(set(meld)) == 1


def _pattern_tai(pair, sets, seat_wind_name, prevailing_wind, has_bonus, two_sided_wait):
    items = []
    for meld in sets:
        tile = meld[0]
        if not _is_pong(meld):
            continue
        if tile[0] == 'dragon':
            items.append((f"Pong of {tile[1]} dragon", 1))
        elif tile[0] == 'wind':
            if tile[1] == seat_wind_name:
                items.append((f"Pong of seat wind ({tile[1]})", 1))
            if tile[1] == prevailing_wind:
                items.append((f"Pong of prevailing wind ({tile[1]})", 1))

    every_tile = [pair] + [t for meld in sets for t in meld]
    suits = {t[0] for t in every_tile if t[0] in NUMBER_SUITS}
    has_honours = any(t[0] not in NUMBER_SUITS for t in every_tile)
    if not suits:
        items.append(("All honours", MAX_TAI))
    elif len(suits) == 1 and not has_honours:
        items.append(("Full colour (one suit)", 4))
    elif len(suits) == 1:
        items.append(("Half colour (one suit + honours)", 2))

    if all(_is_pong(meld) for meld in sets):
        items.append(("All pongs", 2))

    scoring_pair = pair[0] == 'dragon' or pair[1] in (seat_wind_name, prevailing_wind)
    if (not any(_is_pong(meld) for meld in sets) and not scoring_pair
            and not has_bonus and two_sided_wait):
        items.append(("Ping Hu (all chows, no bonus tiles, two-sided wait)", 4))
    return items


def score_hand(concealed, melds, bonus_tiles, seat, prevailing_wind="east", waits_before_win=()):
    """
    Score a complete winning hand.

    concealed         : concealed tiles, including the winning tile
    melds             : exposed melds (lists of tiles)
    bonus_tiles       : revealed flowers / seasons / animals
    seat              : 1-4 (1 = East / dealer)
    waits_before_win  : tiles the hand was waiting on before it won,
                        used to tell whether the wait was two-sided (Ping Hu)

    Returns {"tai": int, "items": [(description, tai), ...]} using the
    highest-scoring way to read the hand, or None if it is not a winning hand.
    """
    sets_needed = 4 - len(melds)
    exposed = [tuple(m) for m in melds]
    bonus_items = bonus_tai(bonus_tiles, seat)
    wind_name = WINDS[seat - 1]

    best = None
    for pair, sets in decompositions(concealed, sets_needed):
        items = bonus_items + _pattern_tai(
            pair, sets + exposed, wind_name, prevailing_wind,
            has_bonus=bool(bonus_tiles), two_sided_wait=len(waits_before_win) >= 2,
        )
        total = min(MAX_TAI, sum(tai for _, tai in items))
        if best is None or total > best["tai"]:
            best = {"tai": total, "items": items}
    return best


# --------------------------------------------------------------------------
# Payouts
# --------------------------------------------------------------------------
def payouts(tai, winner, self_drawn, shooter=None, mode=None):
    """Units paid by each player (negative) and received by the winner."""
    mode = mode or SHOOTER_MODE
    base = 2 ** (max(1, min(tai, MAX_TAI)) - 1)
    result = Counter()
    others = [p for p in range(4) if p != winner]
    if self_drawn:
        for p in others:
            result[p] -= 2 * base
    elif mode == "full":
        result[shooter] -= 4 * base
    else:
        for p in others:
            result[p] -= 2 * base if p == shooter else base
    result[winner] = -sum(result.values())
    return dict(result)
