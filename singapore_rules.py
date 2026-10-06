"""Singapore-style Mahjong rules.

- 148 tiles: the 136 standard tiles plus 12 bonus tiles
  (4 flowers, 4 seasons, 4 animals).
- Bonus tiles are never kept in the hand. When one is dealt or drawn it is
  revealed and replaced with a tile from the back of the wall.
- A player can win on their own draw (self-drawn) or on another player's
  discard. The player whose discard completes the winning hand is the "shooter".
- The last DEAD_WALL (15) tiles of the wall are never drawn; the hand is a
  draw once only they are left.
- Hands are scored in tai. A hand needs at least MIN_TAI to win, and tai is
  capped at MAX_TAI.
- Payouts double with every tai. A self-drawn win is paid double by all three
  opponents. On a discard win the shooter pays more (see SHOOTER_MODE).
- Limit hands (Thirteen Wonders, Big Three Dragons, Big / Small Four Winds,
  All Honours, Heavenly / Earthly Hand) score MAX_TAI.
- Instant payments are made during play, win or not: for every Kong and for
  completed animal pairs or flower / season sets (see the *_PAYMENT values).
"""

from collections import Counter

from D_hand_eval import NUMBER_SUITS, WINDS, _normalise_hand, is_thirteen_wonders, tile_sort_key

FLOWERS = [('flower', n) for n in range(1, 5)]   # red set, numbered 1-4
SEASONS = [('season', n) for n in range(1, 5)]   # blue set, numbered 1-4
ANIMALS = [('animal', a) for a in ('cat', 'mouse', 'rooster', 'centipede')]
BONUS_TILES = FLOWERS + SEASONS + ANIMALS
BONUS_SUITS = ('flower', 'season', 'animal')

FLOWER_NAMES = {1: 'Plum', 2: 'Orchid', 3: 'Chrysanthemum', 4: 'Bamboo'}
SEASON_NAMES = {1: 'Spring', 2: 'Summer', 3: 'Autumn', 4: 'Winter'}

MIN_TAI = 1
# SPGG 2024 rule 9: the last 15 tiles of the wall are never drawn. The hand
# is a draw once only these are left.
DEAD_WALL = 15
MAX_TAI = 5
# "half": shooter pays double, the other two pay single.
# "full": shooter alone pays for the whole table (4x).
SHOOTER_MODE = "half"

# Extra tai on top of the two dragon pongs (1 tai each): 4 tai in total.
SMALL_THREE_DRAGONS_TAI = 2
# 1 tai each for winning on a Kong replacement tile, robbing a Kong,
# or self-drawing the last tile of the wall.
SITUATION_TAI = 1

# Instant payments: units EACH opponent pays straight away.
KONG_PAYMENT = {"exposed": 1, "concealed": 2}
ANIMAL_PAIRS = ((('animal', 'cat'), ('animal', 'mouse')),
                (('animal', 'rooster'), ('animal', 'centipede')))
ANIMAL_PAIR_PAYMENT = 1        # cat + mouse, or rooster + centipede
BONUS_SET_PAYMENT = 2          # all 4 flowers, or all 4 seasons


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

    pongs = [meld[0] for meld in sets if _is_pong(meld)]
    dragon_pongs = sum(t[0] == 'dragon' for t in pongs)
    wind_pongs = sum(t[0] == 'wind' for t in pongs)
    if dragon_pongs == 3:
        items.append(("Big Three Dragons", MAX_TAI))
    elif dragon_pongs == 2 and pair[0] == 'dragon':
        items.append(("Small Three Dragons", SMALL_THREE_DRAGONS_TAI))
    if wind_pongs == 4:
        items.append(("Big Four Winds", MAX_TAI))
    elif wind_pongs == 3 and pair[0] == 'wind':
        items.append(("Small Four Winds", MAX_TAI))

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


def situation_tai(kong_replacement=False, robbed_kong=False, last_tile=False,
                  heavenly=False, earthly=False):
    """Tai that depend on how the hand was won rather than on its tiles."""
    items = []
    if heavenly:
        items.append(("Heavenly Hand (dealer wins on the deal)", MAX_TAI))
    if earthly:
        items.append(("Earthly Hand (win on the dealer's first discard)", MAX_TAI))
    if kong_replacement:
        items.append(("Win on a Kong replacement tile", SITUATION_TAI))
    if robbed_kong:
        items.append(("Robbing the Kong", SITUATION_TAI))
    if last_tile:
        items.append(("Win on the last tile of the wall", SITUATION_TAI))
    return items


def score_hand(concealed, melds, bonus_tiles, seat, prevailing_wind="east", waits_before_win=(),
               situation=()):
    """
    Score a complete winning hand.

    concealed         : concealed tiles, including the winning tile
    melds             : exposed melds (lists of tiles)
    bonus_tiles       : revealed flowers / seasons / animals
    seat              : 1-4 (1 = East / dealer)
    waits_before_win  : tiles the hand was waiting on before it won,
                        used to tell whether the wait was two-sided (Ping Hu)
    situation         : extra (description, tai) items, see situation_tai()

    Returns {"tai": int, "items": [(description, tai), ...]} using the
    highest-scoring way to read the hand, or None if it is not a winning hand.
    """
    sets_needed = 4 - len(melds)
    exposed = [tuple(m) for m in melds]
    bonus_items = bonus_tai(bonus_tiles, seat)
    wind_name = WINDS[seat - 1]

    extra = bonus_items + list(situation)

    best = None
    if not melds and is_thirteen_wonders(list(concealed)):
        items = extra + [("Thirteen Wonders", MAX_TAI)]
        best = {"tai": min(MAX_TAI, sum(tai for _, tai in items)), "items": items}
    for pair, sets in decompositions(concealed, sets_needed):
        items = extra + _pattern_tai(
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


def instant_payment(receiver, units_each):
    """Every opponent pays `units_each` units to `receiver`."""
    return {p: 3 * units_each if p == receiver else -units_each for p in range(4)}


def kong_payment(concealed):
    return KONG_PAYMENT["concealed" if concealed else "exposed"]


def bonus_payments(revealed, new_tile):
    """(description, units_each) earned by revealing `new_tile`, given the
    bonus tiles the player had already revealed."""
    have = set(revealed) | {new_tile}
    items = []
    for pair in ANIMAL_PAIRS:
        if new_tile in pair and all(t in have for t in pair):
            items.append((f"{format_bonus(pair[0])} + {format_bonus(pair[1])}", ANIMAL_PAIR_PAYMENT))
    for suit, label in (('flower', 'flowers'), ('season', 'seasons')):
        if new_tile[0] == suit and sum(t[0] == suit for t in have) == 4:
            items.append((f"Complete set of {label}", BONUS_SET_PAYMENT))
    return items
