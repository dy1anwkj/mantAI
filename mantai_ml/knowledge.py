"""
The study library in docs/ turned into numbers the network can read: the
"book" knowledge that plays the part of AlphaGo's expert games. Each rule
names the PDF it comes from. Riichi-only ideas (riichi, dora, furiten,
permanent safe tiles) are adapted or left out, as docs/00 and docs/17 say.

For every tile the player could discard, four planes (34 values each):

    priority  : how expendable the tile is when it does nothing for the hand.
                Isolated non-scoring honours first, then isolated terminals,
                then other isolated tiles; middle tiles join more sets.
                docs/00 (3. Discard with a reason), docs/17 (five-block
                lesson), docs/12 (five-block theory), docs/03-04 (efficiency)
    tai_route : how much throwing it costs the hand's scoring route: a pair or
                pong of dragons / own or prevailing wind is worth a tai.
                "Complete shape first, then at least one tai."
                docs/00 (5. Get enough tai), docs/01, docs/07-08
    danger    : chance it wins for an opponent who looks close, read from
                their discards and melds: honours < terminals < 2/8 < 3/7 <
                middle tiles, suji and kabe (four copies visible) block
                two-sided waits, a tile an opponent threw is safer but NOT
                safe in Singapore (their hand changes), a visible flush makes
                its suit and the honours dangerous.
                docs/12 (defence: deal-in ranking, suji, kabe), docs/00
                (read evidence; no Japanese safe-tile assumptions), docs/02
                (bao for three sets of one suit / two dragon pongs)
    fresh     : the tile is not in any discard pile late in the hand, when
                SPGG makes its thrower liable for everyone.
                docs/02 rule 12 (last seven / five tiles)

and two numbers for the whole table:

    threat    : how close the most dangerous opponent looks (melds, a flush,
                a short wall). docs/00, docs/12
    fold      : how strongly the books say to defend now: a slow hand (two or
                more from ready), or one from ready with little value, under
                threat. docs/12 (when to fold), docs/00 (push or fold)
"""

from collections import Counter

import numpy as np

from .tiles import PLAY_TILES, TILE_INDEX, sg

N = len(PLAY_TILES)
NUMBER_SUITS = ("bamboo", "dot", "wan")
WINDS = ("east", "south", "west", "north")
PLANES = ("priority", "tai_route", "danger", "fresh")
KNOWLEDGE_SIZE = len(PLANES) * N + 2

# docs/12 deal-in ranking, safest first: honour, 1/9, 2/8, 3/7, 4/5/6.
RANK_DANGER = {1: 0.45, 9: 0.45, 2: 0.6, 8: 0.6, 3: 0.75, 7: 0.75, 4: 0.9, 5: 0.9, 6: 0.9}
HONOUR_DANGER = 0.3
THROWN_BY_THEM = 0.4      # docs/00: earlier discard is safer, not safe
FULL_SUJI, HALF_SUJI = 0.5, 0.75
FRESH_KONG, FRESH_WIN = 7, 5   # docs/02 rule 12: playable tiles left


def scoring_honours(seat, prevailing):
    return {("dragon", "red"), ("dragon", "green"), ("dragon", "white"),
            ("wind", WINDS[(seat - 1) % 4]), ("wind", prevailing)}


def _isolated(tile, counts):
    """No pair and nothing within two ranks in the same suit (docs/00, 12)."""
    if counts[tile] > 1:
        return False
    suit, rank = tile
    if suit not in NUMBER_SUITS:
        return True
    return not any(counts[(suit, r)] for r in (rank - 2, rank - 1, rank + 1, rank + 2))


def priority(tile, counts, scoring):
    """docs/00 section 3: isolated non-scoring honours, then isolated
    terminals, are the first to go; middle tiles can join more sequences."""
    if not _isolated(tile, counts):
        return 0.0
    suit, rank = tile
    if suit not in NUMBER_SUITS:
        return 0.4 if tile in scoring else 1.0
    return {1: 0.7, 9: 0.7, 2: 0.45, 8: 0.45}.get(rank, 0.25)


def tai_route(tile, counts, scoring, bonus_tai):
    """docs/00 section 5, docs/07-08: keep a pair or pong of a scoring honour
    (it is the tai that makes the hand legal). Bonus tiles that already meet
    the minimum let the hand chase speed instead, so the route matters less."""
    if tile not in scoring:
        return 0.0
    value = 1.0 if counts[tile] >= 2 else 0.3
    return value * (0.5 if bonus_tai >= sg.MIN_TAI else 1.0)


def opponent_threat(melds, wall_left):
    """docs/00 (read evidence), docs/12 (when to defend): each exposed meld,
    a hand that shows one suit (flush), and a short wall all add threat.
    Returns (threat 0..1, flush suit or None)."""
    kinds = Counter("honour" if m[0][0] not in NUMBER_SUITS else m[0][0] for m in melds)
    suits = [k for k in kinds if k != "honour"]
    flush = suits[0] if len(suits) == 1 and len(melds) >= 2 else None
    threat = 0.18 * len(melds) + (0.25 if flush else 0.0)
    if wall_left is not None:
        playable = wall_left - sg.DEAD_WALL
        threat += 0.3 if playable <= 15 else 0.15 if playable <= 30 else 0.0
    return min(1.0, threat), flush


def tile_danger(tile, their_discards, seen):
    """docs/12: how likely `tile` completes a hand, from one opponent's
    discards and every copy this player can see (`seen`, a Counter that
    includes the player's own tiles)."""
    suit, rank = tile
    if suit not in NUMBER_SUITS:
        # an honour only wins as a pair or pong: copies in sight rule those out
        d = HONOUR_DANGER * (1.0, 0.8, 0.4, 0.05, 0.0)[min(seen[tile], 4)]
        return d * (THROWN_BY_THEM if tile in their_discards else 1.0)
    d = RANK_DANGER[rank]
    thrown = {r for s, r in their_discards if s == suit}
    if rank in thrown:
        d *= THROWN_BY_THEM
    # Suji: a two-sided wait on `rank` also waits on rank-3 / rank+3.
    low_side = rank - 3 in thrown if rank > 3 else True
    high_side = rank + 3 in thrown if rank < 7 else True
    if low_side and high_side:
        d *= FULL_SUJI
    elif (4 <= rank <= 6) and (low_side or high_side):
        d *= HALF_SUJI
    # Kabe: four copies of a neighbour in sight block the shapes through it.
    shapes, blocked = 0, 0
    if rank >= 3:
        shapes += 1
        blocked += seen[(suit, rank - 1)] >= 4 or seen[(suit, rank - 2)] >= 4
    if rank <= 7:
        shapes += 1
        blocked += seen[(suit, rank + 1)] >= 4 or seen[(suit, rank + 2)] >= 4
    if 2 <= rank <= 8:
        shapes += 1
        blocked += seen[(suit, rank - 1)] >= 4 or seen[(suit, rank + 1)] >= 4
    if shapes:
        d *= 1.0 - 0.6 * blocked / shapes
    if seen[tile] >= 3:
        d *= 0.7          # no pong left; a single wait or a sequence still can
    return min(1.0, d)


def knowledge_planes(hand, melds, visible, bonus, seat, prevailing, wall_left,
                     opponents=None, shanten_after=None):
    """The book planes for a discard decision (zeros when not discarding).

    opponents     : [{"discards": [...], "melds": [[...], ...]}, ...] for the
                    other three seats, or None when unknown (old records)
    shanten_after : the 34 shanten-after-discard values (features.discard_analysis),
                    used for the push / fold reading"""
    out = np.zeros(KNOWLEDGE_SIZE, dtype=np.float32)
    if len(hand) + 3 * len(melds) != 14:
        return out
    counts = Counter(hand)
    scoring = scoring_honours(seat, prevailing)
    bonus_tai = sum(t for _, t in sg.bonus_tai(list(bonus), seat))
    seen = Counter(hand) + Counter(visible) + Counter(t for m in melds for t in m)
    discard_piles = Counter()
    threats = []
    for opp in opponents or ():
        discard_piles.update(opp["discards"])
        threat, flush = opponent_threat(opp["melds"], wall_left)
        threats.append((threat, flush, set(opp["discards"])))
    playable = None if wall_left is None else wall_left - sg.DEAD_WALL
    for tile in set(hand):
        i = TILE_INDEX.get(tile)
        if i is None:
            continue
        out[i] = priority(tile, counts, scoring)
        out[N + i] = tai_route(tile, counts, scoring, bonus_tai)
        safe = 1.0
        for threat, flush, thrown in threats:
            d = tile_danger(tile, thrown, seen)
            if flush and (tile[0] == flush or tile[0] not in NUMBER_SUITS):
                d = max(d, 0.8)   # docs/00 flush evidence, docs/02 bao
            safe *= 1.0 - threat * d
        out[2 * N + i] = 1.0 - safe
        if playable is not None and playable <= FRESH_KONG and opponents and discard_piles[tile] == 0:
            out[3 * N + i] = 1.0 if playable <= FRESH_WIN else 0.5
    top = max((t for t, _, _ in threats), default=0.0)
    out[4 * N] = top
    if shanten_after is not None:
        best = min(float(shanten_after[TILE_INDEX[t]]) for t in set(hand) if t in TILE_INDEX)
        slow = 1.0 if best >= 2 else (0.5 if best == 1 and bonus_tai < 2 else 0.0)
        out[4 * N + 1] = top * slow
    return out
