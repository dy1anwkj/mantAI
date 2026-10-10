"""
Turn a game state into numbers a neural network can read.

State vector (STATE_SIZE = 387):
    34  concealed hand, count of each tile / 4
    34  own exposed melds, count / 4
    34  other visible tiles (all discards + other players' melds), count / 4
    34  tiles still unseen from this player's view, count / 4
        (4 minus hand, own melds and other visible tiles)
     4  seat wind (one-hot)
     4  prevailing wind (one-hot)
     1  bonus tai already secured (/ 5)
     1  wall left (/ 148, 0.5 when unknown)
     1  number of exposed melds (/ 4)
   102  hand analysis for each possible discard (zeros when not discarding):
        34  shanten after throwing that tile (/ 8), 1 if not in hand
        34  useful tiles left to draw after throwing it (/ 60)
        34  1 if throwing it leaves a ready hand that can reach the minimum tai
   138  the study library's reading of each discard (knowledge.py): how
        expendable it is, what it costs the tai route, how dangerous it
        is, whether it is a fresh tile late in the hand, plus the table's
        threat and how strongly the books say to fold

Call vector (CALL_SIZE = 292) adds the offered tile (34, one-hot), the call type (4, one-hot)
and 5 analysis numbers: shanten / useful tiles before and after the call, and
whether the call is possible.
"""

from collections import Counter

import numpy as np

from .knowledge import KNOWLEDGE_SIZE, knowledge_planes
from .tiles import PLAY_TILES, TILE_INDEX, sg
import Decision_Strategy_engine as de

WINDS = ("east", "south", "west", "north")
CALL_TYPES = ("pong", "chi", "kong", "self_kong")
N = len(PLAY_TILES)
ANALYSIS_AT = 4 * N + 4 + 4 + 3            # where the discard analysis starts
BOOK_AT = ANALYSIS_AT + 3 * N              # where the knowledge planes start
STATE_SIZE = BOOK_AT + KNOWLEDGE_SIZE
CALL_SIZE = STATE_SIZE + N + len(CALL_TYPES) + 5


def discard_analysis(hand, melds, visible, ctx):
    """Per-tile shanten / ukeire / ready-with-tai after discarding it."""
    sh = np.ones(N, dtype=np.float32)
    uk = np.zeros(N, dtype=np.float32)
    ready = np.zeros(N, dtype=np.float32)
    if len(hand) + 3 * len(melds) != 14:
        return np.concatenate([sh * 0, uk, ready])
    for tile in set(hand):
        i = TILE_INDEX.get(tile)
        if i is None:
            continue
        rest = list(hand)
        rest.remove(tile)
        after_visible = list(visible) + [tile]
        s = de.shanten(rest, melds)
        if s == 0:
            good = de.scoring_waits(rest, melds, ctx)
            unseen = de.unseen_counts(rest, after_visible)
            u = sum(unseen[w] for w in good)
            ready[i] = 1.0 if good else 0.0
            if not good:
                s, u = 1, 0
        else:
            u, _ = de.ukeire(rest, melds, after_visible, s)
        sh[i] = min(max(s, 0), 8) / 8.0
        uk[i] = min(u, 60) / 60.0
    return np.concatenate([sh, uk, ready])


def rules_prior(x):
    """For each tile, how close the hand is to a legal win after throwing it,
    read from the discard analysis in a state vector (numpy or torch, one
    row or a batch): minus its shanten, plus one per 30 useful tiles left,
    plus one for a ready hand that reaches the minimum tai. This is the
    "complete shape first, then at least one tai" lesson of docs/03, 04, 07
    and 08. Tiles not in the hand score -8 (they are masked out anyway)."""
    a = ANALYSIS_AT
    return -8 * x[..., a:a + N] + 2 * x[..., a + N:a + 2 * N] + x[..., a + 2 * N:a + 3 * N]


def book_prior(x):
    """The whole study library's preference for each discard: the rules prior
    (shape and tai, docs/03, 04, 07, 08) plus the book planes - throw what is
    expendable (docs/00, 12, 17), keep the tai route (docs/00, 01, 07, 08),
    and avoid dangerous and late fresh tiles (docs/02, 12), much harder once
    the books say to fold (docs/00, 12). This is what AlphaGo's search took
    from its expert-trained network: a starting opinion that search and
    self-play then correct."""
    b = BOOK_AT
    pri, tai = x[..., b:b + N], x[..., b + N:b + 2 * N]
    danger, fresh = x[..., b + 2 * N:b + 3 * N], x[..., b + 3 * N:b + 4 * N]
    fold = x[..., b + 4 * N + 1:b + 4 * N + 2]
    ready = x[..., ANALYSIS_AT + 2 * N:ANALYSIS_AT + 3 * N]
    return (rules_prior(x) + 2.0 * pri - 1.5 * tai
            - (0.5 + 4.0 * fold) * danger - 1.0 * fresh * (1 - ready))


def _counts(tiles):
    v = np.zeros(N, dtype=np.float32)
    for t, c in Counter(tiles).items():
        i = TILE_INDEX.get(t)
        if i is not None:
            v[i] = c
    return v


def encode_state(hand, melds, visible, bonus, seat, prevailing, wall_left, opponents=None):
    own_meld_tiles = [t for m in melds for t in m]
    h = _counts(hand)
    m = _counts(own_meld_tiles)
    vis = _counts(visible)
    unseen = np.clip(4 - h - m - vis, 0, 4)
    seat_v = np.zeros(4, dtype=np.float32)
    seat_v[(seat - 1) % 4] = 1
    prev_v = np.zeros(4, dtype=np.float32)
    prev_v[WINDS.index(prevailing)] = 1
    bonus_tai = sum(t for _, t in sg.bonus_tai(list(bonus), seat))
    extra = np.array([
        min(bonus_tai, 5) / 5.0,
        0.5 if wall_left is None else wall_left / 148.0,
        len(melds) / 4.0,
    ], dtype=np.float32)
    ctx = de.TaiContext(bonus=list(bonus), seat=seat, prevailing=prevailing, min_tai=sg.MIN_TAI)
    analysis = discard_analysis(list(hand), [list(x) for x in melds], list(visible) + own_meld_tiles, ctx)
    book = knowledge_planes(list(hand), [list(x) for x in melds], list(visible), list(bonus),
                            seat, prevailing, wall_left, opponents, analysis[:N] * 8)
    return np.concatenate([h / 4, m / 4, vis / 4, unseen / 4, seat_v, prev_v, extra, analysis, book])


def encode_record(r):
    return encode_state(r["hand"], r["melds"], r["visible"], r["bonus"],
                        r["seat"], r["prevailing"], r.get("wall_left"), r.get("opponents"))


_engine = de.StrategyEngine()


def call_analysis(hand, melds, visible, bonus, seat, prevailing, tile, call_type):
    """[shanten before, useful before, shanten after, useful after, possible]."""
    ctx = de.TaiContext(bonus=list(bonus), seat=seat, prevailing=prevailing, min_tai=sg.MIN_TAI)
    hand, melds, visible = list(hand), [list(m) for m in melds], list(visible)
    try:
        if call_type == "self_kong":
            before = _engine.evaluate_discards(hand, melds, visible, 0.0, ctx)[0]
            s0, u0 = before["shanten"], before["ukeire"]
            if hand.count(tile) == 4:
                after_hand, after_melds = [t for t in hand if t != tile], melds + [[tile] * 4]
            elif tile in hand:
                after_hand = list(hand)
                after_hand.remove(tile)
                after_melds = melds
            else:
                return np.array([s0 / 8, min(u0, 60) / 60, 1, 0, 0], dtype=np.float32)
            after_visible = visible + [tile] * (len(hand) - len(after_hand))
            s1, u1 = _engine._current_value(after_hand, after_melds, after_visible, ctx)
        else:
            s0, u0 = _engine._current_value(hand, melds, visible, ctx)
            if call_type == "chi":
                options = _engine.chi_options(hand, tile)
                takes = [(list(o), sorted([tile] + list(o), key=de.tile_sort_key)) for o in options]
            elif call_type == "pong":
                takes = [([tile, tile], [tile] * 3)] if hand.count(tile) >= 2 else []
            else:
                takes = [([tile] * 3, [tile] * 4)] if hand.count(tile) >= 3 else []
            if not takes:
                return np.array([s0 / 8, min(u0, 60) / 60, 1, 0, 0], dtype=np.float32)
            best = None
            for used, meld in takes:
                after = list(hand)
                for t in used:
                    after.remove(t)
                after_visible = visible + used
                if call_type == "kong":
                    r = _engine._current_value(after, melds + [meld], after_visible, ctx)
                else:
                    top = _engine.evaluate_discards(after, melds + [meld], after_visible, 0.0, ctx)[0]
                    r = (top["shanten"], top["ukeire"])
                if best is None or (r[0], -r[1]) < (best[0], -best[1]):
                    best = r
            s1, u1 = best
    except (ValueError, IndexError):
        return np.zeros(5, dtype=np.float32)
    clip = lambda s: min(max(float(s), 0.0), 8.0) / 8
    return np.array([clip(s0), min(u0, 60) / 60, clip(s1), min(u1, 60) / 60, 1], dtype=np.float32)


def call_prior(xc):
    """The study library's opinion of a call, as a logit (numpy or torch, one
    row or a batch), from the call analysis at the end of the call vector:
    call when it brings the hand closer to a legal win, not merely because
    you can (docs/00 section 4, docs/17 lesson 3 and exercise 4); a Pong of a
    scoring honour keeps the tai route (docs/00 section 5, docs/07-08); a Kong
    pays at once and draws a replacement under these rules, but should not
    cost shape (docs/00 "Kong decisions need care", docs/01, docs/02)."""
    a = xc[..., -5:]
    steps = (a[..., 0:1] - a[..., 2:3]) * 8           # shanten gained
    useful = (a[..., 3:4] - a[..., 1:2]) * 6           # useful tiles gained / 10
    t, c = xc[..., STATE_SIZE:STATE_SIZE + N], xc[..., STATE_SIZE + N:STATE_SIZE + N + len(CALL_TYPES)]
    seat = xc[..., 4 * N:4 * N + 4]
    prevailing = xc[..., 4 * N + 4:4 * N + 8]
    winds = t[..., _WIND_AT:_WIND_AT + 4]
    scoring = (t[..., _DRAGON_AT:_DRAGON_AT + 3].sum(-1, keepdims=True)
               + (winds * seat).sum(-1, keepdims=True) + (winds * prevailing).sum(-1, keepdims=True))
    scoring = (scoring > 0) * 1.0
    pong_or_kong = c[..., 0:1] + c[..., 2:3] + c[..., 3:4]
    kong = c[..., 2:3] + c[..., 3:4]
    not_worse = (steps >= 0) * 1.0
    logit = (2.5 * steps + 0.3 * useful - 0.75 + 1.5 * scoring * pong_or_kong * not_worse
             + 1.5 * kong * not_worse - 10.0 * (1 - a[..., 4:5]))
    return logit[..., 0]


def encode_call(state_vec, tile, call_type, analysis=None):
    t = np.zeros(N, dtype=np.float32)
    if tile in TILE_INDEX:
        t[TILE_INDEX[tile]] = 1
    c = np.zeros(len(CALL_TYPES), dtype=np.float32)
    c[CALL_TYPES.index(call_type)] = 1
    if analysis is None:
        analysis = np.zeros(5, dtype=np.float32)
    return np.concatenate([state_vec, t, c, analysis])


def encode_call_record(state_vec, r):
    a = call_analysis(r["hand"], r["melds"], r["visible"] + [x for m in r["melds"] for x in m],
                      r["bonus"], r["seat"], r["prevailing"], r["tile"], r["call_type"])
    return encode_call(state_vec, r["tile"], r["call_type"], a)


_WIND_AT = PLAY_TILES.index(("wind", "east"))
_DRAGON_AT = min(i for i, t in enumerate(PLAY_TILES) if t[0] == "dragon")
assert [t[1] for t in PLAY_TILES[_WIND_AT:_WIND_AT + 4]] == list(WINDS)


def legal_mask(hand):
    """1 for tiles that are in the hand (the only legal discards)."""
    mask = np.zeros(N, dtype=np.float32)
    for t in hand:
        if t in TILE_INDEX:
            mask[TILE_INDEX[t]] = 1
    return mask


def state_from_game(game, player):
    """Everything the player can legitimately see (no peeking at other hands)."""
    visible = []
    for i in range(4):
        visible += game.discards[i]
        if i != player:
            for meld in game.melds[i]:
                visible += meld
    others = [(player + k) % 4 for k in (1, 2, 3)]
    return dict(
        hand=list(game.hands[player]),
        melds=[list(m) for m in game.melds[player]],
        visible=visible,
        bonus=list(game.bonus[player]),
        seat=game.seat(player),
        prevailing=game.prevailing_wind,
        wall_left=len(game.wall),
        opponents=[{"discards": list(game.discards[p]), "melds": [list(m) for m in game.melds[p]]}
                   for p in others],
    )
