"""
Turn a game state into numbers a neural network can read.

State vector (STATE_SIZE = 249):
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

Call vector (CALL_SIZE = 292) adds the offered tile (34, one-hot), the call type (4, one-hot)
and 5 analysis numbers: shanten / useful tiles before and after the call, and
whether the call is possible.
"""

from collections import Counter

import numpy as np

from .tiles import PLAY_TILES, TILE_INDEX, sg
import Decision_Strategy_engine as de

WINDS = ("east", "south", "west", "north")
CALL_TYPES = ("pong", "chi", "kong", "self_kong")
N = len(PLAY_TILES)
STATE_SIZE = 4 * N + 4 + 4 + 3 + 3 * N
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
        s = de.shanten(rest, melds)
        if s == 0:
            good = de.scoring_waits(rest, melds, ctx)
            unseen = de.unseen_counts(rest, visible)
            u = sum(unseen[w] for w in good)
            ready[i] = 1.0 if good else 0.0
            if not good:
                s, u = 1, 0
        else:
            u, _ = de.ukeire(rest, melds, visible, s)
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
    a = STATE_SIZE - 3 * N
    return -8 * x[..., a:a + N] + 2 * x[..., a + N:a + 2 * N] + x[..., a + 2 * N:a + 3 * N]


def _counts(tiles):
    v = np.zeros(N, dtype=np.float32)
    for t, c in Counter(tiles).items():
        i = TILE_INDEX.get(t)
        if i is not None:
            v[i] = c
    return v


def encode_state(hand, melds, visible, bonus, seat, prevailing, wall_left):
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
    return np.concatenate([h / 4, m / 4, vis / 4, unseen / 4, seat_v, prev_v, extra, analysis])


def encode_record(r):
    return encode_state(r["hand"], r["melds"], r["visible"], r["bonus"],
                        r["seat"], r["prevailing"], r.get("wall_left"))


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
            s1, u1 = _engine._current_value(after_hand, after_melds, visible, ctx)
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
                if call_type == "kong":
                    r = _engine._current_value(after, melds + [meld], visible, ctx)
                else:
                    top = _engine.evaluate_discards(after, melds + [meld], visible, 0.0, ctx)[0]
                    r = (top["shanten"], top["ukeire"])
                if best is None or r < best:
                    best = r
            s1, u1 = best
    except (ValueError, IndexError):
        return np.zeros(5, dtype=np.float32)
    clip = lambda s: min(max(float(s), 0.0), 8.0) / 8
    return np.array([clip(s0), min(u0, 60) / 60, clip(s1), min(u1, 60) / 60, 1], dtype=np.float32)


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
    return dict(
        hand=list(game.hands[player]),
        melds=[list(m) for m in game.melds[player]],
        visible=visible,
        bonus=list(game.bonus[player]),
        seat=game.seat(player),
        prevailing=game.prevailing_wind,
        wall_left=len(game.wall),
    )
