from D_hand_eval import is_winning_hand
from Decision_Strategy_engine import StrategyEngine


def test_valid_winning_hand_is_recognized():
    winning = [
        ('bamboo', 1), ('bamboo', 1), ('bamboo', 1),
        ('bamboo', 2), ('bamboo', 2), ('bamboo', 2),
        ('bamboo', 3), ('bamboo', 3), ('bamboo', 3),
        ('dot', 4), ('dot', 4), ('dot', 4),
        ('wan', 5), ('wan', 5),
    ]
    assert is_winning_hand(winning) is True


def test_invalid_hand_is_not_winning():
    invalid = [
        ('dragon', 'green'), ('dot', 5), ('wan', 9), ('dot', 2), ('wind', 'east'),
        ('bamboo', 4), ('dragon', 'white'), ('dragon', 'green'), ('wan', 5),
        ('bamboo', 5), ('wan', 8), ('dot', 7), ('dragon', 'green'), ('wan', 6),
    ]
    assert is_winning_hand(invalid) is False


def test_strategy_engine_declares_hu_for_valid_hand():
    engine = StrategyEngine()
    winning = [
        ('bamboo', 1), ('bamboo', 1), ('bamboo', 1),
        ('bamboo', 2), ('bamboo', 2), ('bamboo', 2),
        ('bamboo', 3), ('bamboo', 3), ('bamboo', 3),
        ('dot', 4), ('dot', 4), ('dot', 4),
        ('wan', 5), ('wan', 5),
    ]
    assert engine.should_declare_hu(winning) is True


# --------------------------------------------------------------------------
# Hand evaluation
# --------------------------------------------------------------------------
import builtins
import contextlib
import io
import random
from collections import Counter

import pytest

import mahjong_simulator as sim
import singapore_rules as sg
from D_hand_eval import ALL_TILE_TYPES, evaluate_discard_quality, winning_tiles


def B(*ranks):
    return [('bamboo', r) for r in ranks]


def D(*ranks):
    return [('dot', r) for r in ranks]


def W(*ranks):
    return [('wan', r) for r in ranks]


def _reference_is_winning(hand, sets_needed):
    """Slow, independent brute force used to cross-check the optimised version."""
    def can_split(counts, sets_left):
        if sets_left == 0:
            return not +counts
        tile = min(+counts, key=ALL_TILE_TYPES.index)
        if counts[tile] >= 3:
            counts[tile] -= 3
            ok = can_split(counts, sets_left - 1)
            counts[tile] += 3
            if ok:
                return True
        suit, rank = tile
        if suit in ('bamboo', 'dot', 'wan'):
            seq = [(suit, rank + 1), (suit, rank + 2)]
            if all(counts[t] > 0 for t in seq):
                for t in [tile] + seq:
                    counts[t] -= 1
                ok = can_split(counts, sets_left - 1)
                for t in [tile] + seq:
                    counts[t] += 1
                if ok:
                    return True
        return False

    counts = Counter(hand)
    if len(hand) != sets_needed * 3 + 2 or any(c > 4 for c in counts.values()):
        return False
    for pair in list(counts):
        if counts[pair] >= 2:
            counts[pair] -= 2
            ok = can_split(counts, sets_needed)
            counts[pair] += 2
            if ok:
                return True
    return False


def test_sequence_hand_wins():
    assert is_winning_hand(B(1, 2, 3) + B(4, 5, 6) + D(7, 8, 9) + W(2, 3, 4) + W(9, 9))


def test_hand_that_needs_backtracking_wins():
    # 1-1-1-2-3 must be split as pair 1-1 + run 1-2-3, not triplet first.
    assert is_winning_hand(B(1, 1, 1, 2, 3) + D(4, 5, 6) + D(7, 8, 9) + W(5, 5, 5))


def test_winning_hand_with_exposed_melds():
    melds = [[('wind', 'east')] * 3, D(1, 2, 3)]
    assert is_winning_hand(B(2, 3, 4) + W(6, 7, 8) + [('dragon', 'red')] * 2, melds)


def test_winning_hand_with_exposed_kong():
    melds = [[('dragon', 'white')] * 4]
    assert is_winning_hand(B(1, 2, 3) + B(4, 5, 6) + D(7, 7, 7) + W(9, 9), melds)


def test_five_copies_is_never_winning():
    assert not is_winning_hand(B(1, 1, 1, 1, 1) + B(2, 3) + D(4, 5, 6) + W(7, 8, 9) + W(9))


def test_honour_tiles_cannot_form_sequences():
    winds = [('wind', 'east'), ('wind', 'south'), ('wind', 'west')]
    assert not is_winning_hand(winds + B(1, 2, 3) + D(4, 5, 6) + W(7, 8, 9) + W(1, 1))


def test_nine_gates_waits_on_every_tile_of_the_suit():
    assert winning_tiles(B(1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9)) == B(*range(1, 10))


def test_winning_tiles_respects_exposed_melds():
    melds = [D(1, 2, 3), D(4, 5, 6), D(7, 8, 9)]
    assert winning_tiles(B(2, 3, 4) + W(5), melds) == W(5)


def test_optimised_win_check_matches_reference_on_random_hands():
    rng = random.Random(1234)
    wall = list(sim.tiles)
    for _ in range(3000):
        melds = rng.randint(0, 2)
        hand = rng.sample(wall, (4 - melds) * 3 + 2)
        assert is_winning_hand(hand, [None] * melds) == _reference_is_winning(hand, 4 - melds)


def test_optimised_win_check_matches_reference_on_near_winning_hands():
    # Random hands almost never win, so also build hands from random melds.
    rng = random.Random(99)
    numbered = [t for t in ALL_TILE_TYPES if t[0] != 'wind' and t[0] != 'dragon']
    for _ in range(2000):
        hand = []
        for _ in range(4):
            if rng.random() < 0.5:
                hand += [rng.choice(ALL_TILE_TYPES)] * 3
            else:
                suit, rank = rng.choice([t for t in numbered if t[1] <= 7])
                hand += [(suit, rank), (suit, rank + 1), (suit, rank + 2)]
        hand += [rng.choice(ALL_TILE_TYPES)] * 2
        if rng.random() < 0.5:
            hand[rng.randrange(14)] = rng.choice(ALL_TILE_TYPES)
        assert is_winning_hand(hand) == _reference_is_winning(hand, 4)


def test_discard_quality_removes_only_one_copy():
    hand = B(1, 2, 3) + D(4, 5, 6) + W(7, 8, 9) + W(2, 2, 2) + D(9, 9)
    result = evaluate_discard_quality(hand, ('dot', 9))
    # Discarding one 9-dot from the pair leaves a hand waiting on the other 9-dot.
    assert result["waits_after"] == D(9)


# --------------------------------------------------------------------------
# Simulator: dealing, randomness and meld accounting
# --------------------------------------------------------------------------
def _quiet_game(dealer=0):
    game = sim.MahjongGame(dealer=dealer, wind_round=0)
    with contextlib.redirect_stdout(io.StringIO()):
        game.deal()
    return game


def _all_tiles(game):
    everything = list(game.wall)
    for i in range(4):
        everything += game.hands[i] + game.discards[i] + game.bonus[i]
        for meld in game.melds[i]:
            everything += meld
    return Counter(everything)


def test_deal_sizes_and_tile_conservation():
    game = _quiet_game()
    assert [len(h) for h in game.hands] == [14, 13, 13, 13]
    assert len(sim.all_tiles) == 148
    revealed = sum(len(b) for b in game.bonus)
    assert len(game.wall) == 148 - 53 - revealed
    assert _all_tiles(game) == Counter(sim.all_tiles)


def test_no_bonus_tiles_left_in_hands_after_deal():
    for _ in range(30):
        game = _quiet_game()
        assert not any(sg.is_bonus(t) for hand in game.hands for t in hand)


def test_deals_are_random():
    first_hands = {tuple(sorted(_quiet_game().hands[0])) for _ in range(20)}
    assert len(first_hands) == 20


def test_ai_discard_is_random_and_from_hand():
    game = _quiet_game()
    picks = {sim.ai_choose_discard(game, 1) for _ in range(300)}
    assert picks <= set(game.hands[1])
    assert len(picks) >= len(set(game.hands[1])) - 1


def test_ai_does_not_copy_previous_discard():
    game = _quiet_game()
    shared = next(t for t in game.hands[1] if t in game.hands[2])
    game.discards[1].append(shared)
    copies = sum(sim.ai_choose_discard(game, 2) == shared for _ in range(1000))
    share = game.hands[2].count(shared) / len(game.hands[2])
    assert copies < 1000 * share * 2


def _setup_call(game, caller_hand, discard):
    game.hands[1] = list(caller_hand)
    game.current_player = 0
    game.discards[0].append(discard)


def test_pong_accounting():
    game = _quiet_game()
    _setup_call(game, B(5, 5) + D(*range(1, 10)) + W(1, 2), ('bamboo', 5))
    with contextlib.redirect_stdout(io.StringIO()):
        assert game.perform_pong(1, 0, ('bamboo', 5))
    assert game.melds[1] == [B(5, 5, 5)]
    assert len(game.hands[1]) == 11 and game.effective_hand_size(1) == 14
    assert game.discards[0] == [] and game.skip_next_draw


def test_pong_keeps_third_copy_in_hand():
    game = _quiet_game()
    _setup_call(game, B(5, 5, 5) + D(*range(1, 10)) + W(1), ('bamboo', 5))
    with contextlib.redirect_stdout(io.StringIO()):
        game.perform_pong(1, 0, ('bamboo', 5))
    assert game.hands[1].count(('bamboo', 5)) == 1 and len(game.hands[1]) == 11


def test_chi_accounting():
    game = _quiet_game()
    _setup_call(game, B(3, 4) + D(*range(1, 10)) + W(1, 2), ('bamboo', 5))
    assert game.can_chi(1, ('bamboo', 5))
    with contextlib.redirect_stdout(io.StringIO()):
        assert game.perform_chi(1, 0, ('bamboo', 5))
    assert sorted(game.melds[1][0]) == B(3, 4, 5)
    assert len(game.hands[1]) == 11 and game.effective_hand_size(1) == 14


def test_kong_accounting_then_replacement_draw():
    game = _quiet_game()
    _setup_call(game, B(5, 5, 5) + D(*range(1, 10)) + W(1), ('bamboo', 5))
    with contextlib.redirect_stdout(io.StringIO()):
        assert game.perform_kong(1, 0, ('bamboo', 5))
        assert not game.skip_next_draw
        game.draw_tile()
    assert game.melds[1] == [B(5, 5, 5, 5)]
    assert len(game.hands[1]) == 11 and game.effective_hand_size(1) == 14


def test_can_hu_with_kong_meld():
    game = _quiet_game()
    game.hands[0] = B(1, 2, 3) + B(4, 5, 6) + D(7, 7, 7) + W(9, 9)
    game.melds[0] = [[('dragon', 'white')] * 4]
    assert game.can_hu(0)


def test_win_chance_is_a_valid_percentage_for_every_tile():
    game = _quiet_game()
    for tile in set(game.hands[0]):
        chance = sim.estimate_discard_win_chance(game, 0, tile)
        assert 0.0 <= chance <= 99.0


def test_win_chance_prefers_keeping_a_ready_hand():
    game = _quiet_game()
    game.hands[0] = B(1, 2, 3) + D(4, 5, 6) + W(7, 8, 9) + W(2, 2, 2) + D(9) + [('wind', 'north')]
    breaks_hand = sim.estimate_discard_win_chance(game, 0, ('bamboo', 2))
    keeps_ready = sim.estimate_discard_win_chance(game, 0, ('wind', 'north'))
    assert keeps_ready > breaks_hand


# --------------------------------------------------------------------------
# Full games: invariants hold on every discard
# --------------------------------------------------------------------------
class CheckedGame(sim.MahjongGame):
    games = []

    def __init__(self):
        super().__init__()
        CheckedGame.games.append(self)

    def discard_tile(self, tile):
        player = self.current_player
        assert self.effective_hand_size(player) == 14, "must hold 14 before discarding"
        ok = super().discard_tile(tile)
        assert ok
        for i in range(4):
            assert self.effective_hand_size(i) == 13
        assert _all_tiles(self) == Counter(sim.all_tiles)
        return ok


@pytest.mark.parametrize("game_number", range(60))
def test_full_game_invariants(monkeypatch, game_number):
    CheckedGame.games.clear()
    monkeypatch.setattr(sim, "MahjongGame", CheckedGame)
    monkeypatch.setattr("time.sleep", lambda *_: None)

    def no_input(*_):
        raise EOFError
    monkeypatch.setattr(builtins, "input", no_input)

    if game_number % 2:
        # Odd games: you say yes to every Hu / Pong / Kong / Chi prompt.
        monkeypatch.setattr(sim, "yes_no_prompt", lambda *_: True)

    with contextlib.redirect_stdout(io.StringIO()) as out:
        sim.main()
    assert "Game Over." in out.getvalue()
    game = CheckedGame.games[0]
    assert _all_tiles(game) == Counter(sim.all_tiles)
    if game.result:
        r = game.result
        assert r["score"]["tai"] >= sg.MIN_TAI
        assert game.check_win(game.hands[r["winner"]], game.melds[r["winner"]])
        assert r["self_drawn"] == (r["shooter"] is None)
        assert r["shooter"] != r["winner"]


# --------------------------------------------------------------------------
# Strategy engine: per-suit shanten matches the original whole-hand search
# --------------------------------------------------------------------------
from Decision_Strategy_engine import NUM_TYPES, _shanten_from_counts, to_counts


def _reference_shanten(counts, exposed_melds):
    counts = list(counts)
    best = [8]

    def dfs(i, sets, partials, has_pair):
        while i < NUM_TYPES and counts[i] == 0:
            i += 1
        if i >= NUM_TYPES:
            usable = max(0, min(partials, 4 - sets))
            best[0] = min(best[0], max(8 - 2 * sets - usable - (1 if has_pair else 0), -1))
            return
        c, suited, pos = counts[i], i < 27, i % 9
        if c >= 3:
            counts[i] -= 3; dfs(i, sets + 1, partials, has_pair); counts[i] += 3
        if suited and pos <= 6 and counts[i + 1] and counts[i + 2]:
            counts[i] -= 1; counts[i + 1] -= 1; counts[i + 2] -= 1
            dfs(i, sets + 1, partials, has_pair)
            counts[i] += 1; counts[i + 1] += 1; counts[i + 2] += 1
        if c >= 2:
            counts[i] -= 2
            if not has_pair:
                dfs(i, sets, partials, True)
            dfs(i, sets, partials + 1, has_pair)
            counts[i] += 2
        if suited and pos <= 7 and counts[i + 1]:
            counts[i] -= 1; counts[i + 1] -= 1; dfs(i, sets, partials + 1, has_pair); counts[i] += 1; counts[i + 1] += 1
        if suited and pos <= 6 and counts[i + 2]:
            counts[i] -= 1; counts[i + 2] -= 1; dfs(i, sets, partials + 1, has_pair); counts[i] += 1; counts[i + 2] += 1
        saved = counts[i]; counts[i] = 0; dfs(i + 1, sets, partials, has_pair); counts[i] = saved

    dfs(0, exposed_melds, 0, False)
    return best[0]


def test_shanten_matches_reference_on_random_hands():
    rng = random.Random(7)
    wall = list(sim.tiles)
    for _ in range(1500):
        melds = rng.randint(0, 3)
        size = (4 - melds) * 3 + rng.choice([1, 2])
        counts = to_counts(rng.sample(wall, size))
        assert _shanten_from_counts(counts, melds) == _reference_shanten(counts, melds)


def test_shanten_matches_reference_on_single_suit_hands():
    # Dense one-suit hands are where the search branches the most.
    rng = random.Random(8)
    suit_tiles = [t for t in sim.tiles if t[0] == 'bamboo']
    for _ in range(400):
        counts = to_counts(rng.sample(suit_tiles, rng.choice([13, 14])))
        assert _shanten_from_counts(counts, 0) == _reference_shanten(counts, 0)


def test_shanten_known_values():
    assert _shanten_from_counts(to_counts(B(1, 2, 3) + D(4, 5, 6) + W(7, 8, 9) + W(2, 2, 2) + D(9, 9)), 0) == -1
    assert _shanten_from_counts(to_counts(B(1, 2, 3) + D(4, 5, 6) + W(7, 8, 9) + W(2, 2, 2) + D(9)), 0) == 0


# --------------------------------------------------------------------------
# Singapore rules: bonus tiles, tai, shooter and payouts
# --------------------------------------------------------------------------
PLAIN_WIN = B(1, 2, 3) + D(4, 5, 6) + W(7, 8, 9) + W(2, 2, 2) + D(9, 9)


def test_bonus_tile_set():
    assert Counter(t[0] for t in sg.BONUS_TILES) == {'flower': 4, 'season': 4, 'animal': 4}
    assert {t[1] for t in sg.ANIMALS} == {'cat', 'mouse', 'rooster', 'centipede'}


def test_seat_numbers_follow_dealer():
    assert [sg.seat_number(p, dealer=0) for p in range(4)] == [1, 2, 3, 4]
    assert [sg.seat_number(p, dealer=2) for p in range(4)] == [3, 4, 1, 2]


def test_plain_hand_without_bonus_scores_zero_tai():
    assert sg.score_hand(PLAIN_WIN, [], [], seat=1)["tai"] == 0


def test_each_animal_is_one_tai():
    score = sg.score_hand(PLAIN_WIN, [], [('animal', 'cat'), ('animal', 'rooster')], seat=1)
    assert score["tai"] == 2


def test_only_own_seat_flower_scores():
    assert sg.score_hand(PLAIN_WIN, [], [('flower', 2)], seat=2)["tai"] == 1
    assert sg.score_hand(PLAIN_WIN, [], [('flower', 3)], seat=2)["tai"] == 0
    assert sg.score_hand(PLAIN_WIN, [], [('flower', 2), ('season', 2)], seat=2)["tai"] == 2


def test_complete_flower_set_is_two_tai():
    assert sg.score_hand(PLAIN_WIN, [], sg.FLOWERS, seat=1)["tai"] == 2


def test_dragon_and_wind_pongs():
    hand = B(1, 2, 3) + D(4, 5, 6) + [('dragon', 'red')] * 3 + [('wind', 'east')] * 3 + W(9, 9)
    score = sg.score_hand(hand, [], [], seat=1, prevailing_wind="east")
    # red dragon 1 + seat wind 1 + prevailing wind 1
    assert score["tai"] == 3
    assert sg.score_hand(hand, [], [], seat=2, prevailing_wind="south")["tai"] == 1


def test_colour_hands_and_all_pongs():
    full = B(1, 2, 3) + B(4, 5, 6) + B(7, 8, 9) + B(2, 2, 2) + B(5, 5)
    assert ("Full colour (one suit)", 4) in sg.score_hand(full, [], [], seat=1)["items"]
    half = B(1, 2, 3) + B(4, 5, 6) + B(7, 8, 9) + [('wind', 'north')] * 3 + B(5, 5)
    assert ("Half colour (one suit + honours)", 2) in sg.score_hand(half, [], [], seat=1)["items"]
    pongs = B(1, 1, 1) + D(4, 4, 4) + W(7, 7, 7) + W(2, 2, 2) + D(9, 9)
    assert sg.score_hand(pongs, [], [], seat=1)["tai"] == 2


def test_best_reading_of_hand_is_used():
    # 111 222 333 can be three pongs or three chows; pongs + full colour wins.
    hand = B(1, 1, 1, 2, 2, 2, 3, 3, 3) + B(7, 7, 7) + B(9, 9)
    assert sg.score_hand(hand, [], [], seat=1)["tai"] == 5  # capped at MAX_TAI


def test_ping_hu_needs_two_sided_wait_and_no_bonus():
    hand = B(1, 2, 3) + D(4, 5, 6) + W(6, 7, 8) + W(2, 3, 4) + D(9, 9)
    assert sg.score_hand(hand, [], [], seat=1, waits_before_win=W(5, 8))["tai"] == 4
    assert sg.score_hand(hand, [], [], seat=1, waits_before_win=D(9))["tai"] == 0
    assert sg.score_hand(hand, [], [('flower', 3)], seat=1, waits_before_win=W(5, 8))["tai"] == 0


def test_exposed_melds_count_for_tai():
    melds = [[('dragon', 'green')] * 3]
    hand = B(1, 2, 3) + D(4, 5, 6) + W(7, 8, 9) + D(9, 9)
    assert sg.score_hand(hand, melds, [], seat=1)["tai"] == 1


def test_payouts_self_drawn_and_shooter():
    assert sg.payouts(1, winner=0, self_drawn=True) == {0: 6, 1: -2, 2: -2, 3: -2}
    assert sg.payouts(1, winner=0, self_drawn=False, shooter=2) == {0: 4, 1: -1, 2: -2, 3: -1}
    assert sg.payouts(3, winner=1, self_drawn=False, shooter=3, mode="full") == {1: 16, 3: -16}
    for tai in range(1, 7):
        assert sum(sg.payouts(tai, 2, False, 0).values()) == 0


def test_bonus_draw_is_replaced_from_back_of_wall():
    game = _quiet_game()
    game.current_player = 1
    before = len(game.hands[1])
    game.wall[-1] = ('animal', 'cat')
    game.wall[0] = ('bamboo', 5)
    with contextlib.redirect_stdout(io.StringIO()):
        drawn = game.draw_tile()
    assert drawn == ('bamboo', 5)
    assert ('animal', 'cat') in game.bonus[1]
    assert len(game.hands[1]) == before + 1


def _ready_game(caller_hand, caller=1, bonus=()):
    game = _quiet_game()
    game.hands[caller] = list(caller_hand)
    game.melds[caller] = []
    game.bonus[caller] = list(bonus)
    game.current_player = 0
    return game


def test_ai_wins_on_discard_and_discarder_is_shooter():
    ready = B(1, 2, 3) + D(4, 5, 6) + W(7, 8, 9) + W(2, 2, 2) + D(9)
    game = _ready_game(ready, caller=2, bonus=[('animal', 'mouse')])
    assert game.can_hu(2, ('dot', 9))
    assert sim.find_discard_winner(game, ('dot', 9)) == 2


def test_discard_win_needs_minimum_tai():
    ready = B(1, 2, 3) + D(4, 5, 6) + W(7, 8, 9) + W(2, 2, 2) + D(9)
    game = _ready_game(ready, caller=2, bonus=[])
    assert game.score_win(2, ('dot', 9))["tai"] == 0
    assert not game.can_hu(2, ('dot', 9))


def test_first_player_in_turn_order_takes_the_win():
    ready = B(1, 2, 3) + D(4, 5, 6) + W(7, 8, 9) + W(2, 2, 2) + D(9)
    game = _ready_game(ready, caller=3, bonus=[('animal', 'cat')])
    game.hands[2] = list(ready)
    game.melds[2] = []
    game.bonus[2] = [('animal', 'rooster')]
    game.current_player = 1
    assert sim.find_discard_winner(game, ('dot', 9)) == 2


def test_human_is_asked_before_winning_on_discard(monkeypatch):
    ready = B(1, 2, 3) + D(4, 5, 6) + W(7, 8, 9) + W(2, 2, 2) + D(9)
    game = _ready_game(ready, caller=0, bonus=[('animal', 'cat')])
    game.current_player = 3
    asked = []
    monkeypatch.setattr(sim, "yes_no_prompt", lambda q: asked.append(q) or False)
    assert sim.find_discard_winner(game, ('dot', 9)) is None
    assert asked and "Player 4" in asked[0]


def test_shooter_win_in_full_game_records_shooter(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_: None)
    monkeypatch.setattr(builtins, "input", lambda *_: (_ for _ in ()).throw(EOFError))
    monkeypatch.setattr(sim, "yes_no_prompt", lambda *_: True)
    shooter_wins = 0
    for _ in range(80):
        with contextlib.redirect_stdout(io.StringIO()):
            CheckedGame.games.clear()
            monkeypatch.setattr(sim, "MahjongGame", CheckedGame)
            sim.main()
        r = CheckedGame.games[0].result
        if r and not r["self_drawn"]:
            shooter_wins += 1
            assert r["shooter"] in range(4) and r["shooter"] != r["winner"]
    assert shooter_wins > 0


# --------------------------------------------------------------------------
# Randomness of the game setup and AI behaviour
# --------------------------------------------------------------------------
def test_dealer_gets_fourteen_tiles_whoever_it_is():
    for dealer in range(4):
        game = _quiet_game(dealer=dealer)
        assert [len(h) for h in game.hands] == [14 if i == dealer else 13 for i in range(4)]
        assert game.current_player == dealer and game.seat(dealer) == 1


def test_dealer_and_round_wind_are_random():
    games = [sim.MahjongGame() for _ in range(400)]
    dealers = Counter(g.dealer for g in games)
    winds = Counter(g.prevailing_wind for g in games)
    assert set(dealers) == {0, 1, 2, 3} and min(dealers.values()) > 50
    assert set(winds) == set(sg.WINDS) and min(winds.values()) > 50


def test_ai_call_rates_differ_between_players_and_games():
    rates = [tuple(sim.MahjongGame().call_rates) for _ in range(20)]
    assert len(set(rates)) == 20
    assert all(0.2 <= r <= 0.8 for game_rates in rates for r in game_rates)


def test_chi_run_is_chosen_at_random_when_several_fit():
    chosen = set()
    for _ in range(60):
        game = _quiet_game()
        _setup_call(game, B(3, 4, 6, 7) + D(*range(1, 10)), ('bamboo', 5))
        assert len(game.chi_options(1, ('bamboo', 5))) == 3
        with contextlib.redirect_stdout(io.StringIO()):
            game.perform_chi(1, 0, ('bamboo', 5))
        chosen.add(tuple(sorted(game.melds[1][0])))
    assert chosen == {tuple(B(3, 4, 5)), tuple(B(4, 5, 6)), tuple(B(5, 6, 7))}


def test_chi_options_empty_for_honour_tiles():
    game = _quiet_game()
    assert game.chi_options(1, ('dragon', 'red')) == []


def test_sorted_hand_shows_no_percentages(monkeypatch):
    game = _quiet_game()
    monkeypatch.setattr(builtins, "input", lambda *_: "1")
    with contextlib.redirect_stdout(io.StringIO()) as out:
        sim.choose_discard_tile(game)
    hand_lines = out.getvalue().split("Your sorted hand:")[1].split("\n\n")[0]
    assert "%" not in hand_lines


# --------------------------------------------------------------------------
# Engine <-> simulator sync: tai-aware engine, own-turn Kongs, batch games
# --------------------------------------------------------------------------
import Decision_Strategy_engine as de

READY = B(1, 2, 3) + D(4, 5, 6) + W(7, 8, 9) + W(2, 2, 2) + D(9)


def test_engine_discard_follows_hand_shape():
    # Throwing the lone North leaves a two-sided wait; the old draw-probability
    # weight (x25000) swamped shanten and could break the hand instead.
    engine = StrategyEngine()
    ctx = de.TaiContext(bonus=[('animal', 'cat')], seat=2, prevailing='east')
    hand = B(1, 2, 3) + D(4, 5, 6) + W(7, 8, 9) + W(2, 2) + D(5, 6) + [('wind', 'north')]
    decision = engine.choose_discard(hand, [], [], 0.0, ctx)
    assert decision.tile == ('wind', 'north')


def test_engine_hu_respects_minimum_tai():
    game = _ready_game(READY, caller=0, bonus=[])
    game.player1 = "engine"
    assert not de.engine_should_call(game, 0, ('dot', 9), "hu")
    game.bonus[0] = [('animal', 'cat')]
    assert de.engine_should_call(game, 0, ('dot', 9), "hu")


def test_ready_hand_without_tai_is_not_treated_as_ready():
    ctx = de.TaiContext(bonus=[], seat=2, prevailing='east')
    assert de.scoring_waits(READY, [], ctx) == []
    ctx.bonus = [('animal', 'rooster')]
    assert de.scoring_waits(READY, [], ctx) == [('dot', 9)]


def test_engine_prefers_pong_of_scoring_dragon():
    engine = StrategyEngine()
    ctx = de.TaiContext(bonus=[], seat=2, prevailing='east')
    hand = B(1, 2, 3) + D(4, 5, 6) + W(2, 3) + [('dragon', 'red')] * 2 + D(8, 9) + [('wind', 'north')]
    assert engine.should_pong(hand, [], ('dragon', 'red'), [], 0.0, ctx).action == "pong"


def test_concealed_kong_on_own_turn():
    game = _quiet_game()
    game.hands[1] = B(5, 5, 5, 5) + D(1, 2, 3) + W(4, 5, 6) + D(7, 8, 9) + W(1)
    game.melds[1] = []
    assert game.self_kong_options(1) == [('bamboo', 5)]
    with contextlib.redirect_stdout(io.StringIO()):
        assert game.perform_self_kong(1, ('bamboo', 5))
    assert game.melds[1] == [B(5, 5, 5, 5)]
    assert game.effective_hand_size(1) == 13


def test_promoted_kong_adds_to_exposed_pong():
    game = _quiet_game()
    game.hands[1] = B(5) + D(1, 2, 3) + W(4, 5, 6) + D(7, 8, 9) + W(1, 1)
    game.melds[1] = [B(5, 5, 5)]
    assert game.self_kong_options(1) == [('bamboo', 5)]
    with contextlib.redirect_stdout(io.StringIO()):
        assert game.perform_self_kong(1, ('bamboo', 5))
    assert game.melds[1] == [B(5, 5, 5, 5)] and ('bamboo', 5) not in game.hands[1]


def test_promoted_kong_can_be_robbed(monkeypatch):
    game = _quiet_game()
    game.verbose = False
    game.hands[1] = D(9) + D(1, 2, 3) + W(4, 5, 6) + B(7, 8, 9) + W(1, 1)
    game.melds[1] = [D(9, 9, 9)[:3]]
    game.hands[2] = list(READY)          # waiting on 9 dots
    game.melds[2] = []
    game.bonus[2] = [('animal', 'cat')]
    monkeypatch.setattr(sim, "ai_should_call", lambda *_: True)
    game.current_player = 1
    assert sim.self_turn_kongs(game, 1) == "win"
    assert game.result["winner"] == 2 and game.result["shooter"] == 1
    assert ('dot', 9) in game.hands[2]


def test_engine_player_one_simulation_runs_cleanly():
    # check=True asserts tile conservation and hand sizes after every discard
    # and validates every win (complete hand, minimum tai, shooter recorded).
    stats = sim.simulate(40, seed=11, check=True, progress=False)
    assert stats["errors"] == []
    finished = stats["p1_wins"] + sum(stats["other_wins"].values()) + stats["draws"]
    assert finished == 40
    assert stats["p1_wins"] > 20


# --------------------------------------------------------------------------
# Regression tests from the full-project review
# --------------------------------------------------------------------------
def test_useful_tiles_discounts_pair_whose_copies_are_all_visible():
    from D_hand_eval import useful_tiles
    red = ('dragon', 'red')
    hand = [red, red, ('bamboo', 1), ('bamboo', 5), ('bamboo', 9)]
    live = useful_tiles(hand)[0]
    dead = next(t for t in useful_tiles(hand, visible_tiles=[red, red]) if t["tile"] == red)
    assert live["tile"] == red
    assert dead["usefulness"] < live["usefulness"]


def test_engine_can_win_enforces_minimum_tai():
    import Decision_Strategy_engine as de
    game = sim.MahjongGame(dealer=0, wind_round=0)
    game.verbose = False
    # Complete shape but no tai: mixed suits, a plain Kong, no bonus tiles.
    game.melds[0] = [[('bamboo', 9)] * 4]
    game.hands[0] = [('bamboo', n) for n in range(1, 7)] + [('dot', n) for n in (1, 2, 3)] + [('dragon', 'red')]
    assert not de.engine_can_win(game, 0, ('dragon', 'red'))
    game.melds[0] = [[('dragon', 'red')] * 4]
    game.hands[0][-1] = ('wind', 'north')
    assert de.engine_can_win(game, 0, ('wind', 'north'))


def test_engine_perform_call_kong_draws_replacement_from_back():
    import Decision_Strategy_engine as de
    game = sim.MahjongGame(dealer=0, wind_round=0)
    game.verbose = False
    tile = ('dot', 5)
    game.hands[1] = [tile] * 3 + [('bamboo', n) for n in range(1, 10)] + [('wan', 1)]
    game.discards[0] = [tile]
    assert de.engine_perform_call(game, 1, 0, tile, "kong", decide=False)
    assert game.kong_replacement and not game.skip_next_draw
    assert game.melds[1] == [[tile] * 4]


def test_sample_wall_keeps_unrevealed_bonus_tiles():
    import random
    import singapore_rules as sg
    from R_opponent_analysis import OpponentAnalyzer
    sim.rng = random.Random(3)
    game = sim.MahjongGame()
    game.verbose = False
    game.deal()
    wall = OpponentAnalyzer().update_from_game(game).sample_wall(random.Random(0))
    assert len(wall) == len(game.wall)
    assert sum(map(sg.is_bonus, wall)) == sum(map(sg.is_bonus, game.wall))


# --------------------------------------------------------------------------
# Singapore rules: limit hands, situational tai and instant payments
# --------------------------------------------------------------------------
def _honours(suit, *names, n=1):
    return [(suit, name) for name in names for _ in range(n)]


THIRTEEN = (B(1, 9) + D(1, 9) + W(1, 9) + _honours('wind', 'east', 'south', 'west', 'north')
            + _honours('dragon', 'red', 'green', 'white'))


def test_thirteen_wonders_is_a_limit_hand():
    hand = THIRTEEN + [('dragon', 'red')]
    assert is_winning_hand(hand)
    assert not is_winning_hand(hand[3:], [B(2, 3, 4)])  # concealed only
    assert sg.score_hand(hand, [], [], seat=1) == {"tai": sg.MAX_TAI, "items": [("Thirteen Wonders", sg.MAX_TAI)]}
    # waiting on all thirteen after holding one of each
    assert set(winning_tiles(THIRTEEN)) == set(THIRTEEN)


def test_dragon_and_wind_limit_hands():
    big_dragons = _honours('dragon', 'red', 'green', 'white', n=3) + B(1, 2, 3) + D(5, 5)
    assert ("Big Three Dragons", sg.MAX_TAI) in sg.score_hand(big_dragons, [], [], seat=1)["items"]
    small_dragons = _honours('dragon', 'red', 'green', n=3) + _honours('dragon', 'white', n=2) + B(1, 2, 3) + D(4, 5, 6)
    score = sg.score_hand(small_dragons, [], [], seat=2, prevailing_wind="east")
    assert ("Small Three Dragons", sg.SMALL_THREE_DRAGONS_TAI) in score["items"]
    assert score["tai"] == 4   # two dragon pongs + small three dragons
    big_winds = _honours('wind', 'east', 'south', 'west', 'north', n=3) + D(5, 5)
    assert ("Big Four Winds", sg.MAX_TAI) in sg.score_hand(big_winds, [], [], seat=1)["items"]
    small_winds = _honours('wind', 'east', 'south', 'west', n=3) + _honours('wind', 'north', n=2) + B(1, 2, 3)
    assert sg.score_hand(small_winds, [], [], seat=1)["tai"] == sg.MAX_TAI


def test_situation_tai_items():
    names = [d for d, _ in sg.situation_tai(kong_replacement=True, robbed_kong=True, last_tile=True)]
    assert len(names) == 3
    score = sg.score_hand(PLAIN_WIN, [], [], seat=1, situation=sg.situation_tai(last_tile=True))
    assert score["tai"] == sg.SITUATION_TAI
    assert sg.score_hand(PLAIN_WIN, [], [], seat=1, situation=sg.situation_tai(heavenly=True))["tai"] == sg.MAX_TAI


def test_win_on_kong_replacement_and_last_tile():
    game = _quiet_game()
    game.verbose = False
    game.turns = 5
    game.hands[1] = list(PLAIN_WIN)
    game.melds[1] = []
    game.bonus[1] = []
    game.current_player = 1
    assert not game.can_hu(1)          # 0 tai
    game.kong_draw = True
    assert game.score_win(1)["tai"] == sg.SITUATION_TAI
    game.kong_draw = False
    game.wall = []
    assert game.can_hu(1)              # last tile of the wall


def test_robbing_the_kong_scores_and_skips_kong_payment():
    ready = B(1, 2, 3) + D(4, 5, 6) + W(7, 8, 9) + W(2, 2, 2) + D(9)
    game = _ready_game(ready, caller=2, bonus=[])
    game.verbose = False
    game.turns = 9
    game.melds[0] = [D(9, 9, 9)]
    game.hands[0] = game.hands[0][:10] + [('dot', 9)]
    game.player1 = "engine"
    rob = {"self_kong": True}
    original = sim.decide_call
    sim.decide_call = lambda g, p, t, kind, prompt: rob.get(kind, False)
    try:
        assert sim.self_turn_kongs(game, 0) == "win"
    finally:
        sim.decide_call = original
    assert game.result["winner"] == 2 and game.result["shooter"] == 0
    assert any("Robbing" in d for d, _ in game.result["score"]["items"])
    assert not game.instant           # a robbed Kong is never paid


def test_heavenly_and_earthly_hands():
    game = _quiet_game(dealer=0)
    game.verbose = False
    game.turns = 1
    for p in range(4):
        game.melds[p] = []
    game.hands[0] = list(PLAIN_WIN)
    game.bonus[0] = []
    assert game.score_win(0)["tai"] == sg.MAX_TAI          # heavenly
    ready = B(1, 2, 3) + D(4, 5, 6) + W(7, 8, 9) + W(2, 2, 2) + D(9)
    game.hands[2] = list(ready)
    game.bonus[2] = []
    game.current_player = 0
    game.discards[0] = [('dot', 9)]
    assert game.score_win(2, ('dot', 9))["tai"] == sg.MAX_TAI  # earthly
    game.turns = 2
    assert game.score_win(2, ('dot', 9))["tai"] == 0


def test_kong_instant_payments():
    game = _quiet_game()
    game.verbose = False
    game.hands[1] = B(5, 5, 5) + game.hands[1][3:]
    game.discards[0] = [('bamboo', 5)]
    assert game.perform_kong(1, 0, ('bamboo', 5))
    assert game.instant == {1: 3, 0: -1, 2: -1, 3: -1}
    game.hands[2] = D(7, 7, 7, 7) + W(1, 2, 3, 4, 5, 6, 7, 8, 9) + D(1)
    assert game.perform_self_kong(2, ('dot', 7))
    assert game.instant == {1: 1, 0: -3, 2: 5, 3: -3}
    assert sum(game.instant.values()) == 0


def test_engine_kong_call_also_pays():
    import Decision_Strategy_engine as de
    game = _quiet_game()
    game.verbose = False
    game.hands[1] = B(5, 5, 5) + game.hands[1][3:]
    game.discards[0] = [('bamboo', 5)]
    assert de.engine_perform_call(game, 1, 0, ('bamboo', 5), "kong", decide=False)
    assert game.instant[1] == 3 * sg.KONG_PAYMENT["exposed"]


def test_animal_pair_and_bonus_set_instant_payments():
    assert sg.bonus_payments([('animal', 'cat')], ('animal', 'mouse')) == [
        ("Cat Animal + Mouse Animal", sg.ANIMAL_PAIR_PAYMENT)]
    assert sg.bonus_payments([('animal', 'cat')], ('animal', 'rooster')) == []
    assert sg.bonus_payments(sg.SEASONS[:3], sg.SEASONS[3])[0][1] == sg.BONUS_SET_PAYMENT
    game = _quiet_game()
    game.verbose = False
    game.instant.clear()
    game.bonus[3] = [('animal', 'rooster')]
    game.hands[3].append(('animal', 'centipede'))
    game.reveal_bonus(3, ('animal', 'centipede'))
    assert game.instant == {3: 3, 0: -1, 1: -1, 2: -1}


def test_simulation_instant_payments_are_zero_sum():
    sim.rng = random.Random(11)
    totals = Counter()
    for _ in range(40):
        game = sim.play_game(player1="engine", delay=0, verbose=False, check=True)
        assert sum(game.instant.values()) == 0
        totals.update(game.instant)
    assert totals   # kongs / animal pairs do happen in 40 games


def test_resumed_game_dealer_still_draws():
    # A game resumed mid-way (Player 1 to move, Player 4 dealer) has no
    # "dealer discards without drawing" turn, so every hand stays legal.
    sim.rng = random.Random(4)
    game = _quiet_game(dealer=3)
    game.verbose = False
    game.player1 = "engine"
    game.wall.append(game.hands[3].pop())   # dealer back to 13 tiles
    game.current_player = 0
    game.turns = 10
    sim.play_game(game=game, delay=0, check=True)
