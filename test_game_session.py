"""GameSession: the game advances one decision at a time, without terminal input."""
import builtins
import copy
import random
import time

import pytest

import mahjong_simulator as sim
import singapore_rules as sg
from mahjong_simulator import Action, GameSession, IllegalActionError


def B(*ranks):
    return [('bamboo', r) for r in ranks]


def D(*ranks):
    return [('dot', r) for r in ranks]


def W(*ranks):
    return [('wan', r) for r in ranks]


EAST, SOUTH = ('wind', 'east'), ('wind', 'south')
RED, GREEN, WHITE = ('dragon', 'red'), ('dragon', 'green'), ('dragon', 'white')

# Player 1 (dealer) opens holding a 5 bamboo to throw and nothing else to do.
OPENER = B(5, 1) + D(1, 5) + W(1, 5, 9) + [EAST, SOUTH, ('wind', 'west'), ('wind', 'north'), RED, GREEN, WHITE]
NOTHING = D(1, 4, 7) + W(2, 5, 8) + [RED, GREEN, WHITE] + D(9) + W(9)


def _snapshot(session):
    g = session.game
    return copy.deepcopy((g.wall, g.hands, g.discards, g.melds, g.bonus, g.current_player,
                          g.turns, dict(g.instant), g.result, g.skip_next_draw,
                          g.kong_replacement, session.decision, session._claim))


def _arranged(hands, bonus=None, melds=None):
    """Player 1 is dealer and about to discard; hands not given hold nothing useful."""
    game = sim.MahjongGame(dealer=0, wind_round=0, rand=random.Random(0))
    game.verbose = False
    game.deal()
    for p in range(4):
        game.hands[p] = list(hands.get(p, OPENER if p == 0 else NOTHING))
        game.melds[p] = [list(m) for m in (melds or {}).get(p, [])]
        game.bonus[p] = list((bonus or {}).get(p, []))
    return GameSession(game=game)


def _discard(session, tile):
    return session.apply(Action("discard", tile), player=0)


# --------------------------------------------------------------------------
# 1. Pause at a decision, resume after an action
# --------------------------------------------------------------------------
def test_game_pauses_at_a_decision_and_resumes_after_an_action():
    session = GameSession(seed=1)
    decision = session.decision
    assert not session.is_over and decision.kind == "turn"
    assert session.current_player == decision.player == session.game.dealer

    before = _snapshot(session)
    session.observation()
    session.legal_actions()
    assert _snapshot(session) == before           # asking never advances the game

    action = next(a for a in session.legal_actions() if a.kind == "discard")
    count = session.game.hands[decision.player].count(action.tile)
    nxt = session.apply(action)
    assert session.game.hands[decision.player].count(action.tile) == count - 1
    assert nxt is session.decision and nxt != decision
    assert session.is_over or session.legal_actions()


def test_same_seed_gives_the_same_game():
    def first_moves(seed):
        session = GameSession(seed=seed)
        moves = []
        for _ in range(20):
            if session.is_over:
                break
            action = session.legal_actions()[-1]
            moves.append((session.current_player, action))
            session.apply(action)
        return moves, session.game.hands
    assert first_moves(42) == first_moves(42)


# --------------------------------------------------------------------------
# 2. Illegal actions leave the game unchanged
# --------------------------------------------------------------------------
def test_illegal_actions_leave_state_unchanged():
    session = GameSession(seed=3)
    player = session.current_player
    hand = session.game.hands[player]
    missing = next(t for t in sim.tiles if t not in hand)
    legal = session.legal_actions()[-1]
    before = _snapshot(session)
    for bad in [Action("discard", missing), Action("pong", hand[0]), Action("chi", hand[0], (hand[1], hand[2])),
                Action("pass"), "discard", None]:
        if bad in session.legal_actions():
            continue
        with pytest.raises(IllegalActionError):
            session.apply(bad)
        assert _snapshot(session) == before
    with pytest.raises(IllegalActionError):
        session.apply(legal, player=(player + 1) % 4)  # not their decision
    assert _snapshot(session) == before


def test_illegal_claim_response_leaves_state_unchanged():
    session = _arranged({1: B(3, 4) + NOTHING[:11], 2: B(5, 5) + NOTHING[:11]})
    _discard(session, ('bamboo', 5))
    assert session.decision.kind == "claim" and session.current_player == 2
    before = _snapshot(session)
    for bad in [Action("discard", ('bamboo', 5)), Action("kong", ('bamboo', 5)),
                Action("chi", ('bamboo', 5), (('bamboo', 3), ('bamboo', 4)))]:  # only Player 2 may Chi
        with pytest.raises(IllegalActionError):
            session.apply(bad)
        assert _snapshot(session) == before


def test_finished_game_rejects_actions():
    session = _arranged({3: D(1, 2, 3) + D(4, 5, 6) + W(7, 8, 9) + W(2, 2, 2) + B(5)},
                        bonus={3: [('animal', 'cat')]})
    _discard(session, ('bamboo', 5))
    session.apply(Action("hu", ('bamboo', 5)))
    assert session.is_over and session.legal_actions() == ()
    with pytest.raises(IllegalActionError):
        session.apply(Action("pass"))


# --------------------------------------------------------------------------
# 3. Observations show only the player's own tiles and public information
# --------------------------------------------------------------------------
def test_observation_hides_opponent_hands_and_wall_order():
    session = GameSession(seed=5)
    game = session.game
    obs = session.observation()
    me = obs.player
    assert obs.hand == tuple(game.hands[me])
    assert obs.wall_remaining == len(game.wall) and obs.decision == "turn"
    assert not any(isinstance(v, list) for v in vars(obs).values())   # no live game lists

    # Rearrange everything the player is not allowed to see.
    others = [p for p in range(4) if p != me]
    hands = [game.hands[p] for p in others]
    for p, hand in zip(others, hands[1:] + hands[:1]):
        game.hands[p] = hand
    random.Random(0).shuffle(game.wall)
    assert session.observation() == obs

    for p in others:
        assert session.observation(p).hand == tuple(game.hands[p])


def test_claim_observation_names_the_tile_on_offer():
    session = _arranged({2: B(5, 5) + NOTHING[:11]})
    _discard(session, ('bamboo', 5))
    obs = session.observation()
    assert (obs.player, obs.decision, obs.pending_tile, obs.pending_from) == (2, "claim", ('bamboo', 5), 0)


# --------------------------------------------------------------------------
# 4. Claim priority and replacement draws
# --------------------------------------------------------------------------
READY_ON_B5 = D(1, 2, 3) + D(4, 5, 6) + W(7, 8, 9) + W(2, 2, 2) + B(5)


def _three_way():
    """Player 2 can Chi, Player 3 can Pong and Player 4 can win on the 5 bamboo."""
    return _arranged({1: B(3, 4) + NOTHING[:11], 2: B(5, 5) + NOTHING[:11], 3: READY_ON_B5},
                     bonus={3: [('animal', 'cat')]})


def test_hu_is_offered_first_and_beats_pong_and_chi():
    session = _three_way()
    _discard(session, ('bamboo', 5))
    assert session.current_player == 3
    assert set(session.legal_actions()) == {Action("hu", ('bamboo', 5)), Action("pass")}
    session.apply(Action("hu", ('bamboo', 5)))
    result = session.result()
    assert session.is_over and (result["winner"], result["shooter"]) == (3, 0)
    assert result["tai"] >= sg.MIN_TAI and sum(result["payouts"].values()) == 0
    assert session.game.discards[0] == []


def test_pong_beats_chi_and_the_chi_player_is_not_asked():
    session = _three_way()
    wall = len(session.game.wall)
    _discard(session, ('bamboo', 5))
    asked = [session.current_player]
    session.apply(Action("pass"))                       # Player 4 declines the win
    asked.append(session.current_player)
    session.apply(Action("pong", ('bamboo', 5)))
    assert asked == [3, 2]
    assert session.game.melds[2] == [B(5, 5, 5)] and session.game.melds[1] == []
    # The Pong player discards next without drawing, and may only discard.
    assert session.current_player == 2 and len(session.game.wall) == wall
    assert {a.kind for a in session.legal_actions()} == {"discard"}


def test_chi_after_everyone_else_passes_with_each_run_a_separate_choice():
    session = _arranged({1: B(3, 4, 6, 7) + NOTHING[:9], 2: B(5, 5) + NOTHING[:11]})
    _discard(session, ('bamboo', 5))
    session.apply(Action("pass"))                       # Player 3 passes on Pong
    assert session.current_player == 1
    chis = [a for a in session.legal_actions() if a.kind == "chi"]
    assert sorted(a.chi for a in chis) == [tuple(B(3, 4)), tuple(B(4, 6)), tuple(B(6, 7))]
    session.apply(Action("chi", ('bamboo', 5), tuple(B(6, 7))))
    assert sorted(session.game.melds[1][0]) == B(5, 6, 7)
    assert B(3, 4) == [t for t in session.game.hands[1] if t[0] == 'bamboo']


def test_nobody_claims_and_the_next_player_draws():
    session = _arranged({2: B(5, 5) + NOTHING[:11]})
    wall = len(session.game.wall)
    _discard(session, ('bamboo', 5))
    session.apply(Action("pass"))
    assert session.current_player == 1 and session.decision.kind == "turn"
    assert len(session.game.wall) < wall and session.game.discards[0] == B(5)


def _put_plain_tile_at_back(game):
    i = next(i for i, t in enumerate(game.wall) if not sg.is_bonus(t))
    game.wall[0], game.wall[i] = game.wall[i], game.wall[0]
    return game.wall[0], game.wall[-1]


def test_kong_on_discard_draws_replacement_from_back_of_wall():
    session = _arranged({2: B(5, 5, 5) + NOTHING[:10]})
    back, front = _put_plain_tile_at_back(session.game)
    _discard(session, ('bamboo', 5))
    assert {a.kind for a in session.legal_actions()} == {"kong", "pong", "pass"}
    session.apply(Action("kong", ('bamboo', 5)))
    game = session.game
    assert game.melds[2] == [B(5, 5, 5, 5)] and game.hands[2][-1] == back
    assert game.wall[-1] == front and game.kong_draw
    assert session.current_player == 2 and game.effective_hand_size(2) == 14
    assert game.instant[2] == 3 * sg.KONG_PAYMENT["exposed"]


def test_concealed_kong_on_own_turn_draws_replacement():
    session = _arranged({0: B(5, 5, 5, 5) + OPENER[2:12]})
    back, _ = _put_plain_tile_at_back(session.game)
    assert Action("kong", ('bamboo', 5)) in session.legal_actions()
    session.apply(Action("kong", ('bamboo', 5)))
    game = session.game
    assert game.melds[0] == [B(5, 5, 5, 5)] and game.hands[0][-1] == back
    assert session.current_player == 0 and session.decision.kind == "turn"
    assert game.effective_hand_size(0) == 14


def test_promoted_kong_can_be_robbed_or_passed():
    def setup():
        return _arranged({0: B(5) + OPENER[2:12], 3: READY_ON_B5},
                         bonus={3: [('animal', 'cat')]}, melds={0: [B(5, 5, 5)]})

    robbed = setup()
    robbed.apply(Action("kong", ('bamboo', 5)))
    assert robbed.decision.kind == "rob_kong" and robbed.current_player == 3
    robbed.apply(Action("hu", ('bamboo', 5)))
    assert robbed.result()["winner"] == 3 and robbed.result()["shooter"] == 0
    assert not robbed.game.instant                     # a robbed Kong is never paid

    passed = setup()
    back, _ = _put_plain_tile_at_back(passed.game)
    passed.apply(Action("kong", ('bamboo', 5)))
    passed.apply(Action("pass"))
    assert passed.game.melds[0] == [B(5, 5, 5, 5)] and passed.game.hands[0][-1] == back
    assert passed.current_player == 0 and passed.game.kong_draw


# --------------------------------------------------------------------------
# 5. Complete automated games through the interface
# --------------------------------------------------------------------------
def _assert_legal_position(session):
    d, game = session.decision, session.game
    holder = {"turn": d.player, "rob_kong": d.source}.get(d.kind)
    game.check_invariants(discarding_player=holder)   # tile count + every hand size


@pytest.mark.parametrize("seed", range(40))
def test_automated_game_runs_to_the_end(seed, monkeypatch, capsys):
    def forbidden(*_):
        raise AssertionError("the session must not prompt or sleep")
    monkeypatch.setattr(builtins, "input", forbidden)
    monkeypatch.setattr(time, "sleep", forbidden)

    session = GameSession(seed=seed, check=True)
    picker = random.Random(seed)
    for _ in range(2000):
        if session.is_over:
            break
        _assert_legal_position(session)
        actions = session.legal_actions()
        wins = [a for a in actions if a.kind == "hu"]
        session.apply(wins[0] if wins else picker.choice(actions))
    assert session.is_over and capsys.readouterr().out == ""

    game, result = session.game, session.result()
    assert sum(result["instant"].values()) == 0
    assert sum(result["total"].values()) == 0
    if result["winner"] is None:
        assert not game.wall and result["payouts"] == {p: 0 for p in range(4)}
    else:
        w = result["winner"]
        assert game.check_win(game.hands[w], game.melds[w])
        assert result["tai"] == result["score"]["tai"] >= sg.MIN_TAI
        assert result["payouts"] == {p: sg.payouts(result["tai"], w, result["self_drawn"],
                                                   result["shooter"]).get(p, 0) for p in range(4)}
