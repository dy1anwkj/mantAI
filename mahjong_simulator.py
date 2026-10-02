import random
import time
from collections import Counter
from dataclasses import dataclass

import singapore_rules as sg
from Decision_Strategy_engine import (
    StrategyEngine,
    collect_visible_tiles,
    engine_choose_discard,
    engine_chi_sequence,
    engine_should_call,
    summarize_player_state,
)
from D_hand_eval import (
    analyze_hand,
    evaluate_discard_quality,
    is_winning_hand,
    useful_tiles,
    winning_tiles,
)
from H_game_state import observe

# Define Mahjong tiles (Bamboo, Dots, Wan, Winds, Dragons)
bamboo = [('bamboo', i) for i in range(1, 10)]
dots = [('dot', i) for i in range(1, 10)]
wan = [('wan', i) for i in range(1, 10)]
winds = [('wind', d) for d in ['east', 'south', 'west', 'north']]
dragons = [('dragon', d) for d in ['red', 'green', 'white']]

tiles = (bamboo + dots + wan + winds + dragons) * 4
# Singapore sets add 4 flowers, 4 seasons and 4 animals: 148 tiles in total.
all_tiles = tiles + sg.BONUS_TILES

# Unseeded RNG drawing from OS entropy: every game gets a fresh random wall.
rng = random.SystemRandom()

# Players 2-4 play with the strategy engine, each with its own style, so they
# build real hands, call Chi / Pong / Kong and win with tai like people do.
# The values override Decision_Strategy_engine.DEFAULT_WEIGHTS.
OPPONENT_STYLES = {
    # calls freely to speed the hand up, worries little about feeding wins
    "aggressive": {"call_widen": 1, "honor_pong_bonus": 25.0, "danger": 60.0},
    "balanced": {"call_widen": 4},
    # keeps the hand concealed and avoids risky discards
    "cautious": {"danger": 300.0, "honor_pong_bonus": 8.0},
}

class MahjongGame:
    def __init__(self, dealer=None, wind_round=None, rand=None):
        # Random source for the wall, seats and random Chi runs; defaults to
        # the module rng (so simulate(seed=...) still controls it).
        self.rand = rand or rng
        self.wall = list(all_tiles)
        self.rand.shuffle(self.wall)
        self.hands = [[] for _ in range(4)]
        self.discards = [[] for _ in range(4)]
        self.melds = [[] for _ in range(4)]
        self.bonus = [[] for _ in range(4)]  # revealed flowers / seasons / animals
        # Dealer and round wind are drawn at random each game unless given.
        self.dealer = self.rand.randrange(4) if dealer is None else dealer
        self.current_player = self.dealer
        self.wind_round = self.rand.randrange(4) if wind_round is None else wind_round
        # Each AI player gets its own random eagerness to call Pong / Kong / Chi
        # (used when opponents == "random").
        self.call_rates = [self.rand.uniform(0.2, 0.8) for _ in range(4)]
        self.strategy = StrategyEngine()
        # Players 2-4: "engine" (each with a random style) or "random".
        self.opponents = "engine"
        self.styles = {p: self.rand.choice(sorted(OPPONENT_STYLES)) for p in (1, 2, 3)}
        self.opponent_engines = {p: StrategyEngine(OPPONENT_STYLES[s]) for p, s in self.styles.items()}
        self.skip_next_draw = False
        self.kong_replacement = False
        self.kong_draw = False      # current player's last tile was a Kong replacement
        self.robbing_kong = False   # a promoted Kong is being offered to the other players
        self.instant = Counter()    # instant payments so far (Kongs, animals, bonus sets)
        self.result = None  # set when someone wins
        # Who controls Player 1 (index 0): "human" prompts at the keyboard,
        # "engine" lets the Decision / Strategy engine play automatically.
        self.player1 = "human"
        self.verbose = True
        self.turns = 0

    def say(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def check_invariants(self, discarding_player=None):
        """Every tile is accounted for and every hand has a legal size."""
        everything = list(self.wall)
        for i in range(4):
            everything += self.hands[i] + self.discards[i] + self.bonus[i]
            for meld in self.melds[i]:
                everything += meld
        assert Counter(everything) == Counter(all_tiles), "tile count mismatch"
        for i in range(4):
            assert not any(sg.is_bonus(t) for t in self.hands[i]), "bonus tile left in hand"
            expected = 14 if i == discarding_player else 13
            assert self.effective_hand_size(i) == expected, (
                f"Player {i + 1} holds {self.effective_hand_size(i)} tiles, expected {expected}")

    @property
    def prevailing_wind(self):
        return sg.WINDS[self.wind_round % 4]

    def seat(self, player_index):
        return sg.seat_number(player_index, self.dealer)

    def deal(self):
        for i, hand in enumerate(self.hands):
            count = 14 if i == self.dealer else 13
            for _ in range(count):
                hand.append(self.wall.pop())
        # Bonus tiles are revealed and replaced, starting with the dealer.
        for i in range(4):
            player = (self.dealer + i) % 4
            if not self.replace_bonus_tiles(player):
                break

    def reveal_bonus(self, player_index, tile):
        earned = sg.bonus_payments(self.bonus[player_index], tile)
        self.hands[player_index].remove(tile)
        self.bonus[player_index].append(tile)
        self.say(f"Player {player_index + 1} reveals {self.format_tile(tile)} and draws a replacement.")
        for reason, units in earned:
            self.pay_instant(player_index, units, reason)

    def pay_instant(self, receiver, units_each, reason):
        """Every opponent pays `receiver` straight away (Singapore instant payment)."""
        self.instant.update(sg.instant_payment(receiver, units_each))
        self.say(f"Instant payment: {reason}. Each player pays Player {receiver + 1} {units_each} unit(s).")

    def pay_kong(self, player_index, concealed):
        kind = "concealed" if concealed else "exposed"
        self.pay_instant(player_index, sg.kong_payment(concealed), f"{kind} Kong")

    def replace_bonus_tiles(self, player_index):
        """Swap every bonus tile in hand for a tile from the back of the wall.
        Returns False if the wall runs out."""
        while True:
            found = [t for t in self.hands[player_index] if sg.is_bonus(t)]
            if not found:
                return True
            for tile in found:
                self.reveal_bonus(player_index, tile)
                if not self.wall:
                    return False
                self.hands[player_index].append(self.wall.pop(0))

    def draw_tile(self, from_back=False):
        """Draw for the current player, replacing any bonus tiles.
        Kong replacements come from the back of the wall."""
        player = self.current_player
        while True:
            if not self.wall:
                self.say("Wall depleted! Round ends in draw.")
                return None
            tile = self.wall.pop(0) if from_back else self.wall.pop()
            self.hands[player].append(tile)
            if not sg.is_bonus(tile):
                break
            self.reveal_bonus(player, tile)
            from_back = True
        if player == 0:
            self.say(f"You draw {self.format_tile(tile)}")
        else:
            self.say(f"Player {player + 1} draws a tile")
        return tile

    def discard_tile(self, tile):
        if tile not in self.hands[self.current_player]:
            self.say(f"Error: Player {self.current_player + 1} tried to discard a tile not in hand!")
            return False
        self.hands[self.current_player].remove(tile)
        self.discards[self.current_player].append(tile)
        self.kong_draw = False
        self.say(f"Player {self.current_player + 1} discards {self.format_tile(tile)}")
        # Do not immediately advance turn here; wait for calls
        return True

    def format_tile(self, tile):
        if sg.is_bonus(tile):
            return sg.format_bonus(tile)
        if tile[0] == 'bamboo':
            return f"{tile[1]} bamboo"
        if tile[0] == 'dot':
            return f"{tile[1]} dots"
        if tile[0] == 'wan':
            return f"{tile[1]} wan"
        return f"{tile[1].capitalize()} {tile[0].capitalize()}"

    def sort_tiles(self, tiles):
        suit_order = {'bamboo': 0, 'dot': 1, 'wan': 2, 'wind': 3, 'dragon': 4}
        rank_order = {'east': 0, 'south': 1, 'west': 2, 'north': 3, 'red': 0, 'green': 1, 'white': 2}

        def tile_key(tile):
            suit, value = tile
            if suit in ['bamboo', 'dot', 'wan']:
                return (suit_order[suit], int(value), suit)
            return (suit_order[suit], rank_order[value], value)

        return sorted(tiles, key=tile_key)

    def show_hands(self):
        for i, hand in enumerate(self.hands):
            formatted = [self.format_tile(t) for t in self.sort_tiles(hand)]
            self.say(f"Player {i + 1} hand: {formatted}")

    def get_concealed_hand_size(self, player_index):
        return len(self.hands[player_index])

    def get_total_active_tiles(self, player_index):
        return len(self.hands[player_index]) + sum(len(meld) for meld in self.melds[player_index])

    def effective_hand_size(self, player_index):
        # Every meld (a Kong included) stands in for three tiles of a 14-tile winning hand.
        return len(self.hands[player_index]) + 3 * len(self.melds[player_index])

    def expose_meld_from_hand(self, player_index, hand_tiles):
        for tile in hand_tiles:
            self.hands[player_index].remove(tile)

    def show_player_state(self, player_index):
        hand = self.sort_tiles(self.hands[player_index])
        melds = []
        for meld in self.melds[player_index]:
            melds.append([self.format_tile(t) for t in self.sort_tiles(meld)])

        self.say(f"Player {player_index + 1} concealed hand ({len(hand)} tiles): {[self.format_tile(t) for t in hand]}")
        if melds:
            self.say(f"Player {player_index + 1} current melds: {melds}")
            self.say(f"Player {player_index + 1} total active tiles: {self.get_total_active_tiles(player_index)}")
        else:
            self.say(f"Player {player_index + 1} current melds: []")
            self.say(f"Player {player_index + 1} total active tiles: {len(hand)}")
        self.say(f"Seat: {self.seat(player_index)} ({sg.WINDS[self.seat(player_index) - 1].capitalize()}), "
              f"bonus tiles: {[self.format_tile(t) for t in self.bonus[player_index]]}")

    def show_discards(self):
        for i, discard in enumerate(self.discards):
            formatted = [self.format_tile(t) for t in discard]
            self.say(f"Player {i + 1} discards: {formatted}")

    def show_melds(self):
        for i, meld_list in enumerate(self.melds):
            if meld_list:
                formatted_melds = [[self.format_tile(t) for t in meld] for meld in meld_list]
                self.say(f"Player {i + 1} melds: {formatted_melds}")
            else:
                self.say(f"Player {i + 1} has no melds.")
            if self.bonus[i]:
                self.say(f"Player {i + 1} bonus tiles: {[self.format_tile(t) for t in self.bonus[i]]}")

    def is_pong(self, tiles):
        return len(tiles) == 3 and len(set(tiles)) == 1

    def is_chi(self, tiles):
        if len(tiles) != 3:
            return False
        suits = [t[0] for t in tiles]
        if suits[0] not in ['bamboo', 'dot', 'wan'] or len(set(suits)) > 1:
            return False
        ranks = sorted(t[1] for t in tiles)
        return ranks[1] == ranks[0] + 1 and ranks[2] == ranks[1] + 1

    def is_pair(self, tiles):
        return len(tiles) == 2 and tiles[0] == tiles[1]

    def check_win(self, hand, melds=None):
        return is_winning_hand(list(hand), melds or [])

    def get_hints(self, player_index):
        hand = self.hands[player_index]
        hints = []
        counts = Counter(hand)
        for tile, count in counts.items():
            if count >= 3:
                hints.append(f"Pong possible with {self.format_tile(tile)}")
        for tile in hand:
            if tile[0] in ['bamboo', 'dot', 'wan']:
                need1 = (tile[0], tile[1] + 1)
                need2 = (tile[0], tile[1] + 2)
                if need1 in hand and need2 in hand:
                    hints.append(f"Chi possible starting at {self.format_tile(tile)}")
        for tile in winning_tiles(hand, self.melds[player_index]):
            hints.append(f"Waiting for {self.format_tile(tile)} to Hu")
        return hints

    def find_completed_sets(self, player_index):
        hand = sorted(self.hands[player_index])
        sets = []
        leftover = hand.copy()
        counts = Counter(leftover)
        for tile, count in counts.items():
            while count >= 3:
                sets.append([tile] * 3)
                for _ in range(3):
                    leftover.remove(tile)
                count -= 3
        i = 0
        while i < len(leftover):
            tile = leftover[i]
            if tile[0] in ['bamboo', 'dot', 'wan']:
                suit, rank = tile
                seq = [(suit, rank), (suit, rank + 1), (suit, rank + 2)]
                if all(s in leftover for s in seq):
                    sets.append(seq)
                    for s in seq:
                        leftover.remove(s)
                    continue
            i += 1
        return sets, leftover

    def can_pong(self, player_index, tile):
        return self.hands[player_index].count(tile) >= 2

    def perform_pong(self, player_index, discarder_index, tile):
        if not self.can_pong(player_index, tile):
            return False
        self.expose_meld_from_hand(player_index, [tile, tile])
        self.discards[discarder_index].pop()
        self.melds[player_index].append([tile] * 3)
        self.say(f"Player {player_index + 1} calls Pong on {self.format_tile(tile)}")
        self.say(f"Player {player_index + 1} now has {len(self.hands[player_index])} concealed tiles left after exposing the meld.")
        self.current_player = player_index
        self.skip_next_draw = True
        return True

    def can_chi(self, player_index, tile):
        if player_index != (self.current_player + 1) % 4:
            return False  # Can only Chi if you are next player after discarder
        suit, rank = tile
        if suit not in ['bamboo', 'dot', 'wan']:
            return False
        hand = self.hands[player_index]
        sequences = [
            [(suit, rank - 2), (suit, rank - 1)],
            [(suit, rank - 1), (suit, rank + 1)],
            [(suit, rank + 1), (suit, rank + 2)],
        ]
        for seq in sequences:
            if all(t in hand for t in seq):
                return True
        return False

    def chi_options(self, player_index, tile):
        suit, rank = tile
        if suit not in ['bamboo', 'dot', 'wan']:
            return []
        hand = self.hands[player_index]
        sequences = [
            [(suit, rank - 2), (suit, rank - 1)],
            [(suit, rank - 1), (suit, rank + 1)],
            [(suit, rank + 1), (suit, rank + 2)],
        ]
        return [seq for seq in sequences if all(t in hand for t in seq)]

    def perform_chi(self, player_index, discarder_index, tile, seq=None):
        options = self.chi_options(player_index, tile)
        if seq is None and options:
            seq = self.rand.choice(options)
        if seq not in options:
            return False
        meld_tiles = [tile] + seq
        self.expose_meld_from_hand(player_index, seq)
        self.discards[discarder_index].pop()
        self.melds[player_index].append(meld_tiles)
        self.say(f"Player {player_index + 1} calls Chi on {self.format_tile(tile)} with sequence {[self.format_tile(t) for t in seq]}")
        self.say(f"Player {player_index + 1} now has {len(self.hands[player_index])} concealed tiles left after exposing the meld.")
        self.current_player = player_index
        self.skip_next_draw = True
        return True

    def can_kong(self, player_index, tile):
        return self.hands[player_index].count(tile) == 3

    def perform_kong(self, player_index, discarder_index, tile):
        if not self.can_kong(player_index, tile):
            return False
        self.expose_meld_from_hand(player_index, [tile, tile, tile])
        self.discards[discarder_index].pop()
        self.melds[player_index].append([tile] * 4)
        self.say(f"Player {player_index + 1} calls Kong on {self.format_tile(tile)}")
        self.say(f"Player {player_index + 1} now has {len(self.hands[player_index])} concealed tiles left after exposing the meld.")
        self.current_player = player_index
        self.skip_next_draw = False
        self.kong_replacement = True
        self.pay_kong(player_index, concealed=False)
        return True

    def self_kong_options(self, player_index):
        """Tiles the player may Kong on their own turn: four concealed copies,
        or one copy that upgrades an exposed Pong."""
        hand = self.hands[player_index]
        counts = Counter(hand)
        options = [t for t, c in counts.items() if c == 4]
        for meld in self.melds[player_index]:
            if len(meld) == 3 and meld[0] == meld[1] == meld[2] and meld[0] in counts:
                options.append(meld[0])
        return self.sort_tiles(options)

    def perform_self_kong(self, player_index, tile):
        """Declare a concealed or promoted Kong. The caller then draws the
        replacement tile from the back of the wall."""
        hand = self.hands[player_index]
        if hand.count(tile) == 4:
            for _ in range(4):
                hand.remove(tile)
            self.melds[player_index].append([tile] * 4)
            self.say(f"Player {player_index + 1} declares a concealed Kong of {self.format_tile(tile)}")
            self.pay_kong(player_index, concealed=True)
        else:
            meld = next((m for m in self.melds[player_index]
                         if len(m) == 3 and m[0] == m[1] == m[2] == tile), None)
            if meld is None or tile not in hand:
                return False
            hand.remove(tile)
            meld.append(tile)
            self.say(f"Player {player_index + 1} adds {self.format_tile(tile)} to their Pong to make a Kong")
            self.pay_kong(player_index, concealed=False)
        return True

    def score_win(self, player_index, winning_tile=None):
        """Tai for the player's hand, optionally completed by another player's
        discard. Returns None if the hand is not a winning hand."""
        concealed = list(self.hands[player_index])
        on_own_turn = winning_tile is None
        if winning_tile is None:
            # Self-drawn: the most recent draw is the winning tile.
            winning_tile = concealed[-1] if concealed else None
            before = list(concealed)
            if winning_tile is not None:
                before.remove(winning_tile)
        else:
            before = list(concealed)
            concealed.append(winning_tile)
        melds = self.melds[player_index]
        if not self.check_win(concealed, melds):
            return None
        return sg.score_hand(
            concealed, melds, self.bonus[player_index], self.seat(player_index),
            self.prevailing_wind, winning_tiles(before, melds),
            self.win_situation(player_index, self_drawn=on_own_turn),
        )

    def win_situation(self, player_index, self_drawn):
        """Situational tai for a win happening right now (see sg.situation_tai)."""
        no_calls = not any(self.melds)
        first_go = self.turns == 1 and no_calls   # still the dealer's opening turn
        return sg.situation_tai(
            kong_replacement=self_drawn and self.kong_draw,
            robbed_kong=not self_drawn and self.robbing_kong,
            last_tile=self_drawn and not self.wall,
            heavenly=self_drawn and first_go and player_index == self.dealer,
            earthly=(not self_drawn and first_go and not self.robbing_kong
                     and self.current_player == self.dealer),
        )

    def can_hu(self, player_index, discard=None):
        score = self.score_win(player_index, discard)
        return score is not None and score["tai"] >= sg.MIN_TAI

def estimate_discard_win_chance(game, player_index, tile, visible_tiles=None):
    hand = game.hands[player_index]
    melds = game.melds[player_index]
    if visible_tiles is None:
        visible_tiles = collect_visible_tiles(game)

    quality = evaluate_discard_quality(hand, tile, exposed_melds=melds, visible_tiles=visible_tiles)
    remaining = list(hand)
    remaining.remove(tile)

    raw_score = (
        quality["after_score"]
        + 140 * len(quality["waits_after"])
        + 25 * len(useful_tiles(remaining, melds, visible_tiles))
        - 10 * len(quality["waits_before"])
    )
    percentage = max(0.0, min(99.0, 52.0 + raw_score / 55.0))
    return round(percentage, 1)


def choose_discard_tile(game, state=None):
    hand = game.sort_tiles(game.hands[game.current_player])
    if state is None:
        state = summarize_player_state(game, game.current_player, game.strategy)
    analysis = state["analysis"]
    strategy_choice = state["best_discard"]

    # Duplicate tiles give the same result, so each distinct tile is rated once.
    chances = {
        tile: estimate_discard_win_chance(game, game.current_player, tile, state["visible_tiles"])
        for tile in dict.fromkeys(hand)
    }
    best_tile, best_percentage = max(chances.items(), key=lambda item: item[1])
    recommended = strategy_choice if strategy_choice in chances else best_tile

    print("\nYour sorted hand:")
    for i, tile in enumerate(hand, start=1):
        print(f"{i}: {game.format_tile(tile)}")

    if game.melds[game.current_player]:
        print("Your current melds:")
        for meld in game.melds[game.current_player]:
            print("  -", [game.format_tile(t) for t in game.sort_tiles(meld)])

    print(f"\nEvaluator summary: score={analysis['strength']['score']:.1f}, waits={len(analysis['waits'])}, useful={[item['tile'] for item in analysis['useful_tiles'][:3]]}")
    print(f"Strategy recommendation: {game.format_tile(recommended)}")
    print(f"Best discard by win estimate: {game.format_tile(best_tile)} ({best_percentage:.1f}% estimated win chance)")

    while True:
        try:
            raw = input(f"Player {game.current_player + 1}, choose tile to discard (1-{len(hand)}): ")
        except EOFError:
            print(f"\n[Auto] No input received. Using recommended discard: {game.format_tile(recommended)}")
            return recommended

        try:
            choice = int(raw)
            if 1 <= choice <= len(hand):
                chosen_tile = hand[choice - 1]
                return chosen_tile
        except ValueError:
            pass

        if raw.strip() == '':
            print(f"\n[Auto] Empty input. Using recommended discard: {game.format_tile(recommended)}")
            return recommended
        print("Invalid choice. Please enter a number corresponding to the tile you want to discard.")


def yes_no_prompt(prompt):
    while True:
        try:
            ans = input(prompt + " (y/n): ").strip().lower()
        except EOFError:
            print("\n[Auto] No input received. Defaulting to no.")
            return False
        if ans in ('y', 'n'):
            return ans == 'y'
        print("Please enter 'y' or 'n'.")

def ai_choose_discard(game, player_index):
    hand = game.hands[player_index]
    if not hand:
        return None
    return rng.choice(hand)


def ai_should_call(game, player_index, tile, call_type):
    return rng.random() < game.call_rates[player_index]


def choose_chi_sequence(game, tile):
    """Ask which run to make when your Chi can be formed more than one way."""
    options = game.chi_options(0, tile)
    if len(options) <= 1:
        return options[0] if options else None
    print(f"Chi {game.format_tile(tile)} with:")
    for i, seq in enumerate(options, start=1):
        print(f"  {i}: {[game.format_tile(t) for t in game.sort_tiles(seq + [tile])]}")
    while True:
        try:
            raw = input(f"Choose sequence (1-{len(options)}): ").strip()
        except EOFError:
            return rng.choice(options)
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        print("Invalid choice.")


# --------------------------------------------------------------------------
# Who controls a seat
# --------------------------------------------------------------------------
def is_human(game, player):
    return player == 0 and getattr(game, "player1", "human") == "human"


def is_engine(game, player):
    return player == 0 and getattr(game, "player1", "human") == "engine"


def is_ml(game, player):
    """Player 1 driven by a trained machine-learning policy (see mantai_ml)."""
    return player == 0 and getattr(game, "player1", "human") == "ml"


def opponent_engine(game, player):
    """The strategy engine playing Players 2-4, or None when they act randomly."""
    if player == 0 or getattr(game, "opponents", "random") != "engine":
        return None
    return game.opponent_engines[player]


def controller_name(game, player):
    if player != 0:
        return "engine" if opponent_engine(game, player) else "random"
    return getattr(game, "player1", "human")


def _record(game, player, kind, tile, **extra):
    """Pass a decision to game.recorder (set by mantai_ml to log gameplay)."""
    recorder = getattr(game, "recorder", None)
    if recorder is not None:
        recorder.on_decision(game, player, kind, tile, **extra)


def decide_call(game, player, tile, call_type, prompt):
    """Yes / no on a call. Human: ask. Engine: Decision / Strategy engine.
    ML: trained policy. Players 2-4: their own engine, or random at their own
    call rate."""
    if is_human(game, player):
        taken = yes_no_prompt(prompt)
    elif is_ml(game, player):
        taken = game.ml_policy.should_call(game, player, tile, call_type)
    elif is_engine(game, player):
        taken = engine_should_call(game, player, tile, call_type, game.strategy)
    elif opponent_engine(game, player):
        taken = engine_should_call(game, player, tile, call_type, opponent_engine(game, player))
    else:
        taken = ai_should_call(game, player, tile, call_type)
    _record(game, player, "call", tile, call_type=call_type, taken=bool(taken))
    return taken


def player_discard(game, player):
    if is_human(game, player):
        tile = choose_discard_tile(game)
    elif is_ml(game, player):
        tile = game.ml_policy.choose_discard(game, player)
        game.say(f"Player 1 (ML) discards {game.format_tile(tile)}")
    elif is_engine(game, player):
        tile = engine_choose_discard(game, player, game.strategy)
        game.say(f"Player 1 (engine) discards {game.format_tile(tile)}")
    else:
        engine = opponent_engine(game, player)
        tile = engine_choose_discard(game, player, engine) if engine else ai_choose_discard(game, player)
        game.say(f"Player {player + 1} (AI) discards {game.format_tile(tile)}")
    _record(game, player, "discard", tile)
    return tile


# --------------------------------------------------------------------------
# Wins
# --------------------------------------------------------------------------
def find_discard_winner(game, tile, discarder=None):
    """First player after the discarder, in turn order, who can and will win
    on `tile`. Winning is a rule, not a strategy choice, so AI players and the
    engine always take a legal win; a human is asked."""
    discarder = game.current_player if discarder is None else discarder
    for i in range(1, 4):
        player = (discarder + i) % 4
        if not game.can_hu(player, tile):
            continue
        if is_human(game, player):
            tai = game.score_win(0, tile)["tai"]
            if yes_no_prompt(f"You can Hu on Player {discarder + 1}'s {game.format_tile(tile)} ({tai} tai)! Declare win?"):
                return player
        else:
            return player
    return None


def announce_win(game, winner, score, self_drawn, shooter=None, winning_tile=None):
    who = "You" if is_human(game, winner) else f"Player {winner + 1}"
    if self_drawn:
        game.say(f"\n{who} won by self-draw!")
    else:
        shooter_name = "you" if is_human(game, shooter) else f"Player {shooter + 1}"
        game.say(f"\n{who} won on {shooter_name}'s discard ({game.format_tile(winning_tile)}). Shooter: {shooter_name}.")
    game.say(f"Winning hand: {[game.format_tile(t) for t in game.sort_tiles(game.hands[winner])]}")
    if game.melds[winner]:
        game.say(f"Exposed melds: {[[game.format_tile(t) for t in game.sort_tiles(m)] for m in game.melds[winner]]}")
    game.say(f"Tai: {score['tai']}")
    for description, tai in score["items"]:
        game.say(f"  +{tai}  {description}")
    for player, units in sorted(sg.payouts(score["tai"], winner, self_drawn, shooter).items()):
        label = "You" if is_human(game, player) else f"Player {player + 1}"
        game.say(f"  {label}: {units:+d} units")
    game.result = {"winner": winner, "self_drawn": self_drawn, "shooter": shooter,
                   "score": score, "turns": game.turns}


def win_on_discard(game, winner, shooter, tile, from_hand=False):
    """Hand `tile` to the winner. from_hand=True when it is robbed from a Kong."""
    score = game.score_win(winner, tile)
    if from_hand:
        game.hands[shooter].remove(tile)
    else:
        game.discards[shooter].pop()
    game.hands[winner].append(tile)
    announce_win(game, winner, score, self_drawn=False, shooter=shooter, winning_tile=tile)


# --------------------------------------------------------------------------
# Game session: the game advances one decision at a time, with no terminal
# input. Whoever drives it (the terminal game below, a test, a bot) asks for
# the current decision, picks one of its legal actions and applies it.
# --------------------------------------------------------------------------
# Claims on a discard: Hu > Kong > Pong > Chi. On a tie the first player
# after the discarder in turn order wins.
CLAIM_PRIORITY = {"hu": 4, "kong": 3, "pong": 2, "chi": 1}


@dataclass(frozen=True)
class Action:
    """One choice in a decision.
    kind : "discard", "hu", "kong", "pong", "chi" or "pass"
    tile : the tile discarded, declared or claimed
    chi  : for a Chi, the two hand tiles that complete the run"""
    kind: str
    tile: tuple = None
    chi: tuple = None


@dataclass(frozen=True)
class Decision:
    """Who must act now and what they may do.
    kind "turn"     : own turn - discard, or (unless the turn began with a
                      Pong / Chi) declare Hu or a Kong
    kind "claim"    : respond to `tile` just discarded by `source`
    kind "rob_kong" : win on `tile`, which `source` is adding to a Pong, or pass"""
    player: int
    kind: str
    actions: tuple
    tile: tuple = None
    source: int = None


class IllegalActionError(ValueError):
    """The action is not one of the current decision's legal actions."""


class GameSession:
    """
    One hand of Singapore Mahjong driven decision by decision.

        session = GameSession(seed=7)
        while not session.is_over:
            obs = session.observation()          # deciding player's view
            session.apply(pick(obs, session.legal_actions()))
        session.result()

    seed / dealer / wind_round : set up a new game (seed makes it repeatable)
    game  : drive an existing, already dealt MahjongGame instead
    check : assert tile / hand-size invariants after every discard and at the end
    start : advance to the first decision straight away
    """

    def __init__(self, seed=None, dealer=None, wind_round=None, game=None, check=False, start=True):
        if game is None:
            game = MahjongGame(dealer, wind_round,
                               rand=random.Random(seed) if seed is not None else None)
            game.verbose = False
            game.deal()
        self.game = game
        self.check = check
        self.decision = None
        self.is_over = False
        self._claim = None
        self._first_turn = True
        if start:
            self._start_turn()

    # ---- public interface ---------------------------------------------------
    @property
    def current_player(self):
        """The player who must make the next decision (None once the game is over)."""
        return None if self.decision is None else self.decision.player

    def legal_actions(self):
        return () if self.decision is None else self.decision.actions

    def observation(self, player=None):
        """`player`'s own concealed hand plus public information only (H_game_state).
        Defaults to the player who must decide."""
        d = self.decision
        if player is None:
            if d is None:
                raise RuntimeError("the game is over: pass the player to observe")
            player = d.player
        if d is None:
            return observe(self.game, player)
        return observe(self.game, player, d.kind, d.tile, d.source)

    def apply(self, action, player=None):
        """Apply one legal action and advance to the next decision or the end of
        the game. Returns the next Decision (None when the game is over).
        An illegal action raises IllegalActionError and changes nothing."""
        d = self.decision
        if d is None:
            raise IllegalActionError("the game is over")
        if player is not None and player != d.player:
            raise IllegalActionError(f"Player {d.player + 1} must decide, not Player {player + 1}")
        if action not in d.actions:
            raise IllegalActionError(f"{action!r} is not legal; choose from {d.actions}")
        self.decision = None
        if d.kind == "turn":
            self._apply_turn(d.player, action)
        else:
            self._apply_claim(d.player, action)
        return self.decision

    def result(self):
        """Winner, score and units won or lost by each player."""
        if not self.is_over:
            raise RuntimeError("the game is still in progress")
        g, r = self.game, self.game.result
        instant = {p: g.instant[p] for p in range(4)}
        if r is None:
            out = {"winner": None, "self_drawn": False, "shooter": None, "tai": 0, "score": None}
            hand = {p: 0 for p in range(4)}
        else:
            out = {k: r[k] for k in ("winner", "self_drawn", "shooter", "score")}
            out["tai"] = r["score"]["tai"]
            pay = sg.payouts(r["score"]["tai"], r["winner"], r["self_drawn"], r["shooter"])
            hand = {p: pay.get(p, 0) for p in range(4)}
        out.update(turns=g.turns, payouts=hand, instant=instant,
                   total={p: hand[p] + instant[p] for p in range(4)})
        return out

    # ---- turns --------------------------------------------------------------
    def _start_turn(self):
        g = self.game
        player = g.current_player
        g.turns += 1
        g.say("\n" + "=" * 50)
        g.say(f"--- Player {player + 1}'s turn ---")
        g.say("=" * 50 + "\n")

        claimed = False
        # Only the dealer's opening turn skips the draw; a game resumed mid-way
        # (tests, recorded positions) has no such turn.
        opening, self._first_turn = self._first_turn and player == g.dealer, False
        if opening:
            g.say("Dealer discards first without drawing.")
        elif g.skip_next_draw:
            # Took a discard for Pong / Chi: no draw, and no self-drawn win.
            g.say(f"Player {player + 1} takes the exposed discard and does not draw a new tile.")
            g.skip_next_draw = False
            claimed = True
        else:
            g.kong_draw = g.kong_replacement
            drawn = g.draw_tile(from_back=g.kong_replacement)
            g.kong_replacement = False
            if drawn is None:
                g.say("Wall exhausted. Game ends in a draw.")
                return self._finish()
        self._offer_turn(player, claimed)

    def _offer_turn(self, player, claimed=False):
        g = self.game
        actions = []
        if not claimed:
            if g.can_hu(player):
                actions.append(Action("hu"))
            actions += [Action("kong", t) for t in g.self_kong_options(player)]
        actions += [Action("discard", t) for t in g.sort_tiles(set(g.hands[player]))]
        self.decision = Decision(player, "turn", tuple(actions))

    def _apply_turn(self, player, action):
        g = self.game
        if action.kind == "hu":
            g.say(f"Player {player + 1} declares Hu and wins!")
            announce_win(g, player, g.score_win(player), self_drawn=True)
            return self._finish()
        if action.kind == "kong":
            if g.hands[player].count(action.tile) != 4:
                # Singapore: another player may rob a promoted Kong to win.
                g.robbing_kong = True
                return self._open_claims(player, action.tile, robbing=True)
            return self._complete_self_kong(player, action.tile)
        g.discard_tile(action.tile)
        if self.check:
            g.check_invariants()
        self._open_claims(player, action.tile)

    def _complete_self_kong(self, player, tile):
        """Declare the Kong, then draw its replacement from the back of the wall."""
        g = self.game
        g.robbing_kong = False
        g.perform_self_kong(player, tile)
        drawn = g.draw_tile(from_back=True)
        g.kong_replacement = False
        g.kong_draw = True
        if drawn is None:
            g.say("Wall exhausted. Game ends in a draw.")
            return self._finish()
        self._offer_turn(player)

    # ---- claims -------------------------------------------------------------
    def _claim_actions(self, player, source, tile, robbing):
        g = self.game
        actions = []
        if g.can_hu(player, tile):
            actions.append(Action("hu", tile))
        if not robbing:
            if g.can_kong(player, tile):
                actions.append(Action("kong", tile))
            if g.can_pong(player, tile):
                actions.append(Action("pong", tile))
            if player == (source + 1) % 4:
                actions += [Action("chi", tile, tuple(seq)) for seq in g.chi_options(player, tile)]
        return actions + [Action("pass")] if actions else []

    def _open_claims(self, source, tile, robbing=False):
        """Offer `tile` to every player who could take it, best possible claim
        first. Nobody is asked once their best claim could no longer win."""
        waiting = []
        for distance in range(1, 4):
            player = (source + distance) % 4
            actions = self._claim_actions(player, source, tile, robbing)
            if actions:
                best = max(CLAIM_PRIORITY.get(a.kind, 0) for a in actions)
                waiting.append(((best, -distance), player, tuple(actions)))
        waiting.sort(reverse=True)
        self._claim = {"source": source, "tile": tile, "robbing": robbing,
                       "waiting": waiting, "best": None}
        self._next_claim()

    def _next_claim(self):
        c = self._claim
        if c["waiting"]:
            potential, player, actions = c["waiting"][0]
            if c["best"] is None or potential > c["best"][0]:
                c["waiting"].pop(0)
                kind = "rob_kong" if c["robbing"] else "claim"
                self.decision = Decision(player, kind, actions, c["tile"], c["source"])
                return
        self._resolve_claims()

    def _apply_claim(self, player, action):
        c = self._claim
        if action.kind != "pass":
            key = (CLAIM_PRIORITY[action.kind], -((player - c["source"]) % 4))
            if c["best"] is None or key > c["best"][0]:
                c["best"] = (key, player, action)
        self._next_claim()

    def _resolve_claims(self):
        g = self.game
        c, self._claim = self._claim, None
        source, tile = c["source"], c["tile"]
        if c["best"] is None:
            if c["robbing"]:
                return self._complete_self_kong(source, tile)
            g.current_player = (source + 1) % 4
        else:
            _, player, action = c["best"]
            if action.kind == "hu":
                # Hu beats every other call. The discarder is the shooter.
                if c["robbing"]:
                    g.say(f"Player {player + 1} robs the Kong!")
                win_on_discard(g, player, source, tile, from_hand=c["robbing"])
                return self._finish()
            if action.kind == "kong":
                g.perform_kong(player, source, tile)
            elif action.kind == "pong":
                g.perform_pong(player, source, tile)
            else:
                g.perform_chi(player, source, tile, list(action.chi))
            g.say(f"{action.kind.capitalize()} called by Player {player + 1}, turn goes to them.")
        if g.verbose:
            g.show_discards()
            g.show_melds()
        self._start_turn()

    def _finish(self):
        self.decision = None
        self.is_over = True
        if self.check:
            if self.game.result is None:
                self.game.check_invariants()   # draw: everyone is left holding 13
            else:
                _check_result(self.game)


# --------------------------------------------------------------------------
# Terminal controllers: answer a session's decisions the way the terminal
# game always has. Human: ask. Engine / ML: decide. Players 2-4: their own
# engine (or random).
# --------------------------------------------------------------------------
def _actions_of(decision, kind):
    return [a for a in decision.actions if a.kind == kind]


def choose_action(game, decision):
    if decision.kind == "turn":
        if is_human(game, decision.player):
            show_human_turn(game)
        action = _choose_hu_or_kong(game, decision)
        if action is not None:
            return action
        return Action("discard", player_discard(game, decision.player))
    return _choose_claim(game, decision)


def show_human_turn(game):
    triplets, leftover = game.find_completed_sets(0)
    if triplets:
        print("Completed sets:")
        for s in triplets:
            print("  -", [game.format_tile(t) for t in s])
    print("Remaining hand tiles:", [game.format_tile(t) for t in leftover])
    game.show_player_state(0)
    hints = game.get_hints(0)
    if hints:
        print("Hints:")
        for h in hints:
            print("  -", h)


def _choose_hu_or_kong(game, decision):
    """Self-drawn Hu or an own-turn Kong, or None to go on and discard.
    Winning is a rule, not a strategy choice: only a human is asked."""
    player = decision.player
    if _actions_of(decision, "hu"):
        if not is_human(game, player) or yes_no_prompt(
                f"You can Hu ({game.score_win(player)['tai']} tai)! Declare win?"):
            return Action("hu")
    for action in _actions_of(decision, "kong"):
        if decide_call(game, player, action.tile, "self_kong",
                       f"You can Kong {game.format_tile(action.tile)}. Declare Kong?"):
            return action
    return None


def _choose_claim(game, decision):
    player, tile = decision.player, decision.tile
    name = game.format_tile(tile)
    if _actions_of(decision, "hu"):
        tai = game.score_win(player, tile)["tai"]
        if not is_human(game, player) or yes_no_prompt(
                f"You can Hu on Player {decision.source + 1}'s {name} ({tai} tai)! Declare win?"):
            return Action("hu", tile)
    for kind in ("kong", "pong"):
        if _actions_of(decision, kind) and decide_call(
                game, player, tile, kind, f"You can {kind.capitalize()} {name}. Call {kind.capitalize()}?"):
            return Action(kind, tile)
    chis = _actions_of(decision, "chi")
    if chis:
        seq = _choose_chi_run(game, player, tile)
        if seq is not None:
            return next((a for a in chis if sorted(a.chi) == sorted(seq)), Action("pass"))
    return Action("pass")


def _choose_chi_run(game, player, tile):
    """The two hand tiles to Chi `tile` with, or None to pass."""
    engine = game.strategy if is_engine(game, player) else opponent_engine(game, player)
    if engine:
        seq = engine_chi_sequence(game, player, tile, engine)
        _record(game, player, "call", tile, call_type="chi", taken=seq is not None)
        return seq
    if is_ml(game, player):
        if not decide_call(game, player, tile, "chi", ""):
            return None
        return game.ml_policy.chi_sequence(game, player, tile)
    if not decide_call(game, player, tile, "chi", f"You can Chi {game.format_tile(tile)}. Call Chi?"):
        return None
    if is_human(game, player):
        return choose_chi_sequence(game, tile)
    return rng.choice(game.chi_options(player, tile))


def self_turn_kongs(game, player):
    """Own-turn Kongs (concealed or promoted, which may be robbed), decided by
    the seat's controller. Returns "win", "draw" or None (carry on and discard)."""
    game.current_player = player
    session = GameSession(game=game, start=False)
    session._offer_turn(player)
    while not session.is_over:
        decision = session.decision
        if decision.kind == "turn":
            action = _choose_hu_or_kong(game, decision)
            if action is None:
                return None
        else:
            action = choose_action(game, decision)
        session.apply(action)
    return "win" if game.result else "draw"


# --------------------------------------------------------------------------
# One full game
# --------------------------------------------------------------------------
def play_game(game=None, player1="human", delay=1.0, verbose=True, check=False, setup=None,
              opponents="engine"):
    """
    Play one hand of Singapore Mahjong and return the finished game.
    player1   : "human" (keyboard) or "engine" (Decision / Strategy engine)
    opponents : Players 2-4, "engine" (each with its own style) or "random"
    delay   : seconds to pause between turns (0 for simulations)
    check   : assert tile / hand-size invariants after every discard
    """
    if game is None:
        game = MahjongGame()
        game.player1 = player1
        game.opponents = opponents
        game.verbose = verbose
        if setup is not None:
            setup(game)         # e.g. attach a recorder or an ML policy
        game.deal()
    human = game.player1 == "human"

    dealer_name = "You are" if human and game.dealer == 0 else f"Player {game.dealer + 1} is"
    game.say(f"\n{dealer_name} the dealer (East). Prevailing wind: {game.prevailing_wind.capitalize()}.")
    game.say(f"Player 1 seat: {game.seat(0)} ({sg.WINDS[game.seat(0) - 1].capitalize()})")
    if game.opponents == "engine":
        game.say("Opponents: " + ", ".join(f"Player {p + 1} ({s})" for p, s in sorted(game.styles.items())))
    if game.verbose:
        print("\nInitial public state:")
        game.show_melds()
        game.show_discards()
    game.say("=" * 50)

    session = GameSession(game=game, check=check)
    while not session.is_over:
        turn = game.turns
        session.apply(choose_action(game, session.decision))
        if delay and game.turns != turn:
            time.sleep(delay)

    if game.instant:
        game.say("Instant payments this game:")
        for p in range(4):
            game.say(f"  Player {p + 1}: {game.instant[p]:+d} units")
    game.say("\nGame Over.")
    recorder = getattr(game, "recorder", None)
    if recorder is not None:
        recorder.on_game_end(game)
    return game


def _check_result(game):
    r = game.result
    winner = r["winner"]
    assert game.check_win(game.hands[winner], game.melds[winner]), "winner's hand is not complete"
    assert r["score"]["tai"] >= sg.MIN_TAI, "win below minimum tai"
    assert r["self_drawn"] == (r["shooter"] is None)
    assert r["shooter"] != winner
    everything = list(game.wall)
    for i in range(4):
        everything += game.hands[i] + game.discards[i] + game.bonus[i]
        for meld in game.melds[i]:
            everything += meld
    assert Counter(everything) == Counter(all_tiles), "tile count mismatch"


def main(opponents="engine"):
    """Interactive game: you are Player 1."""
    play_game(player1="human", delay=1.0, opponents=opponents)


# --------------------------------------------------------------------------
# Batch simulation: Player 1 = strategy engine against Players 2-4
# --------------------------------------------------------------------------
def simulate(n_games=1000, seed=None, check=True, progress=True, player1="engine", setup=None,
             opponents="engine"):
    import traceback
    global rng
    if seed is not None:
        rng = random.Random(seed)

    stats = {
        "games": n_games, "p1_wins": 0, "p1_self_drawn": 0, "p1_on_discard": 0,
        "other_wins": Counter(), "draws": 0, "p1_shooter": 0, "errors": [],
        "p1_tai": Counter(), "turns": [], "p1_units": 0, "p1_instant": 0,
        "opponents": opponents,
    }
    started = time.time()
    for g in range(n_games):
        try:
            game = play_game(player1=player1, delay=0, verbose=False, check=check, setup=setup,
                             opponents=opponents)
        except Exception:
            stats["errors"].append(traceback.format_exc())
            continue
        stats["turns"].append(game.turns)
        stats["p1_instant"] += game.instant[0]
        stats["p1_units"] += game.instant[0]
        r = game.result
        if r is None:
            stats["draws"] += 1
            continue
        pay = sg.payouts(r["score"]["tai"], r["winner"], r["self_drawn"], r["shooter"])
        stats["p1_units"] += pay.get(0, 0)
        if r["winner"] == 0:
            stats["p1_wins"] += 1
            stats["p1_tai"][r["score"]["tai"]] += 1
            stats["p1_self_drawn" if r["self_drawn"] else "p1_on_discard"] += 1
        else:
            stats["other_wins"][r["winner"] + 1] += 1
            if r["shooter"] == 0:
                stats["p1_shooter"] += 1
        if progress and (g + 1) % 100 == 0:
            print(f"  {g + 1}/{n_games} games, Player 1 won {stats['p1_wins']}")
    stats["seconds"] = time.time() - started
    return stats


def print_report(stats, label="Player 1 = strategy engine"):
    n = stats["games"]
    pct = lambda k: 100.0 * k / n if n else 0.0
    print(f"\nSimulated {n} games in {stats['seconds']:.1f}s "
          f"({label}, Players 2-4 = {stats.get('opponents', 'engine')})")
    print(f"  Player 1 wins      : {stats['p1_wins']:5d}  ({pct(stats['p1_wins']):.1f}%)")
    print(f"    self-drawn       : {stats['p1_self_drawn']:5d}")
    print(f"    on a discard     : {stats['p1_on_discard']:5d}")
    for p in (2, 3, 4):
        print(f"  Player {p} wins      : {stats['other_wins'][p]:5d}  ({pct(stats['other_wins'][p]):.1f}%)")
    print(f"  Draws (wall empty) : {stats['draws']:5d}  ({pct(stats['draws']):.1f}%)")
    print(f"  Player 1 was shooter: {stats['p1_shooter']} times")
    print(f"  Player 1 net units : {stats['p1_units']:+d}  (instant payments {stats['p1_instant']:+d})")
    if stats["p1_tai"]:
        print("  Player 1 winning tai: " + ", ".join(f"{t} tai x{c}" for t, c in sorted(stats["p1_tai"].items())))
    if stats["turns"]:
        print(f"  Average turns/game : {sum(stats['turns']) / len(stats['turns']):.1f}")
    print(f"  Crashes / rule violations: {len(stats['errors'])}")
    for err in stats["errors"][:3]:
        print(err)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Singapore Mahjong simulator")
    parser.add_argument("--simulate", type=int, metavar="N",
                        help="run N games with the engine as Player 1 and print statistics")
    parser.add_argument("--seed", type=int, help="random seed for reproducible simulations")
    parser.add_argument("--auto", action="store_true",
                        help="watch one game with the engine playing Player 1")
    parser.add_argument("--random-opponents", action="store_true",
                        help="Players 2-4 discard and call at random instead of playing properly")
    args = parser.parse_args()
    opponents = "random" if args.random_opponents else "engine"
    if args.simulate:
        print_report(simulate(args.simulate, seed=args.seed, opponents=opponents))
    elif args.auto:
        if args.seed is not None:
            rng = random.Random(args.seed)
        play_game(player1="engine", delay=0, opponents=opponents)
    else:
        main(opponents)
