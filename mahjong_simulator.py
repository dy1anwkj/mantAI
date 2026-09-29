import importlib
import random
from collections import Counter

import singapore_rules as sg

try:
    strategy_mod = importlib.import_module("Decision_Strategy_engine")
    StrategyEngine = strategy_mod.StrategyEngine
    collect_visible_tiles = strategy_mod.collect_visible_tiles
    engine_choose_discard = strategy_mod.engine_choose_discard
    engine_should_call = strategy_mod.engine_should_call
    summarize_player_state = strategy_mod.summarize_player_state
except (ImportError, ModuleNotFoundError):
    class StrategyEngine:
        def __init__(self, *args, **kwargs):
            pass

    def collect_visible_tiles(*args, **kwargs):
        return []

    def engine_choose_discard(*args, **kwargs):
        return None

    def engine_should_call(*args, **kwargs):
        return False

    def summarize_player_state(*args, **kwargs):
        return {}

try:
    hand_eval_mod = importlib.import_module("D_hand_eval")
    analyze_hand = hand_eval_mod.analyze_hand
    evaluate_discard_quality = hand_eval_mod.evaluate_discard_quality
    is_winning_hand = hand_eval_mod.is_winning_hand
    useful_tiles = hand_eval_mod.useful_tiles
    winning_tiles = hand_eval_mod.winning_tiles
except (ImportError, ModuleNotFoundError):
    def analyze_hand(*args, **kwargs):
        return {}

    def evaluate_discard_quality(*args, **kwargs):
        return 0

    def is_winning_hand(*args, **kwargs):
        return False

    def useful_tiles(*args, **kwargs):
        return []

    def winning_tiles(*args, **kwargs):
        return []

# Define Mahjong tiles (Bamboo, Dots, Wan, Winds, Dragons)
bamboo = [('bamboo', i) for i in range(1, 10)]
dots = [('dot', i) for i in range(1, 10)]
wan = [('wan', i) for i in range(1, 10)]
winds = [('wind', d) for d in ['east', 'south', 'west', 'north']]
dragons = [('dragon', d) for d in ['red', 'green', 'white']]

tiles = (bamboo + dots + wan + winds + dragons) * 4
# Singapore sets add 4 flowers, 4 seasons and 4 animals: 148 tiles in total.
all_tiles = tiles + sg.BONUS_TILES

# Unseeded RNG drawing from OS entropy: every game gets a fresh random wall,
# and non-human players act randomly instead of following the strategy engine.
rng = random.SystemRandom()

class MahjongGame:
    def __init__(self, dealer=None, wind_round=None):
        self.wall = list(all_tiles)
        rng.shuffle(self.wall)
        self.hands = [[] for _ in range(4)]
        self.discards = [[] for _ in range(4)]
        self.melds = [[] for _ in range(4)]
        self.bonus = [[] for _ in range(4)]  # revealed flowers / seasons / animals
        # Dealer and round wind are drawn at random each game unless given.
        self.dealer = rng.randrange(4) if dealer is None else dealer
        self.current_player = self.dealer
        self.wind_round = rng.randrange(4) if wind_round is None else wind_round
        # Each AI player gets its own random eagerness to call Pong / Kong / Chi.
        self.call_rates = [rng.uniform(0.2, 0.8) for _ in range(4)]
        self.strategy = StrategyEngine()
        self.skip_next_draw = False
        self.kong_replacement = False
        self.result = None  # set when someone wins

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
        self.hands[player_index].remove(tile)
        self.bonus[player_index].append(tile)
        print(f"Player {player_index + 1} reveals {self.format_tile(tile)} and draws a replacement.")

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
                print("Wall depleted! Round ends in draw.")
                return None
            tile = self.wall.pop(0) if from_back else self.wall.pop()
            self.hands[player].append(tile)
            if not sg.is_bonus(tile):
                break
            self.reveal_bonus(player, tile)
            from_back = True
        if player == 0:
            print(f"You draw {self.format_tile(tile)}")
        else:
            print(f"Player {player + 1} draws a tile")
        return tile

    def discard_tile(self, tile):
        if tile not in self.hands[self.current_player]:
            print(f"Error: Player {self.current_player + 1} tried to discard a tile not in hand!")
            return False
        self.hands[self.current_player].remove(tile)
        self.discards[self.current_player].append(tile)
        print(f"Player {self.current_player + 1} discards {self.format_tile(tile)}")
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
            print(f"Player {i + 1} hand: {formatted}")

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

        print(f"Player {player_index + 1} concealed hand ({len(hand)} tiles): {[self.format_tile(t) for t in hand]}")
        if melds:
            print(f"Player {player_index + 1} current melds: {melds}")
            print(f"Player {player_index + 1} total active tiles: {self.get_total_active_tiles(player_index)}")
        else:
            print(f"Player {player_index + 1} current melds: []")
            print(f"Player {player_index + 1} total active tiles: {len(hand)}")
        print(f"Seat: {self.seat(player_index)} ({sg.WINDS[self.seat(player_index) - 1].capitalize()}), "
              f"bonus tiles: {[self.format_tile(t) for t in self.bonus[player_index]]}")

    def show_discards(self):
        for i, discard in enumerate(self.discards):
            formatted = [self.format_tile(t) for t in discard]
            print(f"Player {i + 1} discards: {formatted}")

    def show_melds(self):
        for i, meld_list in enumerate(self.melds):
            if meld_list:
                formatted_melds = [[self.format_tile(t) for t in meld] for meld in meld_list]
                print(f"Player {i + 1} melds: {formatted_melds}")
            else:
                print(f"Player {i + 1} has no melds.")
            if self.bonus[i]:
                print(f"Player {i + 1} bonus tiles: {[self.format_tile(t) for t in self.bonus[i]]}")

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
        print(f"Player {player_index + 1} calls Pong on {self.format_tile(tile)}")
        print(f"Player {player_index + 1} now has {len(self.hands[player_index])} concealed tiles left after exposing the meld.")
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
            seq = rng.choice(options)
        if seq not in options:
            return False
        meld_tiles = [tile] + seq
        self.expose_meld_from_hand(player_index, seq)
        self.discards[discarder_index].pop()
        self.melds[player_index].append(meld_tiles)
        print(f"Player {player_index + 1} calls Chi on {self.format_tile(tile)} with sequence {[self.format_tile(t) for t in seq]}")
        print(f"Player {player_index + 1} now has {len(self.hands[player_index])} concealed tiles left after exposing the meld.")
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
        print(f"Player {player_index + 1} calls Kong on {self.format_tile(tile)}")
        print(f"Player {player_index + 1} now has {len(self.hands[player_index])} concealed tiles left after exposing the meld.")
        self.current_player = player_index
        self.skip_next_draw = False
        self.kong_replacement = True
        return True

    def score_win(self, player_index, winning_tile=None):
        """Tai for the player's hand, optionally completed by another player's
        discard. Returns None if the hand is not a winning hand."""
        concealed = list(self.hands[player_index])
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


def find_discard_winner(game, tile):
    discarder = game.current_player
    for i in range(1, 4):
        player = (discarder + i) % 4
        if not game.can_hu(player, tile):
            continue
        if player == 0:
            tai = game.score_win(0, tile)["tai"]
            if yes_no_prompt(f"You can Hu on Player {discarder + 1}'s {game.format_tile(tile)} ({tai} tai)! Declare win?"):
                return player
        else:
            # Winning is a rule, not a strategy choice, so AI players always take it.
            return player
    return None


def announce_win(game, winner, score, self_drawn, shooter=None, winning_tile=None):
    who = "You" if winner == 0 else f"Player {winner + 1}"
    if self_drawn:
        print(f"\n{who} won by self-draw!")
    else:
        shooter_name = "you" if shooter == 0 else f"Player {shooter + 1}"
        print(f"\n{who} won on {shooter_name}'s discard ({game.format_tile(winning_tile)}). Shooter: {shooter_name}.")
    print(f"Winning hand: {[game.format_tile(t) for t in game.sort_tiles(game.hands[winner])]}")
    if game.melds[winner]:
        print(f"Exposed melds: {[[game.format_tile(t) for t in game.sort_tiles(m)] for m in game.melds[winner]]}")
    print(f"Tai: {score['tai']}")
    for description, tai in score["items"]:
        print(f"  +{tai}  {description}")
    for player, units in sorted(sg.payouts(score["tai"], winner, self_drawn, shooter).items()):
        label = "You" if player == 0 else f"Player {player + 1}"
        print(f"  {label}: {units:+d} units")
    game.result = {"winner": winner, "self_drawn": self_drawn, "shooter": shooter, "score": score}


def main():
    import time
    game = MahjongGame()
    game.deal()
    dealer_name = "You are" if game.dealer == 0 else f"Player {game.dealer + 1} is"
    print(f"\n{dealer_name} the dealer (East). Prevailing wind: {game.prevailing_wind.capitalize()}.")
    print(f"Your seat: {game.seat(0)} ({sg.WINDS[game.seat(0) - 1].capitalize()})")
    print("\nInitial public state:")
    game.show_melds()
    game.show_discards()
    print("=" * 50)

    first_turn = True

    while True:
        print("\n" + "=" * 50)
        print(f"--- Player {game.current_player + 1}'s turn ---")
        print("=" * 50 + "\n")

        if game.current_player == 0:
            triplets, leftover = game.find_completed_sets(0)
            if triplets:
                print("Completed sets:")
                for s in triplets:
                    print("  -", [game.format_tile(t) for t in s])
            print("Remaining hand tiles:", [game.format_tile(t) for t in leftover])

        if first_turn and game.current_player == game.dealer:
            print("Dealer discards first without drawing.")
            first_turn = False
        elif game.skip_next_draw:
            print(f"Player {game.current_player + 1} takes the exposed discard and does not draw a new tile.")
            game.skip_next_draw = False
        else:
            drawn = game.draw_tile(from_back=game.kong_replacement)
            game.kong_replacement = False
            if drawn is None:
                print("Wall exhausted. Game ends in a draw.")
                break

        if game.current_player == 0:
            game.show_player_state(0)
        hints = game.get_hints(game.current_player) if game.current_player == 0 else []
        if hints and not (first_turn and game.current_player == game.dealer):
            print("Hints:")
            for h in hints:
                print("  -", h)

        # Self-drawn Hu
        if game.current_player == 0:
            if game.can_hu(0):
                tai = game.score_win(0)["tai"]
                if yes_no_prompt(f"You can Hu ({tai} tai)! Declare win?"):
                    print("You declared Hu and won!")
                    announce_win(game, 0, game.score_win(0), self_drawn=True)
                    break
            discard_tile = choose_discard_tile(game)
        else:
            if game.can_hu(game.current_player):
                print(f"Player {game.current_player + 1} declares Hu and wins!")
                announce_win(game, game.current_player, game.score_win(game.current_player), self_drawn=True)
                break

            discard_tile = ai_choose_discard(game, game.current_player)
            print(f"Player {game.current_player + 1} (AI) discards {game.format_tile(discard_tile)}")

        game.discard_tile(discard_tile)

        # Hu on the discard beats every other call. The first player after the
        # discarder, in turn order, gets it. The discarder is the "shooter".
        winner = find_discard_winner(game, discard_tile)
        if winner is not None:
            shooter = game.current_player
            score = game.score_win(winner, discard_tile)
            game.discards[shooter].pop()
            game.hands[winner].append(discard_tile)
            announce_win(game, winner, score, self_drawn=False, shooter=shooter, winning_tile=discard_tile)
            break

        pong_called = False
        kong_called = False
        chi_called = False

        # Kong calls: AI players auto-decide, human still prompts only for their own turn
        for i in range(1, 4):
            player = (game.current_player + i) % 4
            if game.can_kong(player, discard_tile):
                if player == 0:
                    call_kong = yes_no_prompt(f"You can Kong {game.format_tile(discard_tile)}. Call Kong?")
                else:
                    call_kong = ai_should_call(game, player, discard_tile, "kong")

                if call_kong:
                    kong_called = game.perform_kong(player, game.current_player, discard_tile)
                    if kong_called:
                        print(f"Kong called by Player {player + 1}, turn goes to them.")
                        break

        # Pong calls if no Kong
        if not kong_called:
            for i in range(1, 4):
                player = (game.current_player + i) % 4
                if game.can_pong(player, discard_tile):
                    if player == 0:
                        call_pong = yes_no_prompt(f"You can Pong {game.format_tile(discard_tile)}. Call Pong?")
                    else:
                        call_pong = ai_should_call(game, player, discard_tile, "pong")

                    if call_pong:
                        pong_called = game.perform_pong(player, game.current_player, discard_tile)
                        if pong_called:
                            print(f"Pong called by Player {player + 1}, turn goes to them.")
                            break

        # Chi calls if no Kong or Pong, only next player on the left
        if not kong_called and not pong_called:
            next_player = (game.current_player + 1) % 4
            if game.can_chi(next_player, discard_tile):
                if next_player == 0:
                    call_chi = yes_no_prompt(f"You can Chi {game.format_tile(discard_tile)}. Call Chi?")
                else:
                    call_chi = ai_should_call(game, next_player, discard_tile, "chi")

                if call_chi:
                    seq = choose_chi_sequence(game, discard_tile) if next_player == 0 else None
                    chi_called = game.perform_chi(next_player, game.current_player, discard_tile, seq)
                    if chi_called:
                        print(f"Chi called by Player {next_player + 1}, turn goes to them.")

        if not kong_called and not pong_called and not chi_called:
            game.current_player = (game.current_player + 1) % 4

        game.show_discards()
        game.show_melds()

        time.sleep(1)

    print("\nGame Over.")

if __name__ == "__main__":
    main()
