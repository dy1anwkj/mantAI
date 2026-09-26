import random
from collections import Counter

from Decision_Strategy_engine import (
    StrategyEngine,
    collect_visible_tiles,
    engine_choose_discard,
    engine_should_call,
    summarize_player_state,
)
from hand_eval import analyze_hand, evaluate_discard_quality

# Define Mahjong tiles (Bamboo, Dots, Wan, Winds, Dragons)
bamboo = [('bamboo', i) for i in range(1, 10)]
dots = [('dot', i) for i in range(1, 10)]
wan = [('wan', i) for i in range(1, 10)]
winds = [('wind', d) for d in ['east', 'south', 'west', 'north']]
dragons = [('dragon', d) for d in ['red', 'green', 'white']]

tiles = (bamboo + dots + wan + winds + dragons) * 4

class MahjongGame:
    def __init__(self):
        self.wall = list(tiles)
        random.shuffle(self.wall)
        self.hands = [[] for _ in range(4)]
        self.discards = [[] for _ in range(4)]
        self.melds = [[] for _ in range(4)]
        self.dealer = 0  # Player 1 dealer
        self.current_player = self.dealer
        self.wind_round = 0  # Tracks East, South, West, North rounds
        self.strategy = StrategyEngine()
        self.skip_next_draw = False

    def deal(self):
        for i, hand in enumerate(self.hands):
            count = 14 if i == self.dealer else 13
            for _ in range(count):
                hand.append(self.wall.pop())

    def draw_tile(self):
        if not self.wall:
            print("Wall depleted! Round ends in draw.")
            return None
        tile = self.wall.pop()
        self.hands[self.current_player].append(tile)
        print(f"Player {self.current_player + 1} draws {self.format_tile(tile)}")
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

    def total_active_tiles(self, player_index):
        return self.get_total_active_tiles(player_index)

    def reconcile_active_tiles(self, player_index, max_total=13):
        while self.total_active_tiles(player_index) > max_total:
            if not self.hands[player_index]:
                break
            extra_tile = self.hands[player_index].pop()
            print(f"Player {player_index + 1} had too many active tiles; removed extra {self.format_tile(extra_tile)} to keep total at {max_total}.")

    def expose_meld_from_hand(self, player_index, meld_tiles):
        for tile in meld_tiles:
            if tile in self.hands[player_index]:
                self.hands[player_index].remove(tile)
        self.reconcile_active_tiles(player_index)

    def enforce_concealed_hand_limit(self, player_index):
        self.reconcile_active_tiles(player_index)

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

    def check_win(self, hand):
        hand = sorted(hand)
        if len(hand) != 14:
            return False

        def can_partition(tiles):
            if not tiles:
                return True

            tiles = list(tiles)
            unique_tiles = sorted(set(tiles), key=lambda t: (t[0], t[1] if isinstance(t[1], int) else 99))

            for tile in unique_tiles:
                count = tiles.count(tile)

                if count >= 3:
                    remaining = tiles.copy()
                    for _ in range(3):
                        remaining.remove(tile)
                    if can_partition(remaining):
                        return True

                if tile[0] in ['bamboo', 'dot', 'wan'] and count >= 1:
                    next1 = (tile[0], tile[1] + 1)
                    next2 = (tile[0], tile[1] + 2)
                    if next1 in tiles and next2 in tiles:
                        remaining = tiles.copy()
                        remaining.remove(tile)
                        remaining.remove(next1)
                        remaining.remove(next2)
                        if can_partition(remaining):
                            return True

            if len(tiles) == 2 and tiles[0] == tiles[1]:
                return True

            return False

        return can_partition(hand)

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
        if len(hand) == 13:
            for tile in tiles:
                test_hand = hand + [tile]
                if self.check_win(sorted(test_hand)):
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
        self.expose_meld_from_hand(player_index, [tile, tile, tile])
        if self.discards[discarder_index]:
            self.discards[discarder_index].pop()
        self.melds[player_index].append([tile] * 3)
        self.enforce_concealed_hand_limit(player_index)
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

    def perform_chi(self, player_index, discarder_index, tile):
        suit, rank = tile
        hand = self.hands[player_index]
        sequences = [
            [(suit, rank - 2), (suit, rank - 1)],
            [(suit, rank - 1), (suit, rank + 1)],
            [(suit, rank + 1), (suit, rank + 2)],
        ]
        for seq in sequences:
            if all(t in hand for t in seq):
                meld_tiles = [tile] + seq
                self.expose_meld_from_hand(player_index, meld_tiles)
                if self.discards[discarder_index]:
                    self.discards[discarder_index].pop()
                self.melds[player_index].append(meld_tiles)
                self.enforce_concealed_hand_limit(player_index)
                print(f"Player {player_index + 1} calls Chi on {self.format_tile(tile)} with sequence {[self.format_tile(t) for t in seq]}")
                print(f"Player {player_index + 1} now has {len(self.hands[player_index])} concealed tiles left after exposing the meld.")
                self.current_player = player_index
                self.skip_next_draw = True
                return True
        return False

    def can_kong(self, player_index, tile):
        return self.hands[player_index].count(tile) == 3

    def perform_kong(self, player_index, discarder_index, tile):
        if not self.can_kong(player_index, tile):
            return False
        self.expose_meld_from_hand(player_index, [tile, tile, tile, tile])
        if self.discards[discarder_index]:
            self.discards[discarder_index].pop()
        self.melds[player_index].append([tile] * 4)
        self.enforce_concealed_hand_limit(player_index)
        print(f"Player {player_index + 1} calls Kong on {self.format_tile(tile)}")
        print(f"Player {player_index + 1} now has {len(self.hands[player_index])} concealed tiles left after exposing the meld.")
        self.current_player = player_index
        self.skip_next_draw = False
        return True

    def can_hu(self, player_index):
        combined_hand = self.hands[player_index][:]
        for meld in self.melds[player_index]:
            combined_hand.extend(meld)
        return self.check_win(sorted(combined_hand))

def estimate_discard_win_chance(game, player_index, tile):
    hand = game.hands[player_index]
    visible_tiles = []
    for discard_pile in game.discards:
        visible_tiles.extend(discard_pile)
    for player_melds in game.melds:
        for meld in player_melds:
            visible_tiles.extend(meld)

    quality = evaluate_discard_quality(
        hand,
        tile,
        exposed_melds=game.melds[player_index],
        visible_tiles=visible_tiles,
    )
    remaining = [t for t in hand if t != tile]
    after_analysis = analyze_hand(
        remaining,
        exposed_melds=game.melds[player_index],
        visible_tiles=visible_tiles,
    )

    raw_score = (
        quality["after_score"]
        + 140 * len(after_analysis["waits"])
        + 25 * len(after_analysis["useful_tiles"])
        - 10 * len(quality["waits_before"])
    )
    percentage = max(0.0, min(99.0, 52.0 + raw_score / 55.0))
    return round(percentage, 1)


def choose_discard_tile(game):
    hand = game.sort_tiles(game.hands[game.current_player])
    state = summarize_player_state(game, game.current_player, game.strategy)
    analysis = state["analysis"]
    strategy_choice = state["best_discard"]

    discard_ratings = []
    for tile in hand:
        chance = estimate_discard_win_chance(game, game.current_player, tile)
        discard_ratings.append((tile, chance))

    best_tile, best_percentage = max(discard_ratings, key=lambda item: item[1])
    if strategy_choice is not None and strategy_choice in hand:
        best_tile = strategy_choice

    print("\nYour sorted hand:")
    for i, tile in enumerate(hand, start=1):
        chance = next(chance for candidate, chance in discard_ratings if candidate == tile)
        print(f"{i}: {game.format_tile(tile)}  -- estimated win chance if discarded: {chance:.1f}%")

    if game.melds[game.current_player]:
        print("Your current melds:")
        for meld in game.melds[game.current_player]:
            print("  -", [game.format_tile(t) for t in game.sort_tiles(meld)])

    print(f"\nEvaluator summary: score={analysis['strength']['score']:.1f}, waits={len(analysis['waits'])}, useful={[item['tile'] for item in analysis['useful_tiles'][:3]]}")
    print(f"Strategy recommendation: {game.format_tile(best_tile)}")
    print(f"Best discard by win estimate: {game.format_tile(best_tile)} ({best_percentage:.1f}% estimated win chance)")

    while True:
        try:
            raw = input(f"Player {game.current_player + 1}, choose tile to discard (1-{len(hand)}): ")
        except EOFError:
            print(f"\n[Auto] No input received. Using recommended discard: {game.format_tile(best_tile)}")
            return best_tile

        try:
            choice = int(raw)
            if 1 <= choice <= len(hand):
                chosen_tile = hand[choice - 1]
                return chosen_tile
        except ValueError:
            pass

        if raw.strip() == '':
            print(f"\n[Auto] Empty input. Using recommended discard: {game.format_tile(best_tile)}")
            return best_tile
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
    return engine_choose_discard(game, player_index, game.strategy)


def main():
    import time
    game = MahjongGame()
    game.deal()
    print("\nInitial public state:")
    game.show_melds()
    game.show_discards()
    print("=" * 50)

    first_turn = True

    while True:
        print("\n" + "=" * 50)
        print(f"--- Player {game.current_player + 1}'s turn ---")
        print("=" * 50 + "\n")

        triplets, leftover = game.find_completed_sets(game.current_player)
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
            drawn = game.draw_tile()
            if drawn is None:
                print("Wall exhausted. Game ends in a draw.")
                break

        if game.current_player == 0:
            game.show_player_state(0)
        hints = game.get_hints(game.current_player)
        if hints and not (first_turn and game.current_player == game.dealer):
            print("Hints:")
            for h in hints:
                print("  -", h)

        # Check Hu declaration
        if game.current_player == 0:
            if game.can_hu(0):
                if yes_no_prompt("You can Hu! Declare win?"):
                    print("You declared Hu and won!")
                    break
            state = summarize_player_state(game, game.current_player, game.strategy)
            analysis = state["analysis"]
            print(
                f"Hand analysis: score={analysis['strength']['score']:.1f}, "
                f"waits={len(analysis['waits'])}, "
                f"useful={[item['tile'] for item in analysis['useful_tiles'][:3]]}"
            )
            discard_tile = choose_discard_tile(game)
        else:
            if game.can_hu(game.current_player):
                print(f"Player {game.current_player + 1} declares Hu and wins!")
                break

            state = summarize_player_state(game, game.current_player, game.strategy)
            analysis = state["analysis"]
            print(
                f"AI analysis: score={analysis['strength']['score']:.1f}, "
                f"waits={len(analysis['waits'])}, "
                f"useful={[item['tile'] for item in analysis['useful_tiles'][:3]]}"
            )

            discard_tile = ai_choose_discard(game, game.current_player)
            print(f"Player {game.current_player + 1} (AI) discards {game.format_tile(discard_tile)}")

        game.discard_tile(discard_tile)

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
                    call_kong = engine_should_call(game, player, discard_tile, "kong", engine=game.strategy, verbose=True)

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
                        call_pong = engine_should_call(game, player, discard_tile, "pong", engine=game.strategy, verbose=True)

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
                    call_chi = engine_should_call(game, next_player, discard_tile, "chi", engine=game.strategy, verbose=True)

                if call_chi:
                    chi_called = game.perform_chi(next_player, game.current_player, discard_tile)
                    if chi_called:
                        print(f"Chi called by Player {next_player + 1}, turn goes to them.")

        if not kong_called and not pong_called and not chi_called:
            game.current_player = (game.current_player + 1) % 4

        # Keep each player's total active tiles at 13 after every full turn:
        # concealed hand + shown melds, with the dealer starting at 14 and dropping to 13 on his first discard.
        if game.current_player != game.dealer or not first_turn:
            game.reconcile_active_tiles(game.current_player)

        game.show_discards()
        game.show_melds()

        time.sleep(1)

    print("\nGame Over.")

if __name__ == "__main__":
    main()
