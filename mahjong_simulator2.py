import random
from collections import Counter

# Tiles definitions
characters = [('character', i) for i in range(1, 10)]
bamboo = [('bamboo', i) for i in range(1, 10)]
dots = [('dot', i) for i in range(1, 10)]
winds = [('wind', d) for d in ['east', 'south', 'west', 'north']]
dragons = [('dragon', d) for d in ['red', 'green', 'white']]
tiles = characters + bamboo + dots + winds + dragons

class MahjongGame:
    def __init__(self):
        self.wall = tiles * 4
        random.shuffle(self.wall)
        self.hands = [[] for _ in range(4)]
        self.discards = [[] for _ in range(4)]
        self.melds = [[] for _ in range(4)]
        self.dealer = 0  # Player 1 starts as dealer
        self.current_player = self.dealer
        self.wind_round = 0

    def deal(self):
        for i, hand in enumerate(self.hands):
            count = 14 if i == self.dealer else 13
            for _ in range(count):
                hand.append(self.wall.pop())

    def draw_tile(self):
        if not self.wall:
            print("Wall empty! Round ends in a draw.")
            return None
        tile = self.wall.pop()
        self.hands[self.current_player].append(tile)
        print(f"Player {self.current_player + 1} draws {self.format_tile(tile)}")
        return tile

    def discard_tile(self, tile):
        if tile not in self.hands[self.current_player]:
            print(f"Error: Tile not in player {self.current_player + 1}'s hand!")
            return False
        self.hands[self.current_player].remove(tile)
        self.discards[self.current_player].append(tile)
        print(f"Player {self.current_player + 1} discards {self.format_tile(tile)}")
        # Note: turn advance only after calls checked
        return True

    def format_tile(self, tile):
        if tile[0] in ['bamboo', 'character', 'dot']:
            return f"{tile[1]} {tile[0]}"
        return f"{tile[1].capitalize()} {tile[0].capitalize()}"

    # Display methods
    def show_hands(self):
        for idx, hand in enumerate(self.hands):
            print(f"Player {idx + 1} hand: {[self.format_tile(t) for t in hand]}")

    def show_discards(self):
        for idx, discards in enumerate(self.discards):
            print(f"Player {idx + 1} discards: {[self.format_tile(t) for t in discards]}")

    def show_melds(self):
        for idx, meld_list in enumerate(self.melds):
            if meld_list:
                meld_repr = [[self.format_tile(t) for t in meld] for meld in meld_list]
                print(f"Player {idx + 1} melds: {meld_repr}")
            else:
                print(f"Player {idx + 1} has no melds.")

    # Tile checking methods
    def is_pong(self, tiles):
        return len(tiles) == 3 and len(set(tiles)) == 1

    def is_chi(self, tiles):
        if len(tiles) != 3: return False
        suits = [t[0] for t in tiles]
        if suits[0] not in ['bamboo', 'character', 'dot'] or len(set(suits)) != 1: return False
        ranks = sorted([t[1] for t in tiles])
        return ranks[1] == ranks[0] + 1 and ranks[2] == ranks[1] + 1

    def is_pair(self, tiles):
        return len(tiles) == 2 and tiles[0] == tiles[1]

    def check_win(self, hand):
        hand = sorted(hand)
        if len(hand) != 14: return False

        def remove_sets(hand, sets_count):
            if sets_count == 4 and len(hand) == 2 and self.is_pair(hand):
                return True
            if sets_count > 4:
                return False

            for tile in set(hand):
                if hand.count(tile) >= 3:
                    tmp = hand.copy()
                    for _ in range(3): tmp.remove(tile)
                    if remove_sets(tmp, sets_count + 1): return True

                if tile[0] in ['bamboo', 'character', 'dot']:
                    next1 = (tile[0], tile[1] + 1)
                    next2 = (tile[0], tile[1] + 2)
                    if next1 in hand and next2 in hand:
                        tmp = hand.copy()
                        tmp.remove(tile)
                        tmp.remove(next1)
                        tmp.remove(next2)
                        if remove_sets(tmp, sets_count + 1): return True
            return False

        pairs = [tile for tile, count in Counter(hand).items() if count >= 2]
        for pair_tile in pairs:
            temp_hand = hand.copy()
            temp_hand.remove(pair_tile)
            temp_hand.remove(pair_tile)
            if remove_sets(temp_hand, 0):
                return True
        return False

    def get_hints(self, player_index):
        hints = []
        hand = self.hands[player_index]
        counts = Counter(hand)

        for tile, count in counts.items():
            if count >= 3:
                hints.append(f"Pong possible with {self.format_tile(tile)}")

        for tile in hand:
            if tile[0] in ['bamboo', 'character', 'dot']:
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
                for _ in range(3): leftover.remove(tile)
                count -= 3
        i = 0
        while i < len(leftover):
            tile = leftover[i]
            if tile[0] in ['bamboo', 'character', 'dot']:
                suit, rank = tile
                seq = [(suit, rank), (suit, rank+1), (suit, rank+2)]
                if all(s in leftover for s in seq):
                    sets.append(seq)
                    for s in seq: leftover.remove(s)
                    continue
            i += 1
        return sets, leftover

    def can_pong(self, player_index, tile):
        return self.hands[player_index].count(tile) >= 2

    def perform_pong(self, player_index, discarder_index, tile):
        if not self.can_pong(player_index, tile):
            return False
        for _ in range(2):
            self.hands[player_index].remove(tile)
        if self.discards[discarder_index]:
            self.discards[discarder_index].pop()
        self.melds[player_index].append([tile] * 3)
        print(f"Player {player_index + 1} calls Pong on {self.format_tile(tile)}")
        self.current_player = player_index
        return True

    def can_chi(self, player_index, tile):
        if player_index != (self.current_player + 1) % 4:
            return False  # Chi only by next player
        suit, rank = tile
        if suit not in ['bamboo', 'character', 'dot']:
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
                for t in seq:
                    self.hands[player_index].remove(t)
                if self.discards[discarder_index]:
                    self.discards[discarder_index].pop()
                self.melds[player_index].append([tile] + seq)
                print(f"Player {player_index + 1} calls Chi on {self.format_tile(tile)} with sequence {[self.format_tile(t) for t in seq]}")
                self.current_player = player_index
                return True
        return False

    def can_kong(self, player_index, tile):
        return self.hands[player_index].count(tile) == 3

    def perform_kong(self, player_index, discarder_index, tile):
        if not self.can_kong(player_index, tile):
            return False
        for _ in range(3):
            self.hands[player_index].remove(tile)
        if self.discards[discarder_index]:
            self.discards[discarder_index].pop()
        self.melds[player_index].append([tile] * 4)
        print(f"Player {player_index + 1} calls Kong on {self.format_tile(tile)}")
        self.current_player = player_index
        return True

    def can_hu(self, player_index):
        combined_hand = self.hands[player_index][:]
        for meld in self.melds[player_index]:
            combined_hand.extend(meld)
        return self.check_win(sorted(combined_hand))

def choose_discard_tile(game):
    hand = game.hands[game.current_player]
    print("\nYour hand:")
    for i, tile in enumerate(hand):
        print(f"{i + 1}: {game.format_tile(tile)}")
    while True:
        choice = input(f"Player {game.current_player + 1}, choose tile to discard (1-{len(hand)}): ")
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(hand):
                return hand[idx]
        print("Invalid choice. Please try again.")

def yes_no_prompt(prompt):
    while True:
        ans = input(prompt + " (y/n): ").strip().lower()
        if ans in ("y", "n"):
            return ans == "y"
        print("Please enter 'y' or 'n'.")

def ai_choose_discard(hand):
    counts = Counter(hand)
    tile_scores = {}
    for tile in hand:
        score = 0
        score -= counts[tile] * 2
        suit, rank = tile
        if suit in ["bamboo", "character", "dot"]:
            for adj in [(suit, rank - 1), (suit, rank + 1)]:
                if adj in hand:
                    score -= 1
        tile_scores[tile] = score
    worst_tile = max(tile_scores, key=tile_scores.get)
    return worst_tile

def main():
    import time
    game = MahjongGame()
    game.deal()
    print("\nInitial Hands:")
    game.show_hands()
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
        else:
            drawn = game.draw_tile()
            if drawn is None:
                print("Wall exhausted. Game ends in draw.")
                break

        hints = game.get_hints(game.current_player)
        if hints and not (first_turn and game.current_player == game.dealer):
            print("Hints:")
            for h in hints:
                print("  -", h)

        if game.current_player == 0:
            if game.can_hu(0):
                if yes_no_prompt("You can Hu! Declare win?"):
                    print("You declared Hu and won!")
                    break
            discard_tile = choose_discard_tile(game)
        else:
            if game.can_hu(game.current_player):
                print(f"Player {game.current_player + 1} declares Hu and wins!")
                break
            discard_tile = ai_choose_discard(game.hands[game.current_player])
            print(f"Player {game.current_player + 1} (AI) discards {game.format_tile(discard_tile)}")

        game.discard_tile(discard_tile)

        discarder = (game.current_player - 1) % 4

        kong_called = False
        for i in range(1, 4):
            player = (discarder + i) % 4
            if game.can_kong(player, discard_tile):
                if player == 0 or yes_no_prompt(f"Player {player + 1}, Kong possible with {game.format_tile(discard_tile)}. Call Kong?"):
                    kong_called = game.perform_kong(player, discarder, discard_tile)
                    if kong_called:
                        print(f"Kong called by Player {player + 1}, turn goes to them.")
                        break

        pong_called = False
        if not kong_called:
            for i in range(1, 4):
                player = (discarder + i) % 4
                if game.can_pong(player, discard_tile):
                    if player == 0 or yes_no_prompt(f"Player {player + 1}, Pong possible with {game.format_tile(discard_tile)}. Call Pong?"):
                        pong_called = game.perform_pong(player, discarder, discard_tile)
                        if pong_called:
                            print(f"Pong called by Player {player + 1}, turn goes to them.")
                            break

        if not kong_called and not pong_called:
            next_player = (discarder + 1) % 4
            if game.can_chi(next_player, discard_tile):
                if next_player == 0 or yes_no_prompt(f"Player {next_player + 1}, Chi possible with {game.format_tile(discard_tile)}. Call Chi?"):
                    chi_called = game.perform_chi(next_player, discarder, discard_tile)
                    if chi_called:
                        print(f"Chi called by Player {next_player + 1}, turn goes to them.")

        game.show_hands()
        game.show_discards()
        game.show_melds()

        if first_turn:
            first_turn = False

        time.sleep(1)

    print("\nGame Over.")

if __name__ == "__main__":
    main()
