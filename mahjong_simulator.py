import random
from collections import Counter

bamboo = [('bamboo', i) for i in range(1, 10)]
characters = [('character', i) for i in range(1, 10)]
winds = [('wind', direction) for direction in ['east', 'south', 'west', 'north']]
dragons = [('dragon', color) for color in ['red', 'green', 'white']]
tiles = bamboo + characters + winds + dragons

class MahjongGame:
    def __init__(self):
        self.wall = tiles * 4
        random.shuffle(self.wall)
        self.hands = [[] for _ in range(4)]
        self.discards = [[] for _ in range(4)]
        self.current_player = 0

    def deal(self):
        for _ in range(13):
            for hand in self.hands:
                hand.append(self.wall.pop())

    def draw_tile(self):
        if not self.wall:
            print("Wall is empty, game over!")
            return None
        tile = self.wall.pop()
        self.hands[self.current_player].append(tile)
        print(f"Player {self.current_player + 1} draws {self.format_tile(tile)}")
        return tile

    def discard_tile(self, tile):
        if tile in self.hands[self.current_player]:
            self.hands[self.current_player].remove(tile)
            self.discards[self.current_player].append(tile)
            print(f"Player {self.current_player + 1} discards {self.format_tile(tile)}")
            self.current_player = (self.current_player + 1) % 4
        else:
            print(f"Player {self.current_player + 1} cannot discard {self.format_tile(tile)} (not in hand)")

    def format_tile(self, tile):
        if tile[0] in ['bamboo', 'character']:
            return f"{tile[1]} {tile[0]}"
        else:
            return f"{tile[1].capitalize()} {tile[0].capitalize()}"

    def show_hands(self):
        for i, hand in enumerate(self.hands):
            formatted_hand = [self.format_tile(tile) for tile in hand]
            print(f"Player {i+1} hand: {formatted_hand}")

    def show_discards(self):
        for i, discard in enumerate(self.discards):
            formatted_discard = [self.format_tile(tile) for tile in discard]
            print(f"Player {i+1} discards: {formatted_discard}")

    def is_pong(self, tiles):
        return len(tiles) == 3 and len(set(tiles)) == 1

    def is_chi(self, tiles):
        if len(tiles) != 3:
            return False
        suits = [t[0] for t in tiles]
        if suits[0] not in ['bamboo', 'character'] or len(set(suits)) != 1:
            return False
        ranks = sorted([t[1] for t in tiles])
        return ranks[1] == ranks[0] + 1 and ranks[2] == ranks[1] + 1

    def is_pair(self, tiles):
        return len(tiles) == 2 and tiles[0] == tiles[1]

    def check_win(self, hand):
        hand = sorted(hand)
        if len(hand) != 14:
            return False

        def remove_sets(hand, sets_needed):
            if sets_needed == 4 and len(hand) == 2 and self.is_pair(hand):
                return True
            if sets_needed > 4:
                return False

            for tile in set(hand):
                if hand.count(tile) >= 3:
                    temp_hand = hand.copy()
                    for _ in range(3):
                        temp_hand.remove(tile)
                    if remove_sets(temp_hand, sets_needed + 1):
                        return True

                if tile[0] in ['bamboo', 'character']:
                    next1 = (tile[0], tile[1] + 1)
                    next2 = (tile[0], tile[1] + 2)
                    if next1 in hand and next2 in hand:
                        temp_hand = hand.copy()
                        temp_hand.remove(tile)
                        temp_hand.remove(next1)
                        temp_hand.remove(next2)
                        if remove_sets(temp_hand, sets_needed + 1):
                            return True

            return False

        counts = Counter(hand)
        pairs = [tile for tile, count in counts.items() if count >= 2]
        for pair_tile in pairs:
            temp_hand = hand.copy()
            temp_hand.remove(pair_tile)
            temp_hand.remove(pair_tile)
            if remove_sets(temp_hand, 0):
                return True
        return False

    def get_hints(self, player_index):
        hand = self.hands[player_index]
        hints = []
        counts = Counter(hand)

        for tile, count in counts.items():
            if count >= 3:
                hints.append(f"Pong possible with {self.format_tile(tile)}")

        for tile in hand:
            if tile[0] in ['bamboo', 'character']:
                needed1 = (tile[0], tile[1] + 1)
                needed2 = (tile[0], tile[1] + 2)
                if needed1 in hand and needed2 in hand:
                    hints.append(f"Chi possible starting at {self.format_tile(tile)}")

        if len(hand) == 13:
            for tile in tiles:
                test_hand = hand + [tile]
                if self.check_win(sorted(test_hand)):
                    hints.append(f"Waiting for {self.format_tile(tile)} to Hu")

        return hints

    def find_triplets(self, player_index):
        hand = self.hands[player_index]
        counts = Counter(hand)
        triplets = [tile for tile, count in counts.items() if count >=3]
        return triplets

    def can_pong(self, player_index, tile):
        return self.hands[player_index].count(tile) >= 2

    def perform_pong(self, player_index, discarder_index, tile):
        if self.can_pong(player_index, tile):
            for _ in range(2):
                self.hands[player_index].remove(tile)
            if self.discards[discarder_index]:
                self.discards[discarder_index].pop()
            else:
                print(f"Warning: discard pile of player {discarder_index+1} empty during Pong")
            print(f"Player {player_index + 1} calls Pong on {self.format_tile(tile)}")
            self.current_player = player_index
            return True
        return False

    def can_chi(self, player_index, tile):
        if player_index != (self.current_player + 1) % 4:
            return False
        suit, rank = tile
        if suit not in ['bamboo', 'character']:
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
                else:
                    print(f"Warning: discard pile of player {discarder_index+1} empty during Chi")
                print(f"Player {player_index + 1} calls Chi on {self.format_tile(tile)} with sequence {[self.format_tile(t) for t in seq]}")
                self.current_player = player_index
                return True
        return False

    def can_kong(self, player_index, tile):
        return self.hands[player_index].count(tile) == 3

    def perform_kong(self, player_index, discarder_index, tile):
        if self.can_kong(player_index, tile):
            for _ in range(3):
                self.hands[player_index].remove(tile)
            if self.discards[discarder_index]:
                self.discards[discarder_index].pop()
            else:
                print(f"Warning: discard pile of player {discarder_index+1} empty during Kong")
            print(f"Player {player_index + 1} calls Kong on {self.format_tile(tile)}")
            self.current_player = player_index
            return True
        return False

    def can_hu(self, player_index):
        hand = self.hands[player_index]
        return self.check_win(sorted(hand))


def choose_discard_tile(game):
    hand = game.hands[game.current_player]
    print("\nYour hand:")
    for i, tile in enumerate(hand):
        print(f"{i + 1}: {game.format_tile(tile)}")
    while True:
        choice = input(f"Player {game.current_player + 1}, choose tile to discard (1-{len(hand)}): ")
        if choice.isdigit():
            index = int(choice) - 1
            if 0 <= index < len(hand):
                return hand[index]
        print("Invalid choice. Please try again.")

def yes_no_prompt(prompt):
    while True:
        answer = input(prompt + " (y/n): ").strip().lower()
        if answer in ('y', 'n'):
            return answer == 'y'
        print("Please enter 'y' or 'n'.")


def main():
    import time
    game = MahjongGame()
    game.deal()
    print("\nInitial Hands:")
    game.show_hands()
    print("="*50)

    while True:
        print("\n" + "="*50)
        print(f"--- Player {game.current_player + 1}'s turn ---")
        print("="*50 + "\n")

        # Show completed triplets before turn
        triplets = game.find_triplets(game.current_player)
        if triplets:
            print(f"Completed triplets: {[game.format_tile(t) for t in triplets]}")

        drawn_tile = game.draw_tile()
        if drawn_tile is None:
            print("All tiles drawn, game ends with no winner.")
            break

        hints = game.get_hints(game.current_player)
        if hints:
            print(f"Hints for Player {game.current_player + 1}:")
            for hint in hints:
                print("  -", hint)

        # Check Hu before discard - can player declare win now?
        if game.can_hu(game.current_player):
            if yes_no_prompt(f"Player {game.current_player + 1}, you can Hu! Declare win?"):
                print(f"\nPlayer {game.current_player + 1} declares Hu and wins!")
                break

        # Interactive discard choice
        discard_tile = choose_discard_tile(game)
        print(f"\nPlayer {game.current_player + 1} discards {game.format_tile(discard_tile)}\n")
        game.discard_tile(discard_tile)

        pong_called = False
        kong_called = False

        # Kong calls (priority highest after discard)
        for i in range(1, 4):
            player = (game.current_player + i) % 4
            if game.can_kong(player, discard_tile):
                if yes_no_prompt(f"Player {player + 1}, Kong possible with {game.format_tile(discard_tile)}. Call Kong?"):
                    kong_called = game.perform_kong(player, game.current_player, discard_tile)
                    print(f"Kong called by Player {player + 1}, turn goes to them.")
                    break

        # Pong calls (priority after Kong)
        if not kong_called:
            for i in range(1, 4):
                player = (game.current_player + i) % 4
                if game.can_pong(player, discard_tile):
                    if yes_no_prompt(f"Player {player + 1}, Pong possible with {game.format_tile(discard_tile)}. Call Pong?"):
                        pong_called = game.perform_pong(player, game.current_player, discard_tile)
                        print(f"Pong called by Player {player + 1}, turn goes to them.")
                        break

        # Chi call (only next player, if no Pong or Kong)
        if not kong_called and not pong_called:
            next_player = (game.current_player + 1) % 4
            if game.can_chi(next_player, discard_tile):
                if yes_no_prompt(f"Player {next_player + 1}, Chi possible with {game.format_tile(discard_tile)}. Call Chi?"):
                    chi_called = game.perform_chi(next_player, game.current_player, discard_tile)
                    print(f"Chi called by Player {next_player + 1}, turn goes to them.")

        game.show_hands()
        game.show_discards()

        time.sleep(1)

    print("\nGame Over.")


if __name__ == "__main__":
    main()
 