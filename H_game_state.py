"""Player observation: only the player's private tiles and public game information."""
from collections import Counter
from dataclasses import dataclass

from Decision_Strategy_engine import ALL_TILE_TYPES


@dataclass(frozen=True)
class GameState:
    player: int
    hand: tuple
    own_melds: tuple
    discards: tuple
    melds: tuple
    bonus: tuple
    wall_remaining: int
    dealer: int
    prevailing_wind: str
    seat: int
    current_player: int

    @property
    def visible_tiles(self):
        return tuple(t for pile in self.discards for t in pile) + tuple(
            t for player_melds in self.melds for meld in player_melds for t in meld
        ) + tuple(t for pile in self.bonus for t in pile)

    @property
    def unseen_counts(self):
        known = Counter(self.hand) + Counter(self.visible_tiles)
        return tuple(max(0, 4 - known[t]) for t in ALL_TILE_TYPES)

    @property
    def legal_discards(self):
        return tuple(dict.fromkeys(self.hand))

    def vector(self):
        """Fixed-size numeric observation for a future trainable model."""
        own = Counter(self.hand)
        discards = [Counter(pile) for pile in self.discards]
        exposed = [Counter(t for meld in player for t in meld) for player in self.melds]
        return tuple(own[t] for t in ALL_TILE_TYPES) + tuple(
            pile[t] for pile in discards + exposed for t in ALL_TILE_TYPES
        ) + self.unseen_counts + (self.wall_remaining, self.seat, self.dealer, self.current_player)


def observe(game, player):
    if player not in range(4):
        raise ValueError('player must be 0, 1, 2, or 3')
    return GameState(
        player, tuple(game.hands[player]),
        tuple(tuple(m) for m in game.melds[player]),
        tuple(tuple(d) for d in game.discards),
        tuple(tuple(tuple(m) for m in p) for p in game.melds),
        tuple(tuple(b) for b in game.bonus),
        len(game.wall), game.dealer, game.prevailing_wind,
        game.seat(player), game.current_player,
    )
