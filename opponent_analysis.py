"""Track public Mahjong information and estimate discard danger.

Danger and opponent threat are heuristic estimates, not knowledge of hidden
hands or calibrated win probabilities. The simulator does not enforce Riichi
furiten, so previous discards do not guarantee safety. Tile values use the
tuple format from ``hand_eval`` and the simulator.
"""

from collections import Counter
import random

from hand_eval import ALL_TILE_TYPES, NUMBER_SUITS, is_sequence, is_triplet


COPIES_PER_TILE = 4
_VALID_TILES = set(ALL_TILE_TYPES)


class OpponentAnalyzer:
	"""Keep a public-information snapshot and assess opponent discard risk."""

	def __init__(self, player_count=4, player_index=0):
		if type(player_count) is not int or player_count < 2:
			raise ValueError("player_count must be at least 2")
		if type(player_index) is not int or not 0 <= player_index < player_count:
			raise ValueError("player_index must identify a player at the table")

		self.player_count = player_count
		self.player_index = player_index
		self.discards = [[] for _ in range(player_count)]
		self.melds = [[] for _ in range(player_count)]
		self.own_hand = []
		self.wall_remaining = None

	def update_from_game(self, game, player_index=None):
		"""Refresh from a simulator with ``hands``, ``discards``, ``melds`` and ``wall``."""
		seat = self.player_index if player_index is None else player_index
		self._check_player(seat)
		if any(len(rows) != self.player_count
		       for rows in (game.hands, game.discards, game.melds)):
			raise ValueError("game player count does not match this analyzer")

		# Validate a complete snapshot before changing any existing state.
		discards = [list(pile) for pile in game.discards]
		melds = [[list(meld) for meld in player_melds] for player_melds in game.melds]
		own_hand = list(game.hands[seat])
		for pile in discards:
			self._check_tiles(pile)
		for player_melds in melds:
			for meld in player_melds:
				self._check_meld(meld)
		self._check_tiles(own_hand)
		wall_remaining = len(game.wall)
		self.player_index = seat
		self.discards = discards
		self.melds = melds
		self.own_hand = own_hand
		self.wall_remaining = wall_remaining
		return self

	def record_discard(self, player_index, tile):
		"""Record a discard when not using ``update_from_game`` snapshots."""
		self._check_player(player_index)
		self._check_tile(tile)
		self.discards[player_index].append(tile)

	def record_meld(self, player_index, tiles, *, discarder_index):
		"""Record a new meld, moving a called tile out of its discard pile.

		Pass the discarder seat for a call, or explicitly pass None for a
		meld made entirely from hand. Include all meld tiles. When recording
		our own actions manually, also keep own_hand up to date.
		"""
		self._check_player(player_index)
		meld = list(tiles)
		self._check_meld(meld)
		if discarder_index is not None:
			self._check_player(discarder_index)
			if discarder_index == player_index:
				raise ValueError("a player cannot call their own discard")
			pile = self.discards[discarder_index]
			if not pile or pile[-1] not in meld:
				raise ValueError("meld must include the discarder's latest tile")
			pile.pop()
		self.melds[player_index].append(meld)

	@property
	def visible_tiles(self):
		"""All public discards and exposed meld tiles."""
		tiles = [tile for pile in self.discards for tile in pile]
		tiles.extend(tile for player_melds in self.melds for meld in player_melds for tile in meld)
		return tiles

	def remaining_counts(self, hand=None, visible_tiles=None):
		"""Return unseen copies by tile, excluding public tiles and our hand.

		These counts include tiles in opponents' concealed hands; they are not
		counts of tiles remaining in the wall.
		"""
		known_hand = self.own_hand if hand is None else list(hand)
		self._check_tiles(known_hand)
		known_visible = self.visible_tiles if visible_tiles is None else list(visible_tiles)
		self._check_tiles(known_visible)
		seen = Counter(known_visible)
		seen.update(known_hand)
		return {tile: max(0, COPIES_PER_TILE - seen[tile]) for tile in ALL_TILE_TYPES}

	def remaining_count(self, tile, hand=None, visible_tiles=None):
		self._check_tile(tile)
		return self.remaining_counts(hand, visible_tiles)[tile]

	def sample_wall(self, rng=None):
		"""Sample a possible wall using public unseen tiles, without replacement."""
		if self.wall_remaining is None:
			return []
		rng = rng or random
		pool = [
			tile
			for tile, count in self.remaining_counts().items()
			for _ in range(count)
		]
		return rng.sample(pool, min(self.wall_remaining, len(pool)))

	def opponent_threat(self, opponent_index):
		"""Estimate readiness from exposed melds and discard count (0.0 to 1.0)."""
		self._check_player(opponent_index)
		if opponent_index == self.player_index:
			return 0.0
		meld_count = len(self.melds[opponent_index])
		discard_count = len(self.discards[opponent_index])
		late_hand_signal = max(0, discard_count - 5) * 0.035
		return min(0.9, meld_count * 0.25 + late_hand_signal)

	def threat_level(self):
		"""Combine opponent readiness and late-wall pressure into 0.0-1.0."""
		threat = max(
			(self.opponent_threat(player) for player in range(self.player_count)
			 if player != self.player_index),
			default=0.0,
		)
		if self.wall_remaining is not None and self.wall_remaining < 30:
			threat += 0.2
		return min(1.0, threat)

	@classmethod
	def threat_from_melds(cls, melds_by_player, player_index, wall_remaining=None):
		"""Compatibility estimate for callers that only have meld and wall data."""
		threat = max(
			(0.25 * len(melds) for player, melds in enumerate(melds_by_player)
			 if player != player_index),
			default=0.0,
		)
		if wall_remaining is not None and wall_remaining < 30:
			threat += 0.2
		return min(1.0, threat)

	def opponent_danger(self, tile, opponent_index, hand=None, visible_tiles=None):
		"""Estimate a heuristic 0.0-1.0 danger score from one opponent for discarding ``tile``."""
		self._check_tile(tile)
		self._check_player(opponent_index)
		if opponent_index == self.player_index:
			raise ValueError("opponent_index must not be the analyzing player")

		counts = self.remaining_counts(hand, visible_tiles)
		base_risk = self._base_tile_risk(tile)
		if len(self.melds[opponent_index]) >= 4:
			# Four exposed sets leave only a pair to complete, never a sequence.
			unseen_scale = min(1.0, counts[tile] / 3)
		else:
			unseen_scale = self._unseen_risk_scale(tile, counts)
		threat_scale = 0.35 + 0.65 * self.opponent_threat(opponent_index)
		return round(min(1.0, base_risk * unseen_scale * threat_scale), 3)

	def danger_estimate(self, tile, hand=None, visible_tiles=None):
		"""Combine per-opponent estimates into a 0.0-1.0 discard danger score."""
		# Materialize iterators once so every opponent sees the same information.
		hand = None if hand is None else list(hand)
		visible_tiles = None if visible_tiles is None else list(visible_tiles)
		risks = [
			self.opponent_danger(tile, player, hand, visible_tiles)
			for player in range(self.player_count)
			if player != self.player_index
		]
		combined_risk = 1.0
		for risk in risks:
			combined_risk *= 1.0 - risk
		return round(1.0 - combined_risk, 3)

	@classmethod
	def danger_from_visible(cls, tile, hand, visible_tiles):
		"""Fallback estimate when per-player discard histories are unavailable."""
		cls._check_tile(tile)
		counts = cls().remaining_counts(hand, visible_tiles)
		return round(cls._base_tile_risk(tile) * cls._unseen_risk_scale(tile, counts), 3)

	@staticmethod
	def _unseen_risk_scale(tile, counts):
		"""Allow sequence waits even when no matching copies are unseen.

		The sequence floor is a heuristic, not a measured probability.
		This follows the evaluator's standard four-sets-and-a-pair hands.
		"""
		scale = min(1.0, counts[tile] / 3)
		suit, value = tile
		if suit in NUMBER_SUITS:
			for start in range(max(1, value - 2), min(value, 7) + 1):
				neighbors = [(suit, rank) for rank in range(start, start + 3)
				             if rank != value]
				if all(counts[neighbor] > 0 for neighbor in neighbors):
					return max(scale, 1 / 3)
		return scale

	@staticmethod
	def _base_tile_risk(tile):
		suit, value = tile
		if suit not in NUMBER_SUITS:
			return 0.42
		if value in (1, 9):
			return 0.34
		if value in (2, 8):
			return 0.5
		return 0.68

	def summary(self):
		"""Return a compact snapshot suitable for logging or a strategy engine."""
		return {
			"visible_tiles": self.visible_tiles,
			"remaining_counts": self.remaining_counts(),
			"wall_remaining": self.wall_remaining,
			"opponent_threats": {
				player: self.opponent_threat(player)
				for player in range(self.player_count)
				if player != self.player_index
			},
		}

	def _check_player(self, player_index):
		if type(player_index) is not int or not 0 <= player_index < self.player_count:
			raise ValueError("player_index is outside the table")

	@staticmethod
	def _check_tile(tile):
		if (not isinstance(tile, tuple) or len(tile) != 2
		    or type(tile[0]) is not str
		    or (type(tile[1]) is not int if tile[0] in NUMBER_SUITS
		        else type(tile[1]) is not str)
		    or tile not in _VALID_TILES):
			raise ValueError(f"invalid Mahjong tile: {tile!r}")

	def _check_tiles(self, tiles):
		for tile in tiles:
			self._check_tile(tile)

	def _check_meld(self, meld):
		self._check_tiles(meld)
		if not (is_sequence(meld) or is_triplet(meld)
		        or (len(meld) == 4 and len(set(meld)) == 1)):
			raise ValueError("meld must be a sequence, triplet, or quad")
