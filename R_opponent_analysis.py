"""Track public Mahjong information and estimate how dangerous each discard is.

Danger is learned, not guessed (the approach Microsoft's Suphx used for
Mahjong). A small network reads only what this player can see about one
opponent - their discards (all and the latest six), their melds, the tiles
still unseen, the wall, their seat, the round and their bonus tiles - and
predicts, for each of the 34 tiles, the chance that opponent would win on it,
plus the chance they are ready at all. It was trained on simulated Singapore
games where the true answer is known exactly: which tiles would really have
completed each opponent's hand with enough tai (python R_opponent_analysis.py).

Without a trained model (models/danger.npz) the old rank-and-count guesses are
used instead. A previous discard is never treated as safe: Singapore has no
furiten, so the opponent's hand may have changed. Tile values use the tuple
format from ``D_hand_eval`` and the simulator.
"""

from collections import Counter
import os
import random

import numpy as np

from D_hand_eval import ALL_TILE_TYPES, NUMBER_SUITS, is_sequence, is_triplet
import singapore_rules as sg
from singapore_rules import BONUS_TILES


COPIES_PER_TILE = 4
_VALID_TILES = set(ALL_TILE_TYPES)
_BONUS_TILES = set(BONUS_TILES)

N = len(ALL_TILE_TYPES)
TILE_INDEX = {tile: i for i, tile in enumerate(ALL_TILE_TYPES)}
WINDS = ("east", "south", "west", "north")
RECENT = 6                       # latest discards read separately (a change of plan shows there)
FEATURE_SIZE = 4 * N + 4 + 4 + 4
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "danger.npz")


def opponent_features(discards, melds, unseen, wall_remaining, seat, prevailing, bonus_count):
	"""Everything public about one opponent, from one player's view, as numbers:
	34 their discards, 34 their latest discards, 34 their meld tiles, 34 copies
	still unseen by the viewer (all / 4), then melds, discards made, playable
	wall left and bonus tiles shown, then their seat wind and the round wind."""
	x = np.zeros(FEATURE_SIZE, dtype=np.float32)
	for tile in discards:
		x[TILE_INDEX[tile]] += 0.25
	for tile in discards[-RECENT:]:
		x[N + TILE_INDEX[tile]] += 0.25
	for meld in melds:
		for tile in meld:
			x[2 * N + TILE_INDEX[tile]] += 0.25
	for tile, count in unseen.items():
		x[3 * N + TILE_INDEX[tile]] = count / 4
	b = 4 * N
	x[b] = len(melds) / 4
	x[b + 1] = min(len(discards), 24) / 24
	x[b + 2] = 0.5 if wall_remaining is None else max(0, wall_remaining - sg.DEAD_WALL) / 100
	x[b + 3] = min(bonus_count, 4) / 4
	if seat:
		x[b + 4 + (seat - 1) % 4] = 1
	if prevailing in WINDS:
		x[b + 8 + WINDS.index(prevailing)] = 1
	return x


class DangerModel:
	"""The trained network (numpy, no torch needed to play). predict(X) gives,
	for each row, 34 chances of winning on each tile and one of being ready."""

	def __init__(self, layers, metrics=None):
		self.layers = layers
		self.metrics = metrics or {}

	@classmethod
	def load(cls, path=MODEL_PATH):
		"""The model at `path`, or None when it has not been trained yet."""
		if not path or not os.path.exists(path):
			return None
		data = np.load(path, allow_pickle=False)
		if int(data["feature_size"]) != FEATURE_SIZE:
			return None                        # trained on other features: retrain
		count = int(data["layers"])
		layers = [(data[f"w{i}"], data[f"b{i}"]) for i in range(count)]
		metrics = {k[2:]: float(data[k]) for k in data.files if k.startswith("m_")}
		return cls(layers, metrics)

	def save(self, path):
		arrays = {"layers": np.array(len(self.layers)), "feature_size": np.array(FEATURE_SIZE)}
		for i, (w, b) in enumerate(self.layers):
			arrays[f"w{i}"], arrays[f"b{i}"] = w, b
		for k, v in self.metrics.items():
			arrays[f"m_{k}"] = np.array(v)
		os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
		np.savez(path, **arrays)

	def predict(self, X):
		x = np.asarray(X, dtype=np.float32)
		for i, (w, b) in enumerate(self.layers):
			x = x @ w + b
			if i < len(self.layers) - 1:
				x = np.maximum(x, 0)
		return 1.0 / (1.0 + np.exp(-x))


_loaded = {}


def load_danger_model(path=MODEL_PATH):
	"""Loaded once per process and shared by every analyzer."""
	if path not in _loaded:
		_loaded[path] = DangerModel.load(path)
	return _loaded[path]


class OpponentAnalyzer:
	"""Keep a public-information snapshot and assess opponent discard risk."""

	def __init__(self, player_count=4, player_index=0, model="auto"):
		"""model: a DangerModel, None for the old guesses, or "auto" to use
		models/danger.npz when it has been trained."""
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
		self.bonus = None
		self.seats = None              # seat number (1 = East) of every player
		self.prevailing = None
		self.model = load_danger_model() if model == "auto" else model
		self._predictions = None

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
		bonus = self._copy_bonus(getattr(game, "bonus", None))
		self.player_index = seat
		self.discards = discards
		self.melds = melds
		self.own_hand = own_hand
		self.wall_remaining = wall_remaining
		self.bonus = bonus
		self.seats = [game.seat(p) for p in range(self.player_count)] if hasattr(game, "seat") else None
		self.prevailing = getattr(game, "prevailing_wind", None)
		self._predictions = None
		return self

	def update_from_state(self, state):
		"""Refresh from a public GameState without accessing opponents' hands."""
		self._check_player(state.player)
		if len(state.discards) != self.player_count or len(state.melds) != self.player_count:
			raise ValueError("state player count does not match this analyzer")
		if type(state.wall_remaining) is not int or state.wall_remaining < 0:
			raise ValueError("wall_remaining must be a nonnegative integer")
		discards = [list(pile) for pile in state.discards]
		melds = [[list(meld) for meld in player] for player in state.melds]
		own_hand = list(state.hand)
		for pile in discards:
			self._check_tiles(pile)
		for player_melds in melds:
			for meld in player_melds:
				self._check_meld(meld)
		self._check_tiles(own_hand)
		bonus = self._copy_bonus(getattr(state, "bonus", None))
		self.player_index = state.player
		self.discards = discards
		self.melds = melds
		self.own_hand = own_hand
		self.wall_remaining = state.wall_remaining
		self.bonus = bonus
		dealer = getattr(state, "dealer", None)
		self.seats = (None if dealer is None else
		              [sg.seat_number(p, dealer) for p in range(self.player_count)])
		self.prevailing = getattr(state, "prevailing_wind", None)
		self._predictions = None
		return self

	def record_discard(self, player_index, tile):
		"""Record a discard when not using ``update_from_game`` snapshots."""
		self._check_player(player_index)
		self._check_tile(tile)
		self.discards[player_index].append(tile)
		self._predictions = None

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
		self._predictions = None

	@property
	def visible_tiles(self):
		"""All public discards, exposed melds, and revealed bonus tiles."""
		tiles = [tile for pile in self.discards for tile in pile]
		tiles.extend(tile for player_melds in self.melds for meld in player_melds for tile in meld)
		if self.bonus is not None:
			tiles.extend(tile for pile in self.bonus for tile in pile)
		return tiles

	def remaining_counts(self, hand=None, visible_tiles=None):
		"""Return unseen copies by tile, excluding public tiles and our hand.

		These counts include tiles in opponents' concealed hands; they are not
		counts of tiles remaining in the wall. Only standard hand tiles are
		returned; revealed bonus tiles are accepted but do not reduce them.
		"""
		known_hand = self.own_hand if hand is None else list(hand)
		self._check_tiles(known_hand)
		known_visible = self.visible_tiles if visible_tiles is None else list(visible_tiles)
		for tile in known_visible:
			self._check_tile(tile, allow_bonus=True)
		seen = Counter(known_visible)
		seen.update(known_hand)
		return {tile: max(0, COPIES_PER_TILE - seen[tile]) for tile in ALL_TILE_TYPES}

	def remaining_count(self, tile, hand=None, visible_tiles=None):
		self._check_tile(tile)
		return self.remaining_counts(hand, visible_tiles)[tile]

	def sample_wall(self, rng=None):
		"""Sample a possible physical wall, including unrevealed bonus tiles.

		Refresh after bonus replacements finish. Unrevealed bonus tiles must
		be in the wall because players cannot keep them in concealed hands.
		Without bonus data, use the standard 136-tile model.
		"""
		if self.wall_remaining is None:
			return []
		rng = rng or random
		pool = [
			tile
			for tile, count in self.remaining_counts().items()
			for _ in range(count)
		]
		bonus = [] if self.bonus is None else [
			tile for tile in BONUS_TILES if tile not in self.visible_tiles
		]
		standard_count = self.wall_remaining - len(bonus)
		if not 0 <= standard_count <= len(pool):
			raise ValueError("wall size is inconsistent with the known tiles")
		wall = rng.sample(pool, standard_count) + bonus
		rng.shuffle(wall)
		return wall

	def opponents(self):
		return [p for p in range(self.player_count) if p != self.player_index]

	def features(self, opponent_index):
		"""The network's input for one opponent (see opponent_features)."""
		self._check_player(opponent_index)
		bonus = 0 if self.bonus is None else len(self.bonus[opponent_index])
		return opponent_features(
			self.discards[opponent_index], self.melds[opponent_index], self.remaining_counts(),
			self.wall_remaining, None if self.seats is None else self.seats[opponent_index],
			self.prevailing, bonus)

	def predictions(self):
		"""{opponent: 35 chances} from the model - win on each tile, then ready -
		computed once per table update."""
		if self.model is None:
			return None
		if self._predictions is None:
			rows = self.opponents()
			out = self.model.predict(np.stack([self.features(p) for p in rows]))
			self._predictions = dict(zip(rows, out))
		return self._predictions

	def ready_chance(self, opponent_index):
		"""How likely this opponent is to be one tile from a legal win."""
		self._check_player(opponent_index)
		if opponent_index == self.player_index:
			return 0.0
		p = self.predictions()
		if p is not None:
			return float(p[opponent_index][N])
		return self.opponent_threat(opponent_index)

	def deal_in_chance(self, tile):
		"""The chance that throwing `tile` lets some opponent win on it."""
		self._check_tile(tile)
		p = self.predictions()
		if p is None:
			return self.danger_estimate(tile) * self.threat_level()
		safe = 1.0
		for row in p.values():
			safe *= 1.0 - float(row[TILE_INDEX[tile]])
		return round(1.0 - safe, 4)

	def opponent_threat(self, opponent_index):
		"""How close this opponent looks to winning (0.0 to 1.0): the model's
		chance that they are ready, or a guess from melds and discards."""
		self._check_player(opponent_index)
		if opponent_index == self.player_index:
			return 0.0
		p = self.predictions()
		if p is not None:
			return float(p[opponent_index][N])
		meld_count = len(self.melds[opponent_index])
		discard_count = len(self.discards[opponent_index])
		late_hand_signal = max(0, discard_count - 5) * 0.035
		return min(0.9, meld_count * 0.25 + late_hand_signal)

	def threat_level(self):
		"""How close the most dangerous opponent is (0.0 to 1.0). The model
		already reads the wall, so the short-wall bump only applies to the guess."""
		threat = max((self.opponent_threat(p) for p in self.opponents()), default=0.0)
		if self.model is None and self.wall_remaining is not None and self.wall_remaining < 30:
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
		"""How dangerous `tile` is against one opponent (0.0 to 1.0): with the
		model, its chance of winning for them if they are ready."""
		self._check_tile(tile)
		self._check_player(opponent_index)
		if opponent_index == self.player_index:
			raise ValueError("opponent_index must not be the analyzing player")
		p = self.predictions()
		if p is not None:
			row = p[opponent_index]
			return round(min(1.0, float(row[TILE_INDEX[tile]]) / max(float(row[N]), 0.05)), 3)
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
		"""Danger of discarding `tile` (0.0 to 1.0) for StrategyEngine, which
		multiplies it by threat_level(). With the model it is the total
		deal-in chance divided by that threat, so the product is the chance
		that this discard pays out."""
		hand = None if hand is None else list(hand)
		visible_tiles = None if visible_tiles is None else list(visible_tiles)
		p = self.predictions()
		if p is not None:
			self._check_tile(tile)
			total = sum(float(row[TILE_INDEX[tile]]) for row in p.values())
			return round(min(1.0, total / max(self.threat_level(), 0.05)), 3)
		risks = [self.opponent_danger(tile, player, hand, visible_tiles) for player in self.opponents()]
		combined_risk = 1.0
		for risk in risks:
			combined_risk *= 1.0 - risk
		return round(1.0 - combined_risk, 3)

	@classmethod
	def danger_from_visible(cls, tile, hand, visible_tiles):
		"""Fallback estimate when per-player discard histories are unavailable."""
		cls._check_tile(tile)
		counts = cls(model=None).remaining_counts(hand, visible_tiles)
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
	def _check_tile(tile, allow_bonus=False):
		if (not isinstance(tile, tuple) or len(tile) != 2
		    or type(tile[0]) is not str
		    or (type(tile[1]) is not int if tile[0] in (*NUMBER_SUITS, "flower", "season")
		        else type(tile[1]) is not str)
		    or (tile not in _VALID_TILES and not (allow_bonus and tile in _BONUS_TILES))):
			raise ValueError(f"invalid Mahjong tile: {tile!r}")

	def _copy_bonus(self, bonus):
		if bonus is None:
			return None
		if len(bonus) != self.player_count:
			raise ValueError("bonus player count does not match this analyzer")
		copied = [list(pile) for pile in bonus]
		seen = set()
		for pile in copied:
			for tile in pile:
				self._check_tile(tile, allow_bonus=True)
				if tile not in _BONUS_TILES or tile in seen:
					raise ValueError("bonus tiles must be distinct flowers, seasons, or animals")
				seen.add(tile)
		return copied

	def _check_tiles(self, tiles):
		for tile in tiles:
			self._check_tile(tile)

	def _check_meld(self, meld):
		self._check_tiles(meld)
		if not (is_sequence(meld) or is_triplet(meld)
		        or (len(meld) == 4 and len(set(meld)) == 1)):
			raise ValueError("meld must be a sequence, triplet, or quad")


# --------------------------------------------------------------------------
# Training: the true answers come from simulated games
# --------------------------------------------------------------------------
def collect_examples(seeds):
	"""(features, answers) from one simulated game per seed. Before every
	discard, from the discarding player's view, one row per opponent: the
	public features, and the 34 tiles that would really let that opponent win
	(complete shape with enough tai) plus whether they are ready at all."""
	import mahjong_simulator as sim
	from D_hand_eval import winning_tiles

	X, Y = [], []
	for seed in seeds:
		game = sim.MahjongGame(rand=random.Random(seed))
		game.player1, game.verbose = "engine", False
		game.deal()
		session = sim.GameSession(game=game)
		analyzer = OpponentAnalyzer(model=None)
		while not session.is_over:
			d = session.decision
			if d.kind == "turn" and any(a.kind == "discard" for a in d.actions):
				analyzer.update_from_game(game, d.player)
				for q in analyzer.opponents():
					y = np.zeros(N + 1, dtype=np.float32)
					for tile in set(winning_tiles(game.hands[q], game.melds[q])):
						if tile in TILE_INDEX and game.can_hu(q, tile):
							y[TILE_INDEX[tile]] = 1
					y[N] = float(y[:N].any())
					X.append(analyzer.features(q))
					Y.append(y)
			session.apply(sim.choose_action(game, d))
	return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def train(out=MODEL_PATH, games=4000, epochs=8, hidden=256, workers=None, seed=0, log=print):
	"""Simulate `games` games, train the network on 90% of them and report on
	the other 10%, then save it to `out`. Needs torch (training only)."""
	from concurrent.futures import ProcessPoolExecutor
	import torch
	from torch import nn

	workers = workers or os.cpu_count() or 1
	master = random.Random(seed)
	seeds = [master.randrange(2 ** 62) for _ in range(games)]
	held_out = max(1, games // 10)
	log(f"simulating {games} games on {workers} processes ...")
	with ProcessPoolExecutor(max_workers=workers) as pool:
		train_parts = list(pool.map(collect_examples, [seeds[held_out:][i::workers] for i in range(workers)]))
		test_parts = list(pool.map(collect_examples, [seeds[:held_out][i::workers] for i in range(workers)]))
	X = torch.tensor(np.concatenate([x for x, _ in train_parts]))
	Y = torch.tensor(np.concatenate([y for _, y in train_parts]))
	XT = torch.tensor(np.concatenate([x for x, _ in test_parts]))
	YT = torch.tensor(np.concatenate([y for _, y in test_parts]))
	log(f"{len(X)} training rows, {len(XT)} test rows; "
	    f"opponents ready in {100 * float(Y[:, N].mean()):.1f}% of them")

	torch.manual_seed(seed)
	net = nn.Sequential(nn.Linear(FEATURE_SIZE, hidden), nn.ReLU(),
	                    nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, N + 1))
	opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
	loss_fn = nn.BCEWithLogitsLoss()
	best_loss, best_state = None, None
	for epoch in range(1, epochs + 1):
		net.train()
		perm = torch.randperm(len(X))
		for i in range(0, len(X), 1024):
			idx = perm[i:i + 1024]
			loss = loss_fn(net(X[idx]), Y[idx])
			opt.zero_grad()
			loss.backward()
			opt.step()
		net.eval()
		with torch.no_grad():
			test_loss = float(loss_fn(net(XT), YT))
		note = ""
		if best_loss is None or test_loss < best_loss:
			# Keep the round that predicts unseen games best, not the last one
			# (later rounds start memorising the training games).
			best_loss = test_loss
			best_state = {k: v.clone() for k, v in net.state_dict().items()}
			note = "  (best so far)"
		log(f"epoch {epoch:2d}  test loss {test_loss:.4f}{note}")
	net.load_state_dict(best_state)

	with torch.no_grad():
		P = torch.sigmoid(net(XT)).numpy()
	Yn = YT.numpy()
	ready = Yn[:, N] == 1
	tiles_p, tiles_y = P[ready, :N], Yn[ready, :N]
	metrics = {
		"test_rows": float(len(Yn)),
		"ready_rate": float(Yn[:, N].mean()),
		"ready_predicted": float(P[:, N].mean()),
		# For ready opponents: average chance given to tiles that win vs tiles that do not.
		"winning_tile_chance": float(tiles_p[tiles_y == 1].mean()) if tiles_y.any() else 0.0,
		"other_tile_chance": float(tiles_p[tiles_y == 0].mean()) if ready.any() else 0.0,
		# How often the tile it calls most dangerous really wins for a ready opponent.
		"top_tile_wins": float(tiles_y[np.arange(len(tiles_p)), tiles_p.argmax(1)].mean()) if ready.any() else 0.0,
		"heuristic_top_tile_wins": _heuristic_top_hit(XT.numpy()[ready], tiles_y),
		"games": float(games),
	}
	log("held-out games: ready {ready_rate:.1%} (predicted {ready_predicted:.1%}); for ready "
	    "opponents winning tiles got {winning_tile_chance:.1%} vs {other_tile_chance:.1%} for others; "
	    "its most dangerous tile really wins {top_tile_wins:.1%} of the time (old guesses: "
	    "{heuristic_top_tile_wins:.1%})".format(**metrics))
	layers = [(m.weight.detach().numpy().T.copy(), m.bias.detach().numpy().copy())
	          for m in net if isinstance(m, nn.Linear)]
	DangerModel(layers, metrics).save(out)
	_loaded.pop(out, None)
	log(f"saved to {out}")
	return metrics


def _heuristic_top_hit(X, Y):
	"""The same check for the old rank-and-count guesses, for comparison:
	among tiles still unseen, the most dangerous by rank is the middle one."""
	if not len(X):
		return 0.0
	risk = np.array([OpponentAnalyzer._base_tile_risk(t) for t in ALL_TILE_TYPES], dtype=np.float32)
	unseen = X[:, 3 * N:4 * N] * 4
	score = risk[None, :] * np.minimum(1.0, unseen / 3)
	return float(Y[np.arange(len(Y)), score.argmax(1)].mean())


if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(description="Train the danger model on simulated games.")
	parser.add_argument("--games", type=int, default=4000)
	parser.add_argument("--epochs", type=int, default=8)
	parser.add_argument("--workers", type=int)
	parser.add_argument("--out", default=MODEL_PATH)
	args = parser.parse_args()
	train(args.out, games=args.games, epochs=args.epochs, workers=args.workers)
