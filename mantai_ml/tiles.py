"""Tile vocabulary shared by the gameplay and video pipelines."""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from D_hand_eval import ALL_TILE_TYPES  # noqa: E402
import singapore_rules as sg  # noqa: E402

PLAY_TILES = list(ALL_TILE_TYPES)                 # 34 tiles a hand can hold
TILE_INDEX = {t: i for i, t in enumerate(PLAY_TILES)}
ALL_TILES = PLAY_TILES + list(sg.BONUS_TILES)     # 46 labels the camera can see


def tile_name(tile):
    """('bamboo', 5) -> 'bamboo_5'. Used as the folder name for tile images."""
    return f"{tile[0]}_{tile[1]}"


def parse_tile(name):
    """'bamboo_5' -> ('bamboo', 5), 'wind_east' -> ('wind', 'east')."""
    suit, value = name.split("_", 1)
    return (suit, int(value)) if value.isdigit() else (suit, value)


TILE_NAMES = [tile_name(t) for t in ALL_TILES]


def to_tile(value):
    """JSON gives lists; the game uses tuples."""
    return tuple(value) if value is not None else None
