"""
Decision records: one JSON object per line (JSON lines).

Every source (simulator or video) writes the same fields, so they can be
mixed freely in training:

    source      "sim" or "video:<file name>"
    game_id     any string identifying the game
    player      0-3 (0 = Player 1); for video the filmed player is 0
    controller  who made the decision: engine / random / human / ml / video
    kind        "discard" or "call"
    tile        the tile discarded, or the tile a call was offered on
    call_type   pong / chi / kong / self_kong  (calls only)
    taken       true / false                   (calls only)
    hand        concealed tiles BEFORE the decision
    melds       the player's exposed melds
    visible     every other face-up tile (all discards and other players' melds)
    bonus       the player's flowers / seasons / animals
    seat        1-4 (1 = East / dealer)
    prevailing  prevailing wind name
    wall_left   tiles left in the wall (null if unknown, e.g. from video)
    outcome     filled in at the end of a game:
                {"won": bool, "self_drawn": bool, "tai": int, "draw": bool}
"""

import json

from .tiles import to_tile


def tile_list(tiles):
    return [list(t) for t in tiles]


def write_records(path, records, append=True):
    with open(path, "a" if append else "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def read_records(paths):
    """Yield records from one or more JSON-lines files, tiles as tuples."""
    if isinstance(paths, str):
        paths = [paths]
    for path in paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                r["tile"] = to_tile(r.get("tile"))
                r["hand"] = [to_tile(t) for t in r.get("hand", [])]
                r["melds"] = [[to_tile(t) for t in m] for m in r.get("melds", [])]
                r["visible"] = [to_tile(t) for t in r.get("visible", [])]
                r["bonus"] = [to_tile(t) for t in r.get("bonus", [])]
                yield r
