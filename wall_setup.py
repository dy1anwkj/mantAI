"""Three-dice wall break and physical packet dealing for a 148-tile set."""


class PhysicalWall(list):
    """Keep public stack occupancy in sync with front and replacement draws."""
    def __init__(self, tiles, slots):
        super().__init__(tiles)
        self.slots = list(slots)
        self.draws = []
        self.recipient = None
        self.player_at_draw = lambda: None

    def pop(self, index=-1):
        tile = super().pop(index)
        slot = self.slots.pop(index)
        self.draws.append({"slot": slot, "end": "replacement" if index == 0 else "live",
                           "player": self.recipient if self.recipient is not None else self.player_at_draw()})
        return tile


def deal_three_dice(game, dice=None):
    if len(game.wall) != 148 or any(game.hands):
        raise ValueError("Build the wall before dealing any tiles.")
    dice = list(dice) if dice is not None else [game.rand.randint(1, 6) for _ in range(3)]
    if len(dice) != 3 or any(type(d) is not int or not 1 <= d <= 6 for d in dice):
        raise ValueError("Roll exactly three dice with values from 1 to 6.")
    total = sum(dice)
    selected = (game.dealer + total - 1) % 4
    stacks = [19 if (p - game.dealer) % 2 == 0 else 18 for p in range(4)]
    # Each side is indexed from its owner's RIGHT to LEFT. At the left
    # corner continue onto that owner's left neighbour (player - 1).
    ring = [(p, stack, level) for p in (0, 3, 2, 1)
            for stack in range(1, stacks[p] + 1) for level in (0, 1)]
    side_start = next(i for i, slot in enumerate(ring) if slot[0] == selected)
    cut = (side_start + 2 * total) % len(ring)
    draw_slots = ring[cut:] + ring[:cut]
    physical_tiles = list(game.wall)
    ordered = physical_tiles[cut:] + physical_tiles[:cut]
    game.wall = PhysicalWall(reversed(ordered), reversed(draw_slots))
    game.wall.player_at_draw = lambda: game.current_player
    order = [(game.dealer + i) % 4 for i in range(4)]
    packets, cursor = [], 0

    def packet(player, count):
        nonlocal cursor
        slots = draw_slots[cursor:cursor + count]
        game.wall.recipient = player
        game.hands[player].extend(game.wall.pop() for _ in range(count))
        packets.append({"player": player, "count": count, "slots": slots})
        cursor += count

    for _ in range(3):
        for p in order:
            packet(p, 4)
    for p in order:
        packet(p, 1)
    packet(game.dealer, 1)
    game.wall_setup = {"dice": dice, "total": total, "banker": game.dealer,
                       "selected": selected, "stacks": stacks,
                       "gap_after": total, "first_stacks": draw_slots[:4],
                       "live_start": draw_slots[0], "replacement_start": draw_slots[-1],
                       "packets": packets, "order": order}
    # Preserve existing Singapore bonus replacement and payment logic.
    for p in order:
        game.wall.recipient = p
        if not game.replace_bonus_tiles(p):
            break
    game.wall_setup["replacements"] = list(game.wall.draws[53:])
    bonus_index = [0, 0, 0, 0]
    for draw in game.wall_setup["replacements"]:
        p = draw["player"]
        draw["exposed_bonus"] = game.bonus[p][bonus_index[p]]
        bonus_index[p] += 1
    game.wall.recipient = None
    game.check_invariants(discarding_player=game.dealer)
    return game.wall_setup
