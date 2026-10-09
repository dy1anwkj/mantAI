import itertools
import random
import unittest
from collections import Counter
from unittest.mock import patch

from mahjong_simulator import MahjongGame, all_tiles
from wall_setup import deal_three_dice


class WallSetupTests(unittest.TestCase):
    def test_all_dice_totals_and_bankers_follow_physical_wall(self):
        for dealer in range(4):
            for total in range(3, 19):
                dice = next(d for d in itertools.product(range(1, 7), repeat=3) if sum(d) == total)
                game = MahjongGame(dealer=dealer, wind_round=0, rand=random.Random(7))
                physical = list(game.wall)
                with patch.object(game, 'replace_bonus_tiles', return_value=True), patch.object(game, 'check_invariants'):
                    setup = deal_three_dice(game, dice)
                selected = (dealer + total - 1) % 4
                self.assertEqual(setup['selected'], selected)
                self.assertEqual(setup['gap_after'], total)
                self.assertEqual(sum(setup['stacks']), 74)
                ring = [(p, n, z) for p in (0, 3, 2, 1)
                        for n in range(1, setup['stacks'][p] + 1) for z in (0, 1)]
                cut = (ring.index((selected, 1, 0)) + total * 2) % 148
                ordered = physical[cut:] + physical[:cut]
                slots = ring[cut:] + ring[:cut]
                self.assertEqual(setup['first_stacks'], slots[:4])
                if total < setup['stacks'][selected]:
                    self.assertEqual(slots[0], (selected, total + 1, 0))
                else:
                    self.assertEqual(slots[0], ((selected - 1) % 4, 1, 0))
                expected = [[] for _ in range(4)]
                cursor = 0
                for i, packet in enumerate(setup['packets']):
                    if i < 12:
                        self.assertEqual(packet['player'], (dealer + i) % 4)
                        self.assertEqual(packet['count'], 4)
                    expected[packet['player']].extend(ordered[cursor:cursor + packet['count']])
                    cursor += packet['count']
                self.assertEqual(cursor, 53)
                self.assertEqual(game.hands, expected)
                self.assertEqual(list(reversed(game.wall)), ordered[53:])
                self.assertEqual(Counter(sum(game.hands, []) + list(game.wall)), Counter(all_tiles))
                self.assertEqual([len(h) for h in game.hands], [14 if p == dealer else 13 for p in range(4)])

    def test_bonus_replacements_and_stack_tracking(self):
        game = MahjongGame(dealer=0, wind_round=0, rand=random.Random(3))
        game.verbose = False
        deal_three_dice(game, [6, 6, 6])
        game.check_invariants(discarding_player=0)
        self.assertEqual(len(game.wall.slots), len(game.wall))
        before = list(game.wall.slots)
        game.wall.pop()
        game.wall.pop(0)
        self.assertEqual(game.wall.slots, before[1:-1])
        self.assertEqual(game.wall.draws[-2]['slot'], before[-1])
        self.assertEqual(game.wall.draws[-2]['end'], 'live')
        self.assertEqual(game.wall.draws[-1]['slot'], before[0])
        self.assertEqual(game.wall.draws[-1]['end'], 'replacement')
        self.assertNotIn('tile', game.wall.draws[-1])

    def test_invalid_dice_do_not_mutate_wall(self):
        game = MahjongGame(dealer=0)
        original = list(game.wall)
        for dice in ([1, 2], [0, 1, 2], [1, 7, 2], [True, 2, 3]):
            with self.assertRaises(ValueError):
                deal_three_dice(game, dice)
            self.assertEqual(game.wall, original)


if __name__ == '__main__':
    unittest.main()
