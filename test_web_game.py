"""Integration checks for the local browser adapter."""
import json
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from mahjong_simulator import choose_action
from web_game import Table, create_server


class TableTests(unittest.TestCase):
    def test_dealer_retains_or_rotates_and_round_advances(self):
        for dealer, winner, expected, round_wind in [(0, 0, 0, 0), (0, 2, 1, 0), (2, None, 2, 0), (3, 1, 0, 1)]:
            table = Table()
            table.new(seed=7)
            table.session.game.dealer = dealer
            table.session.is_over = True
            with patch.object(table.session, 'result', return_value={'winner': winner}):
                self.assertEqual(table.next_dealer(), expected)
                table.new(seed=3)
            self.assertEqual(table.session.game.dealer, expected)
            self.assertEqual(table.session.game.wind_round, round_wind)
            self.assertEqual(table.session.game.seat(expected), 1)

    def test_hint_is_legal_cached_and_rejects_stale_revision(self):
        table = Table()
        table.new(seed=7)
        def rate(game, player, tiles, **kwargs):
            self.assertFalse(kwargs['parallel'])
            return {tiles[0]: {'win': 25.0, 'playouts': 64}}
        with patch('web_game.rate_discards', side_effect=rate) as simulate:
            hint = table.hint(table.revision)['recommendation']
            legal = [a.tile for a in table.session.legal_actions() if a.kind == 'discard']
            self.assertIn(hint['tile'], legal)
            self.assertEqual(hint['win_percent'], 25.0)
            self.assertEqual(table.hint(table.revision)['recommendation'], hint)
            self.assertEqual(simulate.call_count, 1)
            with self.assertRaises(ValueError):
                table.hint(table.revision - 1)
        table.new(seed=3)
        self.assertIsNone(table.hint_cache)

    def test_discard_playback_is_ordered_and_snapshots_are_independent(self):
        table = Table()
        table.new(seed=7)
        index = next(i for i, a in enumerate(table.session.legal_actions()) if a.kind == "discard")
        thrown = table.session.legal_actions()[index].tile
        table.act(index, table.revision)
        first = table.playback[0]
        self.assertEqual(first["player"], 0)
        self.assertEqual(first["action"]["kind"], "discard")
        self.assertEqual(first["action"]["tile"], thrown)
        self.assertEqual(first["state"]["discards"][0][-1], thrown)
        self.assertEqual(first["state"]["last_discard"], {"player": 0, "tile": thrown})
        for frame in table.playback:
            self.assertEqual(frame["state"]["actions"], [])
            self.assertNotIn("hands", frame["state"])
            self.assertNotEqual(frame["action"]["kind"], "pass")
        saved = json.dumps(first)
        table.session.game.discards[0].clear()
        self.assertEqual(json.dumps(first), saved)
        table.new(seed=3)
        self.assertEqual(table.playback, [])
        self.assertIsNone(table.last_discard)

    def test_human_is_dong_and_discards_first(self):
        for seed in (3, 7, 19):
            table = Table()
            table.new(seed=seed)
            state = table.state()
            self.assertEqual(state["dealer"], 0)
            self.assertEqual(state["seats"], [1, 2, 3, 4])
            self.assertEqual(table.session.current_player, 0)
            self.assertEqual(state["decision"], "turn")
            self.assertEqual(state["counts"], [14, 13, 13, 13])
            self.assertEqual(state["discards"], [[], [], [], []])
            self.assertEqual(state["turn"], 1)

    def test_full_hands_through_browser_actions(self):
        for seed in (3, 7, 19):
            table = Table()
            table.new(seed=seed)
            table.session.game.player1 = "engine"
            self.assertIsNone(table.state()["winning_hand"])
            steps = 0
            while not table.session.is_over:
                self.assertEqual(table.session.current_player, 0)
                action = choose_action(table.session.game, table.session.decision)
                index = table.session.legal_actions().index(action)
                table.act(index, table.revision)
                steps += 1
                self.assertLess(steps, 300)
            state = table.state()
            self.assertTrue(state["over"])
            self.assertEqual(state["actions"], [])
            self.assertEqual(sum(state["result"]["total"].values()), 0)
            winner = state["result"]["winner"]
            if winner is None:
                self.assertIsNone(state["winning_hand"])
            else:
                revealed = state["winning_hand"]
                self.assertEqual(revealed["player"], winner)
                self.assertEqual(revealed["hand"], table.session.game.sort_tiles(table.session.game.hands[winner]))
                self.assertEqual(revealed["melds"], table.session.game.melds[winner])
                self.assertEqual(len(revealed["hand"]) + 3 * len(revealed["melds"]), 14)
            # All backend tile-count invariants ran after each discard.
            self.assertNotIn("hands", state)
            self.assertIsInstance(state["wall"], int)

    def test_bad_and_stale_actions_do_not_change_table(self):
        table = Table()
        table.new(seed=7)
        before = json.dumps(table.state())
        for index, revision in [(-1, table.revision), (999, table.revision),
                                (True, table.revision), (0, table.revision - 1)]:
            with self.assertRaises(ValueError):
                table.act(index, revision)
            self.assertEqual(json.dumps(table.state()), before)

    def test_http_assets_and_authorized_new_hand(self):
        server = create_server(0)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            for path in ("/", "/app.js", "/style.css", "/tiles.css", "/tiles.webp"):
                with urlopen(base + path) as response:
                    self.assertEqual(response.status, 200)
                    self.assertTrue(response.read())
            with urlopen(base + "/api/state") as response:
                state = json.load(response)
            self.assertFalse(state["started"])
            with self.assertRaises(HTTPError) as error:
                urlopen(Request(base + "/api/new", data=b"{}", method="POST"))
            self.assertEqual(error.exception.code, 403)
            headers = {"X-Table-Token": state["token"], "Content-Type": "application/json"}
            with urlopen(Request(base + "/api/new", data=b"{}", headers=headers)) as response:
                self.assertTrue(json.load(response)["started"])
            with self.assertRaises(HTTPError) as error:
                urlopen(base + "/../mahjong_simulator.py")
            self.assertEqual(error.exception.code, 404)
        finally:
            server.shutdown()
            server.server_close()
            worker.join()


if __name__ == "__main__":
    unittest.main()
