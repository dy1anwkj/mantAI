# Play Mahjong in your browser

From PowerShell:

```powershell
cd "C:\Users\Christabel\vscode files"
.\venv\Scripts\python.exe .\web_game.py
```

Open **http://127.0.0.1:8000** and click **New hand**. Keep the terminal running.
Click a tile in your hand once to discard it. The three AI opponents play
automatically. When you can claim a tile, choose **CHI**, **PONG**, **KONG**,
**HU**, or **PASS**. Each Chi option lists the two tiles used from your hand.
The end-of-hand display includes the backend's tai and payments.

Actions play back with a 1.5-second pause. Each discard travels from the player's
seat into the centre, where the latest discarded tile stays visible. Controls
are disabled during playback and claims are offered when playback completes.
The delay is set by `ACTION_PAUSE_MS` in `web/app.js`. Refreshing skips any
remaining animation and shows the current game state.

Stop the server with Ctrl+C. If port 8000 is occupied, run with `--port 8001`
and open http://127.0.0.1:8001 instead.

No Node.js or additional Python packages are required. `web_game.py` serves
the HTML/CSS/JavaScript in `web/` and calls the existing `GameSession` and
strategy engine. The browser receives only your concealed hand, public tiles,
and currently legal actions; it does not calculate game rules.

The landscape table uses the supplied Singapore Mahjong reference photograph
as tile artwork via CSS sprite windows. The existing 148-tile set is preserved:
no jokers, Fei tiles, or extra instrument/occupation bonus tiles are added.
On narrow screens, scroll horizontally to see the entire table.

This is one local single-player table against three AI players. Tabs share the
same table. Refresh preserves the running hand; restarting the Python server
resets it. Hands are not saved, and there is no online multiplayer. Dealer and round
progression are kept while the server remains running.

Validation:

```powershell
.\venv\Scripts\python.exe -B -m unittest test_web_game -v
```

On each discard turn, the strategy engine recommends a tile. The displayed
win percentage estimates winning the current hand after that discard from 64
simulated continuations, using your hand and public information. It is a noisy
estimate, not a guarantee or an exact probability. Calculating it can take several
seconds. Recommendations are cached until your next action.

Player 1 starts the first hand as Dong. Click New hand after a result to continue:
Dong retains the dealer seat after winning; any other winner advances the dealer
one player. Draws retain the dealer. After a full dealer rotation, the round wind
advances Dong, Nan, Xi, Bei. Starting over before a hand ends retains the dealer.
