# Play Mahjong in your browser

From PowerShell:

```powershell
cd "C:\Users\Christabel\vscode files"
.\venv\Scripts\python.exe .\web_game.py
```

Open **http://127.0.0.1:8000** and click **New hand**. Keep the terminal running.
Click a tile in your hand once to discard it. The three opponents play
automatically, like people would: each is the strategy engine with its own style
(aggressive, balanced or cautious). Your recommended discard comes from the
trained network and its search. To have that search play the opponents too,
start the server with `--ai-opponents`. When you can claim a tile, choose **CHI**, **PONG**, **KONG**,
**HU**, or **PASS**. Each Chi option lists the two tiles used from your hand.
The end-of-hand display includes the backend's tai and payments.

Actions play back with a 1.5-second pause. Each discard travels from the player's
seat into the centre, where the latest discarded tile stays visible. Controls
are disabled during playback and claims are offered when playback completes.
The delay is set by `ACTION_PAUSE_MS` in `web/app.js`. Refreshing skips any
remaining animation and shows the current game state.

Stop the server with Ctrl+C. If port 8000 is occupied, run with `--port 8001`
and open http://127.0.0.1:8001 instead.

No Node.js is required. The trained models need torch; without it the game
runs on the strategy engine alone. `web_game.py` serves
the HTML/CSS/JavaScript in `web/` and calls the existing `GameSession` and
strategy engine. The browser receives only your concealed hand, public tiles,
and currently legal actions; it does not calculate game rules.

The landscape table uses the supplied Singapore Mahjong reference photograph
as tile artwork via CSS sprite windows. The existing 148-tile set is preserved:
no jokers, Fei tiles, or extra instrument/occupation bonus tiles are added.
On narrow screens, scroll horizontally to see the entire table.

This is one local single-player table against three computer players. Tabs share the
same table. Refresh preserves the running hand; restarting the Python server
resets it. Each hand you play is saved to `data/web_hands/` (not committed) when it ends, or when you deal a new one after moving: the seed and starting hands, every move, and for each of your moves your tiles, what the box recommended and what you chose. Start with `--no-save` to turn this off. Dealer and round
progression are kept while the server remains running.

Validation:

```powershell
.\venv\Scripts\python.exe -B -m unittest test_web_game -v
```

Decisions are made the way AlphaGo made them (`mantai_ml/search.py`):

1. The 17 study PDFs in `docs/` are read into the network's inputs and its
   starting opinion (`mantai_ml/knowledge.py`): shape and tile efficiency,
   what to throw first, the tai route, defence (suji, kabe, flush threats,
   when to fold) and SPGG's late fresh-tile rule.
2. Self-play trained the network from there (`models/alpha_best.pt`) and a
   value network that estimates what a position is worth.
3. Before every discard, the network's likeliest tiles (plus the strategy
   engine's choice) are played out on many guessed deals of the hidden tiles.
   Each playout is scored by the value network and by the units actually won.
   The network's favourite is played unless another tile clearly did better.

Your recommendation searches 160 playouts (with `--ai-opponents`, each AI
discard searches 64). The box shows the recommended tile, the estimated hand
win chance, and every tile the search compared as `score (playouts)`, the pick
highlighted. The score decides: half the points won in each playout, half the
value network's estimate, on the same guessed deals. The network's first choice
is kept unless another tile beats it by a clear margin.

When you can claim a tile (or Hu, or Kong on your own turn), the box recommends
the action instead and outlines its button. A legal Hu is always recommended;
Chi, Pong and Kong come from your seat's strategy engine, which called better
than the network on 4000 test deals (+1.40 against +0.79 units a hand).
`web_game.py --engine-only` plays against the strategy engine alone; without
torch or a trained network the game says so and does the same.

On each discard turn, the recommended tile is shown. The displayed
win percentage estimates winning the current hand after that discard from the
searched continuations, using your hand and public information. It is a noisy
estimate, not a guarantee or an exact probability. Calculating it can take several
seconds. Recommendations are cached until your next action.

Player 1 starts the first hand as Dong. Click New hand after a result to continue:
Dong retains the dealer seat after winning; any other winner advances the dealer
one player. Draws retain the dealer. After a full dealer rotation, the round wind
advances Dong, Nan, Xi, Bei. Starting over before a hand ends retains the dealer.
