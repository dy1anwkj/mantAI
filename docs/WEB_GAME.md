# Play Mahjong in your browser

From the project folder, after installing the requirements (see [README.md](../README.md)):

```bash
python web_game.py
```

Open **http://127.0.0.1:8002** and click **Play**. Keep the terminal running.
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

Stop the server with Ctrl+C. If port 8002 is occupied, run with `--port 8003`
and open http://127.0.0.1:8003 instead.

No Node.js is required. The trained models need torch; without it the game
runs on the strategy engine alone. `web_game.py` serves
the HTML/CSS/JavaScript in `web/` and calls the existing `GameSession` and
strategy engine. The browser receives only your concealed hand, public tiles,
and currently legal actions; it does not calculate game rules.

The landscape table uses the supplied Singapore Mahjong reference photograph
as tile artwork via CSS sprite windows. The existing 148-tile set is preserved:
no jokers, Fei tiles, or extra instrument/occupation bonus tiles are added.
Table rows grow with their contents so exposed tiles, headings and result
badges stay separate. On short windows, scroll vertically to reach your hand;
on narrow screens, scroll horizontally to see the entire table.

This is one local single-player table against three computer players. Tabs share the
same table. Refresh preserves the running hand; restarting the Python server
resets it. Start with `--save-hands` to record hands in `data/web_hands/`
(not committed) when they end, or when you deal a new one after moving. Records
include the seed, starting hands, physical wall setup, moves, recommendations,
and your choices. Saving is off by default. There is no online multiplayer. Dealer and round
progression are kept while the server remains running.

Validation:

```bash
python -m pytest tests/test_web_game.py -v
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


## Arcade interface and three-dice setup

Click Play. Three dice tumble onto the field, then the selected player and stacks are counted automatically. The wall opens and packets fly to each player on the table.
The 148-tile set is arranged as 74 two-tile stacks: 19 at Dong/Xi and 18 at
Nan/Bei. Count players anticlockwise from the banker as 1. On the selected
wall count N stacks from its owner's right to left and break after stack N.
The banker takes the two complete stacks immediately left of the gap.
At a corner, drawing continues onto the adjacent wall to that owner's left.

The backend in wall_setup.py rotates the actual shuffled wall at that break;
this is not just a visual dice effect. Three rounds of four-tile packets are
dealt banker-first anticlockwise, then one tile to each player and one extra
to the banker. Existing bonus replacement rules run afterward. Wall racks show
actual remaining stack occupancy, including draws and replacement tiles.

The menu offers Play, How to play and Options. Options changes action pacing
and enables or disables recommendations. This remains a single-player game
against three AI bots. The interface uses original MantAI branding.

Validate the setup and browser adapter:

```powershell
.\venv\Scripts\python.exe -B -m pytest tests/test_wall_setup.py tests/test_web_game.py -q
```

The four physical walls enclose the central discard field throughout play.
Gold marks the live dealing/draw end and pink marks the opposite flower/Kong
replacement end. Both advance according to the backend's actual wall draws.
The opening uses the same backend dice and wall positions that determine the
hands; it is not an independent cosmetic roll. Exposed bonus tiles are shown
before their replacement is drawn from the opposite end. The old setup dialog
has been replaced by this automatic on-table sequence.
## Claim recommendations and point balances

Every player begins with 1,000 points. Seat badges show current balances, which
carry across hands while this server is running. Refreshing does not reset them;
restarting the server starts a new 1,000-point game. Existing instant Kong and
bonus payments are included, and each finished hand's tai payout is applied once.
The table total stays at 4,000 points.

One backend payout unit equals one point. For tai 1 through 5, the base is
1, 2, 4, 8, 16. On a discard win, the discarder pays twice that base and the other
two losers pay once each. For a self-draw all three losers pay twice the base.
The winner receives the total paid. The result shows each change and new balance.

When responding to a discard, the hint highlights a legal Chi (with its exact
pair of tiles), Pong, Kong, Hu or Pass using the strategy engine. Call hints are
strategy recommendations; they do not claim a simulated win percentage. Your
own discard-turn hint continues to show the estimated win percentage.
