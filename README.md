# mantAI

**An AlphaGo-inspired AI for Singapore Mahjong, aiming to beat three human
players through machine learning and reinforcement learning.** The browser game
lets you play Player 1 against three computer opponents and follow the AI's
discard and claim recommendations. A learned policy, statistical hand analysis,
decision engines, and search over possible hidden deals support its decisions.
Performance against real humans has not yet been measured.

![An earlier mantAI table showing discard recommendations and search scores](docs/screenshot.jpg)

## Quick start

Python 3.9 or newer.

```bash
git clone https://github.com/dy1anwkj/mantAI.git
cd mantAI
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python web_game.py
```

Open **http://127.0.0.1:8000** and select **Play**. The trained models are
included in `models/`, so there is nothing to train first. Without PyTorch the
game still runs, with the strategy engine making the recommendations.

## What you see

- **Three-dice opening:** dice select the wall break, followed by animated
  dealing. The backend uses that break to deal the actual hands.
- **Physical walls:** 74 two-tile stacks surround the discard field. Wall
  occupancy follows actual draws, with gold marking the live draw end and
  pink marking the flower/Kong replacement end.
- **Point balances:** each player starts with 1,000 points. Kong, bonus, and
  winning-hand payments update balances, which carry across hands until the
  server restarts. One scoring unit equals one point.
- **The recommendation box**, above your hand: the tile to discard, your
  estimated chance of winning the hand after it, and every tile the search
  compared as `score (playouts)`, the pick highlighted.
- **Claims:** when you can Chi, Pong, Kong or Hu, the box recommends an action
  and outlines its button, including the exact tiles for Chi. It can also
  recommend Pass. A legal Hu is always recommended; claim hints do not show
  a simulated win percentage.
- **Human-style opponents:** each opponent plays its own style, aggressive,
  balanced or cautious, as stand-ins for the human players mantAI is meant to beat.
- **Saved hands for review:** each hand is saved to `data/web_hands/` with every
  move, what was recommended and what you chose. `--no-save` turns this off.
- **Options:** adjust action pacing and enable or disable recommendations.

This is a local single-player table; there is no online multiplayer. Browser
tabs share the same game. Refreshing preserves the hand and point balances while
the Python server stays running.

More detail on the web game: [docs/WEB_GAME.md](docs/WEB_GAME.md).

## How it decides

mantAI follows the recipe AlphaGo used, adapted to a game with hidden tiles
([mantai_ml/search.py](mantai_ml/search.py)):

1. **Policy network.** Starts with hand analysis and strategy features encoded
   from the study library ([mantai_ml/knowledge.py](mantai_ml/knowledge.py)),
   then improves through reinforcement learning, rewarded by points won. It
   proposes the most promising discards. The hand-tuned strategy engine's pick
   is always tested too.
2. **Guessed deals.** Opponents' hands are hidden, so each playout deals out a
   possible version of them, weighted by what each opponent has thrown and
   claimed.
3. **Search.** Every candidate tile is played out to the end of the hand on the
   *same* guessed deals, so the differences come from the discard, not from luck.
   Each playout is scored half by the points actually won and half by a value
   network's estimate of the position. Playouts are allocated with PUCT,
   AlphaGo's rule.
4. **Decision.** The network's first choice is kept unless another tile scored
   clearly better on the same deals, by a margin of 1.5 standard errors.

The strategy engine measures distance from ready, unseen improving tiles,
scoring potential, and defensive risk. A learned opponent model estimates
readiness and deal-in risk from public information. The displayed win chance
is a noisy estimate from simulations; the search compares expected points.

## Results

Against the human-style opponents, playing every move by the recommendation
(Player 1's points per hand; 0 is break-even and a fair share of wins is 25%):

| Player 1 plays as | Hands | Win rate | Points per hand |
|---|---|---|---|
| Network + search (the recommendation) | 200 | 26% | **+1.30** |
| Trained network alone | 2,000 | 26% | +0.27 |
| Hand-tuned strategy engine | 2,000 | 33% | +0.99 |
| Engine, on the same 200 deals as the search | 200 | 35% | +1.47 |

- These recorded runs show positive average points for both the network and
  search. They use different sample sizes and do not establish a statistically
  significant improvement from search.
- The hand-tuned engine is still slightly ahead. Closing that gap through more
  self-play is the next step.
- Chi, Pong and Kong recommendations come from the engine: with the engine
  discarding, engine calls scored +1.40 a hand against +0.79 for the network's
  calls over 4,000 deals.
- These results are against simulated opponents. mantAI has not yet been
  measured against human players.

## Rules

Singapore Mahjong with the 148-tile set (flowers, seasons and animals included),
1 tai minimum, 5 tai maximum and a 15-tile dead wall. Where sources disagree,
SPGG Mahjong Rules 2024 decides. Rules and scoring:
[singapore_rules.py](singapore_rules.py) and [mahjong_simulator.py](mahjong_simulator.py).

## Project layout

| Path | What it is |
|---|---|
| [web_game.py](web_game.py), [web/](web) | The browser game: a local server plus HTML, CSS and JavaScript |
| [mahjong_simulator.py](mahjong_simulator.py) | The game: deal, turns, claims, scoring and the playout simulator |
| [wall_setup.py](wall_setup.py), [web/table_scene.js](web/table_scene.js) | Three-dice wall break, physical dealing, and table animations |
| [singapore_rules.py](singapore_rules.py) | Tai scoring and payments |
| [Decision_Strategy_engine.py](Decision_Strategy_engine.py) | The hand-tuned strategy engine (also the opponents) |
| [D_hand_eval.py](D_hand_eval.py) | Hand evaluation: winning shapes, waits, tile efficiency |
| [R_opponent_analysis.py](R_opponent_analysis.py) | Learned danger model: how likely a tile is to let an opponent win |
| [mantai_ml/](mantai_ml) | Policy and value networks, self-play training and the search |
| [models/](models) | Trained models: `alpha_best.pt`, `alpha_critic.pt`, `danger.npz` |
| [tests/](tests) | Tests (`python -m pytest`) |
| [tools/](tools) | `tune_engine.py`: tunes the strategy engine's weights by simulation |
| [docs/](docs) | The web game guide and the rule and strategy books the network started from |

## Training

The current command trains the learned player against three strategy-engine
opponents. Training all four seats with learned policies is a future step.
`--games` is the number of games **per iteration**; total training games are
`--iterations × --games`. Evaluation games are additional.

```bash
# Self-play from the book-based starting point
python -m mantai_ml selfplay --iterations 1200 --games 400 --shaping 0 --out models/alpha.pt

# Measure a model against the strategy engine, with and without search
python -m mantai_ml evaluate --model models/alpha_best.pt --games 1000 --compare
python -m mantai_ml evaluate --model models/alpha_best.pt --games 200 --search 64 --compare
```

To continue the included policy for 1,000 training games and save separate
checkpoints:

```bash
python -m mantai_ml selfplay --init models/alpha_best.pt --iterations 10 --games 100 --shaping 0 --eval-games 100 --out models/continued.pt
python web_game.py --model models/continued.pt
```

`python -m mantai_ml --help` lists every command.

## Tests

```bash
python -m pytest
```

The video tests additionally need `opencv-python` and `pillow`. Without PyTorch,
the ML tests are skipped. Wall setup and browser integration checks are in
`tests/test_wall_setup.py` and `tests/test_web_game.py`.
