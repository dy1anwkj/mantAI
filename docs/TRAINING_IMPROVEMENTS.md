# Training and decision improvements

The objective remains an AI that recommends strong Singapore Mahjong decisions
and can eventually beat three human players. Current results are against
simulated opponents; they do not demonstrate human-level performance.

## Changes

- Discard and call examples from a game stay together in the training/validation
  split. Training records now preserve opponents' public discards and melds,
  matching the inputs used during live inference.
- Supervised training supports `--init`, so the existing policy can learn from
  the stronger strategy engine. Gradients are clipped during fine-tuning.
- Self-play supports a mixture of frozen learned opponents, the current policy,
  and engine opponents. Only the Player 1 policy learns; frozen opponents do not.
- `--engine-calls` learns discards while the engine chooses claims. This setting
  is saved in the policy checkpoint and respected by both inference paths.
- PPO handles a single-decision batch without producing NaN advantages.
- Best policy checkpoints have matching best critic checkpoints. Resume training
  finds the appropriate critic, including the legacy filename fallback.
- Search respects its simulation budget and preserves the order of paired hidden
  deals across parallel and sequential execution. It always samples the reference
  choice and its strongest challenger during allocation.
- `--search-reference engine` provides an optional engine reference for search.
  This is an alternative decision mode, not a claim of measured superiority.
- The browser receives candidate search scores again. `--save-hands` optionally
  records decisions and recommendations for later review; saving is off by default.

## Experiment

1. Record 3,000 games played by the tuned engine, including public opponent data.
2. Fine-tune the previous `alpha.pt` policy for 15 supervised epochs, with equal
   weight on winning and losing games, a learning rate of 0.0003, and CPU training.
3. Continue for 25 reinforcement-learning iterations of 200 games (5,000 games),
   using a learning rate of 0.00005 and no progress shaping. Half the opponent
   seats remain engines; the other seats sample frozen policies. Evaluate every
   five iterations on 500 fixed deals to select a policy and its matching critic.
4. Test candidate policies on fresh seeded deals, independently of the training
   and checkpoint-selection deals. Compare learned calls with engine calls.

The supervised policy matched engine discard choices about 74% of the time and
call choices about 93% on held-out games. These measure teacher agreement, not
the probability of selecting a uniquely correct Mahjong move.

The first 5,000-hand comparison (seed 20261012000) found:

| Policy | Win rate | Points/hand | Deal-in rate |
|---|---:|---:|---:|
| Previous alpha.pt | 21.76% | +0.2440 | 17.74% |
| Supervised teacher policy | 28.22% | +0.7950 | 14.56% |
| Best reinforced policy | 28.22% | +0.7858 | 14.20% |
| Strategy engine | 34.12% | +1.6138 | 14.28% |

The reinforced policy improved by +0.5418 points/hand over the previous policy;
the approximate paired 95% interval was [+0.1872, +0.8964]. The supervised stage
accounts for most of the observed gain. This test does not show an additional
gain from reinforcement learning over the supervised policy. Search is excluded
from these policy comparisons, and the engine remains stronger.

A second comparison used 5,000 deals at seed 20261014000:

| Policy | Win rate | Points/hand | Deal-in rate |
|---|---:|---:|---:|
| Previous browser default alpha_best.pt | 24.92% | +0.3354 | 17.66% |
| Best reinforced policy, learned calls | 27.86% | +0.7848 | 14.58% |
| Best reinforced policy, engine calls | 29.62% | +0.8832 | 14.68% |
| Strategy engine | 32.44% | +1.2756 | 14.22% |

The learned-discard/engine-call policy was selected as the local browser default,
with its matching value model. The old default and its critic are backed up in
`models/backups/`. Compared with the old default on identical deals, the selected
policy gained +0.5478 points/hand, with an approximate paired 95% interval of
[+0.2058, +0.8898]. These comparisons exclude search. Search's full-game strength
and human performance remain unmeasured for the new checkpoint.

Local detailed reports, including per-hand results, are saved under `data/`.
That directory is gitignored. Use a new seed for subsequent confirmation tests;
repeatedly selecting models on the same deals can overfit the evaluation set.

## Reproduce a comparison

```bash
python tools/evaluate_models.py \
  --models models/alpha.pt models/improved_rl_best.pt models/improved_hybrid.pt \
  --games 5000 --seed 20261014000 --workers 4 \
  --out data/improvement_final_comparison.json
```

The first model is the baseline. Reports contain per-hand points, wins, deal-ins,
errors, and approximate paired confidence intervals. Every game enables rule and
tile-conservation checks. An evaluation with errors exits unsuccessfully.

New experiment checkpoints are local, gitignored artifacts. They must be copied
or retrained before these commands can be used on another machine.
