# Project audit — October 10, 2026

The audit reviewed the simulator, Singapore scoring and payments, hand and
strategy evaluators, opponent analysis, browser adapter and JavaScript,
policy/value loading and inference, recording, supervised training, self-play,
model evaluation, and the optional video pipeline.

## Fixes

- Bounded the three hand-evaluation caches and the per-suit strategy cache.
  Previously they retained every distinct input indefinitely. A 100-game engine
  run accumulated 76,941 recursive set checks and 42,405 winning-hand checks.
  The limits are now 100,000 entries for each of those two caches and 50,000
  each for winning-tile and per-suit caches. Evicted results are recomputed.
- Playback instructions now display the pause duration selected in Options.
- The initial instruction refers to the current Play button.

The trained policy and value weights were unchanged.

## Validation

- Full post-fix suite: **262 passed**, no skips, in 219.56 seconds, with NumPy,
  PyTorch, OpenCV and Pillow installed. This includes training and self-play
  smoke tests, synthetic video recognition, model checkpoint selection, saved
  recommendations, claim priority, Kong handling, scoring, physical dealing,
  and HTTP assets and authorization.
- The initial sandboxed run passed 261 tests; its HTTP test was prevented from
  binding a local socket. The complete rerun with localhost access passed.
- All 36 Python files parsed; 29 application modules imported. Both browser
  JavaScript files passed `node --check`. All 29 literal JavaScript element-ID
  references resolved to elements in the HTML. `pip check` found no broken
  requirements, and `git diff --check` passed.
- Both shipped policies completed 24 extra physical-wall games each, using
  seeds 20261010000–20261010023, rotating dealer and round wind, with tile
  conservation, legal-win and zero-sum payout checks enabled. No errors.
- An additional 100 seeded engine games passed invariant checks.
- After the cache fix, 160 games (seeds 2026101040–2026101199) were each played
  twice: with caches retained and with caches cleared before each game. Winners,
  tai, turn counts and payouts matched exactly. The recursive set cache reached
  its 100,000-entry limit during the retained run.
- The default policy and critic produced a legal recommendation using the full
  160-playout search budget, with finite scores. Both policy checkpoints had
  finite parameters; learned danger predictions were finite and within [0, 1].
- The existing live server responded normally; its balances summed to 4,000.
  Its running hand and balances were preserved.

## Limits

These checks establish correctness on the exercised paths, not a proof of zero
bugs or improved playing strength. Human performance and search strength still
require separate evaluations. Native browser control was unavailable, so this
audit did not include a fresh visual inspection of animations or responsive
layout; browser validation used syntax, element references and HTTP tests.

Saving played hands still requires `python web_game.py --save-hands`. Already
running Python servers use the old imported evaluator until restarted;
restarting resets their table and balances. Browser text changes apply on refresh.
