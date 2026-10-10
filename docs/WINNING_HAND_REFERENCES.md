# Winning-hand reference review

Reviewed October 10, 2026. These notes record source comparisons and planning
gaps; they do not represent a model-training run or a change to scoring.

## Sources and differences

[AsiaOne's 2021 guide](https://www.asiaone.com/lifestyle/singaporeans-guide-mahjong-winning-hands-probability-scoring-and-how-mahjong)
describes common and rare combinations, including all-chows, all-pongs, flushes,
dragon/wind patterns and Thirteen Wonders. It lists rarity estimates, but the
article does not provide enough methodology to use those numbers as calibrated
probabilities for the player's current hand. Its scoring table lists Small
Three Dragons as 3 tai and Small Four Winds as 2 tai.

[Lucky Cat's main Singapore rules guide](https://singaporemahjongrules.com/)
identifies itself as an independent educational guide and discusses household
variations. It describes Thirteen Wonders as six suited terminals, seven honours
and a duplicate. Seven Pairs is optional there. Its Small Three Dragons total
is 3 Fan, Small Four Winds total is 4, and its default payout ladder starts at
2 base points for 1 Fan. The supplied
[focused hands URL](https://singaporemahjongrules.com/singapore-mahjong-winning-hands/)
could not be fetched; this review uses the accessible main guide.

[Seedly's guide](https://blog.seedly.sg/mahjong-probability-guide-singapore/)
was also reviewed. It is by Ming Feng, dated February 8, 2024, and overlaps
substantially with the AsiaOne article, which credits Seedly as its original
source. Treat these as related versions rather than independent evidence.
Its table likewise lists Small Three Dragons as 3 tai and Small Four Winds
as 2 tai. Its “All Pongs or All Pairs” label is not sufficient justification
for enabling a separate Seven Pairs winning shape in mantAI.

mantAI currently uses SPGG 2024 as the stated tie-breaker between rule sources.
Its Small Three Dragons total is 4 tai, Small Four Winds is capped at 5, and
its 1-tai base payout is 1 point. Those differences need a deliberate ruleset
decision rather than mixing the guides into the current scorer.

## Current implementation

| Pattern | Current scorer | Current planning support |
|---|---|---|
| Ping Hu | 4 tai when its configured conditions hold | General sequence efficiency and scoring simulations; no dedicated route planner |
| All-pongs | 2 tai | Triplet/call evaluation; no explicit distance-to-all-pongs calculation |
| Half/full flush | 2/4 tai | Suit-concentration scoring potential and simulated returns |
| Dragon/wind combinations | Recognised; scoring depends on configured rules | Scoring-honour pairs and triplets influence decisions |
| Thirteen Wonders | Recognised, capped at 5 tai | Progress evaluator primarily measures four sets plus a pair |

These are findings from `singapore_rules.py`, `D_hand_eval.py`,
`Decision_Strategy_engine.py` and `mantai_ml/features.py`, not claims made by
the references. Several other named limit patterns have no dedicated scoring
entry in the current code. A complete pattern catalogue needs tests against
the chosen ruleset before those patterns are advertised as supported.

## Improvements to implement and evaluate

1. Add route-specific distance and available-tile calculations, prioritising
   Thirteen Wonders so an almost-complete special hand is not judged solely
   as an inefficient ordinary hand.
2. Compare achievable routes using legal scoring, unseen copies, remaining
   draws and defensive risk. Preserve viable alternatives as the hand changes.
3. Explain the current candidate route in recommendations without presenting
   an intention as a guaranteed outcome.
4. Validate any new inputs and scoring rules, then train compatible checkpoints
   and benchmark on fresh deals before changing the deployed model.

The current checkpoint uses feature version 3. Adding new feature dimensions
requires a version change and compatible training; documenting a reference
does not update its weights.
