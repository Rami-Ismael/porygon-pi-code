# DiBO v4: categorical adaptation and matched-budget rerun

The v4 predictor preserves each Pokémon's ownership of its moves, item, ability and nature before pooling the team. It is invariant to team-slot and move ordering. Both predictor targets and exponential training weights use standardized observed rewards. Posterior training uses 50 updates of 256 trajectories, split into memory-safe microbatches, and a growing replay buffer prioritized by standardized UCB. The buffer retains the best 1,000 observations. The proposal pool is 100 times the battle batch: 12,800 legal distinct unseen candidates for 128 evaluations. Ten categorical local-search iterations accept changes only when the exact full prior-plus-reward target increases.

These changes follow the paper and its official implementation more closely. They remain a categorical adaptation: autoregressive discrete trajectories replace Gaussian diffusion; local proposals replace continuous gradient ascent; beta is fixed at 30 in standardized reward units; deterministic spread selection remains outside the learned representation. The predictor uses a five-member ensemble. CEM retains its historical nature repair, ridge guidance and elite weighting. The comparison tests these complete pipelines, not an isolated posterior operator.

## Frozen experiment

The new run uses seed 5050 and the same frozen 5,000 starting labels as the original experiment. Each arm receives 10 rounds × 128 teams × 49 physical opponent columns = 62,720 new search battles, followed by 16 search-selected finalists × (192 confirmation + 192 heldout) = 6,144 evaluation battles. Total: 68,864 per arm. The opponent collection was inspected in the earlier experiment; fresh battle seeds do not make it a newly untouched population. Only one training seed is run, so candidate-level uncertainty cannot establish robustness across independent runs.

CEM generates 512 proposals per round; DiBO generates 12,800 as required by its larger filtering pool. Battle budgets are equal; computational budgets are not. Compare diversity on equally sized populations: first 512 legal raw samples, 128 battled teams per round, all 1,280 battled teams, and 16 locked finalists. Report unique rosters, effective roster count, dominant-roster share and build variation within a shared roster. Do not interpret larger unequal proposal-pool counts as evidence of greater diversity.

The old code and results remain preserved. The new driver archives its source and freezes protocol hashes before running. Seven regression tests and a complete two-arm plumbing trial passed before the full run. A real 256-trajectory update and exact-acceptance local-search capacity check passed without battles. The tiny plumbing trial is not performance evidence.

Sources: [paper](https://arxiv.org/html/2502.16824v2) and [official implementation, pinned revision](https://github.com/umkiyoung/DiBO/blob/31ff265563a01a13c34c3038cc13f84c60a9e5a0/baselines/algorithms/dibo.py).
