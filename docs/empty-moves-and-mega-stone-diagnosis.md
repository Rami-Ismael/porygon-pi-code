# Empty moves and the Kingambit/Floettite output

## Why Kingambit received Floettite

The error starts with the distinction between **legal to hold** and **useful to
hold**. Our pinned M-B simulator accepts Kingambit holding Floettite. The v2
exported vocabulary therefore lists Floettite in all three Kingambit ability/item
compatibility lists. The v2 decoder follows that exported table and final legality
validation accepts the output. This is not an unknown-token substitution.

The older decoder had an additional Mega Stone ownership restriction. The v2
regulation-table branch replaced it with simulator legality, removing that
strategic restriction. The synthetic-data sampler then selected from all legal
ability/item combinations, reinforcing the problem: **13,232 of 36,000 synthetic
Pokémon held a Mega Stone for a nonmatching species** under the prior coarse
species-family check. Four examples appear in the completed three-starter grid,
including `s1-pokemon-6` and `s0-pokemon-2`.

The pinned mod's exact item mapping is:

```text
Floettite: Floette-Eternal → Floette-Mega
```

Ordinary Floette is not included in that mapping. The older family-level helper
collapses the owner to `floette`, so reusing that helper alone would also be too
broad. V3 reads the exact item mapping, including forme names, from the pinned
simulator. Its training/decoding policy removes nonmatching stones from candidate
choices. This is an explicit strategic heuristic, not a claim that those choices
are simulator-illegal. It also rules out niche deliberate use of an inert stone.

## Empty moves were represented, but poorly covered

There is already an empty-string move token, separate from `[MASK]`:

- Empty string: a known, intentionally unused move slot.
- `[MASK]`: a hidden slot that the model must complete.

The previous 6,000-team synthetic dataset had these occupied-move counts across
36,000 Pokémon:

| Real moves | Previous v2 | Prepared v3 |
|---|---:|---:|
| 0 | 0 | 0 |
| 1 | 109 | 1,861 |
| 2 | 0 | 3,579 |
| 3 | 56 | 5,359 |
| 4 | 35,835 | 25,201 |

The v3 sampler deliberately includes one-, two-, and three-move sets, preserves
targeted coverage moves, pads empty positions after real moves, and rejects
all-empty movesets. These are unlabeled auxiliary training examples: they have
no invented battle-win labels and do not establish that fewer moves are better.
Historical battle-labeled data remain available to teach competitive context.

There was also a decoder bug: its explicit duplicate-move loop treated a known
empty token like a real move and prevented another empty slot. The separate v3
adapter exempts empty padding while still excluding repeated real moves. Its
final simulator check rejects an all-empty moveset.

## Implemented and verified

- `code/scripts/regmb_training_policy.py`: exact-form Mega Stone policy,
  variable move counts, and consistent encoding of implicit/explicit empty slots.
- `code/scripts/retrain_regmb.py`: v3 preparation and separately named checkpoints.
- `code/scripts/diffusion_baseline_v3.py`: v3-only decoder with repeatable empties.
- `code/scripts/test_regmb_training_policy.py`: padding, move-count coverage,
  duplicate-move behavior, and exact Floettite ownership tests.

Prepared **6,000 simulator-validated v3 teams** under
`~/.local/share/vgc-pilot-runtime/regmb-v3`. All 146,994 historical rows and the
500 historical test rows remain supported by the revised tables. The audit is
`data-audit.json`; the policy snapshot is `training-regulation.json`.

**V3 retraining completed on October 8, 2026**, after all 4,998 v2 meta-starter
generation attempts finished. Both v3 checkpoints are saved; v2 checkpoints and
outputs remain available. V3 is now regenerating the same meta-starter tasks
while Ling continues. These changes do not yet demonstrate improved battle wins.

To train the prepared version:

```sh
~/.local/share/vgc-pilot-runtime/venv/bin/python code/scripts/retrain_regmb.py train --version v3
```

This writes `F1_final_medium_regmb_v3.pt` and
`G2_family_matrix_noprtrain_regmb_v3.pt`, leaving the selected v2 baseline intact.
