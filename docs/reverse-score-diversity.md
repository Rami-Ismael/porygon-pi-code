# Reversing score guidance: diversity protocol

Question: does generating in the opposite direction of the existing predicted
battle-score guidance produce more diverse legal Pokémon species combinations?

## Locked initial experiment

- One fixed model: `results/temperature_p0.pt`, the shared 600-epoch checkpoint
  trained on 692 valid corpus teams for the previous temperature experiment.
- One fixed ridge surrogate, fitted once using the same 200 anchors and 128 gen0
  measured labels. Its coefficients and all input files are fingerprinted.
- Three directions: normal `+108`, no guidance `0`, reversed `−108`.
- Three new generation seeds: 1101, 1202, 1303. The direction order rotates across
  seeds; each direction resets both sampling RNGs to the same seed.
- Exactly 512 raw generation attempts per direction/seed, in chunks of 48:
  **4,608 experimental attempts**. Invalid or repeated teams are retained in the
  records and are not replaced. There is no surrogate acquisition filter.
- The checkpoint, constraints, decoder temperature (1.0), stat-spread sampling,
  validator, and generation budget are held fixed. There is no retraining and no
  battle evaluation in this experiment.

The frozen sampler adds a score-dependent tilt to token logits. Reversal changes
`base_logits + gain * tilt` to `base_logits − gain * tilt` at the same state.
The original nonnegative quality-gap gain is retained, including its zero clamp.
This reverses the existing `gloss` contribution; it does not add a novelty
objective or implement the separate full-Jacobian guidance mode.

The original sampler ignores negative magnitudes. The new adapter therefore
passes a positive magnitude and negates the tilt explicitly. Before sampling,
real-model controls verify that the positive adapter reproduces normal sampling,
the zero adapter reproduces unguided sampling, and opposite logit contributions
are exactly antisymmetric and nonzero at the initial state. The 16 diagnostic
control draws are excluded from the 4,608-attempt experiment.

## Measurements and interpretation

The primary endpoint is **distinct legal six-species combinations per 512 raw
attempts**. A combination ignores Pokémon slot order, moves, items, and spreads.
This measures how many usable different combinations the generation budget buys.
Legality rate is reported alongside it so invalid novelty cannot masquerade as
useful diversity.

Secondary measures cover raw and legal outputs separately: distinct combinations,
effective combinations (exponential Shannon entropy), mean pairwise Jaccard
distance, novel distinct combinations absent from the checkpoint's corpus, and
nearest-corpus Jaccard distance. Exact rarefaction also measures the expected
distinct combinations in 128 valid draws without replacement; it is unavailable
when fewer than 128 valid teams exist. This helps separate diversity within valid
outputs from differences in the number of valid outputs.

The two primary contrasts are reversed minus normal and reversed minus no
guidance. Report individual seed differences and their means. Three-seed paired
t intervals with two-comparison Bonferroni adjustment are exploratory and require
a normal-effects assumption. Other metrics are descriptive. Conclusions are
conditional on this checkpoint, surrogate, magnitude, and seed set.

Surrogate scores are saved as diagnostic predictions. This experiment cannot
establish whether reversed guidance changes actual battle win rate.

## Run and resume

```sh
/Users/ramiismael/.local/share/vgc-pilot-runtime/venv/bin/python \
  scripts/test_reverse_score_diversity.py

/Users/ramiismael/.local/share/vgc-pilot-runtime/venv/bin/python -u \
  scripts/reverse_score_diversity.py
```

The output defaults to `results/reverse_score_diversity.json`, with a Markdown
report, analysis JSON, and plot alongside it. The same command resumes completed
cells only when configuration and input hashes still match. An unfinished cell
restarts from its seed. Output locking prevents duplicate workers.

A separate small plumbing check uses:

```sh
/Users/ramiismael/.local/share/vgc-pilot-runtime/venv/bin/python -u \
  scripts/reverse_score_diversity.py --attempts 12 --chunk 12 --seeds 99001 \
  --output results/reverse_score_diversity_smoke.json
```

Reduced runs are plumbing checks, not evidence for the full experiment.

After the full run exits successfully, independently recount diversity, revalidate
all 4,608 saved proposals, and check the input and report fingerprints:

```sh
/Users/ramiismael/.local/share/vgc-pilot-runtime/venv/bin/python \
  scripts/audit_reverse_score_diversity.py results/reverse_score_diversity.json
```

This writes `results/reverse_score_diversity.verification.json`. The auditor
accepts only the full protocol above, so a plumbing run cannot pass as a completed
experiment.

## Completed first run

The full run and independent audit completed on 2026-09-08. Reversal produced a
small increase in distinct legal combinations; the implications for battle
strength remain untested. See [results and interpretation](reverse-score-diversity-results.md)
for the comparisons and their limitations.
