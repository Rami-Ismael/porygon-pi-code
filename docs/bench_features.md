# Landscape features of the collected BO benchmarks vs VGC — pre-registration (2026-09-30)

Written before any benchmark in this file is run. Scripts: `scripts/bench_features.py` (benchmarks,
no battles) and `scripts/vgc_domain_features.py` (VGC validator + counting, no battles).
Output: `results/bench_features.json`, `results/vgc_domain_features.json`.

Purpose: fill the *measured* cells of the feature table in the vault note
`Teaching/reference/bo-benchmark-derived-vgc-research-comparison.md` (Version 2). Every other cell is *declared* (quoted
from the benchmark's paper or code). A cell is *measured* only when the same metric is computed
the same way on both sides.

## VGC side (no new battles)
f = win rate vs the top-50 meta, behaviour-cloning policy both sides, 384 battles, one fixed
ordered opponent schedule. Reused from `results/ruggedness2_parts/` (run 2026-09-28):
40 anchors (24 meta, 16 search-found), 6 edit types × 3 edits, 3 replicates.

## Benchmarks measured (cheap to run on this machine)
Contamination control (COMBO/Bounce, λ = 1e-2, dynamics seed 42), Ising sparsification
(COMBO, 4×4 grid, 24 edges, λ = 1e-2, instance seed 0), Pest Control 25×5 in two
implementations (COMBO: fresh randomness per call; Bounce: seeded per call), weighted
MaxSAT-60 (`frb10-6-4.wcnf`, weights standardised as in COMBO/Bounce), LABS n = 50,
Ackley-53 (50 binary + 3 continuous, Bounce), penalised logP (LOL-BO normalisation),
GuacaMol Perindopril / Zaleplon MPO and TDC DRD2 (via PyTDC oracles).
Not measured (declared only): XGBoost-MNIST, SVM-Slice, NAS-Bench-101, arithmetic
expressions, DRD3 docking, GSK3B, JNK3 (the last two: TDC pickles fail under current
scikit-learn).

## Design, identical to ruggedness2 on every benchmark
- 24 "start" anchors (uniform random; ZINC molecules for the molecule tasks) + 16 search-found
  anchors (first-improvement hill climb, 100 single-edit proposals, minimising f as observed).
- 15 single edits per anchor; 3 replicates of the anchor (fresh randomness where f is stochastic).
- Single edit: one variable set to another value (binary: flip; categorical: another level;
  Ackley continuous coordinate: resampled uniform). Molecules: one SELFIES token replaced by a
  token of the semantic-robust alphabet (STONED-style point mutation); the child must decode to a
  different canonical SMILES.
- Minimisation throughout (maximised scores negated; ρ is sign-invariant).

## Metrics (fixed now)
1. Smoothness ρ = 1 − (E[d²] − E[r²]) / (2·var_f), var_f pooled over both start types minus
   E[r²]/2; bootstrap 95% CI over anchors (2,000). VGC reported pooled over all six edit types
   and per edit type.
2. Noise share = σ² / (σ² + var_f), σ² = E[r²]/2 (per-evaluation noise variance). VGC also at
   24 battles (σ² × 16, binomial scaling — derived, labelled so).
3. Evaluation cost = wall-clock seconds per evaluation on this machine (benchmarks: one process;
   VGC: the battle pool's measured throughput).
4. VGC only, validator-measured: fraction of uniform draws that are legal under
   (a) unconditioned field vocabularies, (b) species-conditioned fields (roster clauses only).
   Benchmarks: the same fraction under their own naive encoding (binary/categorical: all inputs
   valid by definition; molecules: uniform SMILES-character strings vs uniform SELFIES tokens).
5. Cardinality: log10 of the domain size and of the single-edit neighbourhood size.

No threshold is tested; these are descriptive numbers reported whatever they are.

## Deviations, recorded after running
- Ising instance drawn with numpy's RNG (COMBO's distribution, not its torch stream), so the
  couplings differ from COMBO's seed-0 instance.
- Molecule edits return the SELFIES decoder's SMILES (kekulé) so the child can be re-encoded;
  ZINC start molecules that SELFIES cannot encode are skipped.
- GSK3B and JNK3 were dropped: their Therapeutics Data Commons pickles fail under scikit-learn 1.9.1.
- VGC legality draw (c): the 29 / 500 failures are ability-table errors of ours (mega-forme
  abilities listed for the base species), not an unmodelled rule.
