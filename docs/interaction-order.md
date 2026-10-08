# Interaction order of win rate, against benchmark landscapes — pre-registration (2026-09-30)

Written before any battle. Script: scripts/interaction_order.py; output results/interaction_order.json.
Question: how much of win rate's variation under a few simultaneous edits comes from interactions of
order 1, 2, and 3 or more, and how does that compare with standard black-box benchmark landscapes?

## Design — a local 2^4 factorial around each anchor
- Pick an anchor x and four factors (edits) e1..e4. Evaluate f at all 16 combinations of
  "edit off / edit on". Code each factor as s_i = −1 (off) / +1 (on).
- Walsh coefficients a_S = mean over the 16 points of f(x)·Π_{i∈S} s_i, for every S ⊆ {1..4}.
  By Parseval, the variance of f over the cube = Σ_{S≠∅} a_S².
- Share of order k = Σ_{|S|=k} a_S² / Σ_{S≠∅} a_S². Reported as order 1, order 2, order ≥3.
- This is a LOCAL spectrum (four edits around a starting point), not the global spectrum of f.
  The same local design is applied to every landscape, so the bars are comparable.

## Noise correction
Each cube is evaluated twice with independent seeds (replicates r=1,2). a_S² is estimated by the
cross-replicate product a_S^(1)·a_S^(2), which is unbiased for the noise-free a_S² (independent
noise, any within-replicate correlation). Shares pool numerators and denominators over anchors
(ratio of sums). 95% CIs: bootstrap over anchors, 2,000 resamples. Deterministic landscapes: the
product is the exact square.

## VGC
- f = win rate vs the top-50 meta (results/top50_evs.json), behaviour-cloning policy both sides.
  Common random numbers within a replicate: every team of a cube faces the same ordered opponent
  schedule; replicate seeds 101 and 202.
- Primary cubes ("random species edits"): 16 anchors drawn uniformly (seed 1) from the top-50 meta,
  excluding the three weather anchors below. Factors = four distinct random slots, each replaced by a
  random real corpus set (whole set copied, no repair) of a species not on the team and not used by
  another factor; donor items must not collide with any original item or other donor. All 16 teams
  must pass Showdown's TeamValidator, else the donors are redrawn. 384 battles per team per replicate.
- Designed cubes ("weather core"): anchors MB522, MB489, MB444 (each holds Pelipper/Drizzle,
  Charizard-Mega-Y, Venusaur/Chlorophyll, Archaludon). Factors = those four slots, each replaced by a
  weather-neutral donor (ability and moves free of weather setting or weather use). 768 battles per
  team per replicate. These are chosen, not random: reported as their own bar, never pooled.

## Benchmarks (same 2^4 design; 24 random anchors + 24 hill-climbed anchors, 100 single-edit
first-improvement proposals; factors = 4 distinct random coordinates, each set to a fixed other value)
- Gridded sphere, 25 × 5 levels, f = Σx² — sanity: each term depends on one factor only, so the
  order-1 share must be 1 (checked as ≥ 0.99).
- Spin glass, 25 binary spins, couplings J_ij ~ N(0,1) with edge probability 0.2, fields h_i ~ N(0,1),
  f = Σ J_ij s_i s_j + Σ h_i s_i — sanity: order ≥3 share must be exactly 0. A random instance of the
  class, NOT COMBO's Ising sparsification task.
- Weighted Max-3-SAT, 25 variables, 106 random 3-clauses, weights U(0,1), f = satisfied weight —
  interactions up to order 3. A random instance, NOT the wMaxSAT28/43/60 competition files.
- NK landscape, N = 25, K = 4 — interactions up to order 5.
- Pest Control (COMBO, 25 stages × 5 choices) — stochastic; the two replicates use independent
  simulation seeds, one seed shared across the 16 points of a replicate.

## Thresholds, fixed now
T1 — higher-order interaction present in VGC: bootstrap 95% CI of the primary VGC order-≥3 share
lies above 0.
T2 — beyond Pest Control: bootstrap 95% CI of (VGC order-≥3 share − Pest Control order-≥3 share,
random anchors) lies entirely above 0. Reported against every benchmark, but the claim is worded on
Pest Control, the only benchmark here taken from the literature as-is.
Signal check: the pooled noise-corrected cube variance Σ_{S≠∅} â² must have a CI above 0, else the
shares are undefined and the result is "no measurable variation under four species edits".
Sanity: sphere order-1 ≥ 0.99; spin glass order-≥3 = 0 (to 1e-9).

Wording allowed:
- T1 and T2 → "win rate has more higher-order interaction than Pest Control under four species edits".
- T1 only → "win rate has measurable higher-order interaction, not more than Pest Control".
- T1 fails → "no measurable interaction beyond pairs under four species edits".

## Mutant cycles for the blog
- Third-order cycle ε_ijk = Σ_{x∈{0,1}³} (−1)^{3−|x|} f(x) on the 2³ sub-cube with the fourth
  factor off (0/1 coding: 0 = original).
- Weather cubes: the cycle {Pelipper, Charizard-Y, Venusaur} with Archaludon kept, reported for all
  three anchors whatever the result.
- Primary cubes: the two cycles with the largest |ε| are SELECTED on replicate 1 and REPORTED on
  replicate 2 (winner's curse control). The pooled value is shown only alongside.
- Pairwise cycles ε_ij reported for the same sub-cubes.

## Added after the weather battles, before the primary battles finished (2026-09-30), not pre-registered
- Empirical SE of a pooled three-way cycle under common random numbers: sd(ε3_rep1 − ε3_rep2)/2 over
  every anchor × trio of a group. Reported beside the pre-registered independent-noise SE, never instead.
- Weather cycles are displayed in presence coding (+ = the original core member kept), which flips the
  sign of the three-way ε relative to the edit coding above; pairwise ε are unchanged.
