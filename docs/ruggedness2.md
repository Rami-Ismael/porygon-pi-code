# Ruggedness per edit type, from meta and search-found teams — pre-registration (2026-09-28)

Extends docs/ruggedness.md (run 1: 24 meta anchors, edits A–E, Pest Control, Hartmann-6).
Written before any run-2 battle. Script: scripts/ruggedness2.py; output results/ruggedness2.json.

## What is added
- Edit type F (species edit): one Pokémon replaced by a real corpus set of a species not on the
  team (candidate-copy, Species + Item Clause repaired; operator F of mutation_closure_test.py).
  Repair rule is fixed: the whole set is copied, nothing of the old slot is kept.
- Search-found anchors: the 16 teams of results/rebattle_top.json (combined gradient-guidance loop).
  Edits A–F × 3 each; anchor measured once (seed 101) + 3 replicates (seeds 202/303/404).
- Meta anchors: the 24 of run 1, reused as measured; new F edits × 3 each.
- VGC f = win rate vs the top-50 meta, behaviour-cloning policy both sides, 384 battles,
  one fixed ordered opponent schedule (common random numbers), edits at seed 202 as in run 1.
  Every child passes Showdown's TeamValidator.
- Benchmarks, same 24 random + 16 search-found anchors, 15 edits, 3 replicates:
  - Pest Control (COMBO, 25 stages × 5 choices), edit = one stage set to another value.
  - Gridded sphere, 25 coordinates × 5 levels {−1, −½, 0, ½, 1}, f = Σx²; primary edit = one
    coordinate moved one grid step (inward at the edge); secondary edit = one coordinate set to
    any other level (the Pest Control neighbourhood).
  - Search-found benchmark anchors = first-improvement hill climb from a random point, 100
    single-edit proposals, minimising f (noisy f for Pest Control, as the optimiser would see it).

## Statistics (per landscape × start type × edit type)
- Noise variance σ²ₙ from replicates: E[r²] = mean (replicate − anchor)².
- Noise-corrected RMS edit size Δ* = √(E[d²] − E[r²]), d = edit − anchor (win-rate units for VGC).
- var_f = variance of anchor values pooled over BOTH start types of that landscape, minus mean σ²ₙ
  (pooled so search-found anchors, whose spread is range-restricted, share one scale).
- s = Δ*/√var_f (edit size in units of f's spread); ρ = 1 − s²/2 (noise-corrected neighbour
  correlation). 95% CIs by bootstrap over anchors (2,000 resamples).
- VGC only: fraction of edits with |Δ| > 1.96·SE(Δ), against the same fraction for replicates.

## Threshold, fixed now
T1 — beyond battle noise. An edit type passes if (a) its fraction beyond 1.96·SE exceeds the
replicate fraction with the bootstrap 95% CI of the difference above 0, and (b) Δ* ≥ 0.05 win rate.
T2 — beyond the benchmark. VGC is more rugged than Pest Control under an edit type and start type
if the bootstrap 95% CI of ρ_VGC − ρ_Pest lies entirely below 0 (meta vs random anchors;
search-found vs search-found).
Sanity: the gridded sphere (primary edit) must give ρ ≥ 0.95, else the comparison is broken.

Wording the subsection is allowed to use:
- T1 and T2 hold for some edit type → "win rate is rugged under a single legal edit, beyond
  battle noise and beyond Pest Control" (naming the edit types).
- T1 holds, T2 fails everywhere → "one edit moves win rate beyond battle noise, but win rate is
  not more rugged than Pest Control".
- T1 fails everywhere → the non-smoothness claim is dropped.
Reported whatever the outcome.
