# Ruggedness under a single legal edit — pre-registration (2026-09-27)

Claim under test: win rate moves under one small legal edit by more than battle noise.

Design (scripts/ruggedness.py): 24 real Reg M-B corpus anchors (not in the top-50 opponent
pool); 15 single edits each (3 × move, ability, item, alignment, spread-copy; conditional
operators from mutation_closure_test.py; every child passes Showdown's TeamValidator);
384 battles per team vs the top-50 meta, behaviour-cloning policy both sides, one fixed
opponent schedule; each anchor re-measured 3× with fresh seeds as the noise null.
Comparators, same 24 × 15 × 3 design: COMBO Pest Control (25 × 5, one stage changed,
100 simulated trajectories) and Hartmann-6 (one coordinate ±0.01).

Primary numbers, fixed before the run:
1. fraction of VGC edits whose |Δ win rate| > 1.96 × SE(Δ) (null expectation 5%);
2. noise-corrected neighbour correlation ρ = 1 − (E[Δ²] − E[Δ²_replicate]) / (2 var f),
   bootstrap 95% CI over anchors, for all three landscapes.
Reported whatever the outcome.
