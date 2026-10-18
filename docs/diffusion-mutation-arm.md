# Mutation operator inside deep-surrogate MAP-Elites: hand-written vs uniform refill vs diffusion refill — pre-registration (2026-10-17)

Written before any battle of this experiment. Script: `scripts/diffusion_mutation_arm.py`; output
`results/diffusion_mutation_arm.json`. Protocol: the deep-surrogate MAP-Elites arm of
`docs/dsame-experiment.md`, rerun three times with only the mutation operator changed.

## Question
Inside deep-surrogate MAP-Elites, does refilling masked fields with the version-3 masked diffusion model
find stronger teams than the hand-written operators? The hand-written operators are legal by construction
(90/100 chains legal at 50 steps, `results/mutation_closure_test.json`), so the comparison is on win rate,
not legality. Legality and novelty are reported but are not evidence for or against an operator.

## Arms (identical except the mutation operator)
Shared by all arms: the 313 starting labels (the DSA-ME start set of 328 = 200 anchors + 128 generation-0
teams, minus the 15 that are held-out ranks 51–100 teams by 48-field identity, `heldout.py`); the DSA-ME
pilot's turn counts and 12 × 12 bin edges (offense/bulk ratio × battle length), reused from
`results/dsame_experiment.json`; the surrogate (ensemble of 3 MLPs, 150 epochs, retrained from scratch each
generation, seed 100·seed + g); the insert rule (a child competes only in its own cell, replaces the
incumbent only with a higher predicted win rate); 8,000 Showdown-valid children per generation in batches of
200 parents drawn uniformly from the archive; the gate (Showdown's TeamValidator through
`hps_generate.Validator`, reject and move on; a held-out-team child is dropped); evaluation of the 128
highest-predicted archive elites not yet battled, 24 battles each against the top-50 pool (`pool.score`,
behaviour-cloning policy both sides) on this run's own Showdown servers, ports 8401–8406.

- **A — hand-written:** one of the operators A–F of `scripts/mutation_closure_test.py`, chosen uniformly
  (move swap, ability swap, item swap, alignment swap, spread copy, whole-Pokémon copy from a real set with the
  Item Clause fix), exactly as `scripts/dsame_experiment.py` uses them. Its edit size is the operator's:
  one model field (A–D, 4/6), a Stat Point spread only (E, 1/6), or one whole Pokémon (F, 1/6).
- **B — uniform refill:** draw a mask on the parent, refill every masked field uniformly at random from the
  values the v3 legality tables (`lossdown.Tables.allowed`: Species and Item Clause, learnsets, abilities,
  Mega stones, sorted move order) allow given every visible field, in the decoder's column order (species,
  ability/item, moves, alignment).
- **C — diffusion refill:** the same mask, refilled by the frozen v3 members
  (`F1_final_medium_noheldout_regmb_v3` + `G2_family_matrix_noprtrain_noheldout_regmb_v3`), ask 0.5,
  classifier-free guidance 2, temperature 0.5 on every non-species field and 1 on species, same legality
  tables, same column order; t = fraction of the 48 fields still masked.

**Mask (B and C, the same sampler):** one Pokémon slot chosen uniformly; with probability 1/6 the whole
Pokémon (all 8 model fields: species, ability, item, four moves, alignment); otherwise k ∈ {1, 2, 3, 4}
uniform non-species fields of that slot, chosen without replacement. Mean 3.4 masked fields. The 1/6 whole-
Pokémon share matches arm A's operator F. Stat Points are not a v3 field: a whole-Pokémon refill takes a
random real spread of the new species from the corpus (a uniformly random legal spread if it has none) in both
B and C; a partial mask keeps the parent's spread. Empty item and empty move are excluded from B and C
refills because arm A never produces them.

Known asymmetry, accepted: the v3 encoding stores a Pokémon's moves in sorted order, so a masked move position
in B and C can only take values between its visible neighbours; arm A's move swap has no such restriction.
Arm A's mask distribution is different from B/C's (reported); B vs C is the exact refill-only contrast.

## Budget
6 generations × 128 teams × 24 battles, 3 seeds (11, 22, 33): 9 arm-seeds, 165,888 search battles.

## Measurements
1. **Primary — verified win rate.** Each arm-seed's 8 best distinct teams by search label (over all six
   generations) get 16 new battles per opponent against the 49 distinct top-50 teams (`matchup_db`,
   behaviour-cloning policy both sides, MB493 = MB494 weighted twice), win rate = wins ÷ battles. Never used
   for selection. 24 teams per arm.
2. Held-out ranks 51–100: the same 72 teams, 384 battles each against the 50 held-out teams (`heldout.evaluate`).
3. Per arm: best verified team as six species; distinct species compositions among the 768 evaluated teams
   per seed; Showdown-valid rate before the gate (valid ÷ submitted to the validator; for A also the share of
   operator no-ops, which never reach the validator); wall clock per generation (total, inner loop, battles);
   mask-size distribution.

Errors are clustered by team: every interval is a bootstrap over teams, stratified by seed (10,000 resamples),
so battle noise and team-to-team spread both enter.

## Decision rule, fixed now
Primary contrast Δ = C − A on verified win rate, pooled over the 3 seeds (24 vs 24 teams).
- **PASS** ("the diffusion kernel finds stronger teams than the hand-written operators at this budget"):
  95% CI of Δ above 0 **and** the held-out ranks 51–100 point estimate of C − A above 0.
- **FAIL — worse:** 95% CI of Δ below 0.
- **FAIL — no detectable difference:** anything else. The diffusion kernel then does not earn its compute.

Secondary, same CI rule, reported as stronger / weaker / no detectable difference: C − B (does the learned
refill beat uniform refill on the identical mask) and B − A (does the mask-and-uniform-refill family differ
from the hand-written operators).

## Amendments from the smoke test, before the run (2026-10-17)
1. The offense/bulk measure needs base stats; 24 v3 species are cosmetic formes (Alcremie and Vivillon
   patterns) with none in the old dex. They take their base species' stats. Every other species is unchanged.
2. `pool.REPO` is pinned to the unpacked vgc-bench runtime, as in `cem_v3.py`, because the staging copy has no
   policy package. The opponents and the policy are unchanged.
