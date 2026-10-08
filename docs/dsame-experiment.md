# Deep-surrogate MAP-Elites vs the existing search loop — pre-registration (2026-09-30)

Written before any battle of this experiment. Script: `scripts/dsame_experiment.py`; output
`results/dsame_experiment.json`. Source task: "Prototype deep-surrogate MAP-Elites for team search…"
(vault, New Todo Section; Todo Section 2026-08-20 item). Method after Zhang et al., *Deep Surrogate Assisted
MAP-Elites for Automated Hearthstone Deckbuilding* (GECCO 2022): an online-trained neural surrogate predicts
fitness and behaviour measures; MAP-Elites runs on the surrogate; the surrogate archive's elites get real
evaluations; the surrogate is retrained on them; repeat.

## Arms (same start, same battle budget)
Both arms start from the same labels: the 200 real anchor teams and the 128 generation-0 proposals of
`results/activesearch.json` (24 battles each vs the top-50 meta pool). Each generation, each arm battles
**128 new teams × 24 battles** with `pool.score` against all 50 top-50 files (featured/ included), exactly the
existing loop's evaluation. **6 generations**, **3 seeds** (11, 22, 33).

- **baseline — the existing search loop:** `gradloop.py`'s `combined` arm (ridge fitted on the arm's own
  labels → decode-time gloss guidance λ = 108 → 512 valid proposals → battle the 128 the ridge ranks highest →
  re-steer: fine-tune p0 on the top-25% elites). p0 = `results/temperature_p0.pt` (the restored shared
  checkpoint; the original `activesearch_p0.pt` is gone). Proposal seed 1000·seed + g.
- **dsame — deep-surrogate MAP-Elites:**
  - *Measures:* (1) **static offense/bulk ratio** = mean over the six of ln(max(Atk, SpA)) − ½·ln(HP·(Def + SpD)/2)
    on level-50 Champions stats (HP = base + SP + 75; others = (base + SP + 20) × alignment); base form, no Mega
    stats. (2) **mean battle length in turns** over the team's battles (from the battle runner).
    The vault's revised measure pair (2026-08-26).
  - *Grid:* 12 × 12 cells; bin edges = 1st–99th percentiles of each measure over the pilot teams (below), fixed
    before search. Out-of-range values clip to the edge bins.
  - *Surrogate:* an ensemble of 3 MLPs (input = the ridge's field + species-pair indicator vector; hidden 256 →
    128; outputs win rate and standardised turns), trained from scratch each generation on all the arm's labels
    (AdamW, 150 epochs). Training wall-clock, epochs and parameters are recorded.
  - *Inner loop (no battles):* the surrogate archive starts from every labelled team, placed by (static ratio,
    predicted turns) with predicted win rate. 8,000 children per generation: pick a uniformly random elite,
    apply one of the project's legality-conditioned mutation operators A–F (move, ability, item, alignment,
    spread, whole-Pokémon copy), keep it only if Showdown's validator accepts it. **Insert rule:** a child
    competes only in its own cell, against at most one incumbent, and replaces it only with a higher predicted
    win rate.
  - *Evaluation:* the 128 highest-predicted elites of the surrogate archive not yet battled.

## Measure-calibration pilot (counted separately)
The 328 starting teams have win-rate labels but no turn counts. They are battled once more at 24 battles
(7,872 battles) **only to record turns** and set the bin edges; the pilot's win rates are not used by either
arm. The pilot also gives the correlation matrix (turns, static ratio, win rate) the measure note asked for.

## Measurements
1. **Primary — verified win rate.** Each arm's 8 best teams by search label (over all its generations) get
   16 fresh battles per cell against the 49 distinct top-50 teams (`matchup_db`, MB493 = MB494 weighted twice).
   Win rate = wins ÷ battles, behaviour-cloning policy both sides; never used for selection.
2. Search-phase win rate per generation (24-battle labels; winner's-cursed, reported as such).
3. **Diversity** (Metrics for Diversity contract, `diversity_metrics.measure`, reference = the corpus) over all
   768 evaluated teams per arm: unique fraction, novel-draw and novel-unique fractions for the 48-field and
   with-spreads keys, distinct compositions and the largest composition's share. Plus the MAP-Elites view:
   coverage and QD-score of each arm's evaluated teams placed in the same 12 × 12 grid by real measures.
4. **Surrogate training cost:** seconds per generation, epochs, parameter count, total; the baseline's ridge
   fit and re-steer fine-tune seconds for comparison.
5. **Surrogate validation (the Surrogate Model task, battle-free, run before search):** ridge vs deep
   surrogate, both fitted on the 328 starting labels: (a) Spearman on held-out labels (activesearch2);
   (b) on 720 single legal edits with fresh battles (`results/ruggedness.json` + `ruggedness2_parts`):
   correlation of predicted Δ with measured Δ, and sign agreement on edits whose |Δ| exceeds 1.96 SE.
   Note: those labels were battled against 36 of the top-50 (the featured/ bug); used only for validation.

## Decision rule, fixed now
Δ = dsame − baseline on the primary measure, pooled over the 3 seeds (24 teams per arm); 95% CI by bootstrap
over teams within seed (10,000 resamples). CI above 0 → "deep-surrogate MAP-Elites found stronger teams at
this budget"; below 0 → "weaker"; otherwise "no detectable difference at this budget". Diversity, cost and
surrogate validation are reported whatever the outcome; the vault question "does a fitted estimator of f earn
its training cost" is answered from (primary Δ, surrogate seconds) together.

## Amendments before any search battle (2026-10-01)
1. Surrogate validation: the held-out set was `as2*`, which includes the 200 training anchors re-battled
   (`as2_anchor`). Corrected to `as2` only (2,176 labels): ridge 0.250, deep 0.266 (was 0.315 / 0.327).
2. Mutation operators: the script drew from A–G; restricted to the pre-registered A–F.
3. Verification: each arm-seed's top 8 is deduplicated, and every verified team gets 16 *new* battles per
   opponent (target = battles it already has + 16); `ensure()` alone would give a team already in the database
   no fresh battles.
