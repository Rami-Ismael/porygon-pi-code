# Masked diffusion tree search vs direct sampling — pre-registration (2026-09-30)

Written before any battle. Script: `scripts/mdts_experiment.py`; output `results/mdts_experiment.json`.
Paper: Diffusion Large Language Models for Black-Box Optimization (arXiv 2601.14446, preprint): masked
diffusion tree search (MDTS) beat "w/o MDTS" (same number of directly sampled candidates ranked by the same
predictor) on 3 of 4 tasks (TF Bind 10: 0.642 vs 0.503).

## Question
With the same generator, training data, candidate scorer and battle budget, does searching over partially
generated teams find teams with higher verified win rate than sampling complete teams directly?

## Held fixed across the two arms
- **Generator:** `results/temperature_p0.pt` (600 epochs, 692 corpus teams; shared by the recent experiments),
  unconditional, the project's constrained dependency-order decoder, temperature 1.0. No retraining.
- **Training data for the scorer:** every battle label `gradguide.all_labels()` collects (activesearch 1 and 2).
- **Candidate scorer:** the ridge surrogate (`activesearch.Feats`, field + species-pair indicators) fitted once
  on those labels, with its Bayesian posterior variance σ²·φᵀ(XᵀX + Λ)⁻¹φ (σ² = residual variance). Score =
  expected improvement over the best label, as the paper's Gaussian-process EI.
- **Scoring budget:** the same number of complete teams scored per replicate (tree first, ≈ 400; the
  baseline then draws exactly the tree's count).
- **Selection:** the 16 highest-EI teams that are distinct (canonical form) and Showdown-valid; Stat Points
  drawn from corpus spreads of the species with a seeded generator (the project convention).
- **Battle budget:** 16 teams × 49 distinct top-50 columns × 8 battles (score via `matchup_db.score_pastes`),
  then the 4 best by that score get 16 more battles per cell (fresh: verification).

## Arms
- **direct:** draw N complete teams, score each with EI, select.
- **tree (MDTS):** root = fully masked team. Levels follow the decoder's dependency order: 1 species (6
  fields), 2 abilities + items (12), 3 moves (24), 4 alignments (6). Each iteration: select by
  UCT = V + ω·prior·√(log N_parent / N_child), ω = 1, prior = siblings' normalised model probability of
  the revealed stage; expand 5 children (sample the next stage); value of a child = mean EI over 5
  completions (a level-4 child is complete: its own EI); back up the mean child value along the path.
  Stop when ≥ 400 complete teams have been scored. Candidates = every complete team scored.
- **Deviation from the paper (forced):** the paper re-masks the least confident tokens; the constrained
  decoder must reveal fields in dependency order, so each level reveals the next stage.

## Replicates and measurements
6 replicates (seeds 1101 … 1606); in each, both arms run with their own seeded streams.
1. **Primary — verified win rate:** mean fresh-battle win rate (16 battles per cell, never used for
   selection) of each arm's top 4. Win rate = wins ÷ battles vs the 49 distinct top-50 meta teams (MB493 =
   MB494 weighted twice), behaviour-cloning policy both sides.
2. First-battle win rate of the 16 selected (8 per cell).
3. Diversity of the 16 selected: distinct species sets, mean pairwise Hamming distance on the 48-field grid,
   copy rate and nearest-neighbour Hamming to the corpus.
4. Runtime: proposal wall-clock (generation + scoring) per arm; battle wall-clock.
5. Scorer's own view: mean predicted win rate and EI of the selected teams (does the search do its job on
   the predictor?).

## Decision rule, fixed now
Paired by replicate, Δ = tree − direct on the primary measure; 95% CI by bootstrap over the 6 replicates
(10,000 resamples).
- CI above 0 → "tree search raised verified win rate at this budget".
- CI below 0 → "tree search lowered verified win rate at this budget".
- Otherwise → "no detectable difference at this budget" (reported with the CI, not as "no effect").
Reported whatever the outcome; secondary measures are descriptive.

## Amendment before any battle (2026-09-30, after the smoke test)
EI against the best raw label (0.833, a 24-battle label) was ~1e-17 for every candidate: a degenerate ranking.
The incumbent is changed to the best *fitted* mean among labelled teams, the usual convention for expected
improvement under noisy observations. Both arms use the same scorer; nothing else changes.
