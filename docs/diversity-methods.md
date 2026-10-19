# Four ways to keep composition diverse across generations (pre-registration, 2026-10-08)

**Question.** The version-3 cross-entropy loop (`cem_v3.py`) lifted the battled mean 0.427 → 0.564 but
distinct species sets among the 512 proposals fell 176 → 161 → 44 and the 16 finalists were one core.
Which change to the loop holds composition diversity across generations without losing win rate?
Already failed, not retried: entropy/Boltzmann/niche elite weights, MMR (`redundancy-penalty.md`),
negative score, larger pools, drift fixes.

**Win rate** = wins / battles against the top-50 meta teams, one battle per (team, top-50 column),
BC policy both sides (policy 1), all in the matchup matrix (origin `divm:<arm>_s<seed>_g<gen>`).

## Shared design (`src/diversity_loops.py`)
- Start: the cem_v3 generation 0 (its 512 proposals, 128 battled teams, labels, seed 4040), identical for
  every arm and seed. Arms branch from generation 1; generations 1–6.
- Per generation, as in cem_v3: re-steer both v3 members from the ORIGINAL checkpoints on the elites
  (≥ 0.531 incl. the 49 anchors, min 64; 40 epochs, lr 1e-4), refit the ridge with loop labels ×10,
  propose with ask 0.5, guidance 2, temperature 1, ridge guidance λ 35.4, nature repair; battle 128.
- **No stopping guardrails**: diversity is the outcome here, so a collapse must be allowed to show.
- Seeds 1, 2, 3 (fine-tuning and sampling). Three seeds: intervals are exploratory.

## Arms
| arm | change to the control loop |
|---|---|
| `control` | none (cem_v3 with its stopping rules off) |
| `taboo` | proposal-time hard rule: reject a proposal whose composition (unordered six species) appeared in any proposal of an earlier generation, or has already been accepted 2 times this generation |
| `peel` | logit-mask species out of the sampler. Epochs of two generations; banned set B1 (g1–2) = the 3 most frequent species among the 16 best-labelled battled teams of generation 0; B2 added for g3–4 and B3 for g5–6, each the 3 most frequent not-yet-banned species among the 16 best-labelled teams of the epoch before. Elite teams containing a banned species are dropped from the fine-tuning set |
| `island` | 4 islands, 128 proposals and 32 battled teams each. Seeds: farthest-first (species Jaccard) among the 49 anchors, starting from the best-labelled anchor; each island pins that anchor's first 3 sampled species columns. Generation-0 teams are split among islands by species overlap with the seed. Each island re-steers only on its own elites (≥ 0.531, min 32, else the island's top 32). Migration before g3 and g5: from each other island its best team sharing ≤ 2 species with the receiving island's six most frequent species. The ridge is shared |
| `big` | control, but 4,096 accepted proposals and the top 128 by ridge score are battled (isolates pool size from `dpp`) |
| `dpp` | same 4,096 proposals as `big`; the 128 battled are the greedy MAP of a determinantal point process (Chen et al. 2018): L = diag(q) S diag(q), q = exp(ridge score / 0.05), S = shared species / 6 (Gram of species indicators) |

`island` and `peel` build diversity partly by construction; their informative outcomes are the win-rate
guard and the number of qualifying cores, not the composition counts alone.

## Verification (method of [[Metrics for Diversity]], `diversity_metrics.measure`)
Every generation, the **first 512 validator-accepted proposals in sampling order** of the arm's stream
(before scoring, ranking, deduplication; hard cap 8 attempts per accepted proposal, no post-validation
repair), measured against the cem_v3 corpus reference (labelled `corpus`):
1. distinct compositions /512, exp(H) of the composition frequencies, largest composition share
2. unique, repeat, novel-draw and novel-unique fractions on both the 48-field and with-spreads keys
3. for `taboo`, the same measures on the **raw** stream before the taboo filter (model diversity vs
   output diversity); for `island`, the pooled 512 = 4 × 128
4. of the **battled** set: distinct compositions, and qualifying cores = distinct compositions among
   battled teams with label ≥ 0.531, cumulative over generations
5. an **unranked** guard: 32 random proposals per arm-seed at g3 and g6, 49 battles each (generator win
   rate, not the score-selected batch)
Reference points: original checkpoint 498.3 compositions, largest share 1.4%; fixed-temperature arms
fell to 172.0 and 40.5%; cem_v3 generation 0 is 176 compositions.
Final: per arm-seed the 16 best distinct battled teams re-battled at 196 battles per team (4 per top-50 opponent, through the matchup matrix).

## Decision rule (fixed before battles)
An arm **keeps diversity without losing win rate** if, at generation 6,
(a) its distinct compositions exceed `control`'s in all 3 seeds and are ≥ 176 (generation 0's level), AND
its largest composition share is below `control`'s in all 3 seeds; AND
(b) its generation-6 battled-batch win rate minus `control`'s has a bootstrap 95% lower bound above −0.03
(teams resampled within seed; for `island` and `dpp` the batch is its battled set, which is a different
selection rule by design — read together with the unranked guard and the re-battled best).
Composition diversity does not establish strategic variety; the evidence for variety is (a) plus a
non-zero count of qualifying cores that are not the Basculegion/Chandelure/Floette-Eternal/Garchomp/
Kingambit/Whimsicott core.
Do not rerun an arm that passes or fails; the only open variants are longer runs.
