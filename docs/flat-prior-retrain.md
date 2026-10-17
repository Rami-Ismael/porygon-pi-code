# Flat-prior retrain: family-balanced sampling of the training rows (pre-registration, 2026-10-17)

Written before any training step or battle (labels reworded to plain names (loop-teams model, battle-records model) on 2026-10-17 while the
retrain ran; no setting or rule changed). The result is appended under "Result"; nothing above
that heading is edited afterwards.

**Question.** `scripts/species_prior_check.py` (2026-10-17, `results/species_prior_check.json`) found
that the generation-0 prior is not uniform over species: the unconditional mixture puts Garchomp on
~10% of species slots and the top six species on ~45%, against 5.7% and 28.7% in the 622 real teams,
and asked 0.5 at guidance 2 collapses the sampled species distribution from exp(entropy) 40 to 14
(136 distinct line-ups per 512). The skew is the training data's: the loop-written rows that dominate
it carry it. Does giving the real teams as much weight as the loop-written rows — a flatter species
prior — widen the generation-0 line-ups without losing win rate?

**Premise correction, found while writing this.** The 58%-temperature-family figure describes the
whole 146,944-row `lossdata.npz`. The two clean mixture members do not train on all of it. The
**loop-teams model** (`F1_final_medium_noheldout`, `data=family`) trains on 17,129 rows — 622 real (`tag_of` = 0) and 16,507
August–September loop rows (`oldpool` / `loops08`, `tag_of` = 1) — and the **battle-records
model** (`G2_family_matrix_noprtrain_noheldout`, `data=family+matrix`) on those plus 5,414 cemrev / redund /
matrix rows (tag 3) and 4,321 random-fill rows (tag 4): 26,864 rows. Neither sees a temperature row.
The skew is the same inside what they see: the loop family has Garchomp on 10.6% of slots and the
top six on 47.1% (exp(entropy) 36.4); the real rows 5.7% / 28.7% (63.1); cemrev / matrix 10.0% / 46.7%;
random fill 6.4% / 31.1%. So the lever is the same — weight the source families equally — applied
to the rows each member actually trains on.

**Win rate** (every number here) = wins / battles against the top-50 meta teams, Showdown battles
with the behaviour-cloning policy on both sides (policy 1). Monte Carlo evaluation, not ground truth.

## Arm: `fambal` — per-row sampling weights that equalise the source families

- Groups = `lossdown.tag_of(d["S"])`, the repo's own partition of the family bitmask (real; the
  August–September loops; cemrev / redund / matrix; random fill; HPS; temperature). Exact bitmask values
  are not used as groups: the member's rows hold bitmask groups of 4, 16 and 43 rows (real teams that
  also sit in the old pool or carry matrix entries), and equal mass per exact bitmask would show each of
  those rows hundreds of times per epoch.
- Weight of a row = 1 / (number of training rows in its group), so every group has the same expected
  share of each epoch's draws. Loop-teams model: real 50%, loops 50% (a real row is drawn ~26× as
  often as a loop row). Battle-records model: real, loops, cemrev / matrix, fill 25% each.
- Draws per epoch = the number of training rows, with replacement (`torch.multinomial` on the
  weights), so the step count, schedule and every other setting are the control's:
  `d=256 nlayer=6 nhead=8 sched=16,4,4,1.5,1 epochs=80 (loop-teams) / 60 (battle-records) batch=256 lr=4e-4 ema=0.999
  p_uncond=0.15 final=1 seed=0`, plus `mrow=1 data=family+matrix` for the battle-records model. Ranks 51–100 excluded as
  before. New config key `fambal=1` in `lossdown.train_run`; the control is `fambal=0` (the existing
  code path, `torch.randperm`).
- Checkpoints: the control names with `_fambal` appended (`F1_final_medium_noheldout_fambal.pt`,
  `G2_family_matrix_noprtrain_noheldout_fambal.pt`)
  (`~/.local/share/vgc-pilot-runtime/lossdown/`), rows in `results/lossdown.json` under those names.
  The two control checkpoints are not touched.
- Expected species mixture after balancing (from the group figures above): Garchomp ~8%, top six
  ~38% in both members — a partial flattening, not uniform. Reported, not part of the rule: the
  test-500 ELBO of each member (expected to worsen, since the benchmark is loop-written teams), and
  the exact copies of real teams among the accepted proposals (real rows are oversampled ~26×).

**Control.** The existing clean members, `lossdown_battle.MEMBERS`, sampled again with the same
seeds. Not reused from an earlier file: both arms are fresh draws.

## Measurement: generation 0, the `sampler_qd_run` protocol (`src/flat_prior_run.py`)

- Sampler: `lossdown_battle.sample` — one field per pass in `diffusion.ORDER`, classifier-free
  guidance 2 on the win-rate condition, the legality mask the members were trained with,
  temperature 1, corpus Stat Point spreads (`activesearch.row_to_paste`). No ridge, no loop.
- Cells: arm {control, fambal} × asked {0.5, 0.7} × sampling seed {1, 2, 3}. Per cell the first
  64 Showdown-validator-accepted teams in sampling order (cap 8 tries per accepted team; a team
  that is a held-out rank 51–100 team on 48 fields is refused). Both arms start each (asked, seed)
  from the same torch and numpy state, so they differ only by the checkpoints.
- Battles: one per (team, top-50 column), 49 per team, policy 1, matchup-matrix origin
  `flatprior:<arm>_y<asked>_s<seed>`, `results/flat_prior_run.json`.
- Confirm stage (`sampler_qd_run.confirm`): the best team of every line-up that cleared 0.531 gets
  one more battle per column (49 fresh battles, note `flatprior confirm`); confirmed = the fresh
  battles alone score ≥ 0.531.
- Per cell: distinct line-ups per 64, species exp(entropy) over the 384 species slots, largest
  line-up share, winning line-ups and QD-score (`sampler_qd.qd`, bar 0.531), mean win rate with a
  bootstrap 95% interval (teams resampled, 4,000 draws), best team, count ≥ 0.531, copies of a
  training row (48-field match) and of a real team.
- Per arm: the mean over the six cells; arm − control win rate with a bootstrap 95% interval
  (teams resampled within cell, 4,000 draws); confirmed winners summed over cells.
- Best fresh team: the highest confirm-stage (fresh-battle) score per arm, named with its line-up.

## Decision rule (fixed before training and battles)
`fambal` **widens generation-0 line-ups without losing win rate** if
(a) its distinct line-ups per 64 exceed `control`'s in every one of the six cells, AND
(b) the bootstrap 95% lower bound of (`fambal` − `control`) mean win rate is above −0.03, AND
(c) its confirmed winners (summed over cells) are not below `control`'s.
exp(entropy), QD-score, largest share, ELBO and copies are reported, not part of the rule. A pass
earns the balanced members a place as the loop's generation-0 prior (the next experiment); a fail
retires training-set balancing at this weighting. Not rerun either way.

**Prediction, written before the run.** Line-ups per 64 up by 5–15 in each cell (the mixture only
moves the top-six share from ~46% to ~38%); win rate within ±0.02 of control; confirmed winners
about equal. The copies count is the number to watch: if real teams are being memorised, the
line-up gain is spurious and the ELBO will say so.

## Result (2026-10-17, `results/flat_prior_run.json`; 12 cells × 64 teams × 49 battles, 35 confirm teams)

Retrain: family-balanced draws cost the benchmark, as predicted — test-500 ELBO 1.783 → 1.816
(loop-teams model) and 1.787 → 1.828 (battle-records model); unconditional 1.797 → 1.897
and 1.800 → 1.879. No proposal copied a training row or a real team (0 of 768 in either arm).

| cell (asked, seed) | line-ups /64 control → fambal | largest share | win rate control → fambal |
|---|---|---|---|
| 0.5, s1 | 36 → 34 | 42% → 45% | 0.386 → 0.357 |
| 0.7, s1 | 45 → 61 | 20% → 5% | 0.366 → 0.302 |
| 0.5, s2 | 22 → 21 | 63% → 64% | 0.384 → 0.379 |
| 0.7, s2 | 44 → 60 | 20% → 5% | 0.349 → 0.302 |
| 0.5, s3 | 30 → 29 | 55% → 56% | 0.400 → 0.369 |
| 0.7, s3 | 37 → 59 | 23% → 5% | 0.365 → 0.258 |

Mean over cells (win rate vs the top-50 meta teams, behaviour-cloning policy both sides): control
line-ups 35.7, species exp(entropy) 16.2, winning line-ups 16, QD-score 78.0, win rate 0.375,
confirmed winners 10, best fresh team 0.673 (Basculegion / Chandelure / Floette-Eternal / Garchomp /
Milotic / Sinistcha); `fambal` line-ups 44.0, exp(entropy) 16.4, winning line-ups 6, QD-score 81.7,
win rate 0.328, confirmed winners 3, best fresh team 0.612 (Charizard / Garchomp / Incineroar /
Milotic / Mimikyu / Venusaur). `fambal` − control win rate: **−0.047 [−0.063, −0.032]**.

**Verdict: `fambal` FAILS all three clauses.** (a) Line-ups vs control per cell −2, +16, −1, +16, −1,
+22: unchanged at asked 0.5, wider only at asked 0.7. (b) Win-rate lower bound −0.063 < −0.03.
(c) Confirmed winners 3 vs 10. The prediction (+5 to +15 line-ups per cell, win rate within ±0.02)
was wrong on both counts: at asked 0.5 the guidance collapses the balanced prior onto the same
line-ups as before, and at asked 0.7 the extra line-ups are weak ones (largest share 20% → 5%,
win rate −0.06 to −0.11). Species exp(entropy) over the 384 slots barely moved (16.2 → 16.4): the
balanced members did not become flatter *after guidance*, only less concentrated on one line-up at
the high ask. Training-set balancing at this weighting is retired; not rerun. The balanced
checkpoints stay on disk under their `_fambal` names and are not used by any loop.

**Addendum (2026-10-17, after the verdict; does not change it).** Where the narrowing happens: the
first species the mixture writes, all fields masked (`/tmp/vgc-pilot/flat/first_slot_prior.json`).
Effective species = exp(entropy).

| setting | control: effective species, top-six share | balanced: effective species, top-six share |
|---|---|---|
| no guidance | 40.2, 43% | 56.6, 32% |
| ask 0.5, guidance 2 | 14.3, 75% | 13.5, 78% |
| ask 0.7, guidance 2 | 11.0, 73% | 15.7, 67% |

Balancing flattened the prior as intended. Guidance at ask 0.5 pulled both back to the same six
species (Garchomp, Charizard, Floette-Eternal, Kingambit, Basculegion, Whimsicott). The training-set
skew is not what sets the generation-0 line-ups at ask 0.5; the guidance step is. Explainer:
vault `Teaching/lessons/0012-balancing-the-training-teams.html`.
