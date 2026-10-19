# Diffusion-BBO w = 1 port, no pretrained model (pre-registration, 2026-10-16)

*Renamed 2026-10-16 (owner): not "faithful". The paper picks w ∈ {0.6, …, 1.0} each round with an ensemble uncertainty score; these runs fix w = 1. File and tag names (`dbbo_faithful*`) are kept as identifiers. The full rule is `dbbo_paper` below.*

**Question.** How does Diffusion-BBO (Wu, Kuang, Niu, Ma & Yu, arXiv 2407.00610, Algorithm 1) do on VGC team search when it is
ported with w fixed at 1: no pretrained model and none of this project's loop machinery?

Win rate = wins / battles against the top-50 meta teams, behaviour-cloned policy (policy 1) on both sides.

## As in the paper
- Initial dataset D: the labelled teams between the 25th and 50th percentile of win rate. Taken from the matchup matrix
  (full top-50 coverage, held-out ranks never present): 15,372 teams with win rates 0.30–0.44 at smoke time.
- Each round: a conditional diffusion model trained **from random weights** on D. Label dropout 0.15 (classifier-free
  guidance), learning rate 1e-3, batch 256, 10% of D held out as validation, best-validation checkpoint kept (20 epochs max).
- Ask y = w × best label in D. Sample N teams at guidance 2, battle all of them, append them to D.

## Forced by the problem
- The network is the project's masked transformer over 48 categorical fields (the paper's tasks are continuous vectors).
- Decoding keeps the regulation's legality tables: an illegal team cannot be battled.
- Labels are 49-battle win rates; the paper's oracles have little or no noise. Labels are used raw, as the paper does.

## Dropped on purpose
- The ensemble and its uncertainty term (UaE) that picks w. This project does not use uncertainty acquisition. Without the
  term, the paper's argmax over w ∈ {0.6, …, 1.0} reduces to **w = 1** (ask the best label in D).
- No ridge, no taboo, no lowered temperature, no elite re-steer, no contrast group.

## Budget and finalists
4 rounds × 128 teams × 49 battles, the same budget as `cem_v3_noheldout`. Finalists: the 16 best distinct battled teams
by label, 192 fresh battles vs top-50, then once vs ranks 51–100. Compared with `cem_v3_noheldout` (0.504 / 0.509) and
`cem_scratch` (the same training idea inside our loop).

## Precise-label w = 1 port (`dbbo_faithful_precise`, pre-registration 2026-10-16)

Requested by the owner: the same port, with every label precise. `DBBO_PRECISE=1 python dbbo_faithful.py run`.
- **Starting D = all 1,105 ask-dial precise-label teams** (about 198 battles each; win rates 0.00–0.755, median 0.405), not their
  25th–50th percentile. That set was built to cover every win-rate band evenly (up to 70 teams per 0.05 bin), so its percentiles
  do not mean what the paper's percentile of an offline dataset means. The 25th–50th slice would leave about 275 teams to train
  a model from random weights.
- **Every battled team gets 4 battles per top-50 column (196)** before its label enters D: 4 rounds × 128 × 196 ≈ 100,000 battles.
- **Training steps matched to the raw-label port (about 1,100 per round).** D is 14 times smaller, so equal epochs would give only about 80 steps from random weights.
  Epochs per round = max(20, ⌈1,100 / steps per epoch⌉); the best-validation checkpoint is still kept.
- Everything else unchanged: learning rate 1e-3, batch 256, label dropout 0.15, ask = best label in D (w = 1), guidance 2, battle
  all 128, finalists 16 × 192 vs top-50 and ranks 51–100.

Compared with `dbbo_faithful` (raw labels), `cem_v3_noheldout` (0.504 / 0.509), the plain loop at fresh seeds (pooled 0.559 /
0.558) and the `cem_dbbo` / `cem_scratch` arms.

## Result: raw-label w = 1 port (`dbbo_faithful`, 2026-10-16)

Starting D: 18,203 labelled teams between the 25th and 50th percentile (win rates 0.32–0.46) of 61,490 at run time.

| round | ask (best label in D) | battled win rate | best battled | line-ups / 128 | legal proposals |
|---|---|---|---|---|---|
| 1 | 0.460 | 0.437 | – | 54 | 100% |
| 2 | – | 0.370 ± 0.012 | 0.600 | 52 | 97% |
| 3 | 0.680 | 0.348 ± 0.014 | 0.660 | 94 | 98% |
| 4 | 0.680 | 0.193 ± 0.011 | 0.480 | 124 | **41%** |

Finalists (16 × 192): **0.461** vs top-50 (best 0.573), **0.442** vs ranks 51–100. Minus `cem_v3_noheldout` (0.504 / 0.509):
−0.043 [−0.084, −0.003] / −0.067 [−0.109, −0.026]. **Worse than the baseline on both sets.**

**Reading.** With w fixed at 1, the port gets worse each round. Its ask jumps to the best raw label (0.680) once one lucky 49-battle team
appears. That ask sits far above where almost all of D lies. The model trained from random weights each round cannot produce it, so
win rate falls and, by round 4, more than half the proposals are illegal. The precise-label variant tests whether honest labels keep
the best-so-far ask in reach.

## Result: precise-label w = 1 port (`dbbo_faithful_precise`, 2026-10-16)

| round | ask (best label in D) | battled win rate | line-ups / 128 | legal proposals |
|---|---|---|---|---|
| 1 | 0.755 | 0.247 | 54 | 90% |
| 2 | 0.755 | 0.164 | 120 | 80% |
| 3 | 0.755 | 0.201 ± 0.011 | 113 | 77% |
| 4 | 0.755 | 0.330 ± 0.013 | 96 | 88% |

Finalists (16 × 192): **0.463** vs top-50 (best 0.625), **0.485** vs ranks 51–100. Minus `cem_v3_noheldout`: −0.042 [−0.100, +0.015] /
−0.024 [−0.074, +0.027]. Raw-label port: 0.461 / 0.442.

**Reading.** Precise labels do not rescue the port. The best precise label (0.755) sets the ask from round 1. A model trained from
random weights on 1,105–1,489 teams cannot reach it, and the battled teams stay far below every loop arm. The finalists tie the
raw-label port against the top-50 teams; against ranks 51–100 they are slightly better, within noise.

## Full paper rule (`dbbo_paper`, pre-registration 2026-10-16)

**Owner-approved exception to the 2026-08-29 decision against uncertainty acquisition**, requested 2026-10-16: "Queue the full
paper rule … 5 seed models per round, uncertainty score on softened one-hot vectors as in the paper, w picked from 0.6 to 1.0
each round, precise labels, strictly in the queue." Code: `src/dbbo_paper.py`. Wu et al., arXiv 2407.00610, §5.2–5.3 (Eq. 7–8),
§6, App. D.2.

Everything as the precise-label w = 1 port (D starts as the 1,105 precise teams, about 196 battles per battled team, models from
random weights with matched steps, condition dropout 0.15, guidance 2, 4 rounds × 128 battled, same finalists), except:
1. **Five models per round**, each trained on D from its own random seed.
2. **Candidates** y = w × (best label in D), w ∈ {0.6, 0.7, 0.8, 0.9, 1.0}.
3. **Softened vector x~.** For each of the 48 fields, at the step it is decoded: 0.6 × one-hot(sampled token) + 0.4 × the
   field's legality-masked, guided predictive distribution. ‖x~‖ is the L2 norm of the 48 concatenated vectors. Hard one-hot teams
   all have the same norm, and so do the paper's own 0.6 one-hot / 0.4 uniform vectors, so the remaining 0.4 is spread by the
   model's distribution instead.
4. **Uncertainty**, from N = 64 decoded teams per model and candidate: Δ_epistemic(y) = variance across the 5 models of mean ‖x~‖
   (Eq. 7); Δ_aleatoric = mean within-model variance (logged).
5. **Pick** y* = argmax log y − log Δ_epistemic (§6). The raw y − Δ_epistemic choice is logged for comparison.
6. **The 128 battled teams** are decoded at y*, split equally across the five models (26, 26, 26, 25, 25).

**Battle-free check, done before queuing** (five 1-epoch models on D0): Δ_epistemic varies across w (7.4e-3 to 1.1e-2), the log
rule picks w = 0.9, the raw rule w = 1.0. Two cautions written now:
- Each model's mean ‖x~‖ hardly changes with w, so Δ mostly measures between-model spread, not something about the target.
- A variance over five numbers is itself noisy (relative error around 70%), so the chosen w may be close to random.
- The raw form y − Δ almost always picks w = 1.0, because Δ is about 0.01 against y of 0.45–0.75.

The run stops itself if Δ_epistemic does not vary in a round. Reported per round: chosen w (both rules), Δ for every candidate,
ask, battled win rate, legal share, line-ups. Finalists compared with `cem_v3_noheldout` and both w = 1 ports. It runs strictly
after the fixed-ask ablation queue (`run_ablations.sh`) ends.

## Ablations of the w choice (`dbbo_paper_w06` … `dbbo_paper_w10`, pre-registration 2026-10-16)

*Withdrawn 2026-10-16 before any run (owner): no fixed-w arms. The full rule already scores all five w values each round; only `dbbo_paper` runs. The `DBBO_PAPER_FIXED_W` switch stays in the code, unused.*

Requested by the owner: after the full-rule arm, run it once for each of the paper's five candidates with w **fixed** and **no
uncertainty score**: `DBBO_PAPER_FIXED_W=0.6 | 0.7 | 0.8 | 0.9 | 1.0`. Everything else is identical to `dbbo_paper`:
- start from the 1,105 precise teams;
- 5 models per round from separate seeds;
- 128 battled teams per round, split 26/26/26/25/25 across the models, about 196 battles each;
- 4 rounds and the same finalists.

The w = 1.0 ablation differs from the earlier precise-label w = 1 port only in the 5-model ensemble. Both are kept.

**Comparisons.** Each ablation is compared with the full rule (`dbbo_paper`) and both w = 1 ports, on:
- finalists against the top-50 meta teams and ranks 51–100, using the same two-sample bootstrap;
- per-round battled win rate and legal share.

**Reading rule, fixed now.** The uncertainty-picked w earns its place only if the full rule's finalists beat the best fixed-w
ablation with the interval above zero. If a fixed w ties or wins, the uncertainty score adds nothing here.

Queue order (strict, never in parallel):
1. The other session's fixed-ask ablation queue (`run_ablations.sh`).
2. `dbbo_paper`.
3. w = 0.6, 0.7, 0.8, 0.9, 1.0, in that order.

## Result: full paper rule (`dbbo_paper`, 2026-10-17)

| round | w picked (log rule) | raw-rule w | ask | battled win rate | legal | line-ups / 128 |
|---|---|---|---|---|---|---|
| 1 | 0.9 | 1.0 | 0.679 | 0.233 ± 0.012 | 94% | 78 |
| 2 | 0.9 | 1.0 | 0.679 | 0.269 ± 0.013 | 89% | 89 |
| 3 | 0.8 | 1.0 | 0.604 | 0.179 ± 0.013 | 68% | 100 |
| 4 | 1.0 | 1.0 | 0.755 | 0.208 ± 0.013 | 80% | 91 |

Δ_epistemic ranged from 1.9e-4 to 1.5e-2. The pick went to whichever w happened to have the smallest between-model spread, with
no stable pattern across rounds. The raw form picked w = 1.0 every round, as predicted. One round-3 team (team_id 111680, an
item-less Kingambit that parses to 5 slots) was skipped from round-4 training.

Finalists (16 × 192): **0.452** vs top-50 (best 0.526), **0.442** vs ranks 51–100.
- Minus `cem_v3_noheldout`: −0.052 [−0.090, −0.015] / −0.067 [−0.108, −0.026].
- Minus the raw-label w = 1 port: −0.009 [−0.047, +0.030] / +0.001 [−0.037, +0.038].
- Minus the precise-label w = 1 port: −0.010 [−0.066, +0.048] / −0.043 [−0.091, +0.004].

**Reading.** The uncertainty-picked w does no better than w = 1, and every Diffusion-BBO port stays below the baseline. A model
trained from random weights on about 1,100–1,500 teams cannot reach asks of 0.60–0.755.

## Variant: all five asks battled each round — not the paper's rule (`dbbo_allw`, pre-registration 2026-10-17)

Requested by the owner. Algorithm 1 battles only y*. This variant chooses no w: each round it battles 128 teams split
26/26/26/25/25 across y = w × best label in D for w = 0.6, 0.7, 0.8, 0.9, 1.0. Each w's teams are split across the 5 models
(6/5/5/5/5 or 5/5/5/5/5). Everything else as `dbbo_paper`:
- start from the 1,105 precise teams;
- about 196 battles per battled team;
- 5 models per round from separate seeds;
- 4 rounds, the 6-slot parse filter, the same finalists.

Δ_epistemic is still computed and logged per w, with the w the rule would have picked, but it chooses nothing (and does not
stop the run if flat). `DBBO_PAPER_ALLW=1 python dbbo_paper.py run`.

Reported per round and per w: ask, battled win rate, legal share. Finalists are compared with `dbbo_paper` (0.452 / 0.442), both
w = 1 ports and `cem_v3_noheldout`. Queued strictly after `run_cem_bestw.sh` (marker `CEM_BESTW_DONE`), never in parallel.

## Fixing the likely causes (pre-registration, 2026-10-17)

**Question (owner).** Why do the Diffusion-BBO ports degrade? Fix each likely cause, then see which fix helps. The ports start
low and stay low or get worse (raw w = 1 port 0.437 → 0.193; precise ports 0.16–0.33), while our loop's from-scratch arm,
trained on all labelled teams, sits at 0.51–0.54. Four arms, each the **raw-label w = 1 port** (`dbbo_faithful`) plus fixes
(`DBBO_FIX=…`):

| arm | fix | what changes |
|---|---|---|
| `dbbo_fix_data` | too little / mid-range data | D0 = every labelled matrix team with full top-50 coverage (95,801 at check time, win rates 0.00–0.86) instead of the 25th–50th percentile; labels shrunk toward the mean (beta-binomial, prior mean 0.443); battled teams appended with the same shrinkage |
| `dbbo_fix_ask` | ask out of reach | ask = 90th percentile of D's labels instead of the best label in D |
| `dbbo_fix_pick` | no selection | decode 512 legal teams, battle the ridge surrogate's top 128 (refit on this run's battled teams; no ridge push) |
| `dbbo_fix_all` | all three | the three together (check: ask 0.587 vs best 0.783) |

Everything else as the raw-label port: model from random weights each round, 20 epochs, learning rate 1e-3, batch 256, condition
dropout 0.15, guidance 2, 4 rounds × 128 battled at 49 battles each, finalists 16 × 192 vs top-50 and ranks 51–100. Training
skips teams that do not parse to 6 slots.

**Reading rule, fixed now.** A fix helps if its finalists beat the raw-label port (0.461 / 0.442) with the interval above zero.
Also reported per round: battled win rate, legal share, line-ups. `dbbo_fix_all` is also compared with our loop's from-scratch arm
`cem_scratch` (0.580 / 0.561). One seed each. Note: "all labelled teams" includes teams produced by this project's other searches,
not only an offline dataset.

Queue: strictly after `dbbo_allw`, one at a time: data → ask → pick → all.
