# VGC team-generator pilot

Does a learned generator trained on the Champions VGC 2026 Reg Set M-B meta corpus
propose better teams than cheap non-learned operators?

Battle-free by construction. **Win rate is not measured** — this project has no
battle policy yet, so the objective function `f` cannot be evaluated. What is
measured is what must hold *before* a battle budget is worth spending: rule
legality, novelty, and whether the generator copies its training data.

## Setup

    uv venv --python 3.11 .venv
    .venv/bin/python -m pip install -r requirements.txt

Data is vendored: `teams/reg_mb/` (545 Showdown pastes, scraped by VGC-Bench from
the VGCPastes sheet) and `data/*.json` (Showdown pokedex, learnsets, items,
abilities, moves).

## Run

    cd src
    ../.venv/bin/python pilot.py --epochs 600 --seed 0   # main experiment
    ../.venv/bin/python ablation.py                      # k-sweep
    ../.venv/bin/python report.py                        # aggregate seeds
    ../.venv/bin/python hyperheuristic.py selftest       # LLM-written streamliners
    ../.venv/bin/python hyperheuristic.py report

## What the pilot does

A team is encoded as 6 slots x 8 categorical fields = 48 columns (species,
ability, item, 4 moves, nature). Slots are canonicalised by sorting on species;
slot order is permuted during training as augmentation. Note this regulation has
**no Tera type** (the `champions` mod deletes it) and uses **Stat Points**
(max 32 per stat, 66 total), not EVs — Stat Points are parsed but not yet modelled.

`model.py` trains the minimum model that delivers an on-manifold local move: a
masked-field predictor `p(field | all other fields)`, BERT-style with a random
mask rate. This is MDLM's objective, which that paper calls "a weighted average of
masked language modeling losses" — no diffusion machinery is required.

Four operators are compared from identical source teams with identical masked
columns:

| arm | move |
|---|---|
| `model` | mask the fields, resample them from the model |
| `uniform` | replace each field with a uniform draw from its vocabulary |
| `marginal` | draw each field from the corpus marginal for that column |
| `slotcopy` | paste one whole candidate + set from another corpus team |

## Calibration

The memorisation measure (Gu et al., TMLR 2025: nearest-neighbour distance under
1/3 of the second-nearest) is calibrated against real data: held-out real teams
score 6.2% memorised at mean Hamming distance 21.0; uniform-random teams score 0%
at distance 44.7. Local-move arms start from a real team, so their memorisation
numbers are near 1.0 by construction and are only interpretable for the
unconditional arm.

## Corpus caveat

543 parsed teams contain only 463 distinct species sets; 20% have a twin within 8
of 48 fields; 164 distinct candidates total, with Garchomp in 36% of teams. The
effective sample size is well below the headline count.

## The team sheet generator (GUI)

    cd gui && ../.venv/bin/python server.py 8770     # then open http://localhost:8770

Pin a candidate, build teams around it, and battle each one against the top-50 meta
teams before trusting it. Two build methods, both labelled with their measured
baseline so the interface never implies a team is good without evidence:

| method | what it does | measured baseline |
|---|---|---|
| graft | swaps your pick into a real tournament team | 0.420 |
| invent | the diffusion model writes all six slots | 0.093 |

Requires a local Showdown server (`node pokemon-showdown start 8123 --no-security`)
and the behaviour-cloning checkpoint from `cameronangliss/vgc-bench-models`.
Measuring one sheet runs 96 battles, about 8 seconds.

## Status (2026-08-27)

Measured, errors clustered by team, against the top-50 meta pool with the VGC-Bench
behaviour-cloning policy piloting both sides:

| what | result |
|---|---|
| real team, untouched | 0.470 |
| one candidate pasted from another real team | 0.436 |
| generator, unconditioned | 0.128 |
| generator conditioned on the top win-rate bin, guidance 4 | 0.193 (+4.8 sigma over unconditioned) |
| generated Stat Points vs copied spreads, paired | -2.1 points (-2.3 sigma) |
| legality as a CFG token (3 designs) without rule constraints | 0 / 2,880 valid teams |
| learned legality classifier (3 designs) | chance |

Legality is enforced by constrained decoding against the exact oracle
(`learnset_true.json`, 10,112 Showdown validator queries); it cannot be learned from
team data. The reusable assets are the 4,850 labelled teams (`results/*_labels.json`),
the battle pool (`src/pool.py`, ~50 battles/s), and the oracle.

## The 100,000-team dataset (2026-08-29)

`teams/hps_reg_mb_100k.jsonl` — 100,000 distinct Showdown-legal Reg M-B teams,
one JSON object per line: `{"key": <sha1>, "seed": 0, "team": "<paste>"}`.

Drawn by hierarchical product sampling (`src/hps_generate.py`): species first,
then ability / item / moves from that species' own legal tables
(`learnset_true.json` for moves, corpus items with the mega-stone coupling
enforced), then a uniform nature and a random 66-point Stat Points spread capped
at 32 per stat. Slot order is canonicalised by species, so symmetric duplicates
cannot appear; keys are deduplicated in-run. Every team passed Showdown's own
TeamValidator (`gen9championsvgc2026regmb`): acceptance was 94.9% of 105,340
attempts at ~1,400 teams/s, and an independent re-validation of a random 1,000
scored 1,000/1,000. Rejections were ability edge cases and forme couplings
(e.g. Greninja resolving to Greninja-Bond), all caught by the gate.

Coverage: 188 of the oracle's 189 species appear, near-uniformly (most common
species is in 3.4% of teams). Ditto is absent — it has one legal move and the
sampler draws four distinct moves — and no team carries a set with fewer than
4 moves, so that legal corner of the space is unsampled. Display names unseen
in the 695-team corpus render as Showdown IDs (e.g. `klutz`), which the
importer resolves.

## Status (2026-08-29): the 100k dataset vs the meta, and a conditioning null

The 100k teams are legal but not competitive. 2,000 of them were labelled with
24 battles each against the top-50 meta pool (48,000 battles, BC policy both
sides): **mean win rate 0.015, median 0.000, max 0.208**. 1,457 of 2,000 won
zero games. Real corpus teams re-measure at 0.491 under the same stack
(consistent with the earlier 0.470).

A masked diffusion model (`src/hpsdiffusion.py`, same MDLM objective and
constrained decoding as `diffusion.py`) was trained on all 100k teams — 12
epochs unconditional, then 8 epochs with a wins-bin conditioning token on the
2,000 labels ({0} {1} {2} {>=3} wins of 24, classifier-free dropout,
unlabelled rows carry the null token). Battled 32 teams x 24 per arm
(`src/hps_eval.py`, `results/hps_eval.json`):

| arm | mean win rate |
|---|---|
| real corpus teams | 0.491 ± 0.024 |
| hps dataset (2,000 labelled) | 0.015 |
| model, unconditional | 0.008 ± 0.003 |
| model, top bin, guidance 1 | 0.005 ± 0.002 |
| model, top bin, guidance 2 | 0.001 ± 0.001 |
| model, top bin, guidance 4 | 0.010 ± 0.004 |

Win-rate conditioning gave **no lift at any guidance**. The top bin holds only
46 training examples, and those are themselves weak (>=3 of 24 ≈ 0.125), so
even perfect conditioning could not approach real teams from this data. The
model faithfully reproduces its training distribution; the training
distribution is the problem. Uniform coverage of the legal space is a
negative-signal corpus for team quality — consistent with the 2026-08-26
finding that legality is not evidence of quality. Assets: the model
(`results/hpsdiffusion.pt`), the labels (`results/hps_labels.json`, manifest
maps label file -> jsonl line), and the per-team battle results.

## Status (2026-08-29, later): the surrogate works; active learning doesn't help it

Budget-matched test (`src/al_experiment.py`, `results/al_experiment.json`) on a
4,000-team pool spanning the quality range (real teams, real teams with j of 6
slots replaced by uniform-legal slots for j in 1,2,3,5, and fully uniform teams),
judged on a 300-team held-out set with the same spread (win rates 0.000-0.750,
24 battles each vs the top-50 pool).

Both arms label 500 teams (12,000 battles each), fit the same ridge (810 field
indicators + 1,561 species-pair indicators):

| arm | Spearman | R² | top-20 yield (ceiling 0.610) |
|---|---|---|---|
| random labelling | 0.836 | 0.681 | 0.496 |
| active (variance + diversity + top-mean, 4 rounds) | 0.824 | 0.647 | 0.477 |

Bootstrap of the Spearman difference: -0.011, 95% interval [-0.040, +0.018] —
**no detectable difference**. The acquisition rule chased uniform teams (rare
feature combinations look uncertain), which are the least informative for
ranking the top; and 500 random labels already cover a 6-stratum pool. Active
learning did not earn its complexity at this budget.

The buried lede is the surrogate itself: **Spearman 0.83 / R² 0.68 from 500
labelled teams**, replicating the earlier ridge (R² 0.60) on fresh data. A
ridge on team indicators ranks teams well across the full quality range at a
cost of 12,000 battles once, then microseconds per query. Caveat: its top-20
picks average 0.496 vs real teams' 0.491 — it finds the real teams and
near-real corruptions in the pool, it does not yet find anything better.

## Status (2026-08-29, latest): on the 100k HPS teams themselves, the labels are mostly noise

Same ridge family, but trained and judged inside the 100k uniform-legal
dataset, using the 2,000 existing labels (`src/hps_surrogate.py`,
`results/hps_surrogate.json` — no new battles):

| setup | Spearman | AUC (won ≥1 of 24) | top-20 yield (pool mean 0.014, ceiling 0.110) |
|---|---|---|---|
| fit on 1,500 HPS labels | 0.159 | 0.598 | 0.033 |
| random 500 labels | 0.067 | 0.547 | 0.008 |
| active 500 labels (same rule as al_experiment) | 0.121 | 0.578 | 0.035 |
| ladder-pool surrogate, transferred | 0.094 | 0.560 | 0.017 |

The single-seed active-vs-random gap does not survive replication: over 10
seeds the mean Spearman difference is +0.005 (sd 0.054, t = 0.31) —
**active learning is null here too**, now on both pools.

The reason the surrogate collapses (0.16 here vs 0.83 on the ladder pool) is
in the labels, not the features: with mean win rate 0.015 and 24 battles per
team, an estimated **73% of the observed label variance is binomial noise**
(R² ceiling ≈ 0.27). Uniform-legal teams are nearly indistinguishable at this
label budget; the quality signal the ladder-pool surrogate learned lives
*across* quality strata, not within the uniform-legal ocean. Top coefficients
on the HPS fit are ±0.015 — noise-scale. Ranking inside the 100k dataset would
need either far more battles per label or a pool with real quality spread.

The validator environment lives under `/tmp/vgc-pilot`, which macOS wipes. To
rebuild: symlink `data`, `src`, and `teams` (as `vgc-bench/teams`) into
`/tmp/vgc-pilot/`, copy `results/learnset_true.json` to
`/tmp/vgc-pilot/learnset_true.json`, shallow-clone `smogon/pokemon-showdown`
into `/tmp/vgc-pilot/vgc-bench/`, run `node build` there, and copy
`scripts/validate-teams-batch.js` into the clone's root. If reads hang near 0%
CPU, the repo files are iCloud-dataless: `brctl download` the repo first.

## Status (2026-09-01): active search — the acquisition is the component that was missing

The 2026-08-29 proposer sweep located the failure of the diffusion arm in two
places: the prior, and a missing acquisition function. `activesearch.py` builds the
loop the sweep described — propose de novo, spend the battle budget on the
proposals a surrogate ranks **highest**, re-steer the proposal on the labels — and
runs it against two budget-matched ablations. The utility is Garnett's active search
(`u(D) = sum_i y_i`), not expected information gain: uncertainty acquisition was
dropped 2026-08-29 after two nulls.

This is a controlled replacement of the acquisition in `loop.py` (2026-08-27), which
battled the 200 most PMI-**surprising** of 400 proposals — a novelty acquisition —
and folded all 200 back undifferentiated. Its diffusion arm went 0.165 / 0.172 / 0.153.

Setup: p0 is an unconditional masked diffusion model trained on the 692-team Reg M-B
corpus; 200 real teams labelled once as the anchor and this run's baseline; 4
generations x 3 arms x 128 labelled proposals of 512 drawn, 24 battles each vs the
top-50 pool, BC policy both sides. 44,736 battles. Decoding is unbiased (temp 1.0,
no guidance, no top-p) and legality is the constrained decoder plus Showdown's
validator. Re-steering always finetunes p0, never p_{g-1}, on the top rho=0.25 of
everything labelled so far — real anchor teams included, so a proposal moves the
distribution only by outscoring real teams. Generation 0 is one shared random batch;
`active` and `random` at generation 1 hold the same model and the same 512 proposals
and differ only in which 128 are battled.

| arm-generation | mean win rate | p90 | max | >= 0.458 | copy | NN-Hamming |
|---|---|---|---|---|---|---|
| real corpus teams (n=200) | **0.4625** | — | 0.792 | — | — | — |
| gen0 — p0, random acquisition | 0.1452 ± 0.0088 | 0.292 | 0.458 | 1.6% | 0.0% | 32.0 |
| active g1 / g2 / g3 / g4 | 0.245 / 0.220 / 0.224 / 0.218 | 0.375–0.417 | 0.667 | 1.6–9.4% | 0.0% | 30.4–31.1 |
| random g1 / g2 / g3 / g4 | 0.197 / 0.172 / 0.167 / 0.153 | 0.292–0.333 | 0.667 | 2.3–3.1% | 0.0% | 30.4–31.5 |
| frozen g1 / g2 / g3 / g4 | 0.204 / 0.201 / 0.218 / 0.219 | 0.333–0.417 | 0.583 | 2.3–6.2% | 0.0% | 31.5–32.0 |

**The acquisition works, and it is the durable half.** Pooled over four generations,
active − random = **+0.0547 [+0.0403, +0.0692]** (bootstrap over teams, 20,000
resamples); every generation's interval excludes zero, including the exactly paired
generation 1 (+0.0482 [+0.0173, +0.0785]) where the two arms drew the *same* 512
proposals. A ridge on field and species-pair indicators, fit on nothing but the 200
real anchors, already ranks within a batch of de novo proposals at Spearman +0.373 —
that ranking signal is what the acquisition converts into win rate. (The active
arm's own Spearman reads lower, +0.18 to +0.35, because selecting the top truncates
the predictor's range; the random arm's +0.32 to +0.45 is the unbiased figure.)

**The re-steer works once, then decays.** Read on the random-acquisition arm, whose
labels are an unbiased sample of its own proposal: +0.0521 [+0.0251, +0.0788] at
generation 1, then +0.026, +0.022, and +0.008 [-0.017, +0.033] — back to baseline by
generation 4. The cause is mechanical: elites are `max(100, 0.25|D|)` teams, so as
the labelled set grows the elite cut *loosens* (0.458 -> 0.375) and the tilt weakens.
That is the elite-filter-rate lever from 2026-08-27 running backwards. On top of the
acquisition the re-steer is worth only +0.0165 [+0.0015, +0.0314] pooled — `frozen`,
which never retrains, ends level with `active`.

**Nothing is being copied.** Copy rate is 0.0% in every batch, nearest-neighbour
Hamming distance to the corpus holds at 30–32 across all 13 batches (calibration:
held-out real teams 21.0, uniform-random 44.7), and 493–510 of each 512 proposals
carry a distinct species set. The gain is generation, not memorisation, and there is
no sign of the copy-paste collapse that ended the 2026-08-27 filter-rate sweep.

**The level is still half of real.** Best arm-generation 0.245 against real teams'
0.4625. Of 1,664 labelled proposals, 66 reached the real median and 37 the real mean;
the best single team scored 0.667, though that is the maximum of 1,664 draws at 24
battles each (SE about 0.10), so it is winner's-cursed and would need re-measuring.
Assets: `results/activesearch.json` (every labelled team as a paste plus its label —
1,664 new (team, win rate) pairs on a *generated* pool with real quality spread),
`src/activesearch.py`, `src/activesearch_report.py`.

Two levers this run leaves on the table, in order: hold the elite cut fixed instead
of the elite fraction, which is what made the re-steer decay; and score the battle
margin rather than the binary win, which removes the Bernoulli variance floor at zero
extra battles and would sharpen both the surrogate and the elite set.

## Status (2026-09-01, later): the two levers, as a 2x2 — the elite cut is real, the margin is not

Run 1 named two fixes and this tests both at once (`src/activesearch2.py`,
`results/activesearch2.json`, 57,024 battles). Four arms, one factor each way,
everything else identical to run 1's `active` arm; `src/shard.py` now records a
`margin` (mean faint differential, rescaled to [0,1]) alongside `wins` on every
battle, so the margin costs nothing.

|  | elite = top 0.25 **fraction** | elite = **fixed cut** |
|---|---|---|
| score = **win rate** | `active` 0.2289 *(run 1's configuration)* | `cut` 0.2332 |
| score = **margin** | `margin` 0.2250 | `both` **0.2454** |

Cell means are win rate pooled over 4 generations. The anchor re-measured the real
corpus at **0.4733** (run 1: 0.4625, same 200 teams) and p0's unconditional draws at
0.1315 (run 1: 0.1452); `active` reproduced run 1 generation for generation
(0.238 / 0.232 / 0.231 / 0.215 against 0.245 / 0.220 / 0.224 / 0.218).

**Lever 1 — hold the cut, not the fraction — works, and is small.** Main effect
**+0.0123 [+0.0017, +0.0229]**. The mechanism does exactly what it was meant to: the
fraction arms' elite cut loosened 0.500 -> 0.458 -> 0.417 -> 0.417 while their elite
set swelled 100 -> 178, and the fixed-cut arms held 0.500 across all four generations
with the elite set growing only 103 -> 114. `active` declines 0.238 -> 0.215 over the
run; `cut` is flat at 0.223 / 0.238 / 0.235 / 0.237. The payoff is bounded by what the
re-steer was worth in the first place — run 1 measured that at +0.0165 on top of the
acquisition, and this recovers most of it.

**Lever 2 — score the margin — is null on its own.** Main effect +0.0042 [-0.0063,
+0.0150], and `margin` − `active` is -0.0039 [-0.0189, +0.0112]. The battle-free test
says why, and it is the more useful result: over all 2,376 labelled teams the margin
tracks win rate at **Spearman +0.952**, so it is not an independent measurement, it is
a slightly finer reading of the same one. Cross-validating two ridges on the pooled
labels — one trained on win rate, one on margin, both scored against held-out **win
rate** — the margin-trained ridge wins by only +0.013 Spearman (+0.017 on proposals
alone). Real, reproducible, and far too small to be the variance fix the 2026-08-29
sweep hoped for. The Bernoulli floor is not where the loss was.

**Together they beat run 1's configuration.** `both` − `active` = **+0.0165 [+0.0016,
+0.0313]** pooled, **+0.0308 [+0.0096, +0.0524]** over the late generations where run 1
decayed, and `both` is the only arm that rises across the run (0.237 / 0.237 / 0.254 /
0.254). The interaction term is +0.0162 [-0.0051, +0.0376] — it does not clear zero, so
read the two levers as roughly additive rather than synergistic; the sum is detectable
where neither part reliably is.

**The surrogate, cross-validated properly for the first time.** On the 2,176 labelled
proposals: Spearman **+0.486**, R² +0.242, and its top-20 picks average **0.454** — level
with real teams at 0.4733, not above them. That is the honest ceiling this pipeline has
reached: it can find real-quality teams in its own output, it has not yet found better
ones. Across all four arms 5.5% of proposals reached the real median and 2.9% the real
mean; best single team 0.708 (max of 2,176 at 24 battles, winner's-cursed).

Provenance unchanged and clean: copy rate 0.0% in all 17 batches, NN-Hamming 30.1–32.0
(real held-out 21.0, uniform-random 44.7), 488–507 distinct species sets per 512.

**The lever this leaves.** Neither remaining knob is the label. The acquisition takes
the top 128 of 512, and the same surrogate's top *20* of 2,176 averages 0.454 against
the batch's 0.233 — so the binding constraint is the selection ratio, not the score.
Propose far more per generation and battle far fewer.

## Status (2026-09-01, latest): the LLM as a hyper-heuristic, writing streamliners

A hyper-heuristic searches a space of heuristics rather than a space of solutions, and
every LLM system in that branch (FunSearch, EoH, ReEvo) hands its generated code to a
mandatory evaluator. That evaluator is the thing this project cannot afford. So
`src/hyperheuristic.py` does not evolve a team builder — it evolves a **streamliner**, a
predicate `keep(team) -> bool` that cuts the search space, scored against two fixed
pools with **no battles at all**:

    r  meta retention   fraction of the 649 Reg M-B meta teams the rule keeps
    q  space retention  fraction of the 20,000 hierarchical-product-sampling teams it keeps

A rule earns its place when r stays near 1 while q collapses. The control is
`streamliner.py mine`, which already searches this same score over a hand-written
property vocabulary ("at least k slots have Protect"); the question is whether
arbitrary LLM-written Python beats it. Generated code runs in an AST-checked sandbox
(no imports, no dunders, restricted builtins, wall-clock alarm) against a rule API of
`types / stats / weaknesses / resistances / has / count / move / attacks / spread_moves /
coverage`. A whole generation costs about 1.4 s.

**The portfolio after two generations.** Non-dominated on (keep the meta, cut the space):

| rule | meta kept | space kept | lift | source |
|---|---|---|---|---|
| `stab-attacker` ≥4 slots have a same-type attack | 99.8% | 0.629 | 2x | gen 2 |
| `two-protect` | 97.7% | 0.113 | 9x | gen 1 |
| `two-protect-and-stab` | 97.5% | 0.065 | 15x | gen 2 |
| `three-protect` | 87.7% | 0.0125 | **70x** | control |
| `protect+fo+sc` *(below the 0.85 floor)* | 39.1% | 0.00075 | 522x | control |

**The LLM did not beat the best mined rule; it widened the front where the mined
vocabulary had nothing.** `three-protect` still prunes hardest of anything admissible,
at 70x, and two generations did not touch it. What the generations did buy is the top
of the front: above 90% retention the mined vocabulary offers only rules that prune
essentially nothing (`no-quad-weak` 91.5%/0.709, `two-types-max` 89.7%/0.761), and the
LLM put **15x lift at 97.5% retention** there. That point needs the move-target data —
it is a conjunction of Protect count with a same-type-attack count — and the property
vocabulary cannot express it. Widening a Pareto front is a real result and a smaller one
than "beat the baseline".

**Two beliefs about real teams, falsified battle-free.** `disruption` (at least one Fake
Out, redirection, or Intimidate) keeps only **79.2%** of the meta — better than one team
in five carries none of the three. And `common-items` (≥4 of the six holding an item from
the canonical competitive list) keeps only **21.7%**, while pruning to 0.24% of the space:
the highest lift anything reached, and useless, because real Reg M-B item spreads are far
wider than the canonical list. That is the meta-memorisation failure mode showing up
exactly where it was predicted, and being caught by the retention term rather than by a
battle.

**Coherence rules do not prune, because the sampler is already coherent.**
`no-choice-with-protect` — no slot holds a Choice item and runs Protect — keeps 99.8% of
the meta and **98.98%** of the random pool. Hierarchical product sampling almost never
produces that incoherence, so there is nothing there to cut. Same for `real-attackers`
(≥4 slots with 2+ damaging moves): 97.5% / 96.3%.

**Caveat on the proposer.** `run --proposer claude` shells out to the `claude` CLI, which
could not authenticate on this machine (expired OAuth), and `--proposer api` needs
`ANTHROPIC_API_KEY`. Generations 1 and 2 were therefore produced out of process:
`hyperheuristic.py prompt` emits the exact prompt the automated path would send, an LLM
answers it, and `hyperheuristic.py add --file` scores the reply. The scoring, the
reflection signal fed back into the next prompt (which meta teams a rule killed, and the
species in them), and the recorded results are identical either way.

**The standing risk is unchanged.** Retention is measured against the *collected meta*.
A rule keeping every meta team can still cut away the off-meta counter-team the search
exists to find; r is a proxy for "did not destroy known-good structure", never for "kept
the optimum". Nothing here tests that, and no battle-free score can.
