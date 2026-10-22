# Pretrained text diffusion model as the team writer (pre-registration, 2026-10-14)

**Question.** The scratch generator is a 6 × 8 categorical masked-diffusion model trained only on this
project's teams (F1 + G2 mixture, Regulation M-B vocabulary, `src/regmb_model.py`). Does a *pretrained
text* diffusion model, fine-tuned on the same teams written as Showdown pastes, propose stronger legal
teams than the scratch model under the same battle protocol?

**Win rate** = wins / battles against the top-50 meta teams, one battle per (team, top-50 column),
VGC-Bench behaviour-cloning policy on both sides (policy 1), every battle in the matchup matrix
(origin `textdiff:<cell>`). 64 Showdown-valid teams per cell, seed 2026, SE over teams — exactly
`src/foundation.py`'s protocol. A top-50 team scores 0.531 against the other 49.

## The pretrained model
[Fast-dLLM v2 1.5B](https://huggingface.co/Efficient-Large-Model/Fast_dLLM_v2_1.5B), revision
`25093b6f` — a block-diffusion language model adapted from Qwen2.5-1.5B-Instruct (block size 32,
complementary masking, token-shifted loss). Chosen as the vault note's first pilot: 1.5B parameters
fit the Mac (3.3 GB weights in bf16, 7.8 GB peak while training), and the released code carries
its own training recipe and sampler.

**Port.** The released training path runs attention through `torch flex_attention` compiled for
CUDA. `src/textdiff_port.py` replaces that one function with a dense boolean mask through
`scaled_dot_product_attention` (same `block_diff_mask`), and re-implements the released loss with
the mask rate exposed. Check: with mask rate 0 the noisy half is the clean sequence, so training-path
logits must equal the released eval path's block-causal logits. Measured max |difference| 0.0006 at
logit scale 30 in float32 (0.41 in bfloat16, i.e. bf16 rounding). One real LoRA update step: loss
finite (5.39 → 4.00 over four steps), gradients finite, trainable parameters changed.

## Data: the scratch model's own training rows, as text
The examples are the rows F1_final_medium_noheldout_regmb_v3 trained on (`scripts/retrain_regmb.py
train`): the test's family + the real corpus, valid under the v3 vocabulary, ranks 51–100 removed
(15,129 rows, 12,712 with a win-rate label), plus the 6,000 v3 synthetic teams. 21,129 examples;
2,000 family rows held out as the validation set (`lossdown.splits`, the scratch model's own split).

Each team is written as a Showdown paste **without Stat Points or Level** (the scratch model does not
model spreads either; both arms fill spreads the same way at decode time). The prompt carries the
format and the team's win-rate label to two decimals, or `unknown` for an unlabelled team and for
15% of labelled teams (the scratch model's `p_uncond`):

```
system: You write Pokémon VGC teams as Showdown pastes.
user:   Format: gen9championsvgc2026regmb. Six Pokémon. Target win rate vs top meta teams: 0.47.
```

Pretrained tokenizer kept as is (Qwen2.5 vocabulary); a prompt + team is ~230 tokens, padded to a
multiple of 32. No domain tokens added.

**Budget.** Measured throughput on the M4 Pro is ~100 input tokens/s (each example costs 4× its
length: [noisy | clean] concatenation × complementary masks), about 2.5 s per example. One epoch over
all 21,129 examples would take ~15 h. Decided before training: **one epoch over a subset** — every
real-corpus row plus a uniform random sample of the rest to 8,000 examples (seed 20261014),
LoRA rank 32 on all attention and MLP projections (36.9M trainable, 2.3%), learning rate 2e-4 with
3% warm-up and linear decay, batch 4, bf16. Validation loss every 500 steps on 256 held-out rows;
the final adapter is used unless validation loss rises, in which case the best checkpoint.

## Arms (64 valid teams each, no repair, legality by rejection only)
| cell | model | prompt / decode |
|---|---|---|
| scratch:base | F1 + G2 noheldout v3, constrained decode (`foundation.Gen("v3")`) | unconditioned, temperature 1 |
| scratch:cfg  | same | ask 0.5, guidance 2 (the selected baseline profile) |
| text:untouched | pretrained checkpoint, no fine-tuning | same prompt, `unknown` |
| text:base | fine-tuned | `unknown` |
| text:ask50 | fine-tuned | ask 0.50 (guidance 1) |
| text:ask60 | fine-tuned | ask 0.60 |
| text:ask50_g2 | fine-tuned | ask 0.50 with classifier-free guidance 2 (cond vs `unknown`) in the block sampler |

The scratch model decodes under its legality tables (clause + compatibility masks); the text model has
no constraints — its outputs are parsed (`corpus.parse_team_text`), given a Stat Point spread exactly
as the scratch arms (same-species corpus spread, random legal spread otherwise), and validated by the
pinned Showdown validator. **Raw legality** (valid / attempts) is reported per cell; nothing is
repaired. Sampler: the released `generate` (threshold 0.9, block 32, sub-block 8, temperature 1,
top-p 1), one team per call. The scratch baseline cells are re-generated here with the clean
(noheldout) members rather than reused from `foundation_v3.json`, whose cells predate the held-out
cleanup.

## Decision rules (fixed before any battle)
1. **Primary:** text:ask50 − scratch:cfg and text:base − scratch:base, each with its SE. The
   pretrained model is the better team writer for that arm if z > 2 in its favour; worse if z < −2;
   otherwise undecided at n = 64.
2. **Headline:** the best text cell against 0.531 and against the best scratch cell here.
3. Report per cell: validity, distinct species sets, exact copies of a training team, teams holding a
   species outside the corpus vocabulary, p90, best, count ≥ 0.531, generation seconds per valid team.
4. A lower validation loss, a plausible-looking team or a single strong team is not evidence; only
   rule 1.

## Known limits, stated before results
- One training seed, one epoch over 38% of the rows: a null result does not rule out the model at
  full budget; a positive result at this budget stands.
- The scratch baseline is an ensemble (F1 + G2) that was also pre-trained on the full 147k pool and
  warm-started; the text model sees only the F1 cohort. This compares the *deployed* scratch
  generator with a budget-limited pretrained one, not architectures at matched data.
- The text model's pretraining corpus is unknown; it may contain real Reg M-B pastes. Copies of any
  training team are counted, but copies of unseen real teams cannot be.
- One battle policy, one meta pool; conclusions are scoped to those.

## Results (2026-10-15)

**Training actually run.** 2,000 steps × batch 4 = 8,000 examples (622 real, 5,192 family, 2,186 synthetic),
one epoch, 372 minutes on the M4 Pro, 7.8 GB peak. Validation loss (256 held-out family rows, masked
answer tokens): 0.95 at step 500 → 0.80 → 0.71 → **0.67 at step 2,000**, still falling, so the final adapter
is the one evaluated. Adapter: `~/.local/share/vgc-pilot-runtime/textdiff/lora_r32/final` (147 MB).

**Battles.** 64 valid teams per cell × 49 top-50 columns, policy 1, seed 2026; `results/textdiff.json`.

| cell | win rate | p90 | best | ≥ 0.531 | raw legality | species sets | copies | sec / valid team |
|---|---|---|---|---|---|---|---|---|
| scratch:base | 0.095 ± 0.016 | 0.314 | 0.520 | 0/64 | 96% (64/67) | 64 | 0 | 0.05 |
| scratch:cfg (ask 0.5, g2) | 0.378 ± 0.015 | 0.514 | 0.740 | 6/64 | 100% (64/64) | 20 | 0 | 0.08 |
| text:untouched | — | — | — | — | **0% (0/12)**: prose, no paste | — | — | (49 s / sample) |
| text:base (`unknown`) | **0.438 ± 0.014** | 0.580 | 0.660 | 11/64 | 50% (64/129) | 33 | 0 | 17.5 |
| text:ask50 | **0.454 ± 0.013** | 0.560 | 0.760 | 12/64 | 88% (64/73) | 11 | 0 | 6.1 |
| text:ask60 | **0.460 ± 0.011** | 0.560 | 0.640 | 14/64 | 88% (64/73) | 13 | 0 | 6.3 |
| text:ask50_g2 (guidance 2) | **0.463 ± 0.013** | 0.600 | 0.740 | 17/64 | 79% (64/81) | 10 | 0 | 7.8 |

Rule 1: text:base − scratch:base = **+0.343 (+16σ)**; text:ask50 − scratch:cfg = **+0.076 (+3.8σ)**;
ask60 +0.082 (+4.3σ); ask50_g2 +0.085 (+4.3σ). The fine-tuned pretrained model is the better team writer
in both matched arms. Rule 2 / headline: the best text cell, 0.463 ± 0.013, is above every one-shot scratch
cell measured here but **below a real top-50 team (0.531)** and below the scratch model's full stack
(0.556, foundation-swap v3: low temperature + cfg + guidance + Protect filter + surrogate pick of 512),
which no text arm used. The comparison is one-shot decode against one-shot decode.

**Reading.**
- The text model's unconditioned arm (0.438) is above the scratch model's *conditioned* arm (0.378). Its
  "unknown" prompt was trained mostly on synthetic and label-dropped rows, yet the model writes the family's
  strong line-ups anyway: the pretrained prior plus 8,000 pastes concentrates on what the corpus repeats.
- Asking for a win rate adds little on top (0.438 → 0.454–0.463), unlike the scratch model (0.095 → 0.378).
  Guidance 2 adds +0.009 over guidance 1 (0.5σ): not detectable at n = 64.
- **Diversity is the cost.** The ask arms hold 10–13 distinct species sets in 64 (scratch cfg: 20); the
  dominant line-up is the gradient-loop core (Basculegion, Chandelure/Charizard, Floette-Eternal, Garchomp,
  Kingambit, Whimsicott). text:base keeps 33 sets but half of its raw outputs are illegal.
- **Legality without constraints:** 50% (base) to 88% (ask). Failure types: hallucinated mega formes
  (`Alcremie-Mega @ Alcreite`, 15 of 65 base failures: a pattern generalised from the synthetic mega-stone
  teams), invented items and moves (`lifeherb`, `thundercannon`), Item Clause, wrong slot count. The
  scratch model decodes under legality tables and reaches 96–100%.
- No team copies a training row; no text arm uses a species outside the corpus vocabulary (scratch base: 34
  of 64 teams do, and they lose).
- **Cost.** 6.2 h of fine-tuning and 6–18 s per valid team (one team per call, batch 1 sampler) against
  ~0.1 s for the scratch model. A 512-proposal pool for the surrogate pick would take ~1 h.

**Decision.** By the pre-registered rule the pretrained text diffusion model wins the one-shot comparison
(z > 3.8 in every matched arm). It does not replace the scratch model's search stack on its own: it writes
fewer distinct line-ups, a fifth to a half of its outputs are illegal, and it is 100× slower per team. The
natural next test is the text model as the proposer inside the existing stack (surrogate pick of a 512 pool,
then the cross-entropy / DiBO loop), with a diversity guardrail, against the stack's 0.556.

**Limits.** One seed, one epoch on 38% of the rows; the scratch baseline is the deployed F1 + G2 ensemble
(warm-started on 147k rows), not an architecture-matched control; the pretrained corpus may hold real
Reg M-B pastes; one battle policy, one meta pool.

## Pre-registration — the text model as the proposer inside the stack (2026-10-15)

**Question.** Does the fine-tuned text model, used as the *proposer* inside the existing one-shot stack, beat the
scratch model's stack cell — **v3:stack 0.556 ± 0.012** (`results/foundation_v3.json`, foundation-swap v3: temperature
0.7 / top-p 0.9 + ask 0.5 guidance 2 + surrogate gradient guidance + nature repair + protect3 filter + surrogate pick
of the top 64 from 512)?

**Proposer.** Adapter `~/.local/share/vgc-pilot-runtime/textdiff/lora_r32/final`, arm **ask50_g2** (ask 0.50,
classifier-free guidance 2 in the block sampler, threshold 0.9, temperature 1, top-p 1) — the sampler settings of the
measured one-shot cell, unchanged, so that any difference from 0.463 ± 0.013 is the selection stage alone. Three parts
of the scratch stack have no counterpart here and are not emulated: surrogate gradient guidance tilts field logits in
the categorical decoder and has no analogue over subword tokens; nature repair is skipped (the text model writes its
own natures); low temperature is not added because the block sampler already unmasks by confidence threshold.

**Pool.** 672 raw draws with their own seed (`textdiff.py gen ask50_g2 --tag pool --n 672`, written to
`gen/lora_r32_pool_ask50_g2.jsonl` in the background; ~7 s per raw draw, ~80 min). Decoded exactly as the one-shot
arms (`parse_team_text`, same-species corpus spread or random legal spread, pinned Showdown validator, **no repair**);
the first **512 valid** teams form the pool, or every valid team if fewer than 512 (the count is reported). The scratch
stack drew until 512 teams *passed the filter*; here the filter is applied to a fixed pool of 512 valid teams.

**Cells** (`textdiff_battle.py stack`, then the usual `battle` / `report`):
| cell | selection |
|---|---|
| text:stack | protect3 filter (≥ 3 Pokémon carrying a Protect-family move, `foundation.protect_count`) on the pool, then the shared ridge (`foundation.Surrogate`, `results/foundation_ridge_noheldout.npz`) picks the top 64 |
| text:stack_nofilter | the same ridge picks the top 64 of the whole pool |

Stated before the pool exists: all 64 measured text:ask50_g2 teams carry ≥ 3 Protect users (scratch:cfg 62/64), so
the two cells are expected to be nearly the same 64 teams; both are kept because both were asked for.

**Battles.** 64 teams × 49 top-50 columns per cell, policy 1, seed 2026, origin `textdiff:<cell>` — foundation.py's
protocol, through `battle_supervisor.py`.

**Decision rule (fixed before any battle).** Δ = text:stack − 0.556 with SE = √(SE²_text + 0.012²). **z > 2**: the text
proposer beats the scratch stack; **z < −2**: it is worse; otherwise undecided at n = 64. Secondary: text:stack −
text:ask50_g2 (0.463 ± 0.013) is the gain from selection alone, and text:stack_nofilter − text:stack is the filter's
share. Report per cell: distinct species sets, copies of a training row, teams with a species outside the corpus
vocabulary, p90, best, count ≥ 0.531; for the pool: raw legality, size, filter pass count, surrogate mean of the pool
versus the picked 64, seconds per valid team. A higher surrogate score is not evidence; only the battled Δ is.

**Limits stated before results.** One pool draw, one adapter seed. The proposer wrote 10–13 distinct species sets per 64
one-shot; a 512 pool may hold few line-ups and the ridge pick can narrow it further, so the species-set count is part of
the verdict's reading. The pool is 512 valid teams, not 512 filter-passing teams as in the scratch stack.

### Results — the text model as the proposer inside the stack (2026-10-15)

**Pool.** 672 raw draws in 92 min (8.2 s per draw, 10.8 s per valid team); 512 valid of the first 634 (**81%** raw
legality, one-shot cell 79%); failures: wrong slot count 32, shape 31, Item Clause 22, Species Clause 4, two illegal
moves. 30 distinct species sets in the 512; **512/512 pass protect3**, so text:stack and text:stack_nofilter are the same
64 teams (the filter is inert on this proposer). Ridge mean 0.387 over the pool, 0.472 over the picked 64. Battles:
3,136 in one batch (seed 2026, 71 s on seven servers), `results/textdiff.json`.

| cell | win rate | p90 | best | ≥ 0.531 | raw legality | species sets | copies | sec / valid team |
|---|---|---|---|---|---|---|---|---|
| text:ask50_g2 (one-shot, for reference) | 0.463 ± 0.013 | 0.600 | 0.740 | 17/64 | 79% | 10 | 0 | 7.8 |
| **text:stack** (pool 512 → protect3 → ridge top 64) | **0.491 ± 0.010** | 0.600 | 0.660 | 19/64 | 81% | **4** | 0 | 10.8 (+ pick) |
| text:stack_nofilter | 0.491 ± 0.010 (same 64 teams) | | | | | 4 | 0 | |
| v3:stack (scratch model's stack, foundation-swap v3) | 0.556 ± 0.012 | | | | 100% | 12 | 0 | ~0.1 |

**Decision rule.** text:stack − v3:stack = **−0.065, z = −4.3**: by the pre-registered rule the text model as the
proposer inside the stack is **worse** than the scratch model's stack. Secondary: text:stack − text:ask50_g2 = +0.028
(+1.7σ), undecided at n = 64 — the ridge pick adds at most a small gain over the one-shot draw; the filter adds nothing.

**Reading.**
- The selection stage collapses the proposer. The ridge picks 60 of its 64 from one line-up (Basculegion, Charizard,
  Floette-Eternal, Garchomp, Kingambit, Whimsicott — the gradient-loop core, which the matchup matrix already holds
  3,709 teams of), plus 2 + 1 + 1 from three neighbours; the pool had 30 line-ups. Inside the picked 64 the ridge's rank
  and the battled rank are uncorrelated (Spearman 0.06): once it has chosen the line-up, the ridge is sorting noise.
- The scratch stack's gain over its own one-shot cfg cell (0.369 → 0.556, +0.187) comes mostly from the parts the text
  model cannot use: low temperature, surrogate *gradient guidance* in the categorical decoder, and a pool drawn until 512
  passed the filter. The surrogate *pick* alone is worth +0.028 here, and `v3:surrogate` alone was worth −0.038 there.
  The text proposer starts higher one-shot (0.463 vs 0.369) and ends lower in the stack (0.491 vs 0.556).
- Nothing in the pipeline was repaired; copies of a training row: 0; species outside the corpus vocabulary: 0.
- Cost: 92 min of generation for one 512 pool, against ~1 min for the scratch model's.

**Decision.** The text model does not replace the scratch model inside the one-shot stack. It stays the better *one-shot*
writer (0.463 vs 0.378) and the worse *proposer under selection* (0.491 vs 0.556): its pool is narrow (30 line-ups per 512)
and the shared ridge narrows it to four. What would change this reading is not a bigger pool but a diversity guardrail
in the pick (one team per species set, or taboo over line-ups) and a guidance analogue for subword tokens — both new
experiments, not part of this one.

**Limits.** One pool draw, one adapter seed, one battle policy, one meta pool. The pool is 512 valid teams, not 512
filter-passing teams as in the scratch stack (inert here, since 512/512 pass).

## Mask Lab baseline: the text writer fills the blanks (2026-10-15)

The project's baseline experiment (vault note "What is the baseline", pipeline `team_completion_pipeline/`): five random
legal starting teams, 270 masks (Stat Points, items, moves, whole Pokémon, abilities, natures; k = 1…6, moves 1…24),
each filled by every method, one attempt, no repair, then every legal team battles the 49 distinct top-50 meta teams
once (behaviour-cloning policy both sides, private matrix copy, `mask_lab_battle.py`). Existing arms: untouched
originals, random legal fills, Ling (three prompt arms) and the scratch diffusion decoder (F1 + G2 v3, ask 0.5,
guidance 2). New arm: `team_completion_pipeline/mask_lab_textdiff.py`, the fine-tuned text writer with the same
profile (ask 0.5, guidance 2, temperature 1), filling each blank by constrained left-to-right infilling: the paste up
to the blank is teacher-forced, the blank's line is written (a whole Pokémon as a block), the filled line joins the
prefix. It sees everything before the blank and nothing after it; the scratch decoder sees both sides. Masked Stat
Points are corpus spreads for both. Two harness bugs found on the first pass (an item prompt ending in a space, and a
last slot ending in the stop token) were fixed and the 46 affected masks regenerated before the table below.

| family | legal, text | legal, scratch | text − original (pp) | text − random fill | text − scratch | net gain per mask, text | net gain, scratch |
|---|---:|---:|---|---|---|---|---|
| Stat Points | 30/30 | 30/30 | −1.3 [−3.8, +0.9] | −0.3 [−1.7, +0.7] | +0.1 [−0.5, +0.7] | −1.3 | −1.4 |
| items | 28/30 | 30/30 | −0.9 [−3.3, +1.5] | +0.1 [−1.2, +0.9] | −0.2 [−1.4, +0.7] | −1.1 | −0.9 |
| moves | **13/120** | 120/120 | −0.9 [−4.5, +2.2] | +0.7 [−0.9, +2.6] | −0.9 [−3.6, +1.1] | 0.0 | +0.3 |
| whole Pokémon | 25/30 | 30/30 | **+23.0 [+20.4, +25.5]** | +23.7 [+20.1, +26.4] | **+7.0 [+1.9, +11.1]** | **+19.1 [+15.4, +23.0]** | +15.0 [+10.1, +19.1] |
| abilities | 23/30 | 30/30 | −1.0 [−2.7, +0.7] | +0.6 [−0.5, +1.6] | +0.2 [−0.9, +1.3] | −1.1 | −1.3 |
| natures | 30/30 | 30/30 | −1.3 [−3.3, +0.7] | +0.3 [−0.9, +1.1] | +0.7 [−0.9, +2.5] | −1.3 | −2.0 |
| **all** | **149/270 (55%)** | 270/270 | +2.9 [+0.9, +4.9] | +4.2 [+3.2, +5.0] | **+1.3 [+0.4, +1.9]** | +1.6 [+0.5, +2.7] | +1.2 [−1.0, +3.2] |

Percentage points of win rate; paired on the same mask, each starting team weighted equally, interval = bootstrap over
the five starting teams (exploratory: five clusters). "Net gain" keeps the original team when a fill is rejected.
Results: `runs/prompt-ab/textdiff_report.json`, rows `textdiff_rows.json`, panels in `panels.json`.

**Reading.**
- **Legality is the text writer's weakness: 55% (149/270), the same level as Ling, against 100% for the scratch decoder
  and random fills.** Moves are the failure: 13 of 120 move masks survive, 109 rejections are "X can't learn Y" (the
  writer puts meta moves on random species it has rarely seen; it has no learnset table). Items, natures and Stat Points
  are fine; abilities 23/30; whole Pokémon 25/30.
- **Where it is legal it is level with the scratch decoder on every single-field family** (all within ±1 pp with intervals
  through zero) and **ahead on whole Pokémon: +7.0 pp [+1.9, +11.1] over the scratch fill, +23 pp over the original.**
  As for Ling and the scratch arm, whole-Pokémon masks are the only family where a filler beats the original: a random
  legal team's weak member is replaced by a meta-style set (Floette-Eternal, Basculegion, Whimsicott…).
- Net gain per mask over all 270: text +1.6 pp [+0.5, +2.7], scratch +1.2 [−1.0, +3.2]: level, the text writer's
  whole-Pokémon edge paying for its rejected moves.
- Cost: 16 s per mask on average, 73 min for the grid (whole Pokémon ≈ 40 s, a single field ≈ 1.5 s, two rows per
  step for guidance); the scratch decoder did all 270 in 34 s.

**Decision.** On the baseline grid the fine-tuned text writer is not better than the scratch decoder as a completion
method: equal strength where legal, far worse legality on moves, 100× slower. Its one real edge, writing a whole
Pokémon into a fixed team, is the same edge Ling showed and the same edge the one-shot comparison measured. A legality
table for the move line (learnset-constrained decoding, as the scratch decoder has) is the obvious next arm; it was
deliberately not added here so both the Ling and the text arms are measured as raw generators.

## Training speed on the M4 Pro (2026-10-15)

The run above cost 11 s per step of 4 examples (372 min for 2,000 steps). The GPU's measured matmul
peak is 5.7 TFLOP/s (bf16 = fp16 = fp32 rate; no tensor cores), and one step is ~37 TFLOP of matmuls,
so the floor was ~6.5 s with gradient checkpointing; padding is 5% of tokens and not worth bucketing.
Three changes, all in `src/textdiff_port.py` / `src/textdiff.py`, same loss:

| setting (batch 4, same batches) | s / step | s / example | high-water |
|---|---|---|---|
| as run (released pairs, full checkpointing, eager) | 9.75 | 2.44 | 6.8 GB |
| no gradient checkpointing | out of memory (30 GB) | | |
| **shared-clean layout** | 7.54 | 1.89 | 8.3 GB |
| + attention without the kv-head copy (`enable_gqa`) | 7.31 | 1.83 | 8.3 GB |
| + 8 of 28 layers not checkpointed (`--ckpt-layers 20`) | 6.59 | 1.65 | 13.7 GB |
| + `torch.compile` (Metal backend, dynamic shapes), `--ckpt-layers 20` | 5.5–6.2 | 1.4–1.5 | 17 GB |
| **+ `torch.compile`, full checkpointing (the new default)** | **6.02** | **1.50** | 10.5 GB |

**Shared-clean layout.** In the released loss each example is two rows, `[x_t | x0]` and
`[x_t' | x0]` (complementary masks), and the clean half `x0` never attends to noisy tokens
(`block_diff_mask`), so its hidden states are identical in both rows. One row
`[x_t | x_t' | x0]` with the same attention pattern per segment (`segment_mask`) computes the same
logits on three quarters of the tokens. Checked: for two segments `segment_mask` equals the published
mask bit for bit; on a random mask the two layouts give identical logits and loss in bf16 and in fp32
(`python textdiff_port.py [fp32]`, `verify_shared`), and a deliberately wrong three-segment mask
makes the check fail. `--legacy-layout` restores the released pairs.

**Checkpointing.** Without it the 28 layers' activations (8,960-wide MLP intermediates, LoRA copies)
need ~27 GB at batch 4. Each layer left out of checkpointing costs ~0.7 GB and saves one 28th of a
forward pass; with 14 or 8 layers checkpointed the machine swaps (19–23 GB) and the step takes 6.6 s
or 185 s. With compile, full checkpointing is within 5% of the 20-layer setting at 40% less memory,
hence the default.

**Compile.** `torch.compile(model.forward, dynamic=True)` on the bound forward (parameter names
unchanged, adapters save and load as before): one compile of ~90 s per process; static shapes were
5% faster per step but recompile per padded length (three in this data) and held 2 GB more.
A 2,000-step run at these settings is ~3.4 h instead of 6.2 h. The matmul floor without recompute is
~5 s per step; MLX would face the same 5.7 TFLOP/s and was not pursued.

**Checked end to end (2026-10-15, defaults: shared layout, `enable_gqa`, compile, full checkpointing).**
`textdiff.py train --name speedcheck --limit 64 --eval-every 100 --nval 32` (the 622 real rows, 156
steps of batch 4, a six-worker battle run alive on the CPU): 17.8 min including the ~90 s compile and
two validation passes, 6.2 s per step steady (the recorded run: 12.1 s per step at the same point);
validation 1.63 → 1.51; the saved adapter has the same 392 tensor names as `lora_r32/final`, every
LoRA-B tensor non-zero, and loads, merges and writes a paste in 5 s. The test adapter was deleted.

## Twelve ways to train it faster, measured (2026-10-16)

Protocol: every arm trains on the same first 2,000 examples in the same order (`--limit 8000`, the run's
seed), 500 steps of batch 4 under the full 2,000-step learning-rate schedule, and is scored with the
published loss (complementary pairs, `validate`) on the same 256 held-out family rows; the recorded run
scored 0.9487 there. "val per team" = loss per masked answer token × mean answer tokens. The last two
rows get the control's 60 minutes of wall-clock instead of its 500 steps. One seed each. All flags live
in `textdiff.py train`; the arms' adapters are under `~/.local/share/vgc-pilot-runtime/textdiff/arm_*`
(the two equal-wall-clock adapters were deleted by mistake during clean-up; their logs survive).

| arm | steps | s / step | speed-up | val loss / token | answer tokens | val per team | vs control per team | verdict |
|---|---|---|---|---|---|---|---|---|
| control (current trainer: shared layout + compile) | 500 | 7.23 | 1.00× | 0.958 | 204 | 194.9 | — | reference (+0.009 vs the recorded 0.9487) |
| short prompt, no system message | 500 | 6.61 | 1.09× | 0.967 | 204 | 196.8 | +1.0% | **faster, loss close** |
| one token per domain name (1,588 added) | 500 | 5.22 | 1.39× | 0.705 | 139 | 97.8 | −50% | **faster; a different quantity (see note), decode 3.2 s/team** |
| short prompt + domain tokens | 500 | 4.40 | 1.64× | 0.711 | 139 | 98.8 | −49% | **faster; same bound as domain tokens alone** |
| middle 8 layers removed (20 remain) | 500 | 5.05 | 1.43× | 0.990 | 204 | 201.5 | +3.4% | faster, loss borderline |
| short prompt + 8 layers removed | 500 | 4.13 | 1.75× | 1.022 | 204 | 208.1 | +6.7% | faster, loss off (penalties compound) |
| short prompt + 8 layers removed + domain tokens | 500 | 3.29 | 2.20× | 0.711 | 139 | 98.8 | −49% | faster; same bound as domain tokens alone |
| adapters on the top 14 layers only | 500 | 6.38 | 1.13× | 1.054 | 204 | 214.5 | +10% | faster, loss off |
| adapters on the top 7 layers only | 500 | 6.09 | 1.19× | 1.179 | 204 | 240.0 | +23% | faster, loss off |
| one mask per example (no complementary pair) | 500 | 4.46 | 1.62× | 1.096 | 204 | 223.1 | +14% | faster, loss off |
| middle 16 layers removed (12 remain) | 500 | 2.94 | 2.46× | 1.102 | 204 | 224.3 | +15% | faster, loss off at equal steps |
| Qwen2.5-0.5B-Instruct + the block-diffusion recipe | 500 | 2.75 | 2.63× | 1.094 | 204 | 222.7 | +14% | faster, loss off at equal steps (started at 18.1) |
| **Qwen2.5-0.5B, the control's 60 min of wall-clock** | 1,300 | 2.47 | 2.9× | **0.801** | 204 | 163.1 | **−16%** | **faster and lower loss at equal time** |
| **12-layer model, the control's 60 min of wall-clock** | 1,230 | 2.80 | 2.6× | **0.823** | 204 | 167.6 | **−14%** | **faster and lower loss at equal time** |
| 20-layer model, the control's 60 min of wall-clock | (715) | 5.05 | 1.43× | — | | | | stopped at your request, 45 min in |

**Domain tokens** (`--domain-tokens`): one added token per species, ability, item, move and nature as
it appears in a paste (1,588 strings, two already single tokens); new embedding rows start at the mean of
their old sub-token rows and only those rows train (PEFT `trainable_token_indices`, tied output head
follows); the tokenizer is saved next to the adapter and `gen` picks it up. A team is 139 answer tokens
instead of 204, pastes round-trip exactly, and a team decodes in 3.2 s (5–18 s before). The halved loss is
**not evidence of better teams**: with atomic names nothing leaks between a name's sub-tokens, so the
per-token loss is a different quantity. It is the training objective's own number; battles decide.

**Equal wall-clock.** Per step the 0.5B swap (`--base Qwen/Qwen2.5-0.5B-Instruct`: the released
block-diffusion architecture and loss on a plain Qwen2.5 checkpoint, `[MASK]` row trained) and the
12-layer model (`--drop-layers 8-23`) lose 14–15% at 500 steps, but they fit 2.5–2.6× the steps in the
control's hour and end 14–16% *below* it (0.801 and 0.823 vs 0.958). Neither has been decoded or
battled; the 0.5B model learned masking from scratch on 5,200 teams, so its samples need checking before
it replaces anything. The short prompt (`--prompt-style short`: "Win rate 0.47." and no system
message) is the one free change: 9% faster, loss within the control's own drift.

**Not worth it:** adapters on the top layers only (backward stops early, but +10–23% loss), one mask per
example instead of the complementary pair (+14%), and stacking the short prompt on a shallower model
(+6.7%, more than the sum).

## Against the LLM baseline: the text writer and Ling on the same 270 masks (2026-10-21)

The best-trained text writer (`lora_r32/final`: Fast-dLLM v2 1.5B + LoRA, the full 2,000-step run, validation 0.667)
against the Ling masked-completion baseline (`inclusionai/ling-3.0-flash-vl` through OpenRouter, reasoning off,
temperature 0.7), from the Mask Lab run's stored battle results: no new battles. Both filled the same 270 masks on the
same five random legal starting teams, one attempt each, no repair. A fill is **legal** when the completed team keeps
every unmasked field and passes Showdown's Regulation M-B validator; each legal fill battled the 49 distinct top-50
meta teams once, with the behaviour-cloning policy on both sides. Ling ran three prompt arms. Its best, used in every
column below, is the regulation-grounded prompt with a 16,384-token output limit (`regmb@16384`). Statistics as in
`mask_lab_textdiff.py report`: percentage points of win rate, paired on the same mask, each starting team weighted
equally, 95% bootstrap over the five starting teams. **Net** keeps the original team when a fill is rejected, so it is
the comparison that counts illegal answers.

| mask type | legal, text writer | legal, Ling | text writer − original | Ling − original | text writer − Ling where both are legal (masks) | net, text writer | net, Ling | **net, text writer − Ling** |
|---|---:|---:|---|---|---|---|---|---|
| Stat Points | 30/30 | 25/30 | −1.3 [−3.8, +0.9] | −1.3 [−3.7, +1.0] | +0.0 [−0.5, +0.7] (25) | −1.3 | −1.0 | −0.3 [−1.0, +0.2] |
| items | 28/30 | 26/30 | −0.9 [−3.3, +1.5] | +0.0 [−2.5, +3.1] | −1.5 [−2.6, −0.7] (24) | −1.1 | −0.0 | **−1.1 [−1.6, −0.6]** |
| moves | **13/120** | 41/120 | −0.9 [−4.5, +2.2] | +1.5 [−1.4, +4.0] | −0.5 [−2.0, +0.8] (8) | −0.0 | +0.5 | −0.5 [−1.2, +0.3] |
| whole Pokémon | 25/30 | **16/30** | +22.9 [+20.4, +25.5] | +23.5 [+19.3, +29.5] | +0.4 [−5.4, +6.1] (13) | +19.1 | +12.7 | **+6.5 [+1.7, +9.9]** |
| abilities | 23/30 | 15/30 | −1.0 [−2.7, +0.7] | −1.0 [−3.2, +1.3] | −0.2 [−0.6, +0.0] (11) | −1.1 | −0.3 | **−0.8 [−1.3, −0.3]** |
| natures | 30/30 | 26/30 | −1.3 [−3.3, +0.7] | −1.6 [−4.4, +1.6] | +0.4 [−1.1, +2.0] (26) | −1.3 | −1.4 | +0.1 [−1.2, +1.5] |
| **all** | **149/270 (55%)** | **149/270 (55%)** | +2.9 [+0.9, +4.9] | +2.4 [−0.6, +4.5] | −0.3 [−1.2, +0.7] (107) | +1.6 [+0.5, +2.7] | +1.3 [−0.2, +2.5] | **+0.3 [−0.3, +0.9]** |

Ling's other two arms use the same prompt without the regulation reference, at a 16,384-token limit
(`baseline@16384`) and at a 4,096-token limit (`baseline@4096`). Both are weaker: 113 and 87 of 270 legal, net +0.4
[−1.0, +1.8] and −0.3 [−1.3, +0.5]. Cost: the text writer runs locally, 16 s per mask on the M4 Pro (73 min for the
grid); Ling took 22 s per mask through the API and $0.28 for the grid. Script: `scripts/text_vs_ling.py` (writes
`text_vs_ling.json` into the run folder). Its text-writer columns are identical to the stored output of
`mask_lab_textdiff.py report`, and every cell above matches the script's output (both checked by code, 2026-10-21).

**Reading.**
- **Overall the two are level.** Same legality (149 of 270 each), no difference in strength where both are legal
  (−0.3 pp over 107 masks), and a net gain per mask of +1.6 for the text writer against +1.3 for Ling: +0.3 pp
  [−0.3, +0.9], an interval through zero.
- **They fail on different fields.** The text writer loses moves (13 of 120 legal; it has no learnset table). Ling
  loses whole Pokémon (16 of 30), abilities (15 of 30) and some Stat Points.
- **Whole Pokémon is the largest difference and the only one in the text writer's favour: +6.5 pp [+1.7, +9.9]
  net.** It comes from legality (25 against 16 legal fills), not from stronger fills (+0.4 pp where both are legal).
- **Items and abilities favour Ling by about 1 pp net**, both intervals clear of zero (−1.1 [−1.6, −0.6] and
  −0.8 [−1.3, −0.3]): the text writer's extra legal fills there are slightly weaker than the originals they replace.
- Five starting teams make these intervals exploratory, as in the Mask Lab table.

**Published.** The adapter (`lora_r32/final`, 147.8 MB) and a model card with these results are public at
[Rami/fast-dllm-v2-1.5b-vgc-regmb-lora](https://huggingface.co/Rami/fast-dllm-v2-1.5b-vgc-regmb-lora) (commit
`336362f`, 2026-10-21). The hub copy's SHA-256 matches the local file, and loading it by hub id with plain
transformers + peft writes a team on the M4 Pro in 11 s.
