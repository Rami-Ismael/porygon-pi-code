# Redundancy penalty in elite selection (pre-registration, 2026-10-04)

**Question.** The combined loop (gloss guidance + value acquisition + retrain from p0 on
elites) loses diversity because elite selection keeps picking the same core: distinct
species sets among proposals 485 → 176 by generation 11 (gradloop.json). Selecting on a
*negative* score keeps no more diversity than random elites and drops win rate to 0.4%
(cem_score_direction); niche selection cost −0.074 (entropyloop). Does penalising
*redundancy* instead keep diversity without losing win rate?

**Selection rule — maximal marginal relevance (Carbonell & Goldstein, SIGIR 1998).**
Greedy: pick the labelled team maximising `y_i − λ · max_{j ∈ kept} overlap(i, j)`, where
overlap = shared species / 6. λ = 0 is the existing hard top-k cut. Elite count k is the
loop's own, max(100, 0.25 × labelled pool).

## Arms (everything else identical, common random numbers per seed)
- `cut`   λ = 0 (the existing loop, rerun here so all arms share code and battles)
- `mmr10` λ = 0.10 (a full-core duplicate costs 0.10 win rate ≈ one 24-battle SE)
- `mmr30` λ = 0.30
Random elites are not rerun: measured 2026-09 at 1.0% fresh win rate (cem_score_direction).

## Protocol
Start: the 200 anchor real teams + the shared generation-0 batch (activesearch.json), p0 =
`results/temperature_p0.pt`. Per generation: refit ridge on all labels → select elites →
retrain p0 on them (40 epochs) → 512 Showdown-valid proposals (gloss guidance λ=108) →
battle the 64 the ridge ranks highest. **Every battle goes into the matchup matrix:** one
battle per top-50 opponent (49 battles per team), policy 1, origin `redundancy:<arm>_s<seed>_g<gen>`.
6 generations × 3 seeds × 3 arms = 54 generations, 3,456 teams, ~169k battles.
After generation 6, each arm-seed's 8 best-labelled teams are raised to 4 battles per
opponent (196) to cut the winner's curse.

## Endpoints (fixed before battles)
1. Diversity: distinct species sets among the 512 generation-6 proposals.
2. Win rate: mean of the generation-6 battled batch (64 teams × 3 seeds), vs `cut`,
   95% interval by bootstrap over teams within seed.
3. Secondary: best team after the 196-battle re-battle; per-generation curves.

**Decision:** an MMR arm "keeps diversity without losing win rate" if its generation-6
distinct sets exceed `cut`'s in all 3 seeds AND its win-rate difference vs `cut` has a
95% lower bound above −0.03. Three seeds: intervals are exploratory.

## Amendment (2026-10-04, before any battle)
Battle-free check on the collapsed pool of gradloop's combined arm (anchor + gen0 + g1–g11,
k = 434): elite distinct species sets / elite mean label at λ = 0, 0.1, 0.3, 0.6 →
195 / 0.539, 202 / 0.538, 236 / 0.532, 312 / 0.502. λ = 0.10 is a near no-op, so the arms
are `cut` (0), `mmr30` (0.30) and `mmr60` (0.60). Nothing else changes.

## Result (2026-10-04) — both arms FAIL the pre-registered rule
Generation 6, 3 seeds × 64 battled teams × 49 battles (matchup matrix):
| arm | win rate | proposal species sets /512 | vs cut (95% CI) | sets vs cut by seed | best team (196 battles) by seed |
|---|---|---|---|---|---|
| cut | 0.329 | 455 | — | — | 0.570 / 0.560 / 0.565 |
| mmr30 | 0.294 | 495 | −0.035 [−0.057, −0.013] | +28 / +47 / +46 | 0.590 / 0.535 / 0.540 |
| mmr60 | 0.289 | 502 | −0.040 [−0.062, −0.018] | +43 / +50 / +48 | 0.530 / 0.530 / 0.530 |
Diversity rises in every seed, but the win-rate cost is real and beyond the −0.03 margin; the best
team does not improve. Caveat: in 6 generations `cut` only fell 495 → 455 sets (the old loop's collapse
came by generation 11), so the penalty bought diversity that was not yet scarce.
