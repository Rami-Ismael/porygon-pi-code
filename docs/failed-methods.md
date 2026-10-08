# Methods that failed (as of 2026-09-28)

**Metric, unless an entry says otherwise:** win rate = fraction of battles won against the
top-50 meta (the 50 top-placing Reg M-B teams on VGCPastes, 2026-08-21 snapshot), with the
VGC-Bench behaviour-cloning policy playing both sides. **The bar is a real team:** real
Reg M-B teams from outside the pool score 0.46–0.49 on the same measure.

A *method* is a whole process taken from a paper. Changes we made to a method appear inside its
entry. Methods that were tried and did **not** fail are listed at the end, with the reason.

---

## 1. Masked diffusion as a from-scratch team generator

- **Paper:** Sahoo et al., *Simple and Effective Masked Diffusion Language Models* (MDLM),
  NeurIPS 2024 🟢 [arXiv 2406.07524](https://arxiv.org/abs/2406.07524). No Papers/ note.
- **What it is:** fields are masked independently in the forward process, and the reverse
  process unmasks them one by one. We trained it on 692 Reg M-B teams and sampled whole teams,
  using constrained decoding so every team is legal.
- **Why we tried it:** the starting hypothesis was that a diffusion model can build novel
  teams that compete with the meta.
- **Compared against:** real corpus teams, plus four edit arms (slot copy, shuffled spreads,
  one slot regenerated, steered).
- **Result:** generated from scratch **0.093 ± 0.016** vs real team **0.475 ± 0.025**
  (win rate vs top-50 meta, behaviour-cloning policy both sides; 2,048 battles per arm,
  16 teams × 128 battles, errors clustered by team). Gap 13.1 σ. 89.6% Showdown-legal,
  100% unique.
- **Why it failed:** composition. Copying real Stat Points closes only 6 of the 38 points, so
  the rest is which Pokémon the model picks. The model ranks well on real teams: its log-odds
  track true win rate at +0.78. On its own samples that drops to +0.15, because sampling
  leaves the region where its judgement holds (152 bits of surprisal vs 94 for real teams).
  Two changes did not help: learned legality (a legality token, a per-slot token, a legality
  classifier) gave 0 legal teams out of 2,880 without constraints, and making the model 13×
  wider added only +3.3 points.
- **What we learned:** use the model to rank teams, never to sample them unconstrained. A
  generator needs an acquisition step around it. The later loops that used this same model
  as a proposer, not as the whole method, reached 0.41.
- **Verdict:** failed as a stand-alone generator. The model survives as a proposer.

## 2. Classifier-free guidance on win rate (DDOM recipe)

- **Papers:** Krishnamoorthy et al., *Diffusion Models for Black-Box Optimization* (DDOM),
  ICML 2023 🟢 [arXiv 2306.07180](https://arxiv.org/abs/2306.07180). The discrete guidance form
  we used is from Schiff et al., *Simple Guidance Mechanisms for Discrete Diffusion Models*,
  ICLR 2025 🟢 [arXiv 2412.10193](https://arxiv.org/abs/2412.10193). No Papers/ note for
  either.
- **What it is:** train the model with each team's win-rate bin as a condition, then sample
  with the top bin requested and the guidance weight γ raised. DDOM adds reweighting of the
  training data toward high scores.
- **Why we tried it:** it was the hypothesis's "most important piece", steering generation
  toward teams that beat the meta.
- **Compared against:** unconditional sampling from the same model, the bottom-bin control,
  and real teams.
- **Result:**
  - 4,850-team labelled pool: top bin at γ=4 **0.192** (105 teams × 24 = 2,520 battles) vs
    unconditioned 0.128. Bottom bin 0.006 (2,880 battles), so the label is used. Showdown
    validity falls 88% → 77% → 29% as γ goes 1 → 2 → 4.
  - Uniform-legal (HPS) pool: top bin at γ = 1/2/4 gives **0.005 / 0.001 / 0.010** vs
    unconditioned **0.008** (32 teams × 24 = 768 battles per cell), real **0.491** (768
    battles). A null.
  - DDOM reweighting (balanced win-rate brackets): **0.262** vs base **0.276** (150 teams × 24
    = 3,600 battles per arm). A null.
  - All scores: win rate vs top-50 meta, behaviour-cloning policy both sides.
- **Why it failed:** guidance tilts the proposer. It is not an acquisition, and it measured
  about 1× lift where swapping the prior measured ≥476×. It also collapses diversity: each
  guidance step costs about 5 effective species, and at γ=8 Blastoise-Mega is on 100% of
  teams. With few high labels (147 teams ≥ 0.667), there is little to condition on.
- **What we learned:** fix the prior before tuning guidance, and keep the low-scoring teams
  in the training data. Dropping them costs 5–11 points, because the condition needs a
  contrast. Gradient guidance at decode time was the variant that worked (+0.090); it
  appears under "not failed" below.
- **Verdict:** failed. It closes about a third of the gap to a real team, at the cost of
  validity and diversity.

## 3. Uncertainty-based active learning for the win-rate surrogate

- **Papers:** Settles, *Active Learning Literature Survey*, Tech. Report 1648, Univ. of
  Wisconsin–Madison 2009 🟢 [PDF](https://burrsettles.com/pub/settles.activelearning.pdf);
  the BALD variant from Houlsby et al., *Bayesian Active Learning for Classification and
  Preference Learning*, 2011 🟢 [arXiv 1112.5745](https://arxiv.org/abs/1112.5745). No
  Papers/ note.
- **What it is:** label the teams the ridge surrogate is least sure about (predictive
  variance plus a max-min diversity rule, with 20% taken from the predicted top) instead of
  random teams.
- **Why we tried it:** battles are the budget, so we wanted the surrogate to learn f from
  fewer of them.
- **Compared against:** random labelling at the same battle budget.
- **Metric (not a win rate):** Spearman rank correlation between the surrogate and measured
  win rate on 300 held-out teams. Labels are win rates vs top-50 meta, behaviour-cloning
  policy both sides, 24 battles each.
- **Result:** ladder pool: active **0.824** vs random **0.836**, difference −0.011
  [−0.040, +0.018] (31,200 battles). HPS pool, 10 seeds: +0.005 (sd 0.054), a null.
- **Why it failed:** at 24 battles, about 73% of label variance is binomial noise. An
  uncertainty score can't tell noise from model ignorance, so it chased rare features in
  weak teams.
- **What we learned:** informativeness is the wrong acquisition for maximising win rate.
  Picking by predicted value (active search) was the first acquisition to beat its control:
  +0.055 [+0.040, +0.069].
- **Verdict:** failed and dropped (BALD dropped 2026-08-29).

## 4. Trust-Region Noise Search

- **Paper:** Schweiger, Cremers & Ram, *Trust-Region Noise Search for Black-Box Alignment of
  Diffusion and Flow Models*, ReALM-GEN workshop at ICLR 2026, full version accepted to
  ECCV 2026 🟢 [arXiv 2603.14504](https://arxiv.org/abs/2603.14504). Papers note:
  [[Trust-Region Noise Search for Black-Box Alignment of Diffusion and Flow Models]].
- **What it is:** keep the generator frozen and search its input noise with trust regions
  that grow on success and shrink on failure. Our noise is 10,542 Gumbel-max coordinates
  through the constrained decoder.
- **Why we tried it:** it improves a frozen generator using black-box queries only, which
  matched our setting.
- **Compared against:** random search in noise space and best-of-N sampling from the prior,
  at matched budgets.
- **Result:** top-8 mean **0.0814 ± 0.0089**, best team 0.109 (48,384 battles spent; finals
  re-evaluated at 192 fresh battles each). Random search 0.0247 (48,768 battles), best-of-N
  0.0293 (48,000 battles). All win rates vs top-50 meta, behaviour-cloning policy both
  sides. TRS beats random search by +0.057 [+0.037, +0.074].
- **Why it failed:** the mechanism transfers, but a frozen prior caps it. The best team it
  found is at 0.109, about a fifth of a real team (0.491). The landscape has no flat regions
  either: even a trust region of 0.05 changes about 17 of 48 fields.
- **What we learned:** noise search can't fix a weak prior. It would need the re-steered
  checkpoint (never run).
- **Verdict:** failed as a team builder. It is a real win over random search.

## 5. Boltzmann selection with an annealed temperature

- **Paper:** Sanokowski, Hochreiter & Lehner, *A Diffusion Model Framework for Unsupervised
  Neural Combinatorial Optimization* (DiffUCO), ICML 2024 🟢
  [arXiv 2406.01661](https://arxiv.org/abs/2406.01661). Papers note:
  [[DiffUCO — selection temperature experiment]].
- **What it is:** instead of retraining on a hard elite cut, weight each measured team by
  exp(win rate / T) and lower T linearly (0.30 → 0.10), following DiffUCO's annealed
  free-energy objective.
- **Why we tried it:** the combined loop stopped on diversity, not win rate. Annealing
  promised to explore early and exploit late.
- **Compared against:** fixed T = 0.10 / 0.20 / 0.30 and an adaptive T (2026-09-07 run),
  and the hard-cut combined loop (2026-09-02 run).
- **Result:**
  - 2026-09-07 (15 runs, 921,600 battles): fresh generator win rate — annealed **33.25%** vs
    fixed 0.10 **36.05%**, 0.20 34.58%, 0.30 32.15%, adaptive 32.87%. That is 128 samples ×
    192 battles per run, 73,728 battles per arm over 3 seeds. Finalists (16 × 192 per run):
    annealed 46.77% vs fixed 0.10 47.57%. Annealing was lower than fixed 0.10 and 0.20 in
    every seed. All four adjusted confidence intervals include zero.
  - 2026-09-02 (11 generations × 128 teams × 24 battles per arm): Boltzmann weights −0.009
    ± 0.011 vs the hard cut (a null).
  - Two changes did worse: one representative per species set −0.074 ± 0.016 (it plateaued
    at 0.343 even when run to 22 generations), and an entropy bonus inside fine-tuning
    −0.110 ± 0.015.
  - All: win rate vs top-50 meta, behaviour-cloning policy both sides.
- **Why it failed:** at low T the weights become the hard cut again, and at high T they give
  up the concentration that produces the climb. Every arm converged on the same core
  (Garchomp, Kingambit, Basculegion, Whimsicott, Charizard, Floette-Eternal).
- **What we learned:** diversity loss comes from selection, and removing it removes the
  climb. Annealing kept more species sets (255 vs 172 at fixed 0.10) and was the only arm
  never to trip a diversity guardrail, but it bought no win rate.
- **Verdict:** failed on win rate. Its diversity gain is real.

## 6. LLM hyper-heuristic that writes pruning rules

- **Paper:** Ye et al., *ReEvo: Large Language Models as Hyper-Heuristics with Reflective
  Evolution*, NeurIPS 2024 🟢 [arXiv 2402.01145](https://arxiv.org/abs/2402.01145). No
  Papers/ note.
- **What it is:** an LLM writes `keep(team)` filter rules, which are scored without battles
  and evolved from their scores.
- **Why we tried it:** a hand-found rule (three Protect users) already gave a 70× lift in
  meta teams per team kept. We wanted to see whether an LLM could find more such rules.
- **Compared against:** the hand-mined three-protect rule.
- **Metric (not a win rate):** meta retention (the share of 649 real Reg M-B teams a rule
  keeps) against space retention (the share of 20,000 uniform-legal teams it keeps). Lift
  is the ratio. No battles were run.
- **Result:** best LLM rule `two-protect-and-stab`, 97.5% / 0.065 (15× lift), vs
  three-protect 87.7% / 0.0125 (70×). Two generations.
- **Why it failed:** the rules it proposed keep most real teams but prune far less. Its
  "disruption" rule dropped a fifth of real teams, and a common-items rule kept only 21.7%.
- **What we learned:** it widened the trade-off front where our hand-picked properties had
  nothing, but did not beat the one rule that mattered. It was also blocked by expired
  credentials (`claude -p`).
- **Verdict:** failed to beat the baseline rule after two generations. It was never run
  long enough to be conclusive.

## 7. Partial-noise editing of real teams (SDEdit)

- **Paper:** Meng et al., *SDEdit: Guided Image Synthesis and Editing with Stochastic
  Differential Equations*, ICLR 2022 🟢 [arXiv 2108.01073](https://arxiv.org/abs/2108.01073).
  No Papers/ note.
- **What it is:** add partial noise to an existing example and denoise it back, so the model
  makes a local edit. Here: mask k fields of a real team and let the diffusion model
  regenerate them (one candidate regenerated).
- **Why we tried it:** it was the "local move" the diffusion idea promised. Search needs
  small edits of good teams, not whole new teams.
- **Compared against:** leaving the real team alone; pasting one candidate from another real
  team (slot copy); shuffling Stat Points within species.
- **Result:** one candidate regenerated **0.406 ± 0.033** vs untouched real team
  **0.475 ± 0.025**, slot copy 0.420 ± 0.034. The steered version (edit toward the team's
  playstyle) scored 0.371 ± 0.036. Each is a win rate vs top-50 meta, behaviour-cloning
  policy both sides, 2,048 battles per arm (16 teams × 128), errors clustered by team.
  12,288 battles in all.
- **Why it failed:** every edit loses points; the model's edit is no better than pasting
  a real candidate (0.29 σ apart).
- **What we learned:** a dictionary of real candidates is a better local move than a
  learned one. That motivated the copy-paste proposer that later reached 0.540.
- **Verdict:** failed; the "local move" is retracted as a reason to build the model.

## 8. Self-consuming retraining loop

- **Paper:** the setting and its risk come from Alemohammad et al., *Self-Consuming
  Generative Models Go MAD*, ICLR 2024 🟢 [arXiv 2307.01850](https://arxiv.org/abs/2307.01850).
  Papers note: [[Self-Consuming Generative Models Go MAD - reading guide]]. The loop
  itself (sample, admit every new legal team, retrain) was our own design of that setting.
- **What it is:** sample from the generator, add every legal team not already in the data
  back into the data, retrain, repeat.
- **Why we tried it:** to grow a small dataset of 692 real teams without collecting more,
  with Showdown's validator as a perfect legality filter.
- **Compared against:** the uniform-legal team pool (HPS) — can the loop reach teams that
  uniform legal sampling cannot?
- **Metric (not a win rate):** reach beyond the uniform-legal pool at a calibrated Hamming
  radius (E7), plus a plateau rule on frozen held-out loss. No battles were run.
- **Result:** stopped at generation 2. Reach beyond the uniform-legal pool was **0.000** in
  both generations; held-out loss was flat. Validity rose 0.477 → 0.535 (a concentration
  alarm) while coverage stayed intact. 19,663 teams admitted.
- **Why it failed:** admitting every legal sample adds no quality signal, so the loop
  behaved like slow uniform-legal sampling.
- **What we learned:** a retraining loop needs an admission signal tied to quality or
  reach, not just legality and novelty.
- **Verdict:** failed as run; do not re-run the plain loop.

---

## Tried, and did not fail

- **Cross-entropy method** (Rubinstein 1999; de Boer et al., *A Tutorial on the Cross-Entropy
  Method*, Annals of Operations Research 2005 🔒). No Papers/ note.
  - Its diffusion-proposer run did not climb: **0.165 / 0.172 / 0.153** over 3 rounds, 200
    proposals × 24 battles per round.
  - The same loop with a copy-paste proposer climbed: **0.468 → 0.494**, best 0.875.
  - With the elite fraction ρ = 0.01 it reached **0.540 ± 0.008** (200 proposals × 24 =
    4,800 battles per arm), above a real team.
  - All: win rate vs top-50 meta, behaviour-cloning policy both sides.
  - The proposer was the change that failed; the method did not.
- **Gradient guidance at decode time** (Guo et al., *Gradient Guidance for Diffusion
  Models: An Optimization Perspective*, NeurIPS 2024 🟢
  [arXiv 2404.14743](https://arxiv.org/abs/2404.14743)).
  - Naive gradient guidance scored 0.230 and the paper's look-ahead form 0.231, vs 0.140
    unguided (128 teams × 24 = 3,072 battles per arm).
  - In the combined loop it reached **0.411** by generation 10 (73,728 battles), and its best
    team re-battled at **0.589 ± 0.036** over 192 battles.
  - The paper's full form with the Jacobian term was the change that lost: 0.209, and its
    loop collapsed.
- **Active search**, picking the next teams to battle by predicted value (Garnett's
  framing): +0.055 [+0.040, +0.069] over random labelling at the same budget (44,736
    battles, 24 per team).
  - Margin scoring (Fontaine et al., GECCO 2019) was a change to it and was a null (+0.004
    [−0.006, +0.015]), because margin tracks win rate at Spearman 0.95.

