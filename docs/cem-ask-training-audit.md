# Is there a training bug behind the cross-entropy loop's flat ask? (audit, 2026-10-16)

**Verdict: no bug found in the training code.** The cross-entropy loop returns about 0.51 win rate whether it asks 0.3,
0.5 or 0.8. Win rate = wins / battles against the top-50 meta teams, behaviour-cloned policy on both sides. The
condition reaches both members in training and at generation time in the same form; label dropout runs during the elite
fine-tune; the null condition and the guidance formula match the classifier-free guidance paper. The flat ask is a
modelling effect: the elite fine-tune only ever shows labels 0.52–0.72, and the retrained model loses the middle of the
win-rate scale (see `ask-dial.md`, "Does the loop's model become a forward model?").

## A. Condition shape for the matrix-row member (G2) — not a bug
`lossdown.Net.forward` (`src/lossdown.py` 227–242): for an `mrow` model, a 2-D `w` splits into the scalar `w[:, 0]` and the
row `w[:, 1:]`. A 1-D `w` keeps the scalar and fills the row with NaN. The elite fine-tune (`cem_v3.members`) passes
`[label, NaN × 49]`, so both forms reach `mvec` as an all-NaN row → the learned `mnull` (lines 222–225) and the same scalar.
Probe on `G2_family_matrix_noprtrain_noheldout_regmb_v3`, 16 half-decoded teams: the 1-D and 2-D forms give identical logits
(max difference 0.0 at ask 0.3 and 0.5). G2 alone responds to the ask (max logit change 3.0 between 0.3 and 0.5).

## B. Label dropout during the fine-tune — applied
Both members load `p_uncond = 0.15` (`load_net` merges `DEFAULT`, line 360). `batch_loss` sets the condition to NaN with that
probability (line 504), and `cem_v3.members` trains through `batch_loss` with the member's own config. So the fine-tune keeps
training the unconditional branch that guidance subtracts.

## C. Null condition and guidance — consistent
`wvec` (lines 216–220) maps a NaN condition to the learned `wnull`. `cem_dbbo.sample_asks` mixes
`g · logp(ask) + (1 − g) · logp(NaN)` with g = 2. That is the paper's `(1 + w)·cond − w·uncond` with w = 1.

## D. Fine-tune regime — the cause (modelling, not a bug)
40 epochs at learning rate 1e-4 on 64 → 110 → 168 elites, every label 0.52–0.72 (rebuilt exactly as `run()` builds them;
0 labels equal the ask). The model never sees a 0.3 or 0.5 team during the fine-tune and is then asked for one. Measured
effect: asked about the 1,105 precise-label teams, the generation-3 model's best-fitting win rate correlates +0.46 with the
battles. Only 188 of those teams are placed in 0.3–0.6. Models trained on the full label range reach +0.83–0.84
(`src/implied_label.py`).

## E. Other checks — nothing found
- The label is the battled win rate from the matchup matrix, never the ask (`cem_v3.elites_from` uses `r["y"]`).
- Fine-tunes start from the original members each generation; checkpoints are cached per arm tag and generation
  (`{TAG}_g{g}_{member}.pt`), so arms do not share weights.
- Each row of a mixed-ask batch is conditioned on its own ask (`scripts/test_ask_rows.py`).

## Primary source
🟢 Ho & Salimans, "Classifier-Free Diffusion Guidance", arXiv 2207.12598 (https://arxiv.org/abs/2207.12598). Guided score
(Eq. 6): (1 + w)·ε(z, c) − w·ε(z), with w the guidance strength. One network learns both terms by replacing the condition with
a null token with probability p_uncond; they tried 0.1, 0.2 and 0.5, and 0.5 was worst. The paper tests only class labels on
ImageNet. It says nothing about a continuous condition like a win rate, or about conditioning on values outside the training
distribution, which is the situation the elite fine-tune creates.
