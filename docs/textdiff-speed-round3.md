# Training the text team writer faster, round three (2026-10-18)

The owner's goal: a faster fine-tune of the text team writer (Fast-dLLM v2 1.5B + LoRA, `src/textdiff.py train`) with the
loss close to the last measurement, wild ideas welcome, including another text diffusion model, SGD instead of AdamW
for memory, and Neural Magic. Rounds one and two are in [textdiff.md](textdiff.md) ("Training speed on the M4 Pro",
"Twelve ways to train it faster"); the sources behind this round, read by eight Haiku web-fetch agents
with briefs written to the writing-for-agents skill, are in [the research note](textdiff-speed-research.md).

**Speed before this round.** The trainer as left on 2026-10-16 (the shared-clean row, one training row holding both
complementary noisy copies and one clean copy of the team; `enable_gqa`; compile; full checkpointing; batch 4) ran at **6.10 s per step** on 2026-10-18 (steps 25–100; a second control at the end of the
probes: 6.08), 1.5 s per example, 9.5 GB peak, 0.30 GB of AdamW state. A 2,000-step run takes about 3.4 h. Both
controls reproduced the 2026-10-16 control's training loss at every logged step (4.584, 3.114, 2.455, 2.307).

## What changed in the code (in `main` since commit 12dd21c, 2026-10-19)

`textdiff_port.functional_forward` writes the released training forward out layer by layer, with rotary positions
passed explicitly, and `--layout` arranges the training row through it:

- `trim` drops the work that never reaches the loss: the noisy block whose hidden states predict only prompt labels
  (the full prompt fills block 0), the last clean block (no noisy block attends to it), and any all-padding tail block
  of a micro-batch. Same loss, same gradient.
- `split` runs the row as two passes, clean then noisy, the noisy queries reading the clean keys and values. Same
  loss, same gradient.
- `detach` is `split` with the clean pass under `no_grad`: the same loss value, no gradient through the clean context.
- `--micro-batch N` splits each batch of four into micro-batches whose gradients add up to the whole batch's, so
  gradient checkpointing can go (`--ckpt-layers 0`).
- Also new: `--decoder-layers N` (E2D2's shared-key-value decoder), `--optimizer adamw | adamw-fused | sgd | lion |
  adafactor | mezo`, `--lora-plus`, `--prune-mlp`, `--no-shift`, `--base <folder>` (a dllm-hub Qwen2 or Qwen3
  block-diffusion checkpoint, weights and JSON only), `textdiff.py verify`, and `src/textdiff_interior.py`.

`textdiff.py verify` checks the layouts against the released forward on one real batch: `full` matches exactly; in
float32 `trim` matches to 1.4 × 10⁻⁶ in loss and 3.6 × 10⁻⁶ relative in gradient, `split` to 2.7 × 10⁻⁶; `detach`
keeps the loss value and changes 80% of the gradient's norm.

**Protocol.** As in round two: the same first 2,000 examples in the same order, the same noisy copies (every layout
draws the masks with the same calls on the same random-number generator), batch 4, the 2,000-step learning-rate schedule, one seed.
Speed is the steady seconds per step from step 25 to 100, compile excluded, on the M4 Pro (24 GB). Full arms run 500
steps and are scored with the published loss, Fast-dLLM v2's own training loss (mean cross-entropy over the masked
answer tokens of both complementary noisy copies), on the same 256 held-out rows. That validation loss was **0.958**
for the 2026-10-16 control. "Loss close" means at most 1% above 0.958 (lower is better), about the spread round two measured between two runs
of the same trainer (0.949 against 0.958). Every timing below was measured before 13:07 on 2026-10-18, except the
pruning row, which ran while another session's battles shared the CPU.

## Results

| idea | exact? | s / step | speed-up | peak GB | validation loss at 500 steps (control 0.958) | block-interior loss (control 0.940) | verdict |
|---|---|---|---|---|---|---|---|
| control (the trainer before this round) | — | 6.10 | 1.00× | 9.5 | 0.958 | 0.940 | reference |
| trim the dead blocks | yes | 5.30 | 1.15× | 9.5 | train loss same as control | — | faster, same loss |
| trim + fused AdamW | yes | 5.25 | 1.16× | 9.5 | train loss same | — | fused AdamW saves 0.9% |
| **trim + micro-batch 1 + no checkpointing** | **yes** | **3.83** | **1.59×** | 18.2 | **0.947 (−1.1%)** | 0.930 | **faster, loss close: verified** (500 steps in 34.7 min; the 2026-10-16 control took 57.8) |
| trim + micro-batch 2 + no checkpointing | yes | — | — | 19–20 | — | — | does not fit: the machine swaps |
| detach the clean blocks | no | 4.58 | 1.33× | 7.3 | train +4.8% at step 100 | — | faster, loss off |
| detach + micro-batch 1 or 2 + no checkpointing | no | 3.44 | 1.77× | 11.7 / 16.3 | train +4.7% at step 100 | — | faster, loss off at equal steps |
| SGD with momentum (lr 0.1, best of 0.03 / 0.1 / 0.3) instead of AdamW | no | 3.80 | same as AdamW | 16.8 | 1.091 (+13.9%) | — | worse; saves 0.15 GB of optimizer state |
| MLP width pruning to half (the Metal version of Neural Magic's sparsity) | no | — | — | 10.8 | train +7% at step 250 | — | partial: stopped at step 250, timed under contention |
| Qwen2.5-Coder 0.5B block diffusion (dllm-hub) | — | 1.46 | 4.17× | 8.4 | 0.974 (+1.7%) | 0.998 (+6.1%) | much faster, loss off at equal steps |
| Qwen3 0.6B block diffusion (dllm-hub) | — | 1.97 | 3.09× | 9.8 | 0.959 (+0.1%) | 0.982 (+4.5%) | faster, loss off on the comparable measure |

The SGD, pruning and replacement rows ran on the trim + micro-batch 1 layout, so their speed comes from the layout or
the smaller model, not from the optimizer.

**The replacement models and the token shift.** The two dllm-hub checkpoints were trained to predict a masked token at
its own position; Fast-dLLM v2 predicts it from the position before. For 31 of every 32 tokens both see the same
context. A block's first token is the exception: with the shift it is predicted from the previous block's last
position (loss 1.47 for the control); without the shift it also sees the rest of its own block (0.28–0.29). The
published validation loss mixes the two and flatters the shift-free models, so the block-interior loss
(`src/textdiff_interior.py`: same rows, same masks, tokens that are not first in their block) is the number to
compare. On it the replacements are 4.5% and 6.1% worse than the control at equal steps.

**At equal time the smaller model goes further.** The Qwen2.5-Coder 0.5B model ran the whole 2,000-step schedule in
51 min, less than the 2026-10-16 control's 57.8 min for 500 steps (the recommended exact layout needs 34.7 min for
500). Its validation loss at 500 / 1,000 / 1,500 / 2,000 steps was 0.974 / 0.803 / 0.717 / 0.677, against the
recorded Fast-dLLM v2 run's 0.949 / 0.803 / 0.707 / 0.667, which took 6.2 h. These are all-token losses, which
flatter the shift-free model; the block-interior losses of the full runs are queued with the unrun arms, so whether it
wins at equal time on the comparable measure is still open. Neither replacement has been decoded or battled; their
sampler would need the shift removed too.

**SGD and memory.** AdamW keeps 0.30 GB of state for the 36.9M LoRA weights, SGD with momentum 0.15 GB. The step's
high-water mark is 16–18 GB on the fast layout, almost all of it activations and the Metal cache, so swapping the
optimizer frees under 1% of the memory and costs 14% of the loss. SGD's lower peak in the table (16.8 GB
against 18.2) comes from a cache flush added to the trainer after the exact-layout run, not from the optimizer. The research shows the same picture: Riemannian
preconditioned SGD closes the SGD–AdamW gap on GPT-2 but not on Mistral 7B.

**Sparsity and pruning, Neural Magic's approach.** Neural Magic's 2:4 sparsity needs NVIDIA Ampere or newer GPUs and speeds up inference, not training; PyTorch's
2:4 kernels need compute capability 8.0 and never mention Metal. The Mac version of the idea is to remove whole MLP
neurons (`--prune-mlp`, after Minitron), which shrinks the dense matrices. Pruning half the neurons left the training
loss 7% above the control at step 250, where the run stopped.

## Not run

The queue stopped when the session ended (2026-10-18, about 13:45, inside the pruning run); from 13:07 the owner's
ablation queue was sharing the CPU, so later timings were inflated. These arms were written and verified but have no
result: pruning to completion, detach at 500 steps, LoRA+ (B matrices at 4× the learning rate), E2D2's shallow decoder
(noisy tokens through the top 14 layers, with and without gradient through the clean pass), Lion, MeZO, the Qwen3
model over the full schedule, and back-to-back speed probes of the arms timed under contention.

## Decision

Use **trim + micro-batch 1 + no checkpointing** (`--layout trim --micro-batch 1 --ckpt-layers 0`): 1.59× faster
with the same loss function and gradient, verified at 500 steps (validation 0.947 against 0.958; the gap is bf16
rounding on identical math). It peaks at 16–18 GB of the 24; micro-batch 2 without checkpointing does not fit. A 2,000-step run drops from about 3.4 h to
2.1–2.3 h. The smaller dllm-hub models are 3–4× faster but 4.5–6% worse per step on the comparable loss; they are
worth a battle test only at equal time, after a shift-free sampler exists.
