# Faster fine-tuning of the text diffusion model: sources (2026-10-18)

**Question.** Which published ideas could make the Fast-dLLM v2 1.5B LoRA fine-tune (`src/textdiff.py train`) train
faster on the M4 Pro at the same loss: a smaller pretrained text diffusion model, a lighter optimizer than AdamW,
Neural Magic-style sparsity, or a cheaper block-diffusion training row? The measured answers are in
[textdiff.md](textdiff.md), section "Round three". This note holds what each source says.

**How it was read.** Haiku web-fetch agents (WebFetch only) read fixed pages, one narrow job each, and returned
claims with short quotes and URLs; the main session chose the pages and checked the claims used below against them.
🟢 = openable without an institutional login.

## Replacement text diffusion models

- Fast-dLLM v2 ships two checkpoints, 1.5B and 7B, both adapted from Qwen2.5-Instruct; the paper is at ICLR 2026
  [[1]](#ref1) [[2]](#ref2). Nothing smaller exists in that family.
- The dLLM library's hub (dllm-hub) holds six small diffusion models converted from autoregressive ones: ModernBERT
  base and large chat (0.2B, 0.4B), Qwen3-0.6B as a masked (MDLM) and a block (BD3LM) diffusion model (0.8B with
  embeddings), and Qwen2.5-Coder-0.5B-Instruct in the same two forms (0.6B) [[3]](#ref3).
- Qwen3-0.6B-diffusion-bd3lm-v0.1: block size 32, mask token `<|mask|>` id 151669, bfloat16, one 1.5 GB weight file,
  custom modelling code (`trust_remote_code`), supervised fine-tuning on tulu-3-sft-mixture, smoltalk and
  opc-sft, Apache-2.0 [[4]](#ref4).
- Qwen2.5-Coder-0.5B-Instruct-diffusion-bd3lm-v0.1: the Qwen2 architecture (24 layers, hidden 896, 14 query and 2
  key-value heads, tied embeddings), `<|mask|>` at id 151665 (the same id as Fast-dLLM v2's [MASK]), one 1.26 GB
  weight file, fine-tuned on opc-sft stages 1 and 2 (code instructions), Apache-2.0 [[18]](#ref18).
- No token shift: dllm's BD3LM trainer scores the logits at position i against the clean token at position i; a
  `right_shift_logits` option exists and is off by default [[19]](#ref19). Fast-dLLM v2 predicts token i + 1 from
  position i. Measured here (untrained, 16 held-out rows, the same masks): validation loss without the shift 5.15
  (Qwen2.5-Coder 0.5B) and 5.01 (Qwen3 0.6B), with it 7.21 and 7.34. Both tokenizers encode the pastes exactly as
  Fast-dLLM v2's does.
- Only the weights and JSON files were downloaded (with the owner's approval, 2026-10-18); `textdiff_port.load`
  puts the tensors into the released architecture (Qwen2) or a stock transformers Qwen3, so none of the repositories'
  own code runs.
- SDAR-1.7B-Chat (block diffusion from Qwen3) is larger than our 1.5B model, so it is slower per token [[5]](#ref5).

## Cheaper block-diffusion training

- E2D2 splits a block-diffusion model into an encoder for the clean tokens and a light decoder for the noisy ones
  (NeurIPS 2025) [[6]](#ref6). Fine-tuning a decoder-only model, its shared-KV variant uses the layers closest to the
  output as the decoder, weight-tied with the encoder (28/14 layers for Qwen3-1.7B on GSM8K), and decoder layer i
  reads the encoder's keys and values at the matching depth. A noisy block attends to its own tokens and to clean
  blocks before it. Gradients reach the encoder (no stop-gradient in Algorithm 2). Measured: 40% faster training
  than BD3LM on one H100 (8.4 × 10⁴ vs 5.9 × 10⁴ tokens/s, L = 1024), 2× fewer training FLOPs in theory.
- Here: `--layout split` runs a training row as two passes (clean, then noisy reading the clean keys and values)
  and equals `trim` exactly; `--decoder-layers 14` sends the noisy tokens through the top 14 layers only (E2D2
  shared-KV); `--layout detach` drops the gradient through the clean pass.

## Optimizers: SGD and lighter relatives instead of AdamW

- Arithmetic for this run: LoRA rank 32 trains 36.9M parameters in float32, so AdamW keeps 2 × 36.9M × 4 bytes =
  0.30 GB of state (measured: 0.295 GB). SGD with momentum keeps half of that, plain SGD none. The step's high-water
  is 8–10 GB, so the most any optimizer swap can free is about 3% of it.
- Lion keeps one momentum state ("more memory-efficient than Adam") and needs a smaller learning rate than Adam
  [[7]](#ref7).
- MeZO trains with the memory footprint of inference, two forward passes and no backward pass, and is compatible
  with LoRA; the abstract calls zeroth-order methods in general slow (NeurIPS 2023) [[8]](#ref8).
- Schedule-Free AdamW removes the learning-rate schedule; it is still AdamW and makes no memory claim [[9]](#ref9).
- LOMO fuses the gradient computation and the parameter update in one step, for full-parameter fine-tuning
  (ACL 2024) [[10]](#ref10). With LoRA there are no full gradients to save.
- LoRA+ gives the B matrices a larger learning rate than A: of order 2²–2³ when B starts at zero (our case), 2¹–2²
  for LLaMA, with up to about 2× faster fine-tuning reported [[11]](#ref11).
- Riemannian preconditioned LoRA multiplies each LoRA gradient by an r × r matrix. Plain SGD with it ("scaled GD")
  closes the SGD–AdamW gap on GPT-2 medium (BLEU 66.6 → 69.2 against AdamW's 68.9) but not on 4-bit Mistral 7B
  GLUE (80.36 against 88.28) [[12]](#ref12).
- LoRA-GA initialises the adapters so their first gradient matches full fine-tuning and reports 2–4× faster
  convergence [[13]](#ref13).

## Neural Magic: sparsity

- Neural Magic's 2:4 Sparse Llama is built for NVIDIA Ampere and newer GPUs, and every speed-up the article reports
  is inference on vLLM [[14]](#ref14).
- LLM Compressor quantises models (W8A8, W4A16, FP4 formats) for deployment on vLLM [[15]](#ref15).
- PyTorch's semi-structured (2:4) sparsity needs an NVIDIA GPU of compute capability 8.0 or newer and NVIDIA's
  sparse kernels; the tutorial never mentions Apple or Metal [[16]](#ref16).
- So no sparsity format speeds up training on the M4 Pro. Removing whole MLP neurons does, because the dense
  matrices get smaller (`--prune-mlp`, after Minitron [[17]](#ref17)).
- Minitron scores an MLP neuron by its activation over 1024 calibration samples (sum over batch and sequence,
  aggregated as batch L2 and sequence mean) and found width pruning better than depth pruning only after retraining
  (Llama 3.1 4B: MMLU 60.5 vs 58.7); recovery used distillation from the unpruned model on 1.8B–94B tokens, and LoRA
  was left to future work [[17]](#ref17) [[20]](#ref20). `--prune-mlp` scores a neuron by mean |input to down_proj|
  over 64 sequences: Minitron-style, not its exact aggregator, and with 500 LoRA steps instead of distillation.

## References

<a id="ref1"></a>1. 🟢 Efficient-Large-Model on Hugging Face (Fast_dLLM_v2_1.5B, Fast_dLLM_v2_7B). https://huggingface.co/Efficient-Large-Model
<a id="ref2"></a>2. 🟢 Wu et al., "Fast-dLLM v2: Efficient Block-Diffusion LLM", ICLR 2026. https://arxiv.org/abs/2509.26328
<a id="ref3"></a>3. 🟢 dllm-hub on Hugging Face. https://huggingface.co/dllm-hub
<a id="ref4"></a>4. 🟢 dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1 (model card, config.json, tokenizer_config.json). https://huggingface.co/dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1
<a id="ref5"></a>5. 🟢 JetLM/SDAR-1.7B-Chat. https://huggingface.co/JetLM/SDAR-1.7B-Chat
<a id="ref6"></a>6. 🟢 Arriola et al., "Encoder-Decoder Diffusion Language Models for Efficient Training and Inference", NeurIPS 2025. https://arxiv.org/abs/2510.22852
<a id="ref7"></a>7. 🟢 Chen et al., "Symbolic Discovery of Optimization Algorithms" (Lion); no venue listed on arXiv. https://arxiv.org/abs/2302.06675
<a id="ref8"></a>8. 🟢 Malladi et al., "Fine-Tuning Language Models with Just Forward Passes" (MeZO), NeurIPS 2023. https://arxiv.org/abs/2305.17333
<a id="ref9"></a>9. 🟢 Defazio et al., "The Road Less Scheduled" (Schedule-Free); no venue listed on arXiv. https://arxiv.org/abs/2405.15682
<a id="ref10"></a>10. 🟢 Lv et al., "Full Parameter Fine-tuning for Large Language Models with Limited Resources" (LOMO), ACL 2024. https://arxiv.org/abs/2306.09782
<a id="ref11"></a>11. 🟢 Hayou et al., "LoRA+: Efficient Low Rank Adaptation of Large Models"; no venue listed on arXiv. https://arxiv.org/abs/2402.12354
<a id="ref12"></a>12. 🟢 Zhang & Pilanci, "Riemannian Preconditioned LoRA for Fine-Tuning Foundation Models"; no venue listed on arXiv. https://arxiv.org/abs/2402.02347
<a id="ref13"></a>13. 🟢 Wang et al., "LoRA-GA: Low-Rank Adaptation with Gradient Approximation"; no venue listed on arXiv. https://arxiv.org/abs/2407.05000
<a id="ref14"></a>14. 🟢 Red Hat Developer article on 2:4 Sparse Llama for GPU inference (2025-02-28). https://developers.redhat.com/articles/2025/02/28/24-sparse-llama-smaller-models-efficient-gpu-inference
<a id="ref15"></a>15. 🟢 vllm-project/llm-compressor (README). https://github.com/vllm-project/llm-compressor
<a id="ref16"></a>16. 🟢 PyTorch tutorial on semi-structured (2:4) sparsity. https://docs.pytorch.org/tutorials/advanced/semi_structured_sparse.html
<a id="ref17"></a>17. 🟢 Muralidharan et al., "Compact Language Models via Pruning and Knowledge Distillation" (Minitron). https://arxiv.org/abs/2407.14679
<a id="ref18"></a>18. 🟢 dllm-hub/Qwen2.5-Coder-0.5B-Instruct-diffusion-bd3lm-v0.1 (model card, config.json, tokenizer_config.json). https://huggingface.co/dllm-hub/Qwen2.5-Coder-0.5B-Instruct-diffusion-bd3lm-v0.1
<a id="ref19"></a>19. 🟢 ZHZisZZ/dllm, BD3LM and MDLM trainers. https://github.com/ZHZisZZ/dllm/tree/main/dllm/core/trainers
<a id="ref20"></a>20. 🟢 Sreenivas et al., "LLM Pruning and Distillation in Practice: The Minitron Approach"; no venue listed on arXiv. https://arxiv.org/abs/2408.11796
