# Self-consuming loops: the problem and the fixes (2026-09-30)

Starting papers: **Self-Consuming Generative Models Go MAD** (Alemohammad et al., ICLR 2024) and **On the Stability
of Iterative Retraining of Generative Models on their own Data** (Bertrand et al., ICLR 2024 spotlight). Below are
papers that cite them and propose a fix, grouped by the fix. Venues were checked at the source (arXiv
comments, conference pages, OpenReview). 🟢 = free to open.

**Why it matters here.** The project's self-training loop (retrain the team generator on its own best teams) is a
*curated* self-consuming loop. It stopped at generation 2 on a concentration alarm (2026-08-29). Grades below
are for that loop: a generator of full Reg M-B teams, retrained on teams selected by battle win rate.

## The problem, in one paragraph each
- **MAD** — [Self-Consuming Generative Models Go MAD](https://arxiv.org/abs/2307.01850) 🟢 ICLR 2024. Train each
  generation on the last one's samples: without enough fresh real data, quality or diversity falls generation by
  generation. Cherry-picking typical samples (sampling bias λ < 1) keeps quality but speeds the loss of diversity.
- **Stability** — [On the Stability of Iterative Retraining…](https://arxiv.org/abs/2310.00429) 🟢 ICLR 2024
  spotlight. Retraining is stable when the first model is close enough to the real data and each round keeps a
  large enough share of real data; synthetic-only retraining collapses.

## Fix 1 — keep real data in every round
| Paper | Venue | What it shows | Grade |
|---|---|---|---|
| [Is Model Collapse Inevitable? Breaking the Curse of Recursion by Accumulating Real and Synthetic Data](https://arxiv.org/abs/2404.01413) 🟢 | COLM 2024 | Accumulating data (never discarding the real set) keeps error bounded; replacing it collapses | HIGH |
| [Collapse or Thrive? Perils and Promises of Synthetic Data in a Self-Generating World](https://arxiv.org/abs/2410.16713) 🟢 | NeurIPS 2024 workshops (M3L; ATTRIB) | Replace collapses; accumulate is stable even as the real share shrinks; a fixed budget degrades slowly | MED |
| [A Tale of Tails: Model Collapse as a Change of Scaling Laws](https://arxiv.org/abs/2402.07043) 🟢 | ICML 2024 | Synthetic data cuts the tails and bends scaling laws; mixing human data restores improvement | MED |
| [A Theoretical Perspective: How to Prevent Model Collapse in Self-consuming Training Loops](https://arxiv.org/abs/2502.18865) 🟢 | ICLR 2025 | "Recursive stability": architecture and the real-data share decide whether the loop converges | MED |

**For the team loop (HIGH overall):** refit on the real meta teams *plus* the selected teams every round, never on
selected teams alone. Cheapest fix, directly aimed at the concentration alarm.

## Fix 2 — curate, filter or verify the synthetic data
| Paper | Venue | What it shows | Grade |
|---|---|---|---|
| [Self-Consuming Generative Models with Curated Data Provably Optimize Human Preferences](https://arxiv.org/abs/2407.09499) 🟢 | NeurIPS 2024 spotlight | Retraining on reward-curated samples is implicit preference optimisation: expected reward is maximised, reward-model biases are amplified, a positive real share keeps it stable | HIGH |
| [Self-Consuming Generative Models with Adversarially Curated Data](https://arxiv.org/abs/2505.09768) 🟢 | ICML 2025 | When the curator is noisy or adversarial, conditions for (in)stability of the loop | MED |
| [Beyond Model Collapse: Scaling Up with Synthesized Data Requires Verification](https://arxiv.org/abs/2406.07515) 🟢 | ICLR 2025 | A verifier that selects synthetic data prevents collapse, if it ranks well | HIGH |
| [Escaping Model Collapse via Synthetic Data Verification: Near-Term Improvements and Long-Term Convergence](https://arxiv.org/abs/2510.16657) 🟢 | ICLR 2026 (OpenReview PDF: DATA-FM workshop at ICLR 2026) | Verification gives near-term gains, but the model converges to the verifier's "knowledge centre" unless the verifier is perfect | HIGH |
| [Self-Correcting Self-Consuming Loops for Generative Model Training](https://arxiv.org/abs/2402.07087) 🟢 | ICML 2024 | An expert correction function applied to samples before retraining stabilises the loop | MED |
| [Stabilizing Self-Consuming Diffusion Models with Latent Space Filtering](https://arxiv.org/abs/2511.12742) 🟢 | AAAI 2026 | Filter out less realistic synthetic samples using latent-space degradation patterns | MED |

**For the team loop:** the battle simulator *is* the curator and verifier. Ferbach et al. say the loop then
optimises win rate and amplifies whatever the scorer gets wrong; the verification papers say it converges to the
verifier's centre. With 24-battle labels the verifier is noisy (0.79 label → 0.57 re-battled), so curation must use
re-battled scores and keep a real share.

## Fix 3 — use the synthetic data as a negative signal
| Paper | Venue | What it shows | Grade |
|---|---|---|---|
| [Neon: Negative Extrapolation From Self-Training Improves Image Generation](https://arxiv.org/abs/2510.03597) 🟢 | ICLR 2026 oral | Fine-tune on your own samples, then move the weights the *opposite* way (θ − w·Δ): a post-hoc merge, ~1k samples, < 1% extra compute, ImageNet-256 FID 1.02 | MED |
| [Self-Improving Diffusion Models with Synthetic Data (SIMS)](https://arxiv.org/abs/2408.16333) 🟢 | ICLR 2025 workshop (Self-Improving Foundation Models Without Human Supervision) | Train an auxiliary model on self-generated data and use it as *negative guidance* away from the synthetic manifold; iterated without going MAD | MED |
| [Guiding a Diffusion Model with a Bad Version of Itself (Autoguidance)](https://arxiv.org/abs/2406.02507) 🟢 | NeurIPS 2024 | Related mechanism, not a self-consuming paper: guide away from a weaker copy of the model | LOW |

**For the team loop (MED):** these repair *fidelity to the real distribution*, not win rate. Neon is the cheapest
experiment in the whole list: fine-tune the masked team model on ~1k of its own teams, extrapolate away, and check
legality, distance to the meta teams and win rate. Untested on discrete masked models.

## Fix 4 — diagnose where collapse comes from
| Paper | Venue | What it shows | Grade |
|---|---|---|---|
| [A Closer Look at Model Collapse: From a Generalization-to-Memorization Perspective](https://arxiv.org/abs/2509.16499) 🟢 | NeurIPS 2025 spotlight | Diffusion models in the loop drift from generalising to memorising; entropy of the training data predicts it | HIGH |
| [Model Collapse in the Self-Consuming Chain of Diffusion Finetuning](https://arxiv.org/abs/2407.17493) 🟢 | ICLR 2025 | Fine-tuning chains degrade like a quantitative trait; ReDiFine slows it | HIGH |
| [AI models collapse when trained on recursively generated data](https://www.nature.com/articles/s41586-024-07566-y) 🟢 | Nature 2024 | Tails disappear first, across model families | MED |

## Bottom line for the self-training loop
1. Keep the real meta teams in every refit (Fix 1). Cheapest, best supported.
2. Treat battle selection as curation (Fix 2): re-battle before admitting a team, because the loop will amplify
   label noise.
3. Try Neon as a cheap post-hoc repair of the prior (Fix 3), measured on legality, meta distance and win rate.
