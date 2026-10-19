# Cost of evaluating every mask size

Estimate dated October 8, 2026. Expanding the existing three-team pilot to every possible integer mask size requires **243 additional Ling API calls**, for **306 total calls including the 63 already recorded**. Estimated additional API cost is **$0.028 with the pilot's cache behavior**, or **$0.099 without caching at the pilot's average token lengths**.

## What “continuous” means

The slider can move smoothly from 0–100%, but a team contains a finite number of fields: each evaluated mask still hides a whole number of fields. Evaluating every integer size eliminates the current large jumps. A fractional slider position maps to a count; any interpolated performance between measured counts must be labeled an estimate. Moving the slider over saved results makes no API calls.

This estimate retains the existing three starting teams, one completion per team/task/mask size, historical Regulation M-B, and the same meta-opponent panel and behavior-cloning battle policy. Diffusion retains ask 0.5, guidance 2, temperature 1. It is the retrained F1/G2 ensemble used in the current comparison.

## Additional calls

| Task family | Full mask range | Existing calls | Full-grid calls | Additional calls |
|---|---:|---:|---:|---:|
| Stat spreads | 1–6 | 9 | 18 | 9 |
| Items | 1–6 | 9 | 18 | 9 |
| Whole Pokémon | 1–6 | 9 | 18 | 9 |
| Abilities | 1–6 | 9 | 18 | 9 |
| Natures / alignment | 1–6 | 9 | 18 | 9 |
| Individual moves | 1–24 | 9 | 72 | 63 |
| Mixed fields | 1–48 | 9 | 144 | 135 |
| **Total** | | **63** | **306** | **243** |

Mixed has 48 eligible parts: six stat spreads, six items, six abilities, six natures, and 24 moves. The current experiment only samples mixed sizes 1, 20, and 40. Keeping the mixed slider capped at 40 instead would require **219 additional calls**, giving 282 total. Mask size zero reuses the original-team results and needs no calls.

Existing trials should be retained, including invalid completions. Add only missing sizes using the same nested mask ordering. The estimate includes one attempt per new point, without retries or repairs.

## API cost scenarios

The recorded 63-call pilot cost $0.0073227: an average of $0.000116233 per call, with 19,027.56 input tokens and 157.57 output tokens per call. Approximately 91.675% of input tokens were cached.

Current Novita pricing for inclusionai/ling-3.0-flash-vl is $0.021 per million uncached input tokens, $0.0042 per million cached input tokens, and $0.0616 per million output tokens. These are currently discounted rates, verified against [OpenRouter pricing](https://openrouter.ai/inclusionai/ling-3.0-flash-vl).

| Scope | Extra calls | Pilot cache and token mix | No cache, pilot token lengths | No cache, 4,096 output tokens per call |
|---|---:|---:|---:|---:|
| Mixed through 40 | 219 | $0.0255 | $0.0896 | $0.1428 |
| **Full mixed range through 48** | **243** | **$0.0282** | **$0.0995** | **$0.1584** |

The final column retains the pilot's mean input length and assumes every call exhausts the configured output allowance. It is a planning scenario, not a strict upper bound: prompt lengths, provider pricing, retries, and output lengths can change. Larger masks may produce longer answers than the sparse pilot average. A **$0.20 API allowance** is reasonable at these discounted rates under the stated assumptions. If the 72% discount disappears, multiply these estimates by approximately 3.57; the full-output scenario becomes about $0.57.

Formula: cost = ((input tokens − cached tokens) × 0.021 + cached tokens × 0.0042 + output tokens × 0.0616) / 1,000,000.

## Local experiment work

For the full range, also generate **243 local diffusion completions** and **243 local random-fill controls**. Diffusion and random fill incur no OpenRouter charges.

Scoring all three new arms against the existing 50-battle panel produces up to **36,450 additional battle outcome slots** (243 × 3 × 50). Invalid outputs and exact-team panel reuse reduce actual simulation work. The original teams' panels are reused. Local compute time or hosted-machine charges are separate from the API estimate and have not been estimated here.

More mask sizes improve curve resolution; they do not increase the number of independent starting teams beyond three. The sweep therefore remains exploratory rather than strong evidence that one model wins generally.

This document is an estimate only; no new model calls or battle sweep were launched for it.
