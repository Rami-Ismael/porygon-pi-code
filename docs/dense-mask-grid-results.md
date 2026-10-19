# Full mask grid results

306 tasks; 63 reused and 243 additional Ling calls. Additional API cost: $0.02807940; combined cost: $0.03540210.

Ling: 54/306 legal. Diffusion: 303/306 legal, 306/306 supported. 54 jointly scored tasks.

Equal-start-weighted diffusion minus Ling: -0.40 percentage points on jointly scored tasks. Three independent starts: descriptive only, not proof of a winner.

Diffusion uses the full-regulation F1/G2 v2 ensemble, ask 0.5, guidance 2, temperature 1. All battles use the frozen behavior-cloning policy on both sides and the original 50-battle meta-opponent schedule. Policy seeds match; simulator randomness is not fixed. Invalid completions are retained without retries. Random-fill controls are team completions, not a random-move battle policy.

| Task | Mask sizes | Ling legal | Diffusion legal |
|---|---|---:|---:|
| stats | 1–6 | 9/18 | 18/18 |
| item | 1–6 | 4/18 | 18/18 |
| move | 1–24 | 13/72 | 72/72 |
| pokemon | 1–6 | 3/18 | 17/18 |
| ability | 1–6 | 3/18 | 18/18 |
| alignment | 1–6 | 18/18 | 18/18 |
| mixed | 1–48 | 4/144 | 142/144 |
