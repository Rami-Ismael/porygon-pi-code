# Discrete DiBO replaces the current CEM loop

The current implementation is `src/dibo_v3.py`, with independent mechanics in
`src/dibo_core.py`. `src/search.py` is the default search entry point. The former
`src/cem_v3.py` command and public `run/final/report` functions route to the same
implementation. Its importable legacy
helpers serve historical, explicitly pinned diversity comparisons only; they
are not called by DiBO. Existing CEM data and foundation checkpoints are preserved.

Primary method: [Yun et al., Posterior Inference with Diffusion Models for
High-dimensional Black-box Optimization, arXiv v2](https://arxiv.org/html/2502.16824v2).
The trajectory objective is inherited from
[Venkatraman et al., RTB](https://arxiv.org/abs/2405.20971).

## Method mapping

| Paper component | This implementation |
| --- | --- |
| Reweighted training, Eq. 9–11 | Every observed buffer row receives positive `softmax(y)` weight. The prior learns masked denoising without win-rate conditioning; predictors learn weighted squared error. There is no elite fraction, fixed reward cutoff, or selected-only retraining. |
| Learned diffusion prior | Trainable mixture of the two original regulation-v3 F1/G2 denoisers. Start each round from their frozen foundation state. |
| Deep ensemble UCB | Independently initialized and bootstrapped slot-invariant neural predictors over species, ability, item, each move position, and nature. `r = mean + gamma * population_std`. Disagreement estimates model uncertainty, not battle outcome variance. |
| Candidate posterior, Eq. 13 | `log_target(x) = log_prior(x) + beta * r(x)`, up to the unknown normalizer. |
| RTB, Eq. 14 | Separate posterior initialized from the trained prior. Optimize `[logZ + logp_posterior(path) - logp_prior(path) - beta*r(endpoint)]²`. Use both detached on-policy paths and observed replay grids. No gradients through the sampling operation. |
| Local search, Eq. 15–16 | Monotonic discrete coordinate ascent: propose alternative field tokens, revalidate the completed grid, and accept only an increase in the full log target. Continuous gradients/ODE density estimation do not apply to categorical fields. |
| Filtering | Rank by the full target density, not predicted mean alone. Deduplicate by semantic team identity and exclude previously evaluated grids. |
| Evaluation / moving dataset | New teams are battled against the pinned top50 matrix schedule. Add measured labels and retain a bounded top-L buffer for the next round. The buffer is the paper's moving dataset, not CEM's elite threshold. |

## The discrete probability law

The existing masked denoiser is sampled by revealing one field at a time in the
fixed `diffusion.ORDER`. The prior and posterior use identical orders, legality
masks, and categorical normalization. Every ordered terminal 48-field grid has
one such trajectory, so summing its transition log probabilities is an exact
density for **this discrete generation law**. It is not the continuous paper's
Gaussian diffusion density, a random-mask denoising score, or a likelihood lower
bound. The core tests enumerate all terminals in a small categorical space to
check normalization and the posterior target.

Dropout is disabled during RTB trajectory scoring. Gradients still update the
posterior parameters. The residual is computed once, detached, then each
transition is recomputed and its gradient accumulated. This gives the same RTB
gradient without retaining all 48 transformer graphs at once. The prior is
unchanged by RTB. The log normalizer is initialized from a prior-sample estimate
of the partition function and learned with a separate optimizer group, so a
small transformer learning rate does not prevent the scalar from adapting.

Structured masks help generate compatible fields. A dead prefix uses the same
non-mask support fallback in both models, rather than assigning inconsistent
path probabilities. The simulator validator decides endpoint legality. Invalid
or held-out endpoints receive a finite `-40` log-reward barrier during RTB and
are rejected entirely at proposal time. In the ideal sampler, conditioning on
accepted endpoints retains the desired prior-times-UCB target restricted to
legal, non-heldout teams. Finite RTB training is an approximation, and local
refinement, deduplication and filtering intentionally alter the final batch law.

The latent design is the existing 48-field grid. Stat Points are not additional
learned tokens: the renderer deterministically chooses a legal existing spread,
or a seeded legal fallback, from a hash of that grid. It never repairs nature
after density scoring. Thus the same endpoint always has the same full team
paste during sampling, reward computation, local search and battle evaluation.
Historical training labels have their original spreads; the predictor may
therefore carry additional label noise. Modeling and optimizing Stat Points
requires a further representation change, beyond replacement of the current
48-field CEM search.

## Data and evaluation boundary

Initial observations come from the current matchup database under policy 1,
using only fully covered legal rows against `top50`. Scores against other sets
and partially covered rows are not mixed in. The initial random subset is
frozen in the manifest. Generated held-out twins are rejected. Existing
foundation loaders and the matchup database keep ranks 51–100 separate.

`final` locks in the best distinct search candidates before any evaluation-set
measurement. It independently confirms those candidates against top50 and
evaluates them against ranks 51–100. Those results do not feed back into the
search or candidate selection. Search scores are noisy point estimates; no
competitive-performance or calibrated-uncertainty guarantee is claimed.

## Run and resume

Use the project virtual environment, or the existing prepared runtime:

```sh
/tmp/vgc-pilot/.venv/bin/python src/search.py run --smoke
/tmp/vgc-pilot/.venv/bin/python src/search.py final --smoke
/tmp/vgc-pilot/.venv/bin/python src/search.py report --smoke
```

For a full run omit `--smoke`. Parameters such as `--rounds`, `--beta`, `--gamma`,
`--rtb-steps`, `--local-steps`, `--battle`, and `--output` are configurable.
The smoke run has two rounds, 16 initial labels, eight legal proposals per
round, two selected teams, and two RTB updates. It is an integration check,
not sufficient posterior training or a performance comparison.

Runtime dependencies are the project's regulation-v3 checkpoints, vocabulary,
legality tables, seeded top50 matchup database, pinned simulator and BC policy,
and battle server stack. It does not download or silently substitute models.
The existing `/tmp/vgc-pilot` fallback is used for legacy imports, as in the
other pilot scripts. Pure mathematical tests require only torch.

The prepared local server can be started in another terminal (its saved
configuration binds to `127.0.0.1`):

```sh
cd ~/.local/share/vgc-pilot-runtime/servers/8134
~/.nvm/versions/node/v22.22.0/bin/node pokemon-showdown start 8123 --no-security
```

The default manifest is `results/dibo_v3.json` (smoke:
`results/dibo_v3_smoke.json`). Checkpoints go into a separate hashed run directory
under `~/.local/share/vgc-pilot-runtime/lossdown/dibo_*`. Manifests and finished
round checkpoints are written atomically. Resume checks the configuration,
foundation and implementation hashes, evaluator-policy identity, vocabulary,
regulation and top50 schedule. It
refuses to reuse checkpoints under changed observations. Saved selected batches
are evaluated through the matrix's missing-cell mechanism on restart. An
interrupted, unfinished model-training stage restarts that stage; a completed
model stage is cached. Use a different output path for a changed configuration.

Training is more expensive than the former elite loop: full trajectory
evaluation requires 48 normalized transitions. The number of objective
evaluations and model-training time are separate budgets.

## Verification

```sh
/tmp/vgc-pilot/.venv/bin/python scripts/test_dibo.py
/tmp/vgc-pilot/.venv/bin/python scripts/test_heldout_separation.py
```

The finite-space tests verify trajectory density, sampling frequencies,
prior-times-reward convergence, zero-reward behavior, positive weights for
low-scoring observations, ensemble disagreement, and posterior-density selection.
Real integration results are recorded in the smoke manifest. Completing that
check establishes plumbing, not an improvement over the old CEM baseline.

The final-version integration receipt and its preserved result path are in
[dibo-verification.md](dibo-verification.md). The two-round check and independent
confirmation/heldout stages completed; all seven core tests and fifteen existing
heldout-separation tests passed.
