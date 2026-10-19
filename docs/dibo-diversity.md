# Diversity methods ported to DiBO v4

The opt-in runner is `src/dibo_diversity_experiment.py`. It adds the diversity
interventions previously tried in the cross-entropy experiments as **separate
arms**, with DiBO v4 as the control. It does not edit the frozen v3/v4 comparison
sources, change `src/search.py`, or alter an active experiment.

## Method mapping

| Arm | Earlier experiment | DiBO adaptation |
| --- | --- | --- |
| `control` | Baseline | Standard v4 top-L observations, weighted prior, ensemble, RTB and target-density selection. |
| `boltz` | Entropy-regularized elite selection / annealed temperature | Prior-training weights `softmax(y/T)`, with T decreasing from .30 to .10 across the configured rounds. |
| `fixed_010`, `fixed_020`, `fixed_030` | Fixed selection temperatures | Prior-training weights at the corresponding fixed temperature. |
| `adaptive` | Adaptive ESS temperature | Solve for ESS/n=.5 with T constrained to [.10,.30]; record boundary status. |
| `niche` | Species-set entropy | Give each roster's best measured build Boltzmann weight; other builds have zero prior-training weight. Predictor still sees every buffered observation. |
| `refit` | Reverse-kernel entropy bonus | Subtract annealed .30→.10 times masked categorical Shannon entropy from the prior denoising loss, with the same 1/t weighting. |
| `mmr30`, `mmr60` | Redundancy-penalized elite selection | Choose the observation buffer greedily by measured score minus .30/.60 times maximum shared-species/6 overlap. |
| `taboo` | Composition taboo | Reject rosters accepted in any previous round; permit at most two builds of a roster in the current output. Recheck after local refinement. |
| `peel` | Winning-core peeling | Before rounds 1,3,5,… (zero-based), ban three new species most frequent in the preceding epoch's 16 highest-labelled teams. Exclude banned teams before buffer selection, mask proposal species, and reject refinement that restores them. |
| `island` | Four isolated loops with migration | Four distant meta-anchor seeds, private observation histories and models, first three sampled species columns pinned. At rounds 3,5,… each other island may send its best evaluated team sharing at most two species with the receiving island's common six. |
| `big` | Larger proposal pool | Eight times the standard v4 proposal pool; retain standard target-density selection. |
| `dpp` | Species-Gram DPP acquisition | Same larger pool as `big`; greedy DPP selects battled teams using predicted win-rate quality and species overlap. Rank exhaustion is recorded and remaining slots use target-density order. |
| `negative` | Negative-score / reversed-guidance diagnostics | Retain low-labelled buffer rows, reverse prior score weighting, and target `log_prior + beta*(-standardized_mean + gamma*std)`. This is a DiBO analogue, not the original token-gradient reversal. |
| `random` | Random-selection diagnostic | Random observation buffer, uniform prior weights, zero reward tilt and random acquisition. |
| `no_guidance` | Zero-guidance diagnostic | Keep ordinary score-weighted training but remove the posterior reward tilt. Legality penalties remain. |

DiBO has no CEM elite refit step. The temperature/niche variants therefore change
the **prior's training weights**, MMR changes the **training buffer**, and the
predictor still trains on measured win-rate targets. These are explicitly
adapted interventions, not claims of mathematical equivalence to CEM.
Temperature is in win-rate units, not log-density units. The old MMR .10 pilot
was replaced by .30/.60 before its full experiment; the runner carries those
actually evaluated settings.

The older fixes for checkpoint drift are already in v4: every round restarts
the prior from the original foundation; signatures protect checkpoint reuse.
They are retained rather than reintroduced as a different diversity method.
MAP-Elites was discussed in project notes, but is not an implemented CEM arm in
the experiment sources audited for this port.

## DiBO-specific boundaries

- Prior density, standardized UCB and their complete log target remain recorded
  for every candidate. MMR affects the buffer; DPP explicitly changes acquisition.
- Peel/island masks restrict **proposal sampling**, after RTB training. Candidate
  density and local-search acceptance are scored with the original trained
  prior. This restriction changes the proposal law; it is not an exact sampler
  for a newly conditioned posterior. Island predictors are private, whereas
  the historical CEM islands shared a ridge predictor.
- Taboo, peel and island constraints survive local refinement. The original
  proposals can fill refinement-induced collisions, subject to the same
  restrictions. Dead prefixes are rejected, never silently unmasked.
- The control uses the same model-training routines as v4. All arms now require
  the entire requested pre-refinement proposal count; shortage stops before
  spending the next battle batch instead of silently reducing the pool.
- Raw first-512 legal samples, post-filter samples and selected teams have
  separate roster counts, effective roster counts and dominant-roster shares.
  Check the recorded sample size: early severe rejection can leave fewer than
  512 raw examples. Island raw samples are interleaved across islands.
- No special early diversity stop reduces one arm's battle allocation. Structural
  constraints can exhaust the feasible pool; that is a recorded failure, not a
  reason to loosen constraints or claim equal completed budgets.

## Run

From `/Users/ramiismael/Documents/code/vgc-team-generator-pilot`, using the local
experiment environment:

```sh
/tmp/vgc-pilot/.venv/bin/python src/dibo_diversity_experiment.py plan
/tmp/vgc-pilot/.venv/bin/python src/dibo_diversity_experiment.py run --arms control taboo peel island big dpp --directory results/dibo_diversity_structural_s6060
/tmp/vgc-pilot/.venv/bin/python src/dibo_diversity_experiment.py report --directory results/dibo_diversity_structural_s6060
```

Omit `--arms` to run all 18 arms. These are independent alternatives, not a
stack of interventions enabled together. `--seed` selects a new training seed;
use a new directory for each seed/configuration. `--pool-multiplier` defaults
to 8 for `big` and `dpp`. Since v4 already proposes 12,800 teams, these arms
propose **102,400** each round and are computationally expensive.

By default each arm receives 10 × 128 × 49 = **62,720 fresh search battles**,
plus 16 × 192 × 2 = **6,144 final battles**: **68,864 per arm**, or
**1,239,552 battles for all 18 arms**. Proposal/training compute is unequal.
The original 5,000 initial observations are shared historical setup, outside
these new-battle counts. No new setup battles are launched.

The default initial data are copied from the completed original matched
comparison's frozen `protocol.json`, not from a changing shared database.
`--initial-protocol` accepts another compatible frozen protocol. Vocabulary,
policy, meta schedule, label hash and heldout separation are checked. Each arm
only adds its own new labels, even though receipts also enter the shared matrix.
Every final team is selected by search score before any arm's fresh final
evaluation. The existing fresh-evaluation adapter gives every arm identical
opponent schedules and records incomplete batches rather than retrying them
silently. A directory lock prevents concurrent writers.

`run --smoke` uses separate default output, tiny training and battle budgets.
It **does launch battles**. For a battle-free real-model check instead:

```sh
/tmp/vgc-pilot/.venv/bin/python scripts/test_dibo_diversity.py
/tmp/vgc-pilot/.venv/bin/python scripts/test_dibo_v4.py
/tmp/vgc-pilot/.venv/bin/python scripts/verify_dibo_diversity.py --directory results/dibo_diversity_proposal_check
```

The proposal verifier trains the real models with tiny settings and checks
legality, semantic uniqueness, exclusion of initial/heldout teams, and the
stored target-density identity. It performs **zero battles** and does not
establish that any intervention improves diversity or competitive strength.

Historical implementation references: `src/entropyloop.py`,
`src/boltzmann_selection.py`, `docs/redundancy-penalty.md`,
`src/diversity_loops.py`, and `docs/reverse-score-diversity.md`.

## Verification completed on October 13, 2026

- 15 new behavioral regression tests passed, including a mocked full-driver run
  across all 18 arms, finalist locking before evaluation, zero repeated search
  battles on resume, and rejection of changed frozen configuration.
- All seven existing DiBO v4 regression tests passed.
- All 18 arms passed real-foundation training and proposal smoke checks, with
  two selected legal, distinct, unseen teams per arm and verified target-density
  identities. Receipts are in
  `results/dibo_diversity_proposal_verification_20261013/verification.json`.
- These real-model checks covered round zero with tiny settings, including
  islands' actual pinned sampling. Later-round taboo history, peeling and
  migration have behavioral tests, not full multi-round battle validation.
- No new battles or full diversity campaign were launched. Competitive effects
  and long-run feasible-pool exhaustion remain to be measured.
