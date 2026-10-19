# Full DiBO versus CEM comparison — October 12, 2026

Run: `src/dibo_comparison.py run --directory results/dibo_vs_cem_full_20261012`.
The full protocol, starting labels and all fingerprints are frozen in that
directory's `protocol.json` before either arm starts.

Each arm receives ten rounds, 512 valid distinct proposals and 128 new distinct
selected teams per round. Every selected team plays one **fresh** battle against
each of the 49 distinct top50 columns: **62,720 search battles per arm**. The
duplicated top50 paste retains its weight of two in objective aggregation.

Each arm then locks its sixteen highest-search-score distinct finalists. Each
plays 192 fresh battles for top50 confirmation and 192 fresh battles against
held-out ranks 51–100: **3,072 + 3,072 final battles per arm**, or **68,864 total
per arm**. Both arms use identical, seeded opponent schedules in final stages.
Heldout results never feed back into proposal generation or finalist selection.

Before any final evaluation, final input serialization was amended to use the
same canonical team/move order as search battles. The original protocol and
driver are retained beside the run, with the correction documented in
`canonical-final-amendment.md`. This does not change the chosen-team criterion
or measured search results.

The primary descriptive comparison is the heldout score of each arm's first
search-selected finalist. Secondary measures are the finalist-set means in
fresh top50 and heldout evaluation, with a bootstrap interval clustered by team.
Per-round means, best scores, diversity and compute time are also reported.
One training seed (4040) establishes a matched single-run comparison; it does not
estimate variability across independent training runs.

Both arms use the same original v3 F1/G2 foundation, legality validator, BC
policy, frozen 5,000 initial observations, and exclusion of seen or heldout
grids. All 49 meta anchors are included in the common initial data. Missing
anchor-versus-anchor cells are completed once as shared setup and recorded
separately in `shared_setup.json`; this expenditure benefits both methods equally
and is outside each method's search allocation. Labels are
from complete-coverage rows in the starting matchup matrix. Neither arm sees
the other arm's new labels, even though real outcomes are saved to the shared
matrix for future research.

CEM uses the original fixed 0.531 elite cut (minimum 64), conditional generator
fine-tuning from the original foundation, ridge prediction, classifier-free
guidance and ridge guidance, nature repair, and mean-score acquisition. Its
ridge uses the shared initial labels and weights new loop labels by ten. This
replaces its old extra private 44,887-label initialization to equalize starting
observations. Item-less slots are parsed separately so valid teams are retained.

DiBO uses the full default settings: forty prior-training epochs, fifty
predictor epochs, five predictors, one hundred RTB updates, beta ten, gamma one,
two local-refinement steps, and posterior-density filtering. Local refinement
is batched for execution speed; it still accepts only legal coordinate changes
that increase the full log target.

The historical CEM run stopped at three rounds on diversity collapse. This
fixed-budget repeat records that guardrail but runs both methods to the same
ten-round budget. The three-round checkpoint provides context. Historical
figures are not substituted for the fresh matched comparison.

Raw fresh-battle receipts are saved in sixteen-candidate chunks. Cached scores
do not count toward this budget and never supply new labels. Incomplete or
unattributed batches stop the run rather than silently retrying a complete
budget. Completed proposal/model stages and completed battle receipts resume.
The driver takes an exclusive run lock to prevent simultaneous writers.

The `--smoke` variant is a one-round plumbing check with reduced model-training
and battle counts. It is stored separately and never serves as the full result.
