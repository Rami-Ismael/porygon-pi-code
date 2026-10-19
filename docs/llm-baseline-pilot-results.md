# Ling baseline pilot — completed October 8, 2026

The 63-call masked-completion pilot is complete. **Ling did not demonstrate a
reliable advantage over random legal fills.** It produced 27 legal completions
(42.9%); the other 36 failed JSON, mask, Stat Point, or Showdown legality checks.

Only `inclusionai/ling-3.0-flash-vl` was used, through OpenRouter/Novita, with
reasoning disabled, temperature 0.7 and a 4,096-token output limit. Reported pilot
API cost was **$0.0073227**, excluding the earlier default-reasoning setup smoke.

Three random legal Champions **M-B** starting teams were tested across seven
tasks and three mask sizes per task. Both battle sides used the verified
**behaviour-cloning checkpoint `bc_100.zip`**, including learned team preview.
The SHA-256 starts `57f5edcab415cf6c`. No random-move policy was used.

| Masked feature | Legal completions / attempts |
|---|---:|
| Stat Points | 4 / 9 |
| Items | 4 / 9 |
| Moves | 3 / 9 |
| Whole Pokémon | 2 / 9 |
| Abilities | 3 / 9 |
| Stat Alignment | 9 / 9 |
| Mixed parts | 2 / 9 |

Among legal completions, weighting starting teams equally, the mean win-rate
change was **+1.17 percentage points versus the originals** and **+0.40 points
versus random legal fills**. Those conditional means exclude illegal completions;
they are not an overall success rate. The three starting-team differences against
random fills were −0.20, −1.11, and +2.50 points. Three starting teams cannot
establish a reliable improvement or support broad claims about the model.

All 93 candidate labels are complete: 3 originals, 63 random controls and 27
legal Ling outputs. Two identical candidate panels reused cached outcomes,
leaving **91 unique panels and 4,550 simulator battles**. Each panel contained
50 battles against the 49 distinct stored meta teams, with one shared repeated
opponent. Python/NumPy/Torch policy seeds were shared; the existing runner does
not control Showdown's internal random seed.

The full [generated report](../results/llm-baseline-instant-pilot/report.md) contains
task/k comparisons and validation failures. The [protocol](llm-baseline.md)
documents sampling restrictions, commands, caching and interpretation. Frozen
inputs, raw API responses, labels, request hashes, full runtime fingerprints,
and machine-readable results are under `code/results/llm-baseline-instant-pilot/`.
Complete panels are also in `~/vgc-data/matchup_regmb.sqlite`, in the exact-panel
cache and matchup records. Local result directories are Git-ignored.

Verification: four completion-contract tests passed; all 3 originals, 63 random
controls, 27 accepted completions and 49 opponents passed Showdown validation.
All unmasked values were independently checked for preservation. Every candidate
label was checked against its SQLite cache entry and 50 attributed battle outcomes.

The main follow-up is improving legality before a larger efficacy study—for
example, a separately named condition supplying legal options for each masked
field. That would change the prompt and must be a new experiment rather than a
silent repair of this baseline. No expanded run or individual-team recommendation
was made; a selected team still requires a fresh 192-battle confirmation.
