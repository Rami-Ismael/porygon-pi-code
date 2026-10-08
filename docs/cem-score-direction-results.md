# Reversing CEM ranking: experiment results

Completed on 2026-09-09 UTC: three seeds, five cumulative CEM updates per arm, 19,776 raw generation attempts, 103,680 battles, and 2,880 optimizer steps.

Reversed CEM ranked candidates using **−score**, while retaining the original measured scores and ordinary diffusion denoising loss. The starting checkpoint, sampling temperature, training budget, battle policy, and opponent pool were matched. A third arm selected elites uniformly at random.

The reversed arm retained more Pokémon-ID diversity than normal CEM in each of the three observed seeds. Its mean final diversity was **12.71 percentage points higher**. The random-elite control had an almost identical sample mean to the reversed arm. Both lost diversity relative to the initial generator, and reversed CEM produced a lower fresh battle win rate.

| Final generator | Pokémon-ID diversity | Distinct legal ID rosters per 256 raw attempts* | Fresh win rate |
|---|---:|---:|---:|
| Normal CEM | 66.71% | 238.0 | 9.53% |
| Random elites | 79.47% | 246.7 | 1.00% |
| Reversed CEM | 79.42% | 248.3 | 0.41% |

*Counts and rates are means across the three seeds. Final legality was 97.92%, 96.35%, and 97.01%, respectively. Distinct-roster counts therefore describe the legal portion of the fixed raw-attempt budget.

The diversity percentage is the average fraction of six Pokémon IDs that must be replaced between two legal generated teams: identical rosters score 0%, one replacement scores 16.67%, and disjoint rosters score 100%. It ignores slot order, EVs, IVs, nature, ability, moves, and items. Different form IDs remain distinct. It is not the percentage of unique teams.

Initial mean diversity was **85.12%**. After five updates, normal CEM had lost 18.41 points, random elites 5.65 points, and reversed CEM 5.70 points. Sign reversal therefore did not preserve the initial level of diversity in this experiment.

The adjusted 95% paired intervals were **[−18.37, 43.78] points** for reversed minus normal and **[−8.59, 8.48] points** for reversed minus random. These three-seed, exploratory intervals assume normally distributed seed effects and adjust for two comparisons. They are too wide to establish a reliable advantage over normal CEM or equivalence with random selection. The observed means support investigating diversity retention further; they do not establish sign reversal as a dependable diversity-preservation method.

Fresh quality evaluation used 32 uniformly selected legal holdout outputs per arm/seed, with 48 new battles each. Those scores were excluded from training. The very low absolute win rates apply to this particular fixed battle policy and opponent pool.

The training corpus contained 692 six-member teams, 570 distinct ID rosters, and 189 distinct Pokémon IDs. Its ID diversity was 84.67%. The generator remains limited by this vocabulary, but the starting model already had substantially broader ID coverage and diversity than the final normal-CEM generators. See the [corpus context](/Users/ramiismael/Documents/code/vgc-team-generator-pilot/results/cem_score_direction.corpus-context.json).

The independent audit revalidated all 19,776 proposals, explicitly checked 4,273,140 team pairs, and verified all 48 battle phases, score signs, checkpoint chains, budgets, and summary calculations. No battle phase required a retry. The eight experiment-owned servers were stopped afterward.

**Source-freeze deviation:** Three shared source files changed during the uninterrupted run. Review found that the changes affect two unimported experiment modules and an unused proposal helper; the executed sampler, selection, training, battle, and metric code are unchanged. The original controller completed every experimental phase and then failed its strict final hash check. Its original raw status and error remain preserved. The independent audit passed with this explicitly reviewed deviation, not as a clean strict source-freeze pass. [Review details](/Users/ramiismael/Documents/code/vgc-team-generator-pilot/docs/cem-source-deviation-review.md).

[Full report and trajectories](/Users/ramiismael/Documents/code/vgc-team-generator-pilot/results/cem_score_direction.md) · [Verification receipt](/Users/ramiismael/Documents/code/vgc-team-generator-pilot/results/cem_score_direction.verification.json) · [Protocol](/Users/ramiismael/Documents/code/vgc-team-generator-pilot/docs/cem-score-direction-protocol.md)
