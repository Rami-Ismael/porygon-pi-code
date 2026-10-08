# Reversed score guidance: first experiment

Reversing the existing predicted-score guidance produced slightly more distinct
legal Pokémon combinations in this run. Diversity was already high with no
guidance, so the practical increase was small. Battle strength was not tested.

The experiment completed 4,608 attempts: three directions, three matched
generation seeds, and exactly 512 raw attempts per direction and seed. It used
one frozen checkpoint and surrogate, guidance magnitudes +108 / 0 / −108, and a
fixed decoding temperature of 1.0. There was no retraining. Invalid and repeated
outputs were retained, with no replacement or score-based filtering.

## Results

Values are means across the three seeds, per 512 raw attempts. A combination is
the unordered six-species roster; moves, items, and stat spreads do not create
additional combinations.

| Direction | Distinct legal combinations | Novel legal combinations | Legal attempts | Distinct in 128 legal draws |
|---|---:|---:|---:|---:|
| Normal (+108) | 442.0 | 434.3 | 88.35% | 126.79 |
| No guidance (0) | 448.0 | 443.0 | 88.41% | 127.57 |
| Reversed (−108) | 457.3 | 454.3 | 89.58% | 127.88 |

“Novel” means absent from the checkpoint's training corpus. The last column is
the exact expected distinct count in a sample of 128 legal outputs without
replacement, which controls for different legal-output counts.

Reversal yielded **15.3 more unique legal combinations than normal guidance**
(3.5%) and **9.3 more than no guidance** (2.1%). All three seeds favored reversal
on this endpoint. The adjusted paired intervals were −12.0 to +42.7 against
normal guidance and +7.3 to +11.4 against no guidance. These exploratory intervals
use a t model with only three seed pairs and Bonferroni adjustment for the two
comparisons; the normal-guidance comparison remains inconclusive.

Some of the increase came from more legal outputs. After equalizing the number
of legal draws, reversal added only 0.31 expected distinct combinations over no
guidance per 128 draws, close to the maximum of 128 for both methods. Mean
distance to the nearest training-corpus combination also increased from 0.508
without guidance to 0.535 with reversal (Jaccard distance, descriptive).

The mean legal-output surrogate score fell from 0.396 with normal guidance to
0.228 without guidance and −0.082 with reversal. These are unbounded ridge-model
predictions, not measured battle win rates. The experiment shows a small diversity
increase for this checkpoint and guidance magnitude; it does not establish a
useful diversity-versus-battle-strength tradeoff or a general advantage across
models.

## Verification and reproduction

The eight implementation tests and the separate 36-attempt plumbing run passed.
Real-model controls confirmed that normal and zero guidance reproduce the
original sampler and that reversal negates the nonzero guidance contribution.
An independent audit revalidated all 4,608 proposals, recomputed diversity and
paired intervals, and checked 1,268 input-file fingerprints. It found 4,091 legal
attempts and passed.

- [Protocol and run commands](reverse-score-diversity.md)
- [Generated report, individual seeds, and plot](../results/reverse_score_diversity.md)
- [All proposals and validation outcomes](../results/reverse_score_diversity.json)
- [Independent verification receipt](../results/reverse_score_diversity.verification.json)
- [Source snapshot manifest](../results/reverse_score_diversity_source/manifest.json)

The implementation leaves the existing sampler intact and adds a signed-guidance
adapter. Running the full command again validates the fingerprints and reuses
completed cells; a different configuration must use a separate output path.

## Additional Pokemon-ID-only metric

The [Pokemon-ID diversity report](../results/reverse_score_diversity.pokemon-id-diversity.md)
adds average roster-ID replacement distance and ID-usage diversity, using only
the six species/form IDs. These are descriptive measurements added after the
original experiment. Mean legal-output ID diversity was 82.39% for normal
guidance, 85.53% without guidance, and 86.29% with reversal. See the
[metric verification receipt](../results/reverse_score_diversity.pokemon-id-diversity.verification.json)
for ten passing tests and an independent check of all 2,105,329 raw/legal team
pairs. The original proposals and experimental inputs were preserved.
