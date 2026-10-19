# Regulation-grounded Ling prompt comparison

## Completed generation result — October 8, 2026

Both prompts returned **65/147 legal completions (44.2%)**; diffusion v3 returned
147/147 legal completions on these same tasks. There is **no aggregate legality
improvement** in this pilot. Revised-prompt calls cost $0.02723501 versus $0.01983257
for the original calls on these tasks. The original prompt remains the baseline.

| Mask family | Original Ling | Revised Ling | Diffusion v3 |
|---|---:|---:|---:|
| Stat Points | 5/21 | 14/21 | 21/21 |
| Items | 11/21 | 8/21 | 21/21 |
| Moves | 9/21 | 3/21 | 21/21 |
| Whole Pokémon | 0/21 | 0/21 | 21/21 |
| Abilities | 16/21 | 17/21 | 21/21 |
| Natures | 19/21 | 20/21 | 21/21 |
| Mixed | 5/21 | 3/21 | 21/21 |

Battle comparisons are queued behind the main meta-starter run. These generation
results are not battle-win results, and this document does not claim an improvement
in battle performance. The live chart adds paired battle scores when available.

The original prompt already said “Pokemon Champions Regulation M-B” and supplied
`gen9championsvgc2026regmb`. Merely repeating the regulation name is not a tested
fix. The revised `mb-grounded-v2` prompt adds:

- A prominent frozen M-B ruleset declaration.
- Explicit Stat Point limits, species/item clauses, and empty-slot instructions.
- Simulator-derived choices for masked moves, abilities, items and natures on
  known species; a legal species list for whole-slot masks.
- Strict JSON instructions, including no comments, ellipses or placeholders.

Hints are constructed only from the masked context. They never inspect masked
original values. This is a combined rules/context/formatting intervention, not an
isolated test of the regulation name.

Each task has one stochastic response per prompt, so sampling variation is also
present. This pilot can identify promising changes, not prove that wording alone
caused a performance difference.

The paired pilot freezes 147 existing tasks: seven starters selected with seed
20261011, seven families, and minimum/middle/maximum mask sizes. Starter numbers
are 3, 4, 18, 22, 23, 32 and 49. Selection occurred before revised outputs were
observed. Original masks, random controls, opponent exclusions, model, temperature
and output budget are unchanged. Every revised response is saved separately in
`code/results/ling-mb-prompt-comparison`; original results are retained.

The [comparison chart](http://127.0.0.1:8767/prompt-comparison.html) shows:

1. Full-cohort legality for the completed 4,998 original Ling and diffusion outputs.
2. Original Ling, revised Ling and diffusion legality on the same completed pilot
   tasks, with exact numerators and denominators.
3. Per-family legality and actual paired API costs.
4. Conditional battle win rates only once all three methods have scored panels
   for the same tasks, with starters weighted equally.

Legality is not battle performance. Revised-prompt battle scoring is queued after
the current main experiment to avoid duplicate cache writes. Until then, the
battle chart explicitly says pending rather than inventing scores. Seven starter
clusters are an exploratory test and do not establish general superiority.

The static chart refreshes saved JSON every 15 seconds; it sends no model requests.
Resume the pilot from the repository directory with:

```sh
~/.local/share/vgc-pilot-runtime/venv/bin/python code/scripts/prompt_comparison.py run
```

It skips saved responses and requests the API key only when additional calls are
needed. Existing uncertain-request protection applies. Test coverage verifies
hidden-value independence, unchanged original request hashes, exact task reuse,
and exclusion of each starter's own paste.
