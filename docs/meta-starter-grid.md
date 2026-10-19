# Frozen meta teams as starters

This run uses the **49 distinct teams already in the M-B meta-threat pool** as
starters, rather than the previous three random teams. The pool combines teams
from multiple events; it is not one tournament's first-through-50th standings.
Source IDs, events, and recorded placements are retained in the manifest.

The full integer grid has 102 masks per starter: six sizes each for spreads,
items, abilities, natures and whole Pokémon; 24 move sizes; and 48 mixed sizes.
That gives **4,998 Ling attempts and 4,998 diffusion attempts**, with one draw per
task. Invalid answers remain failures and are not repaired. Diffusion uses F1/G2
regulation-vocabulary v2, ask 0.5, guidance 2, temperature 1.

Each starter's exact team is removed from the opponent pastes shown to Ling and
from its battle panel. This prevents direct copying of its complete answer from
the prompt. Each original, random-fill, Ling and diffusion candidate uses the
same 50-battle schedule over the other 48 teams. Both battle sides use the frozen
behavior-cloning policy. Policy seeds match; simulator randomness is not fixed.
This remains an in-pool experiment: it is not a held-out generalization test.

Two starter teams contain unused move slots. These are preserved as empty
positions rather than silently filled. A masked move must receive a nonempty
move from Ling. Whole-slot Ling completions require four moves. Random controls
sample compatible item–ability pairs and distinct moves, then pass simulator
validation; they are team completions, never random-move battle policies.

The previous full grid averaged $0.0001157 per Ling attempt, giving approximately
**$0.58** for 4,998 calls under similar cache and output behavior. This is a
historical extrapolation, not a guaranteed provider price. Actual receipts are
saved. Up to 752,150 battle outcome slots are planned, before rejected candidates
and exact-panel cache reuse; local battle computation is much larger than the
three-starter run.

## Run, status, and recovery

```sh
~/.local/share/vgc-pilot-runtime/venv/bin/python code/scripts/meta_starter_grid.py run
~/.local/share/vgc-pilot-runtime/venv/bin/python code/scripts/meta_starter_grid.py status
```

The run command resumes saved work. If paid calls remain, it prompts for the
OpenRouter key without saving it. Uncertain sent requests stop for reconciliation
rather than being automatically charged again. Missing temporary runtime aliases
are restored from durable storage. Four disjoint Ling workers generate responses;
diffusion generates locally; six isolated battle workers score completed candidates.

Outputs are in `code/results/llm-baseline-meta-starters` and
`code/results/diffusion-baseline-meta-starters`. All raw responses, validation
outcomes, usage receipts, generation settings, masks, and completed panels are
saved. They are local gitignored files, not an off-device backup.

The live local page is [meta-starter progress](http://127.0.0.1:8767/meta.html).
It refreshes saved progress every 15 seconds and links to the results after
scoring completes. The original [random-starter dashboard](http://127.0.0.1:8767/)
remains available. Closing the browser does not stop the worker processes.

If the local page server needs restarting:

```sh
python3 -m http.server 8767 --bind 127.0.0.1 --directory code/dashboard
```
