# Prompt changelog

Every prompt sent to `inclusionai/ling-3.0-flash-vl` for the masked-completion
baseline, oldest first. The exact system text of each version is saved in
`code/prompts/<version>.system.txt`; `test_prompt_versions.py` fails if a saved
version is edited, so a change is always a new row here.

| Version | Date | Parent | System text | SHA-256 (first 12) | What changed | Tested in | Result |
|---|---|---|---:|---|---|---|---|
| `mb-original-v1` | 2026-10-08 | none | 841 chars | `73c291103c20` | First prompt. Names Regulation M-B, the Stat Point limits, both clauses and the answer format. Gives no list of what is legal. | [llm-baseline.md](llm-baseline.md) | 65 of 147 legal on the paired pilot tasks |
| `mb-grounded-v2` | 2026-10-08 | v1 | 1,524 chars | `f09c7b03bb70` | Instructions rewritten, and legal moves, ability/item pairs and natures added, but only for the masked fields. | [ling-mb-prompt-comparison.md](ling-mb-prompt-comparison.md) | 65 of 147 legal, no change; battles never scored |
| `mb-regulation-v3` | 2026-10-10 | v1 | 8,799 chars | `8f03bd905817` | v1 word for word, then the Regulation M-B reference. Nothing else changes. | `prompt_regulation_test.py`, 2026-10-10 | 170 of 294 legal against 139 of 294 for v1 sampled at the same time: +10.5 points [+6.8, +14.3]. Win rate of legal teams unchanged (0.498 against 0.496). |

The v1 hash is for the 147-task pilot, where the manifest sets
`preserve_empty_move_slots`; the first three-team pilot used the same text
without its last two sentences.

## Which regulation

The project is pinned to **Regulation Set M-B**: the simulator format is
`gen9championsvgc2026regmb`, and the meta pool, matchup database and trained
models are all M-B. The official
[Regulation Set M-B notice](https://champions-news.pokemon-home.com/en/page/776.html)
dates it 17 June to 9 September 2026.
[Regulation Set M-C](https://champions-news.pokemon-home.com/en/page/816.html)
is the live season from 9 September to 2 December 2026, and the pinned simulator
has no M-C format. So the prompt is given M-B, and it says in words that M-C is
not the target.

## What `mb-regulation-v3` adds

One paragraph appended to the v1 system message, the same for every task:

- From the official notice: the season dates, Mega Evolution needs the Mega
  Stone and happens once per battle, duplicate held items are not allowed.
- From the pinned simulator: Double Battle, six Pokémon with four brought,
  level 50, open team sheets, Species Clause, no Mythical or restricted
  Legendary Pokémon, Stat Points 0 to 32 each and 66 in total, no
  Terastallization.
- From `code/data/regmb-vocabulary.json`: all 347 species names, 148 items and
  25 natures the validator accepts.

It does not add legal moves or ability/item pairs per Pokémon, and it does not
touch the answer-format instructions. `mb-grounded-v2` changed both at once,
which is why its result could not be read as the effect of regulation
information.

## The test

`code/scripts/prompt_regulation_test.py` compares `mb-regulation-v3` with
`mb-original-v1` on the frozen 147 tasks of the 8 October pilot: seven meta
starting teams (the test refuses fewer than six), seven mask families, three
mask sizes. Both prompts are sampled fresh, twice each, so 588 calls. The
model, temperature 0.7, 4,096-token cap, reasoning off and the Novita route are
unchanged, and the v1 request bytes are identical to the 8 October run.

```sh
PY=~/.local/share/vgc-pilot-runtime/venv/bin/python
$PY code/scripts/prompt_regulation_test.py prepare              # no calls
$PY code/scripts/prompt_regulation_test.py run                  # asks for the OpenRouter key
$PY code/scripts/prompt_regulation_test.py battle --workers 3   # 50 battles per legal team
$PY code/scripts/prompt_regulation_test.py report
```

`prepare` needs the frozen manifest (pass `--parent` to its folder).

## Result, 10 October 2026

**The regulation reference makes more answers legal. It does not make the legal
answers stronger.**

| Prompt | Responses | Legal | Share | Cost (USD) |
|---|---:|---:|---:|---:|
| `mb-original-v1` | 294 | 139 | 47.3% | 0.036 |
| `mb-regulation-v3` | 294 | 170 | 57.8% | 0.042 |

Legal share, v3 minus v1, each starting team weighted equally: **+10.5 points**
[+6.8, +14.3] (95% bootstrap over the seven starting teams). v3 has more legal
answers than v1 for every one of the seven starting teams.

| Mask family | Tasks | v1 legal per repeat | v3 legal per repeat |
|---|---:|---:|---:|
| Stat Points | 21 | 5.5 | 14.5 |
| Items | 21 | 11.0 | 11.5 |
| Moves | 21 | 9.5 | 9.5 |
| Whole Pokémon | 21 | 2.0 | 5.5 |
| Abilities | 21 | 16.0 | 17.0 |
| Natures | 21 | 19.5 | 20.0 |
| Mixed | 21 | 6.0 | 7.0 |

The gain is almost all Stat Points and whole Pokémon. Items and moves do not
move: the reference lists which items exist, not which moves a Pokémon learns,
and the model still repeats an item or a move already on the team.

| Reason for rejection | v1 | v3 |
|---|---:|---:|
| Stat Points out of range | 69 | 45 |
| Item Clause | 22 | 25 |
| Repeated or empty move | 20 | 23 |
| Wrong answer shape | 14 | 10 |
| Unparseable JSON | 11 | 10 |
| Ability not available | 8 | 6 |
| Move not learnable | 5 | 4 |
| Name does not exist | 6 | 1 |

**Battles.** Win rate is wins divided by 50 battles against the 48 other meta
teams shown in the prompt (two opponents repeat to reach 50), with the
behaviour-cloning policy playing both sides. All 309 legal completions were battled. Their mean win rate is 0.496 for
v1 (139 teams) and 0.498 for v3 (170 teams). On the 56 tasks where all four
responses were legal, v3 minus v1 is +1.3 points with a standard error of 0.7
over tasks; 33 of those tasks show no difference at all, mostly because both
prompts returned the same team. That is within noise.

Limits: one model, seven starting teams, two samples per task. The win-rate
comparison only covers answers that were legal, so it is not a randomised
comparison. The meta pool is in the prompt, so this is not a held-out test.
Three requests were cut off by a network timeout mid-run and re-sent; their
markers are kept in each run folder under `interrupted/`. v1 scored 65 of 147
on 8 October and 71 and 68 here, so run-to-run variation is a few answers.

## Adding a version

1. Add it to `VERSIONS` in `code/scripts/prompt_versions.py` and to the
   dispatch in `llm_baseline.prompt`.
2. Run `prompt_versions.py snapshot`, then `prompt_versions.py list` for the hash.
3. Add a row to the table above.
