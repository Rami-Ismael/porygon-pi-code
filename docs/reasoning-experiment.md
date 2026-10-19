# Reasoning and team quality experiment

`scripts/reasoning_experiment.py` compares reasoning settings within Ling or Mercury,
using frozen prompts, starting teams, masks, and opponent pool. The default arms
are **off vs on**, with three independent generation requests per task per arm.
It tests whether requesting reasoning changes completion legality and the battle
performance of legal teams. It does not assume that reasoning improves either.

## Run

Use the Python environment with the existing validator, battle policy and runtime
dependencies described in [the baseline documentation](llm-baseline.md).
From the repository root:

```sh
# Create a small frozen baseline, or substitute an existing parent manifest.
python code/scripts/llm_baseline.py prepare \
  --output code/results/reasoning-parent --teams 3
python code/scripts/reasoning_experiment.py prepare \
  --parent code/results/reasoning-parent \
  --output code/results/reasoning-meta-pilot \
  --arms off on --replicates 3 --max-tokens 16384

# Inspect experiment.json (including planned_calls) and each example-request.json.
# Set OPENROUTER_API_KEY in your process environment before generation.
python code/scripts/reasoning_experiment.py run --output code/results/reasoning-meta-pilot
python code/scripts/reasoning_experiment.py battle --output code/results/reasoning-meta-pilot
python code/scripts/reasoning_experiment.py report --output code/results/reasoning-meta-pilot
```

`prepare` and `report` do not call the API or launch battles. `run` makes paid
requests; it does not automatically launch battles. Preparation prints the total
call count: parent tasks × arms × replicates. The example plans **378 calls**.
An existing full 4,998-task meta-starter parent would produce **29,988 calls**
with these defaults. Result manifests are local experiment artifacts and may
not exist in a fresh checkout. There is no implicit task sampling or reuse of historical
off responses. Each arm gets the same new output cap, including the off control.

Outputs are isolated by arm and repeat (`off/r0`, `on/r0`, etc.). Rerunning
`run` resumes saved responses without paying again. Uncertain pending requests
block retries for manual reconciliation, as in the baseline. `battle` resumes
through its exact-panel cache. A study lock prevents simultaneous invocations.
Changing frozen manifests, prompts or requests is rejected; create a new output
directory for changed settings. Preparation refuses nonempty destinations.

The parent is explicit: its pool, prompt version, format, masks and any excluded
self-opponents are retained. The current runner supports the repository's
historical Champions M-B format and 50-battle panels. It does **not** fetch or
claim to evaluate today's live meta. To evaluate newer teams within that format,
prepare a baseline with that pool first. A different regulation also requires
updating the underlying validator, prompt and battle runtime.

## Concurrent generation

Generation uses one asynchronous dispatcher with a bounded set of network worker
threads. It starts at **2 concurrent requests**, grows by one after 20 successful
responses, and never exceeds **4** by default. `asyncio.to_thread` keeps the
existing blocking HTTP adapter; no additional HTTP dependency is required.
Validation and file publication remain coordinated by the dispatcher.

```sh
python code/scripts/reasoning_experiment.py run \
  --output code/results/reasoning-meta-pilot \
  --concurrency 2 --max-concurrency 4 --request-interval 1 --max-attempts 6
```

These flags also work with `run_reasoning_pilot.py run`. The interval is a global
minimum between request starts, not an extra delay after every answer. Set both
concurrency limits to 1 for sequential dispatch. The scheduler reduces the limit
by half after a confirmed HTTP 429, waits globally for `Retry-After` (seconds or
HTTP date), or uses jittered exponential backoff when that header is absent.
It retries only confirmed 429 rejections, up to the configured per-request attempt
limit. Timeouts, connection failures, HTTP 408 and 5xx retain pending markers and
stop dispatch; their potentially billable outcomes require reconciliation.

Every job's frozen request hash and saved/pending state is checked before the
first request. Each pending marker is persisted before dispatch, and each raw
response is persisted before legality validation. Completed responses are reused;
saved raw responses can be revalidated without an API call. A validation or network
failure stops new work and drains remaining requests into their original files.
Malformed responses are saved as invalid rather than silently regenerated.

`generation-status.json` records concurrency, outstanding and queued requests,
session attempts, reused/completed counts, cooldown, errors and timestamps.
SIGINT/SIGTERM or a `STOP_GENERATION` file in the output directory stops new
dispatch and waits for in-flight requests to finish. Remove that file before
resuming. Avoid force-killing the runner: uncertain pending markers intentionally
block automatic resubmission. These shutdown controls apply to the new runner;
processes started before this change still execute their old code.

The seeded order still interleaves arms and repeats, and all prompts, request
hashes, output budgets and scientific settings are unchanged. Responses can
finish out of order. Parallel battle workers are independent of API concurrency.
Record a mid-study scheduling change when interpreting latency; provider capacity
and rate limits determine speedup, and concurrent requests do not train the model.

## Reasoning controls

For Mercury, prepare with `--model inception/mercury-2.5 --provider inception
--arms off low medium high`. Model and provider are explicit frozen settings;
the default remains Ling through Novita. Preflight/provider metadata and returned
usage must be checked before treating requested effort as effective reasoning.

## Completed comparison snapshot

The [Mask Lab snapshot](../../blog/mask-lab/README.md) includes full-set,
shared-cohort and pairwise results for the completed studies. Study-specific
scripts `matched_diffusion_study.py`, `audit_matched_diffusion.py` and
`mask_lab_comparison.py` use the dated local result directories named in their
source. They are not generic fresh-checkout entry points: frozen parent studies,
the Pokémon diffusion checkpoints and battle runtime must already exist.
The local diffusion comparison uses three distinct generation seeds while
retaining the original battle seed, masks, starters and 50-battle configurations.

## Provider behavior

The request uses OpenRouter's `reasoning.enabled` toggle, pins Novita with no
fallback, and requires parameter support. For a model that advertises graded
efforts, preparation also accepts `--arms off low medium high` (plus `minimal`,
`xhigh`, and `max`). Before sending graded requests, the runner checks the model
catalog's advertised `reasoning.supported_efforts` and saves those capabilities.
Absent or incompatible effort metadata causes a failure before generation.
Do not interpret Ling's on/off toggle as evidence that it supports a graded scale.

The default total output budget is 16,384 tokens for **every** arm, with a
600-second request timeout. Both are configurable at preparation. Most providers
count reasoning within the output budget, so this is a comparison under a fixed
total cap. It is not an equal-visible-answer-token or equal-cost experiment.
See [OpenRouter's reasoning documentation](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens)
and [the Ling model page](https://openrouter.ai/inclusionai/ling-3.0-flash-vl).

Requests are interleaved in seeded shuffled arm/repeat order within each task to
reduce provider time drift. Repeats are independent generation requests, not new
independent starters. Prompts do not ask for visible explanations: strict JSON
answers and the existing legality validator are unchanged. No repair calls are
made. Usage metadata helps check whether reasoning actually occurred; a requested
setting alone cannot prove provider compliance or a particular amount of thinking.

## Results and interpretation

The runner writes `report.md` and machine-readable `status.json` containing:

- Planned, returned, missing, legal and fully battled counts per arm.
- Legality over all returned attempts, including malformed/truncated responses.
- Mean win rate among legal, fully scored teams for each arm.
- Paired legality changes versus off for the same task and generation repeat.
- Paired win-rate changes restricted to pairs where both teams are legal and
  fully battled. Illegal answers are never fabricated as battle losses.
- Exploratory starter-cluster bootstrap intervals with equal starter weights;
  a single starter has no interval. Partial runs are explicitly incomplete.
- Per-task/mask-size summaries, failure reasons, finish reasons, reported cost
  and its coverage, mean request latency (including rate-limit retries), and
  reported reasoning-token counts and coverage. Missing usage is unknown.
- Per-attempt rows for further analysis and a hash identifying the opponent pool.

The original BC policy plays both sides. All settings use matching opponent
schedules and policy seeds; Showdown RNG remains unseeded. Identical generated
teams reuse the same cached battle panel, including across repeats. The report
rejects mismatched battle configurations instead of pooling incompatible results.

The paired battle subset is conditional on legality, so changes in which teams
are legal can create selection effects. Inspect legality and battle quality
together. The unpaired per-arm means may cover different subsets and are
descriptive only. Small starter counts give unstable exploratory intervals;
neither many masks nor repeated completions replace independent starters.
The visible opponent pool measures in-pool optimization, not held-out performance.

## Parallel battles

Run four isolated workers, including while generation is still running:

```sh
python code/scripts/reasoning_battles.py \
  --output code/results/reasoning-expanded-20261012 --workers 4 --watch
```

Omit `--watch` to score a single snapshot. The scorer waits for the generation
lock before writing its final report. `run_reasoning_pilot.py run` now launches
this scorer after generation, with configurable `--workers` (default 4).

Parallelism is between independent panels. Each worker retains one policy thread,
one battle at a time, the original 50-battle opponent schedule, policy seeds,
checkpoint and simulator. Each owns an isolated simulator with its own copied
code, writable config and logs. Separate ports alone do not isolate Showdown's
temporary-file renames. The copy's JavaScript hashes are checked against the
frozen source and `--skip-build` prevents rebuilding during scoring. Short `/tmp`
paths avoid macOS Unix-socket path limits. Only third-party dependencies are
shared. This changes server deployment, not battle rules or policy fingerprints.
Canonical team plus
opponent schedule identifies duplicate panels across arms and repeats. Workers
reuse the original exact-panel cache, and publishing refuses to replace a saved
result or mismatched battle configuration. Process count does not change the
scientific panel configuration or cache key. Simulator RNG remains unseeded.

Battle transport also acknowledges intentional zero Stat Points: canonicalization
collapses neutral natures to Serious, but Showdown rejects that spelling with zero
investment as a user warning. The wire paste uses the equivalent neutral Hardy
nature for those slots only. Saved teams, legality records, canonical identities
and cache keys stay unchanged. Every wire paste is validated and checked for
canonical equivalence before sending challenges, so a rejected team fails with a
specific error instead of waiting for a battle that will never start.

The scorer stores progress in `parallel-battles.json`, writes per-worker logs
under `battle-workers/`, and publishes each completed panel immediately. Restart
it to recover cached panels after interruption; it makes no model API calls.
The active expanded run began generation under the older in-memory pipeline;
its old scoring stage may exit with a missing `random_team` error. The separate
parallel scorer owns battle scoring for that run and continues independently.

## Validation

```sh
python3 -m unittest discover -s code/scripts -p 'test_reasoning_experiment.py'
python3 -m unittest discover -s code/scripts -p 'test_llm_baseline.py'
python3 -m unittest discover -s code/scripts -p 'test_openrouter_generation.py'
python3 -m unittest discover -s code/scripts -p 'test_reasoning_battles.py'
```

Tests use synthetic manifests, responses and battle panels; they require no API
key or installed simulator. A live paid/provider and simulator experiment must
still be run to obtain scientific results.
