"""The text team writer against the LLM baseline on the Mask Lab's 270 masks, from the run's stored battle results.

The text team writer is Fast-dLLM v2 1.5B with the LoRA adapter lora_r32/final, prompted with a target win rate of 0.5
and classifier-free guidance weight 2 (the project's selected decoding profile). The LLM baseline is a large language model,
Ling (inclusionai/ling-3.0-flash-vl), filling the same masks through OpenRouter under three prompt arms. The Mask Lab is the project's completion test: five
random legal starting teams with 270 masked fields (Stat Points, items, moves, whole Pokémon, abilities, natures),
each filled once per method. A fill is legal when the completed team keeps every unmasked field and passes Showdown's
Regulation M-B validator. Its panels file holds every legal filled team's battle results: one battle against each of
the 49 distinct teams among the top-50 meta teams (the 50 best-placed teams of the collected meta), with the
behaviour-cloning policy on both sides. Classifier-free guidance mixes the model's prediction with and without the
target win rate in the prompt; weight 2 leans twice as hard toward the prompted one.

Statistics are those of mask_lab_textdiff.py report: differences paired on the same mask, each starting team weighted
equally, 95% bootstrap over the five starting teams. "Net" keeps the original team when a fill is rejected.

    python3 scripts/text_vs_ling.py [RUN_DIR]      # default: the vault's team_completion_pipeline/runs/prompt-ab
Writes text_vs_ling.json into RUN_DIR. docs/textdiff.md, "Against the LLM baseline" (2026-10-21)."""
import json, os, random, statistics, sys

run_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    "~/Documents/yakumsi-vault/Personal Project/Using Advance method search in the whole search of possible vgc pokemon "
    "format to find the right counter to a pokemon team/team_completion_pipeline/runs/prompt-ab")
SEED = 20261008
LOW, HIGH = 100, -100                    # 2.5% and 97.5% of the 4,000 bootstrap draws
LING_ARMS = ["regmb@16384", "baseline@16384", "baseline@4096"]     # prompt @ output-token limit
BEST_ARM = "regmb@16384"                                           # the regulation-grounded prompt

panels = json.load(open(os.path.join(run_dir, "panels.json")))["panels"]
masks = json.load(open(os.path.join(run_dir, "eval2.tasks.json")))["tasks"]
ling_calls = [json.loads(line) for line in open(os.path.join(run_dir, "eval2.jsonl"))]
seconds_by_arm, cost_by_arm = {}, {}
for call in ling_calls:
    seconds_by_arm.setdefault(call["arm"], []).append(call["seconds"])
    cost_by_arm[call["arm"]] = cost_by_arm.get(call["arm"], 0) + ((call.get("usage") or {}).get("cost") or 0)
win_rate = lambda label: (panels.get(label) or {}).get("score")      # None when the fill was rejected


def start_weighted_bootstrap(pairs):
    """Mean of (starting team, value) pairs with each starting team weighted equally, and a 95% bootstrap over the
    starting teams (the same draws, seed and interval bounds as mask_lab_textdiff.py report)."""
    by_start = {}
    for start, value in pairs: by_start.setdefault(start, []).append(value)
    if not by_start: return None
    means = [sum(v) / len(v) for v in by_start.values()]
    rng = random.Random(SEED); draws = sorted(sum(rng.choice(means) for _ in means) / len(means) for _ in range(4000))
    return dict(mean=sum(means) / len(means), lo=draws[LOW], hi=draws[HIGH], n=sum(len(v) for v in by_start.values()))


def paired(rows, value):
    """Bootstrap of value(row) over the rows where it is defined (both fills legal)."""
    return start_weighted_bootstrap([(r["start"], value(r)) for r in rows if value(r) is not None])


def net(row, method):
    """Win-rate change of a mask's fill over the original team; a rejected fill keeps the original (change 0)."""
    return (row[method] if row[method] is not None else row["original"]) - row["original"]


difference_of = lambda a, b: (lambda r: None if r[a] is None or r[b] is None else r[a] - r[b])   # per-row getter
rows_by_mask_type = {}
for mask in masks:
    row = dict(start=mask["source"], original=win_rate(f"orig:{mask['source']}"), text=win_rate(f"textdiff:{mask['id']}"))
    for arm in LING_ARMS: row[arm] = win_rate(f"ling:{arm}:{mask['id']}")
    rows_by_mask_type.setdefault(mask["task"], []).append(row)
groups = list(rows_by_mask_type.items()) + [("all", [r for v in rows_by_mask_type.values() for r in v])]

results = {}
for mask_type, rows in groups:
    stats = dict(n=len(rows), legal={k: sum(r[k] is not None for r in rows) for k in ["text"] + LING_ARMS},
                 text_vs_original=paired(rows, difference_of("text", "original")),
                 net_text=start_weighted_bootstrap([(r["start"], net(r, "text")) for r in rows]))
    for arm in LING_ARMS:
        stats[f"{arm}_vs_original"] = paired(rows, difference_of(arm, "original"))
        stats[f"text_vs_{arm}_both_legal"] = paired(rows, difference_of("text", arm))
        stats[f"net_{arm}"] = start_weighted_bootstrap([(r["start"], net(r, arm)) for r in rows])
        stats[f"net_text_minus_{arm}"] = start_weighted_bootstrap([(r["start"], net(r, "text") - net(r, arm)) for r in rows])
    results[mask_type] = stats
json.dump(dict(groups=results, seconds={a: statistics.mean(v) for a, v in seconds_by_arm.items()}, cost_usd=cost_by_arm),
          open(os.path.join(run_dir, "text_vs_ling.json"), "w"), indent=1)

format_interval = lambda interval: "—" if not interval else (f"{100*interval['mean']:+.1f} [{100*interval['lo']:+.1f}, "
                                                              f"{100*interval['hi']:+.1f}] (n={interval['n']})")
print("seconds per mask (LLM baseline, through the API):", {k: round(statistics.mean(v), 1) for k, v in seconds_by_arm.items()},
      "cost USD:", {k: round(v, 4) for k, v in cost_by_arm.items()})
columns = ["legal, text team writer", "legal, LLM baseline", "text team writer − original", "LLM baseline − original",
           "text team writer − LLM baseline, both legal", "net, text team writer", "net, LLM baseline",
           "net, text team writer − LLM baseline"]
print("LLM baseline = its regulation-grounded prompt with a 16,384-token limit (" + BEST_ARM + ")")
print(f"{'mask type':14s} " + " | ".join(columns))
for mask_type, stats in results.items():
    cells = [f"{stats['legal']['text']}/{stats['n']}", f"{stats['legal'][BEST_ARM]}/{stats['n']}",
             format_interval(stats["text_vs_original"]), format_interval(stats[f"{BEST_ARM}_vs_original"]),
             format_interval(stats[f"text_vs_{BEST_ARM}_both_legal"]), format_interval(stats["net_text"]),
             format_interval(stats[f"net_{BEST_ARM}"]), format_interval(stats[f"net_text_minus_{BEST_ARM}"])]
    print(f"{mask_type:14s} " + " | ".join(cells))
print("the LLM baseline's other prompt arms, all masks: legal", {k: results["all"]["legal"][k] for k in LING_ARMS},
      "| net", {arm: format_interval(results["all"][f"net_{arm}"]) for arm in LING_ARMS},
      "| text team writer − arm, both legal", {arm: format_interval(results["all"][f"text_vs_{arm}_both_legal"]) for arm in LING_ARMS})
