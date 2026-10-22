"""The text writer against the Ling LLM baseline on the Mask Lab's 270 masks, from the run's stored battle results.

The text writer is Fast-dLLM v2 1.5B with the LoRA adapter lora_r32/final, prompted with a target win rate of 0.5
and classifier-free guidance weight 2 (the project's selected decoding profile). Ling is inclusionai/ling-3.0-flash-vl
filling the same masks through OpenRouter under three prompt arms. The Mask Lab is the project's completion test: five
random legal starting teams with 270 masked fields (Stat Points, items, moves, whole Pokémon, abilities, natures),
each filled once per method. Its panels file holds every legal filled team's battle results: one battle against each
of the 49 distinct top-50 meta teams, behaviour-cloning policy on both sides.

Statistics are those of mask_lab_textdiff.py report: differences paired on the same mask, each starting team weighted
equally, 95% bootstrap over the five starting teams. "Net" keeps the original team when a fill is rejected.

    python3 scripts/text_vs_ling.py [RUN_DIR]      # default: the vault's team_completion_pipeline/runs/prompt-ab
Writes text_vs_ling.json into RUN_DIR. docs/textdiff.md, "Against the LLM baseline" (2026-10-21)."""
import json, os, random, statistics, sys

run_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    "~/Documents/yakumsi-vault/Personal Project/Using Advance method search in the whole search of possible vgc pokemon "
    "format to find the right counter to a pokemon team/team_completion_pipeline/runs/prompt-ab")
SEED = 20261008
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
    starting teams (the same draws, seed and cut as mask_lab_textdiff.py report)."""
    by_start = {}
    for start, value in pairs: by_start.setdefault(start, []).append(value)
    if not by_start: return None
    means = [sum(v) / len(v) for v in by_start.values()]
    rng = random.Random(SEED); draws = sorted(sum(rng.choice(means) for _ in means) / len(means) for _ in range(4000))
    return dict(mean=sum(means) / len(means), lo=draws[100], hi=draws[-100], n=sum(len(v) for v in by_start.values()))


def paired(rows, value):
    """Bootstrap of value(row) over the rows where it is defined (both fills legal)."""
    return start_weighted_bootstrap([(r["start"], value(r)) for r in rows if value(r) is not None])


def net(row, method):
    """Win-rate change of a mask's fill over the original team; a rejected fill keeps the original (change 0)."""
    return (row[method] if row[method] is not None else row["original"]) - row["original"]


diff = lambda a, b: (lambda r: None if r[a] is None or r[b] is None else r[a] - r[b])
rows_by_mask_type = {}
for mask in masks:
    row = dict(start=mask["source"], original=win_rate(f"orig:{mask['source']}"), text=win_rate(f"textdiff:{mask['id']}"))
    for arm in LING_ARMS: row[arm] = win_rate(f"ling:{arm}:{mask['id']}")
    rows_by_mask_type.setdefault(mask["task"], []).append(row)
groups = list(rows_by_mask_type.items()) + [("all", [r for v in rows_by_mask_type.values() for r in v])]

results = {}
for mask_type, rows in groups:
    stats = dict(n=len(rows), legal={k: sum(r[k] is not None for r in rows) for k in ["text"] + LING_ARMS},
                 text_vs_original=paired(rows, diff("text", "original")),
                 net_text=start_weighted_bootstrap([(r["start"], net(r, "text")) for r in rows]))
    for arm in LING_ARMS:
        stats[f"{arm}_vs_original"] = paired(rows, diff(arm, "original"))
        stats[f"text_vs_{arm}_both_legal"] = paired(rows, diff("text", arm))
        stats[f"net_{arm}"] = start_weighted_bootstrap([(r["start"], net(r, arm)) for r in rows])
        stats[f"net_text_minus_{arm}"] = start_weighted_bootstrap([(r["start"], net(r, "text") - net(r, arm)) for r in rows])
    results[mask_type] = stats
json.dump(dict(groups=results, seconds={a: statistics.mean(v) for a, v in seconds_by_arm.items()}, cost_usd=cost_by_arm),
          open(os.path.join(run_dir, "text_vs_ling.json"), "w"), indent=1)

fmt = lambda c: "—" if not c else f"{100*c['mean']:+.1f} [{100*c['lo']:+.1f}, {100*c['hi']:+.1f}] (n={c['n']})"
print("seconds per mask (Ling API):", {k: round(statistics.mean(v), 1) for k, v in seconds_by_arm.items()},
      "cost USD:", {k: round(v, 4) for k, v in cost_by_arm.items()})
columns = ["legal, text writer", f"legal, {BEST_ARM}", "text writer − original", f"{BEST_ARM} − original",
           "text writer − Ling, both legal", "net, text writer", f"net, {BEST_ARM}", "net, text writer − Ling"]
print(f"{'mask type':14s} " + " | ".join(columns))
for mask_type, s in results.items():
    cells = [f"{s['legal']['text']}/{s['n']}", f"{s['legal'][BEST_ARM]}/{s['n']}", fmt(s["text_vs_original"]),
             fmt(s[f"{BEST_ARM}_vs_original"]), fmt(s[f"text_vs_{BEST_ARM}_both_legal"]), fmt(s["net_text"]),
             fmt(s[f"net_{BEST_ARM}"]), fmt(s[f"net_text_minus_{BEST_ARM}"])]
    print(f"{mask_type:14s} " + " | ".join(cells))
print("other Ling arms, all masks: legal", {k: results["all"]["legal"][k] for k in LING_ARMS},
      "| net", {k: fmt(results["all"][f"net_{k}"]) for k in LING_ARMS},
      "| text writer − arm, both legal", {k: fmt(results["all"][f"text_vs_{k}_both_legal"]) for k in LING_ARMS})
