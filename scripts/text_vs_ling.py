"""Text writer (Fast-dLLM v2 1.5B + LoRA lora_r32/final, ask 0.5, guidance 2) against the Ling masked-completion
baseline on the Mask Lab's 270 masks, using the run's existing panels (one battle against each of the 49 distinct
top-50 meta teams, behaviour-cloning policy both sides). Same statistics as mask_lab_textdiff.py report: differences
paired on the same mask, each of the five starting teams weighted equally, 95% bootstrap over the starting teams.

    python3 scripts/text_vs_ling.py [RUN_DIR]      # default: the vault's team_completion_pipeline/runs/prompt-ab
Writes text_vs_ling.json into RUN_DIR. docs/textdiff.md, "Against the LLM baseline" (2026-10-21)."""
import json, os, random, statistics, sys
R = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    "~/Documents/yakumsi-vault/Personal Project/Using Advance method search in the whole search of possible vgc pokemon "
    "format to find the right counter to a pokemon team/team_completion_pipeline/runs/prompt-ab")
SEED = 20261008
pan = json.load(open(os.path.join(R, "panels.json")))["panels"]
tasks = json.load(open(os.path.join(R, "eval2.tasks.json")))["tasks"]
E = [json.loads(l) for l in open(os.path.join(R, "eval2.jsonl"))]
secs = {}
for e in E: secs.setdefault(e["arm"], []).append(e["seconds"])
cost = {}
for e in E: cost[e["arm"]] = cost.get(e["arm"], 0) + ((e.get("usage") or {}).get("cost") or 0)
trow = json.load(open(os.path.join(R, "textdiff_rows.json")))
score = lambda lab: (pan.get(lab) or {}).get("score")
ARMS = ["regmb@16384", "baseline@16384", "baseline@4096"]
def cluster(pairs):
    by = {}
    for s, v in pairs: by.setdefault(s, []).append(v)
    if not by: return None
    means = [sum(v) / len(v) for v in by.values()]
    rng = random.Random(SEED); draws = sorted(sum(rng.choice(means) for _ in means) / len(means) for _ in range(4000))
    return dict(mean=sum(means) / len(means), lo=draws[100], hi=draws[-100], n=sum(len(v) for v in by.values()))
fmt = lambda c: "—" if not c else f"{100*c['mean']:+.1f} [{100*c['lo']:+.1f}, {100*c['hi']:+.1f}] (n={c['n']})"
fam = {}
for t in tasks:
    tid, src = t["id"], t["source"]
    r = dict(start=src, original=score(f"orig:{src}"), text=score(f"textdiff:{tid}"), diffusion=score(f"diffusion:{tid}"))
    for a in ARMS: r[a] = score(f"ling:{a}:{tid}")
    fam.setdefault(t["task"], []).append(r)
groups = list(fam.items()) + [("all", [r for v in fam.values() for r in v])]
out = {}
for family, rs in groups:
    o = {}
    o["legal"] = {k: sum(r[k] is not None for r in rs) for k in ["text", "diffusion"] + ARMS}
    o["n"] = len(rs)
    o["text_vs_original"] = cluster([(r["start"], r["text"] - r["original"]) for r in rs if r["text"] is not None])
    for a in ARMS:
        o[f"{a}_vs_original"] = cluster([(r["start"], r[a] - r["original"]) for r in rs if r[a] is not None])
        o[f"text_vs_{a}_both_legal"] = cluster([(r["start"], r["text"] - r[a]) for r in rs if r["text"] is not None and r[a] is not None])
        # net gain per mask: a rejected fill keeps the original team; paired over all masks
        net = lambda r, k: (r[k] if r[k] is not None else r["original"]) - r["original"]
        o[f"net_{a}"] = cluster([(r["start"], net(r, a)) for r in rs])
        o[f"net_text_minus_{a}"] = cluster([(r["start"], net(r, "text") - net(r, a)) for r in rs])
    o["net_text"] = cluster([(r["start"], (r["text"] if r["text"] is not None else r["original"]) - r["original"]) for r in rs])
    out[family] = o
json.dump(dict(groups=out, seconds={a: statistics.mean(v) for a, v in secs.items()}, cost_usd=cost), open(os.path.join(R, "text_vs_ling.json"), "w"), indent=1)
a = "regmb@16384"
print("seconds per mask (Ling API):", {k: round(statistics.mean(v), 1) for k, v in secs.items()}, "cost USD:", {k: round(v, 4) for k, v in cost.items()})
print(f"{'family':14s} {'legal text':>10s} {'legal Ling':>10s} {'text−orig':>28s} {'Ling−orig':>28s} {'text−Ling (both legal)':>30s} {'net text':>26s} {'net Ling':>26s} {'net text−Ling':>28s}")
for family, o in out.items():
    print(f"{family:14s} {o['legal']['text']:4d}/{o['n']:<5d} {o['legal'][a]:4d}/{o['n']:<5d} {fmt(o['text_vs_original']):>28s} {fmt(o[a + '_vs_original']):>28s} {fmt(o['text_vs_' + a + '_both_legal']):>30s} {fmt(o['net_text']):>26s} {fmt(o['net_' + a]):>26s} {fmt(o['net_text_minus_' + a]):>28s}")
print("other Ling arms, all masks: legal", {k: out['all']['legal'][k] for k in ARMS}, "| net", {k: fmt(out['all']['net_' + k]) for k in ARMS}, "| text−arm both legal", {k: fmt(out['all']['text_vs_' + k + '_both_legal']) for k in ARMS})
