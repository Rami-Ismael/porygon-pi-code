"""The Mask Lab baseline scored on the held-out teams (placement ranks 51-100) instead of the top-50.

A near-exact copy of the 2026-10-15 Mask Lab battles (the vault's team_completion_pipeline/mask_lab_battle.py) with
one change: the opponents. Kept: the five random legal starting teams, the 270 masks and every arm's fills exactly
as saved in runs/prompt-ab/candidates.json (untouched originals, random legal fills, Ling's three prompt arms, the
scratch diffusion decoder, the fine-tuned text writer lora_r32/final); one battle per (fill, opponent); the
behaviour-cloning policy on both sides; listed slot order; seed 20261008; the 2026-10-15 fast evaluator, whose runner
file is loaded from that run's evidence copy so the battle configuration identity is the recorded one (4208bc07...).
Changed: the opponents are the 50 held-out teams (results/heldout_next50_evs.json), battled only through
matchup_db.evaluate_final, the held-out set's permitted use, into a database of their own
(~/vgc-data/mask_lab_heldout.sqlite) that no training or search code reads. A fill that is a held-out team (same
48 fields) is not battled. Statistics as in scripts/text_vs_ling.py. docs/textdiff.md, "On the held-out teams".

    /tmp/vgc-pilot/.venv/bin/python scripts/mask_lab_heldout.py battle --limit 6   # smoke test
    /tmp/vgc-pilot/.venv/bin/python scripts/mask_lab_heldout.py battle             # RUN/panels_heldout.json
    python3 scripts/mask_lab_heldout.py report                                     # RUN/heldout_report.json
    python3 scripts/mask_lab_heldout.py report --panels panels.json                # the same tables on the top-50
"""
import argparse, importlib.util, json, random, sys, time, uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PIPELINE = Path.home() / ("Documents/yakumsi-vault/Personal Project/Using Advance method search in the whole search of "
                          "possible vgc pokemon format to find the right counter to a pokemon team/team_completion_pipeline")
RUN = PIPELINE / "runs/prompt-ab"
DATABASE = Path.home() / "vgc-data/mask_lab_heldout.sqlite"
EVIDENCE_ROOT = Path.home() / "vgc-data/battle-evaluations"
TOP50_EVIDENCE = PIPELINE / "runs/battle-evaluations/20261015-094649-077c67be"     # the Mask Lab's top-50 battles
TOP50_IDENTITY = "4208bc07b46d240b23059d94b9fcdee5f00b6dea4525e1272f2284083f4906f8"
BC_HASH, SEED = "57f5edcab415cf6c", 20261008
LING = ["regmb@16384", "baseline@16384", "baseline@4096"]
ARMS = ["text", "diffusion", *LING, "random"]
NAMES = {"original": "original", "text": "text writer", "diffusion": "scratch decoder", "random": "random fill",
         **{a: f"Ling {a}" for a in LING}}
PREFIX = {"original": "orig:", "text": "textdiff:", "diffusion": "diffusion:", "random": "random:",
          **{a: f"ling:{a}:" for a in LING}}


# ---------------------------------------------------------------- battle (/tmp/vgc-pilot/.venv)
def load_runner():
    """fast_battle_runner.py as it was on 2026-10-15 (the evidence copy): same identity as the top-50 panels."""
    sys.path.insert(0, str(PIPELINE.parent))                      # battle_speed_lab, which the runner imports
    spec = importlib.util.spec_from_file_location("fast_battle_runner_20261015", TOP50_EVIDENCE / "source/fast_battle_runner.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

def battle(a):
    sys.path.insert(0, str(REPO / "src"))
    import heldout as H, matchup_db as MDB
    from showdown_validator import Validator
    if MDB.sha256_file(MDB.POLICY_FILE) != BC_HASH:
        raise SystemExit("bc_100.zip differs from the verified behaviour-cloning checkpoint; refusing to battle")
    if a.database.resolve() == Path(MDB.DEFAULT_DB).resolve():
        raise SystemExit("use a private database, not the shared matchup matrix")
    runner_module = load_runner()
    con = MDB.connect(a.database)
    cands = [c for c in json.loads((a.run_dir / "candidates.json").read_text())["candidates"] if c["paste"]]
    if a.limit: cands = cands[:a.limit]
    val = Validator()
    try:
        with con: ids = {c["label"]: MDB.add_team(con, c["paste"], "llm-completion:maskLab-heldout", val)[0] for c in cands}
        cols = MDB.seed_heldout(con, val)                          # validated, so the runner accepts them as opponents
    finally:
        val.close()
    held = H.team_ids(con)
    marks = ",".join("?" * len(ids))
    legal = {t for (t,) in con.execute(f"SELECT team_id FROM team WHERE legal=1 AND team_id IN ({marks})", list(ids.values()))}
    rows = sorted(legal - held)
    print(f"{len(cands)} candidates -> {len(legal)} distinct legal teams, {len(legal & held)} of them a held-out team; "
          f"{len(cols)} held-out opponents; planned battles <= {len(rows) * len(cols)}", flush=True)

    evidence = EVIDENCE_ROOT / f"maskLab-heldout-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    with runner_module.FastBattleRunner(evidence, a.workers, a.concurrency) as runner:
        if runner.identity != TOP50_IDENTITY:
            raise SystemExit(f"evaluator identity {runner.identity} differs from the top-50 run's {TOP50_IDENTITY}")
        policy_id = runner.policy_id(con, MDB)
        t_all = time.time()
        for i in range(0, len(rows), a.batch):
            t0 = time.time()
            MDB.evaluate_final(con, policy_id, rows[i:i + a.batch], per_cell=1, seed=SEED, runner=runner, note="maskLab-heldout:fast")
            print(f"batch {i // a.batch + 1}/{-(-len(rows) // a.batch)} in {time.time() - t0:.1f}s ({time.time() - t_all:.1f}s total)", flush=True)

    out = {"policy_id": policy_id, "policy_identity": TOP50_IDENTITY,
           "policy": f"bc_100.zip:{BC_HASH} on both sides, listed slot order", "seed": SEED, "opponent_set": H.SET_NAME,
           "database": str(a.database), "evaluator": {"backend": "fast", "workers": a.workers, "concurrency": a.concurrency,
           "runner_file": str(TOP50_EVIDENCE / "source/fast_battle_runner.py"), "evidence": str(evidence)},
           "opponents": [], "panels": {}}
    for mb, t in zip(H.ids(), cols):
        paste = con.execute("SELECT paste FROM team WHERE team_id=?", (t,)).fetchone()[0]
        out["opponents"].append({"id": mb, "team_id": t, "species": [s["species"] for s in MDB.canonical_slots(paste)]})
    incomplete = []
    for c in cands:
        t = ids[c["label"]]
        if t not in legal: out["panels"][c["label"]] = {"team_id": t, "score": None, "illegal": True}; continue
        if t in held: out["panels"][c["label"]] = {"team_id": t, "score": None, "heldout_team": True}; continue
        s = MDB.score(con, policy_id, t, H.SET_NAME, evaluation=True)
        cells = MDB.row_cells(con, policy_id, t, cols)
        if s["coverage"] != len(cols) or s["battles"] != len(cols): incomplete.append(c["label"])
        out["panels"][c["label"]] = {"team_id": t, "score": s["score"], "coverage": s["coverage"], "battles": s["battles"],
                                     "se": s["se"], "by_opponent": {str(o): list(v) for o, v in cells.items()}}
    (a.run_dir / "panels_heldout.json").write_text(json.dumps(out, indent=1) + "\n")
    scored = sum(p.get("score") is not None for p in out["panels"].values())
    print(f"saved {scored} scored panels to {a.run_dir / 'panels_heldout.json'}; incomplete: {len(incomplete)}", flush=True)
    con.close()
    if incomplete:
        raise SystemExit(f"{len(incomplete)} panels lack one battle against each of the {len(cols)} held-out teams")


# ---------------------------------------------------------------- report (any Python 3)
def cluster(pairs):
    """Equal-start-weighted mean of a paired difference, 95% bootstrap over the five starting teams
    (as in mask_lab_textdiff.py report and scripts/text_vs_ling.py)."""
    by = {}
    for s, v in pairs: by.setdefault(s, []).append(v)
    if not by: return None
    means = [sum(v) / len(v) for v in by.values()]
    rng = random.Random(SEED); draws = sorted(sum(rng.choice(means) for _ in means) / len(means) for _ in range(4000))
    return dict(mean=sum(means) / len(means), lo=draws[100], hi=draws[-100], n=sum(len(v) for v in by.values()))

fmt = lambda c: "—" if not c else f"{100*c['mean']:+.1f} [{100*c['lo']:+.1f}, {100*c['hi']:+.1f}]"

def mask_rows(panels, tasks):
    score = lambda lab: (panels.get(lab) or {}).get("score")
    fam = {}
    for t in tasks:
        tid, src = t["id"], t["source"]
        r = dict(task=tid, start=src, original=score(f"orig:{src}"), **{k: score(PREFIX[k] + tid) for k in ARMS})
        fam.setdefault(t["task"], []).append(r)
    return list(fam.items()) + [("all", [r for v in fam.values() for r in v])]

def compare(groups):
    """Every statistic of the Mask Lab and text-vs-Ling tables, per mask family and over all 270 masks.
    Net: a rejected fill keeps the original team, so it counts illegal answers."""
    net = lambda r, k: (r[k] if r[k] is not None else r["original"]) - r["original"]
    out = {}
    for family, rs in groups:
        o = {"n": len(rs), "legal": {k: sum(r[k] is not None for r in rs) for k in ARMS}}
        for k in ARMS:
            o[f"{k}_vs_original"] = cluster([(r["start"], r[k] - r["original"]) for r in rs if r[k] is not None])
            o[f"net_{k}"] = cluster([(r["start"], net(r, k)) for r in rs])
        for k in ARMS[1:]:
            o[f"text_vs_{k}_both_legal"] = cluster([(r["start"], r["text"] - r[k]) for r in rs if r["text"] is not None and r[k] is not None])
            o[f"net_text_minus_{k}"] = cluster([(r["start"], net(r, "text") - net(r, k)) for r in rs])
        out[family] = o
    return out

def report(a):
    tasks = json.loads((a.run_dir / "eval2.tasks.json").read_text())["tasks"]
    target = json.loads((a.run_dir / a.panels).read_text())
    groups = compare(mask_rows(target["panels"], tasks))
    L = "regmb@16384"
    print(f"## {a.panels}: {len(target['opponents'])} opponents, one battle each, behaviour-cloning policy both sides\n")
    print("| family | legal, text | legal, Ling | text − original | Ling − original | text − Ling, both legal | net, text | net, Ling | **net, text − Ling** |")
    print("|---|---:|---:|---|---|---|---|---|---|")
    for family, o in groups.items():
        both = o[f"text_vs_{L}_both_legal"]
        print(f"| {family} | {o['legal']['text']}/{o['n']} | {o['legal'][L]}/{o['n']} | {fmt(o['text_vs_original'])} | "
              f"{fmt(o[L + '_vs_original'])} | {fmt(both)} ({both['n'] if both else 0}) | {100*o['net_text']['mean']:+.1f} | "
              f"{100*o['net_' + L]['mean']:+.1f} | **{fmt(o['net_text_minus_' + L])}** |")
    print("\n| family | text − random fill | text − scratch decoder | net, scratch decoder |\n|---|---|---|---|")
    for family, o in groups.items():
        print(f"| {family} | {fmt(o['text_vs_random_both_legal'])} | {fmt(o['text_vs_diffusion_both_legal'])} | {fmt(o['net_diffusion'])} |")
    al = groups["all"]
    print("\nOther Ling arms, all masks: " + "; ".join(f"{k} legal {al['legal'][k]}/270, net {fmt(al['net_' + k])}, "
          f"text − arm both legal {fmt(al['text_vs_' + k + '_both_legal'])}" for k in LING[1:]))

    gaps, means = {}, {}                     # the same fills on both panels: top-50 → held-out, per arm
    if a.panels != "panels.json":
        top = json.loads((a.run_dir / "panels.json").read_text())["panels"]
        start = {t["id"]: t["source"] for t in tasks}
        print("\n| arm | fills scored on both | win rate vs top-50 | win rate vs held-out | held-out − top-50 |\n|---|---:|---:|---:|---|")
        for arm, prefix in PREFIX.items():
            pairs, t50, h50 = [], [], []
            for label, p in target["panels"].items():
                if not label.startswith(prefix) or p.get("score") is None or (top.get(label) or {}).get("score") is None: continue
                key = label[len(prefix):]
                pairs.append((key if arm == "original" else start[key], p["score"] - top[label]["score"]))
                t50.append(top[label]["score"]); h50.append(p["score"])
            if not pairs: continue
            gaps[arm] = cluster(pairs)
            means[arm] = dict(n=len(pairs), top50=sum(t50) / len(t50), heldout=sum(h50) / len(h50))
            print(f"| {NAMES[arm]} | {len(pairs)} | {means[arm]['top50']:.3f} | {means[arm]['heldout']:.3f} | {fmt(gaps[arm])} |")

    shifts = {}                              # does each comparison hold on the held-out teams? same mask, both panels
    if a.panels != "panels.json":
        net = lambda r, k: (r[k] if r[k] is not None else r["original"]) - r["original"]
        diffs = {f"net, text − Ling {L}": lambda r: net(r, "text") - net(r, L),
                 f"text − Ling {L}, both legal": lambda r: r["text"] - r[L] if r["text"] is not None and r[L] is not None else None,
                 "net, text − scratch decoder": lambda r: net(r, "text") - net(r, "diffusion"),
                 "text − scratch decoder, both legal": lambda r: r["text"] - r["diffusion"] if r["text"] is not None and r["diffusion"] is not None else None,
                 "net, text": lambda r: net(r, "text"), f"net, Ling {L}": lambda r: net(r, L), "net, scratch decoder": lambda r: net(r, "diffusion")}
        top_groups = dict(mask_rows(json.loads((a.run_dir / "panels.json").read_text())["panels"], tasks))
        print("\n| comparison | masks | vs top-50 | vs held-out | held-out − top-50 |\n|---|---|---:|---|---|")
        for family in ("all", "pokemon"):
            top_rows = {r["task"]: r for r in top_groups[family]}
            for name, f in diffs.items():
                cells = [(r["start"], f(top_rows[r["task"]]), f(r)) for r in dict(mask_rows(target["panels"], tasks))[family]]
                cells = [(s, t, h) for s, t, h in cells if t is not None and h is not None]
                o = dict(top50=cluster([(s, t) for s, t, _ in cells]), heldout=cluster([(s, h) for s, _, h in cells]),
                         change=cluster([(s, h - t) for s, t, h in cells]))
                shifts[f"{family}: {name}"] = o
                print(f"| {name} | {family} ({len(cells)}) | {fmt(o['top50'])} | {fmt(o['heldout'])} | {fmt(o['change'])} |")
    if a.panels == "panels_heldout.json":            # the top-50 tables are text_vs_ling.json's; print them only
        (a.run_dir / "heldout_report.json").write_text(json.dumps(dict(panels=a.panels, opponents=len(target["opponents"]),
            groups=groups, arm_means=means, heldout_minus_top50=gaps, comparison_shift=shifts), indent=1) + "\n")
        print(f"\nwrote {a.run_dir / 'heldout_report.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("battle")
    b.add_argument("--workers", type=int, default=6)
    b.add_argument("--concurrency", type=int, default=24)
    b.add_argument("--batch", type=int, default=24, help="teams battled per saved batch")
    b.add_argument("--limit", type=int, help="only the first N candidates (smoke test)")
    b.add_argument("--database", type=Path, default=DATABASE)
    r = sub.add_parser("report")
    r.add_argument("--panels", default="panels_heldout.json", help="panel file in the run folder")
    for p in (b, r): p.add_argument("--run-dir", type=Path, default=RUN)
    a = ap.parse_args()
    {"battle": battle, "report": report}[a.cmd](a)
