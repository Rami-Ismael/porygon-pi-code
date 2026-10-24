"""The Mask Lab baseline with held-out teams (placement ranks 51-100) as the starting teams.

A near-exact copy of the Mask Lab run (the vault's team_completion_pipeline, runs/prompt-ab) with one change: the
five starting teams are drawn from ranks 51-100 instead of being random legal teams. Kept: the six mask families and
their k values (54 masks per starting team, task seed 22), the random legal fill (mask_lab_prepare.random_fill), the
scratch decoder's settings (ask 0.5, guidance 2, temperature 1), the text writer's constrained infilling
(mask_lab_textdiff.generate: ask 0.5, guidance 2, threshold 0.9, seed 20261008 + trial index), one battle against
each of the 49 distinct top-50 teams, the behaviour-cloning policy on both sides, listed slot order, seed 20261008,
and the 2026-10-15 fast evaluator (identity 4208bc07...). The vault scripts run unchanged; only their paths point
here, and the project's own src/ modules are imported first so the stale /tmp/vgc-pilot/src copies are never used.

Two text writers fill the masks: lora_r32/final (the adapter behind every earlier text-writer result) and
lora_r32_s20261021/final, a fresh fine-tune for this run on the same 8,000 examples with the same recipe and a new
seed. The scratch decoder is the clean pair (F1/G2 _noheldout_regmb_v3). Ling is not run: it needs
OPENROUTER_API_KEY and paid calls.

Held-out rule (docs/heldout-51-100.md): these teams are inputs here and nothing trains on them. `split` checks that
no model that fills them saw them. Their battles go to ~/vgc-data/mask_lab_heldout_starters.sqlite, which no
training or search code reads; ensure/score run with evaluation=True there because the originals are held-out
teams. docs/textdiff.md, "Held-out teams as the starting teams".

    /tmp/vgc-pilot/.venv/bin/python scripts/mask_lab_heldout_starters.py prepare      # starters, masks, random fills
    /tmp/vgc-pilot/.venv/bin/python scripts/mask_lab_heldout_starters.py split        # the training data holds no starter
    /tmp/vgc-pilot/.venv/bin/python scripts/mask_lab_heldout_starters.py diffusion    # scratch decoder fills
    ~/.local/share/vgc-pilot-runtime/venv-dlm/bin/python scripts/mask_lab_heldout_starters.py text-generate --kind textdiff
    /tmp/vgc-pilot/.venv/bin/python scripts/mask_lab_heldout_starters.py text-collect --kind textdiff
    /tmp/vgc-pilot/.venv/bin/python scripts/mask_lab_heldout_starters.py battle
    python3 scripts/mask_lab_heldout_starters.py report
"""
import argparse, collections, importlib.util, json, random, subprocess, sys, time, uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
PIPELINE = Path.home() / ("Documents/yakumsi-vault/Personal Project/Using Advance method search in the whole search of "
                          "possible vgc pokemon format to find the right counter to a pokemon team/team_completion_pipeline")
RUN = PIPELINE / "runs/heldout-starters"               # tasks, candidates, panels, report (git-ignored)
SOURCE_RUN = PIPELINE / "runs/prompt-ab"               # the random-starter Mask Lab run this copies
WORK = Path.home() / "vgc-data/mask_lab_heldout_starters"
DATABASE = Path.home() / "vgc-data/mask_lab_heldout_starters.sqlite"
RUNTIME = Path.home() / ".local/share/vgc-pilot-runtime"
TOP50_EVIDENCE = PIPELINE / "runs/battle-evaluations/20261015-094649-077c67be"
TOP50_IDENTITY = "4208bc07b46d240b23059d94b9fcdee5f00b6dea4525e1272f2284083f4906f8"
BC_HASH, SEED, TASK_SEED, N_STARTERS = "57f5edcab415cf6c", 20261008, 22, 5
MEMBERS = ["F1_final_medium_noheldout_regmb_v3", "G2_family_matrix_noprtrain_noheldout_regmb_v3"]
ADAPTERS = {"textdiff": RUNTIME / "textdiff/lora_r32/final", "textdiff_s21": RUNTIME / "textdiff/lora_r32_s20261021/final"}
STATS = ["HP", "Atk", "Def", "SpA", "SpD", "Spe"]


def vault(name):
    """A team_completion_pipeline module, loaded unchanged, with its run folder pointed at this experiment."""
    if str(PIPELINE) not in sys.path: sys.path.insert(1, str(PIPELINE))
    spec = importlib.util.spec_from_file_location(name, PIPELINE / f"{name}.py")
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module; spec.loader.exec_module(module)
    module.RUN = RUN
    return module

def to_pipeline_team(text):
    """A Showdown paste as the pipeline's team record (blocks parsed one by one, as heldout.grid_key_text does)."""
    import corpus
    team = []
    for block in text.strip().split("\n\n"): team += corpus.parse_team_text(block)
    return [dict(slot=i + 1, species=s["species"], item=s["item"], ability=s["ability"], moves=list(s["moves"]),
                 nature=s["nature"], statPoints={k: int(s["evs"].get(k, 0)) for k in STATS}, ivs={k: 31 for k in STATS},
                 level=50) for i, s in enumerate(team)]


# ---------------------------------------------------------------- prepare (/tmp/vgc-pilot/.venv)
def prepare(a):
    import corpus, heldout as H                       # the project's own modules, before the vault code adds the staging folder
    M = vault("mask_lab_prepare"); A, P = sys.modules["prompt_ab"], sys.modules["pipeline"]
    config = A.load_config(str(PIPELINE / "benchmark.ling.json")); config["mask_repetitions"] = 1
    rows, rejected = [], []
    for f in H.files():
        team = to_pipeline_team(Path(f).read_text())
        try:
            P.check_structure(team); P.validate_team(config, team)
        except ValueError as error:
            rejected.append(dict(id=Path(f).stem, error=str(error)[:200])); continue
        rows.append(dict(id=Path(f).stem, source=str(f), format=config["format"], team=team))
    starts = random.Random(SEED).sample(rows, N_STARTERS)
    tasks = A.build_tasks(config, starts, TASK_SEED)
    RUN.mkdir(parents=True, exist_ok=True)
    P.save_json(RUN / "eval2.tasks.json", dict(format=config["format"], seed=TASK_SEED, tasks=tasks, starting_teams=starts))
    (RUN / "eval2.jsonl").write_text("")               # no Ling responses in this run
    M.main()                                            # untouched originals and random legal fills -> candidates.json
    D = vault("mask_lab_diffusion"); D.WORK = WORK / "diffusion"
    D.prepare(argparse.Namespace(limit=None))           # the decoder-format manifest both model arms read
    P.save_json(RUN / "starters.json", dict(seed=SEED, eligible=len(rows), rejected=rejected,
                                             starters=[s["id"] for s in starts], tasks=len(tasks)))
    print(f"{len(rows)} of 50 held-out teams eligible ({len(rejected)} rejected: "
          f"{[r['id'] for r in rejected]}); starters {[s['id'] for s in starts]}; {len(tasks)} masks")


# ---------------------------------------------------------------- split (/tmp/vgc-pilot/.venv)
def split(a):
    """No model that fills these masks trained on a starting team: 48-field identity against every text-writer example,
    shared slots for the near copies, and the held-out separation tests for the scratch decoder's data."""
    import corpus, heldout as H
    saved = json.loads((RUN / "eval2.tasks.json").read_text())
    starts = {s["id"]: s["team"] for s in saved["starting_teams"]}
    slot_key = lambda s: (H._norm(s["species"]), H._norm(s["ability"]), H._norm(s["item"]),
                          tuple(sorted(H._norm(m) for m in s["moves"])),
                          "serious" if H._norm(s.get("nature")) in H.NEUTRAL else H._norm(s.get("nature")))
    held, start_keys = H.identities(), {i: H.grid_key_team(t) for i, t in starts.items()}
    start_slots = {i: collections.Counter(slot_key(s) for s in t) for i, t in starts.items()}
    out = {"starters": {i: {"is_heldout_team": k in held} for i, k in start_keys.items()}}
    for name in ("train", "val"):
        examples = [json.loads(l) for l in open(RUNTIME / f"textdiff/{name}.jsonl")]
        teams = []
        for e in examples:
            t = []
            for block in e["answer"].strip().split("\n\n"): t += corpus.parse_team_text(block)
            teams.append(t)
        keys = [H.grid_key_team(t) for t in teams]
        o = {"examples": len(examples), "heldout_teams": sum(k in held for k in keys),
             "starting_teams": sum(k in start_keys.values() for k in keys)}
        if name == "train":                             # the trainer's --limit 8000 subset: every real row, then the rest in order
            real = [i for i, e in enumerate(examples) if e["source"] == "real"]
            rest = [i for i, e in enumerate(examples) if e["source"] != "real"]
            used = real + rest[:8000 - len(real)]
            o["limit_8000_subset"] = {"examples": len(used), "heldout_teams": sum(keys[i] in held for i in used)}
            for sid, want in start_slots.items():
                shared = [sum((collections.Counter(slot_key(s) for s in teams[i]) & want).values()) for i in used]
                o.setdefault("max_shared_slots_with_a_starter", {})[sid] = {
                    "max": max(shared), "examples_sharing_5": sum(x == 5 for x in shared), "sharing_4": sum(x == 4 for x in shared)}
        out[f"text_writer_{name}"] = o
    # The scratch decoder's members were retrained without the set on 2026-10-14 (docs/heldout-51-100.md); load_net
    # refuses any other name. The separation tests check the guards; one of them also lists every checkpoint whose
    # file name lacks "noheldout", which later loop runs (dbbo_*, cem_scratch_*) trip without touching these members.
    import lossdown as L
    test = subprocess.run([sys.executable, "-W", "ignore", str(REPO / "scripts/test_heldout_separation.py")], cwd=REPO,
                          capture_output=True, text=True)
    failed = sorted({l.split(" (")[0][len("FAIL: "):] for l in (test.stdout + test.stderr).splitlines() if l.startswith(("FAIL: ", "ERROR: "))})
    naming = "test_no_contaminated_checkpoint_is_left_in_the_load_path"
    unnamed = sorted(p.name for p in Path(L.CK).glob("*.pt") if "noheldout" not in p.name)
    out["scratch_decoder"] = {"members": MEMBERS,
                              "members_clean": all("noheldout" in m and (Path(L.CK) / f"{m}.pt").exists()
                                                   and not (Path(L.CK) / "_contaminated_heldout" / f"{m}.pt").exists() for m in MEMBERS),
                              "separation_tests_failed": failed,
                              "checkpoints_without_noheldout_in_name": dict(n=len(unnamed), kinds=sorted({n.split("_")[0] + "_" + n.split("_")[1] for n in unnamed}))}
    out["modules"] = {m: sys.modules[m].__file__ for m in ("corpus", "heldout")}
    (RUN / "split.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    if (not all(v["is_heldout_team"] for v in out["starters"].values()) or out["text_writer_train"]["starting_teams"]
            or out["text_writer_val"]["starting_teams"] or out["text_writer_train"]["heldout_teams"]
            or out["text_writer_val"]["heldout_teams"] or not out["scratch_decoder"]["members_clean"] or set(failed) - {naming}):
        raise SystemExit("split check failed")


# ---------------------------------------------------------------- fills
def diffusion(a):
    """The scratch decoder (clean F1 + G2 v3 pair) on the manifest, then its valid fills join candidates.json."""
    import asked_vs_got, corpus                       # noqa: F401  project modules first
    D = vault("mask_lab_diffusion"); D.WORK = WORK / "diffusion"; D.MEMBERS = MEMBERS
    subprocess.run(["nice", "-n", "10", sys.executable, "diffusion_baseline_v3.py", "generate", "--baseline",
                    str(D.WORK / "baseline"), "--output", str(D.WORK / "out"), "--members", *MEMBERS],
                   cwd=REPO / "scripts", check=True)
    D.collect(argparse.Namespace())

def llm_baseline_2026_10_15():
    """llm_baseline as the Mask Lab's collect steps called it on 2026-10-15: runtime(), paste(), validate_shape() and
    masked(). Merge #6 dropped runtime() and the others moved to team_masks (identical code); mask_lab_textdiff.collect
    still imports them from llm_baseline, so this module stands in for it, with runtime() and paste() as they were."""
    import types, hps_generate as h, matchup_db as db, team_masks
    def runtime():
        h.SHOWDOWN = Path.home() / ".local/share/vgc-pilot-runtime/validator-913da36"; db.SHOWDOWN = h.SHOWDOWN
        return h, db
    def paste(team, h):
        return "\n\n".join(h.slot_to_text(dict(s, moves=[m for m in s["moves"] if m])) for s in team) + "\n"
    shim = types.ModuleType("llm_baseline")
    shim.runtime, shim.paste, shim.validate_shape, shim.masked = runtime, paste, team_masks.validate_shape, team_masks.masked
    sys.modules["llm_baseline"] = shim
    return shim

def text_module(kind):
    T = vault("mask_lab_textdiff")
    T.WORK, T.OUT, T.ADAPTER, T.KIND = WORK / "diffusion", WORK / kind, ADAPTERS[kind], kind
    return T

def text_generate(a):                                 # ~/.local/share/vgc-pilot-runtime/venv-dlm
    if not (ADAPTERS[a.kind] / "adapter_model.safetensors").exists():
        raise SystemExit(f"no adapter at {ADAPTERS[a.kind]}")
    text_module(a.kind).generate(argparse.Namespace(limit=a.limit, family=None, guidance=2.0, ask=0.5))

def text_collect(a):                                  # /tmp/vgc-pilot/.venv
    import asked_vs_got, corpus, hps_generate          # noqa: F401  project modules first
    llm_baseline_2026_10_15()
    text_module(a.kind).collect(argparse.Namespace())


# ---------------------------------------------------------------- battle (/tmp/vgc-pilot/.venv)
def battle(a):
    import heldout as H, matchup_db as MDB, top50 as T50
    from showdown_validator import Validator
    if MDB.sha256_file(MDB.POLICY_FILE) != BC_HASH:
        raise SystemExit("bc_100.zip differs from the verified behaviour-cloning checkpoint; refusing to battle")
    if a.database.resolve() == Path(MDB.DEFAULT_DB).resolve():
        raise SystemExit("use a private database, not the shared matchup matrix")
    sys.path.insert(0, str(PIPELINE.parent))           # battle_speed_lab, imported by the runner
    spec = importlib.util.spec_from_file_location("fast_battle_runner_20261015", TOP50_EVIDENCE / "source/fast_battle_runner.py")
    runner_module = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner_module)
    con = MDB.connect(a.database)
    cands = [c for c in json.loads((a.run_dir / "candidates.json").read_text())["candidates"] if c["paste"]]
    val = Validator()
    try:
        with con:
            ids = {c["label"]: MDB.add_team(con, c["paste"], "llm-completion:maskLab-heldout-starters", val)[0] for c in cands}
            con.execute("DELETE FROM team_set WHERE set_name='top50'")
            for t in json.load(open(T50.TOP50)):          # as matchup_db.seed: a team listed twice counts twice
                tid = MDB.add_team(con, Path(T50.resolve(t["id"])).read_text(), "vgcpastes", val)[0]
                con.execute("INSERT INTO team_set VALUES ('top50', ?, 1) ON CONFLICT (set_name, team_id) "
                            "DO UPDATE SET weight = weight + 1", (tid,))
    finally:
        val.close()
    cols = MDB.set_members(con, "top50")
    import sqlite3                                    # the opponents must be the Mask Lab's 49 teams, paste for paste
    lab = sqlite3.connect("file:" + str(Path.home() / "vgc-data/mask_lab_matrix.sqlite") + "?mode=ro&immutable=1", uri=True)
    want = {h for (h,) in lab.execute("SELECT canon_hash FROM team JOIN team_set USING (team_id) WHERE set_name='top50'")}
    got = {h for (h,) in con.execute(f"SELECT canon_hash FROM team WHERE team_id IN ({','.join('?' * len(cols))})", cols)}
    if want != got: raise SystemExit(f"top-50 columns differ from the Mask Lab's: {len(want ^ got)} pastes")
    marks = ",".join("?" * len(ids))
    legal = {t for (t,) in con.execute(f"SELECT team_id FROM team WHERE legal=1 AND team_id IN ({marks})", list(ids.values()))}
    held = H.team_ids(con)
    rows = sorted(legal)
    print(f"{len(cands)} candidates -> {len(legal)} distinct legal teams ({len(legal & held)} are held-out teams); "
          f"{len(cols)} top-50 opponents; planned battles <= {len(rows) * len(cols)}", flush=True)
    evidence = Path.home() / f"vgc-data/battle-evaluations/maskLab-heldout-starters-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    with runner_module.FastBattleRunner(evidence, a.workers, a.concurrency) as runner:
        if runner.identity != TOP50_IDENTITY:
            raise SystemExit(f"evaluator identity {runner.identity} differs from the top-50 run's {TOP50_IDENTITY}")
        policy_id = runner.policy_id(con, MDB)
        t_all = time.time()
        for i in range(0, len(rows), a.batch):
            t0 = time.time()
            MDB.ensure(con, policy_id, rows[i:i + a.batch], cols, 1, seed=SEED, runner=runner,
                       note="maskLab-heldout-starters:fast", evaluation=True)
            print(f"batch {i // a.batch + 1}/{-(-len(rows) // a.batch)} in {time.time() - t0:.1f}s ({time.time() - t_all:.1f}s total)", flush=True)
    out = {"policy_id": policy_id, "policy_identity": TOP50_IDENTITY,
           "policy": f"bc_100.zip:{BC_HASH} on both sides, listed slot order", "seed": SEED, "database": str(a.database),
           "evaluator": {"backend": "fast", "workers": a.workers, "concurrency": a.concurrency, "evidence": str(evidence)},
           "opponents": [], "panels": {}}
    for t in cols:
        paste = con.execute("SELECT paste FROM team WHERE team_id=?", (t,)).fetchone()[0]
        weight = con.execute("SELECT weight FROM team_set WHERE set_name='top50' AND team_id=?", (t,)).fetchone()[0]
        out["opponents"].append({"team_id": t, "weight": weight, "species": [s["species"] for s in MDB.canonical_slots(paste)]})
    incomplete = []
    for c in cands:
        t = ids[c["label"]]
        if t not in legal: out["panels"][c["label"]] = {"team_id": t, "score": None, "illegal": True}; continue
        s = MDB.score(con, policy_id, t, "top50", evaluation=True)
        cells = MDB.row_cells(con, policy_id, t, cols)
        if s["coverage"] != len([x for x in cols if x != t]): incomplete.append(c["label"])
        out["panels"][c["label"]] = {"team_id": t, "score": s["score"], "coverage": s["coverage"], "battles": s["battles"],
                                     "se": s["se"], "heldout_team": t in held, "by_opponent": {str(o): list(v) for o, v in cells.items()}}
    (a.run_dir / "panels.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"saved {sum(p.get('score') is not None for p in out['panels'].values())} scored panels; incomplete: {len(incomplete)}", flush=True)
    con.close()
    if incomplete: raise SystemExit(f"{len(incomplete)} panels lack a battle against each top-50 team")


# ---------------------------------------------------------------- report (any Python 3)
def report(a):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from mask_lab_heldout import cluster, fmt
    tasks = json.loads((RUN / "eval2.tasks.json").read_text())["tasks"]
    panels = json.loads((RUN / "panels.json").read_text())["panels"]
    kinds = [k for k in ADAPTERS if any(l.startswith(k + ":") for l in panels)]
    score = lambda lab: (panels.get(lab) or {}).get("score")
    fam = {}
    for t in tasks:
        r = dict(start=t["source"], original=score(f"orig:{t['source']}"), random=score(f"random:{t['id']}"),
                 diffusion=score(f"diffusion:{t['id']}"), **{k: score(f"{k}:{t['id']}") for k in kinds})
        fam.setdefault(t["task"], []).append(r)
    groups = list(fam.items()) + [("all", [r for v in fam.values() for r in v])]
    net = lambda r, k: (r[k] if r[k] is not None else r["original"]) - r["original"]
    both = lambda rs, x, y: cluster([(r["start"], r[x] - r[y]) for r in rs if r[x] is not None and r[y] is not None])
    out = {}
    for family, rs in groups:
        o = {"n": len(rs), "legal": {k: sum(r[k] is not None for r in rs) for k in ["diffusion", "random", *kinds]}}
        for k in ["diffusion", *kinds]:
            o[f"{k}_vs_original"] = cluster([(r["start"], r[k] - r["original"]) for r in rs if r[k] is not None])
            o[f"{k}_vs_random"] = both(rs, k, "random")
            o[f"net_{k}"] = cluster([(r["start"], net(r, k)) for r in rs])
        for k in kinds:
            o[f"{k}_vs_diffusion"] = both(rs, k, "diffusion")
            o[f"net_{k}_minus_diffusion"] = cluster([(r["start"], net(r, k) - net(r, "diffusion")) for r in rs])
        if len(kinds) == 2:
            o["new_vs_old_both_legal"] = both(rs, kinds[1], kinds[0])
            o["net_new_minus_old"] = cluster([(r["start"], net(r, kinds[1]) - net(r, kinds[0])) for r in rs])
        out[family] = o
    names = {"textdiff": "text writer (lora_r32)", "textdiff_s21": "text writer (fresh fine-tune)", "diffusion": "scratch decoder"}
    originals = {t["source"] for t in tasks}
    print("## Held-out teams as starting teams: win rate vs the 49 distinct top-50 teams, one battle each, BC policy both sides\n")
    print("Originals:", ", ".join(f"{s} {score('orig:' + s):.3f}" for s in sorted(originals)), "\n")
    for k in [*kinds, "diffusion"]:
        print(f"### {names[k]}\n\n| family | legal | − original | − random fill | net per mask |"
              + (" − scratch decoder, both legal | net − scratch decoder |" if k != "diffusion" else "")
              + "\n|---|---:|---|---|---|" + ("---|---|" if k != "diffusion" else ""))
        for family, o in out.items():
            extra = f" {fmt(o[k + '_vs_diffusion'])} | {fmt(o['net_' + k + '_minus_diffusion'])} |" if k != "diffusion" else ""
            print(f"| {family} | {o['legal'][k]}/{o['n']} | {fmt(o[k + '_vs_original'])} | {fmt(o[k + '_vs_random'])} | "
                  f"{fmt(o['net_' + k])} |{extra}")
        print()
    if len(kinds) == 2:
        print("### fresh fine-tune − lora_r32\n\n| family | both legal | net per mask |\n|---|---|---|")
        for family, o in out.items():
            print(f"| {family} | {fmt(o['new_vs_old_both_legal'])} | {fmt(o['net_new_minus_old'])} |")
    rows = {k: json.loads((RUN / f"{k}_rows.json").read_text()) for k in kinds if (RUN / f"{k}_rows.json").exists()}
    errors = {k: collections.Counter((v["error"] or "")[:40].split(":")[0] for v in r.values() if not v["valid"]).most_common(4) for k, r in rows.items()}
    seconds = {k: round(sum(v["seconds"] for v in r.values()) / len(r), 1) for k, r in rows.items()}
    print("\nrejections:", errors, "| seconds per mask:", seconds)
    (RUN / "report.json").write_text(json.dumps(dict(groups=out, kinds=kinds, rejections=errors, seconds=seconds), indent=1) + "\n")
    print(f"wrote {RUN / 'report.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("prepare", "split", "diffusion", "report"): sub.add_parser(name)
    for name in ("text-generate", "text-collect"):
        t = sub.add_parser(name); t.add_argument("--kind", choices=list(ADAPTERS), default="textdiff"); t.add_argument("--limit", type=int)
    b = sub.add_parser("battle")
    b.add_argument("--workers", type=int, default=6); b.add_argument("--concurrency", type=int, default=24)
    b.add_argument("--batch", type=int, default=24); b.add_argument("--database", type=Path, default=DATABASE)
    b.add_argument("--run-dir", type=Path, default=RUN, help="folder with candidates.json; panels.json is written there "
                                                             "(scripts/textdiff_llmformat.py's random-grid run uses its own)")
    a = ap.parse_args()
    {"prepare": prepare, "split": split, "diffusion": diffusion, "text-generate": text_generate, "text-collect": text_collect,
     "battle": battle, "report": report}[a.cmd](a)
