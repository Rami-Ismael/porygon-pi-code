"""Battle-side of the pretrained text diffusion comparison (project venv). Pre-registration: docs/textdiff.md.

    python textdiff_battle.py scratch          # 64 valid teams per scratch arm (noheldout v3 members): base, cfg
    python textdiff_battle.py decode           # parse every generated text (textdiff.py gen), fill spreads, validate
    python textdiff_battle.py stack            # text model as the stack's proposer: 512-valid pool -> protect3 -> ridge top 64
    python battle_supervisor.py textdiff_battle.py battle --then report
(run with /tmp/vgc-pilot/.venv/bin/python)

Win rate = wins / battles against the top-50 meta teams, one battle per (team, top-50 column),
BC policy both sides (policy 1), in the matchup matrix (origin textdiff:<cell>). Same protocol,
seed and cell size as foundation.py (64 teams, seed 2026).
"""
import json, math, sys, time
from collections import Counter
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lossdown as L
import lossdown_battle as LB
import asked_vs_got as X
import matchup_db as MDB
import hps_generate
from hps_generate import Validator, slot_to_text
from corpus import parse_team_text, norm
from encode import team_fields

RUNTIME = Path.home() / ".local/share/vgc-pilot-runtime"
GEN = RUNTIME / "textdiff" / "gen"
OUT = L.RES / "textdiff.json"
N, SEED, POOL = 64, 2026, 512
SCRATCH_ARMS = {"base": dict(), "cfg": dict(ystar=0.5, g=2.0)}


def _load():
    return json.load(open(OUT)) if OUT.exists() else {"cells": {}}


def _save(out):
    json.dump(out, open(OUT, "w"))


# ---------------------------------------------------------------- the scratch baseline, re-generated clean
def scratch():
    import torch, foundation as FD
    out = _load(); val = Validator()
    try:
        G = FD.Gen("v3"); sur = FD.Surrogate(G.V)
        for arm, cfg in SCRATCH_ARMS.items():
            tag = f"scratch:{arm}"
            if tag in out["cells"]: continue
            k = FD.SEED_ORDER.index(arm); torch.manual_seed(SEED + 10 * k); rng = np.random.default_rng(SEED + 10 * k)
            keep, tried, t0 = [], 0, time.perf_counter()
            while len(keep) < N and tried < N * 12:
                x = G.sample(96, cfg.get("ystar"), cfg.get("g", 1.0)).cpu().numpy()
                for r in x:
                    tried += 1
                    p, slots = G.to_paste(r, rng)
                    if val(p) is not None: continue
                    keep.append(p)
                    if len(keep) >= N: break
            out["cells"][tag] = dict(model="scratch", arm=arm, members=FD.V3_MEMBERS, pastes=keep, tried=tried,
                                     valid=len(keep) / max(tried, 1), seconds=time.perf_counter() - t0)
            _save(out)
            print(f"  {tag:14s} {len(keep)} teams · {len(keep)}/{tried} valid · {time.perf_counter()-t0:.0f}s", flush=True)
    finally:
        val.close()


# ---------------------------------------------------------------- text -> validated paste
def decode():
    """Parse raw generated text into six slots, fill Stat Points exactly as the scratch arms do
    (same-species corpus spread, random legal spread for a species without one), validate with Showdown.
    Raw legality is measured before any repair; there is no repair."""
    corpus, V, C, look, spreads = X.setup()
    out = _load(); val = Validator()
    try:
        for path in sorted(GEN.glob("lora_r32_*.jsonl")):            # peeks (_peek_*.jsonl) are not cells
            if path.stem.startswith("lora_r32_pool_"): continue      # stack pools are decoded by stack()
            tag = "text:" + path.stem[len("lora_r32_"):]
            if ONLY and tag.split(":")[1] not in ONLY: continue
            rows = [json.loads(l) for l in open(path)]
            if tag in out["cells"] and out["cells"][tag].get("n_raw") == len(rows): continue
            rng = np.random.default_rng(SEED)
            keep, fails, secs = [], Counter(), 0.0
            for r in rows:
                secs += r["seconds"]
                try:
                    slots = parse_team_text(r["text"])
                except Exception as e:
                    fails["parse-exception"] += 1; continue
                if len(slots) != 6: fails[f"slots={len(slots)}"] += 1; continue
                bad = False
                for s in slots:
                    if not s["species"] or not s["ability"] or not (1 <= len(s["moves"]) <= 4): bad = True
                    pl = spreads.get(norm(s["species"]))
                    s["evs"] = pl[int(rng.integers(len(pl)))] if pl else hps_generate.sample_spread(rng)
                if bad: fails["shape"] += 1; continue
                p = "\n\n".join(slot_to_text(s) for s in slots) + "\n"
                err = val(p)
                if err is not None: fails[err.split(";")[0][:60]] += 1; continue
                keep.append(p)
                if len(keep) >= N: break
            n_used = len(rows) if len(keep) < N else rows.index(r) + 1
            out["cells"][tag] = dict(model="text", arm=tag.split(":")[1], pastes=keep[:N], n_raw=len(rows), tried=n_used,
                                     valid=len(keep) / max(n_used, 1), failures=dict(fails.most_common(8)),
                                     seconds=secs, seconds_per_raw=secs / max(len(rows), 1))
            _save(out)
            print(f"  {tag:18s} {len(keep)} valid of {n_used} raw ({len(keep)/max(n_used,1):.0%}) · {dict(fails.most_common(4))}", flush=True)
    finally:
        val.close()


def _decode_rows(rows, spreads, val, want, rng, fails):
    """The decode() loop, shared with stack(): (valid pastes with slots, rows used, seconds over rows used)."""
    keep, secs = [], 0.0
    for n_used, r in enumerate(rows, 1):
        secs += r["seconds"]
        try:
            slots = parse_team_text(r["text"])
        except Exception:
            fails["parse-exception"] += 1; continue
        if len(slots) != 6: fails[f"slots={len(slots)}"] += 1; continue
        bad = False
        for s in slots:
            if not s["species"] or not s["ability"] or not (1 <= len(s["moves"]) <= 4): bad = True
            pl = spreads.get(norm(s["species"]))
            s["evs"] = pl[int(rng.integers(len(pl)))] if pl else hps_generate.sample_spread(rng)
        if bad: fails["shape"] += 1; continue
        p = "\n\n".join(slot_to_text(s) for s in slots) + "\n"
        err = val(p)
        if err is not None: fails[err.split(";")[0][:60]] += 1; continue
        keep.append((p, slots))
        if len(keep) >= want: break
    return keep, n_used, secs


def stack():
    """The text model as the proposer inside the one-shot stack (docs/textdiff.md, stack pre-registration):
    the pool file gen/lora_r32_pool_<arm>.jsonl -> first POOL valid teams (no repair) -> protect3 filter ->
    the shared ridge picks the top N. text:stack_nofilter skips the filter. Pool statistics go to out["pools"]."""
    import foundation as FD
    corpus, V, C, look, spreads = X.setup()
    out = _load(); val = Validator(); sur = FD.Surrogate(V)
    try:
        for path in sorted(GEN.glob("lora_r32_pool_*.jsonl")):
            arm = path.stem[len("lora_r32_pool_"):]
            rows = [json.loads(l) for l in open(path)]
            if out["cells"].get("text:stack", {}).get("n_raw") == len(rows): continue
            rng = np.random.default_rng(SEED); fails = Counter()
            keep, n_used, secs = _decode_rows(rows, spreads, val, POOL, rng, fails)
            sc = np.array([sur.score(s) for _, s in keep])
            passing = [i for i, (_, s) in enumerate(keep) if FD.protect_count(s) >= 3]
            out["pools"] = out.get("pools", {})
            out["pools"][arm] = dict(n_raw=len(rows), tried=n_used, size=len(keep), valid=len(keep) / max(n_used, 1),
                                     failures=dict(fails.most_common(8)), protect3_pass=len(passing),
                                     seconds=secs, seconds_per_valid=secs / max(len(keep), 1), mu_pool=float(sc.mean()),
                                     sets=len({tuple(sorted(norm(s["species"]) for s in t)) for _, t in keep}))
            for tag, idx in (("text:stack", passing), ("text:stack_nofilter", list(range(len(keep))))):
                order = sorted(idx, key=lambda i: -sc[i])[:N]
                out["cells"][tag] = dict(model="text", arm=f"stack:{arm}", pool_arm=arm, pastes=[keep[i][0] for i in order],
                                         mu=[float(sc[i]) for i in order], n_raw=len(rows), tried=n_used,
                                         valid=len(keep) / max(n_used, 1), pool=len(keep), kept=len(idx), seconds=secs)
                print(f"  {tag:20s} {len(order)} of {len(idx)} (pool {len(keep)} valid of {n_used} raw, {len(keep)/max(n_used,1):.0%}) "
                      f"· ridge mean pool {sc.mean():.3f} -> picked {np.mean(out['cells'][tag]['mu']):.3f}", flush=True)
            _save(out)
    finally:
        val.close()


# ---------------------------------------------------------------- battle and report (foundation.py's protocol)
def battle():
    LB._stack()
    con = MDB.connect(); out = _load()
    todo = [t for t, c in out["cells"].items() if "y" not in c and c["pastes"]]
    for k in range(0, len(todo), 4):
        chunk = todo[k:k + 4]
        pastes = [p for t in chunk for p in out["cells"][t]["pastes"]]
        origins = [t for t in chunk for _ in out["cells"][t]["pastes"]]
        v = MDB.Validator()
        try:
            ids = [MDB.add_team(con, p, f"textdiff:{o}", v)[0] for p, o in zip(pastes, origins)]
        finally:
            v.close()
        legal = {t for (t,) in con.execute(f"SELECT team_id FROM team WHERE legal=1 AND team_id IN ({','.join('?' * len(ids))})", ids)}
        MDB.ensure(con, LB.POLICY, sorted(legal), MDB.set_members(con, "top50"), LB.PER_CELL, seed=SEED,
                   note=f"textdiff {chunk[0]}..{chunk[-1]}")
        i = 0
        for t in chunk:
            n = len(out["cells"][t]["pastes"])
            sc = [MDB.score(con, LB.POLICY, tid) if tid in legal else None for tid in ids[i:i + n]]
            out["cells"][t].update(team_id=ids[i:i + n], y=[s["score"] if s else None for s in sc],
                                   battles=[s["battles"] if s else 0 for s in sc])
            i += n
        _save(out)
        print(f"  battled {chunk}", flush=True)


def _stats(y):
    y = np.array([v for v in y if v is not None], float)
    return float(y.mean()), float(y.std(ddof=1) / math.sqrt(len(y))), len(y)


def report():
    LB._stack()
    con = MDB.connect(); out = _load()
    corpus, V, *_ = X.setup(); d = L.load_data()
    known = {tuple(V.decode_field(c, int(r[c])) for c in range(L.COLS)) for r in d["X"]}
    corpus_species = set(V.itos["species"])
    summ = {}
    for tag, c in out["cells"].items():
        if "team_id" not in c: continue
        sc = [MDB.score(con, LB.POLICY, t) for t in c["team_id"]]
        c["y"] = [s["score"] if s and s["score"] is not None else None for s in sc]
        m, se, n = _stats(c["y"])
        teams = [parse_team_text(p) for p in c["pastes"]]
        y = np.array([v for v in c["y"] if v is not None])
        summ[tag] = dict(model=c["model"], arm=c["arm"], got=m, se=se, n=n, valid=c["valid"], tried=c["tried"],
                         p90=float(np.quantile(y, .9)), best=float(y.max()), above_531=int((y >= 0.531).sum()),
                         sets=len({tuple(sorted(norm(s["species"]) for s in t)) for t in teams}),
                         copies=sum(tuple(team_fields(t)) in known for t in teams),
                         new_species_teams=sum(any(norm(s["species"]) not in corpus_species for s in t) for t in teams),
                         seconds=c.get("seconds"))
    out["summary"] = summ; _save(out)
    print(f"{'cell':18s} {'win rate':>14s} {'p90':>6s} {'best':>6s} {'≥.531':>6s} {'valid':>7s} {'sets':>5s} {'copies':>6s} {'new-sp':>6s}")
    for tag, v in summ.items():
        print(f"{tag:18s} {v['got']:.3f} ± {v['se']:.3f}  {v['p90']:.3f} {v['best']:.3f} {v['above_531']:>4d}/{v['n']:<2d} "
              f"{v['valid']:6.1%} {v['sets']:5d} {v['copies']:6d} {v['new_species_teams']:6d}")
    base = summ.get("scratch:base"); cfgc = summ.get("scratch:cfg")
    v3 = json.load(open(L.RES / "foundation_v3.json"))["summary"].get("v3:stack")      # 0.556 ± 0.012, foundation-swap v3
    for tag, v in summ.items():
        if v["arm"].startswith("stack:"):
            for name, ref in (("v3:stack", v3), ("text:ask50_g2", summ.get("text:ask50_g2")), ("text:stack", summ.get("text:stack"))):
                if ref and ref is not v:
                    dlt = v["got"] - ref["got"]; sd = math.hypot(v["se"], ref["se"])
                    print(f"  {tag} − {name}: {dlt:+.3f} ({dlt/sd:+.1f}σ)")
            pool = out.get("pools", {}).get(v["arm"].split(":", 1)[1], {})
            if pool: print(f"    pool: {pool['size']} valid of {pool['tried']} raw ({pool['valid']:.0%}), {pool['sets']} species sets, "
                           f"protect3 pass {pool['protect3_pass']}, ridge mean {pool['mu_pool']:.3f}, {pool['seconds_per_valid']:.1f} s / valid team")
            continue
        if v["model"] != "text": continue
        ref = cfgc if v["arm"].startswith("ask") else base
        if ref:
            dlt = v["got"] - ref["got"]; sd = math.hypot(v["se"], ref["se"])
            print(f"  {tag} − scratch:{ref['arm']}: {dlt:+.3f} ({dlt/sd:+.1f}σ)")


# ---------------------------------------------------------------- the Ablation-note metrics: fresh 192-battle finals
FINAL_N, FINAL_BATTLES = 16, 192


def final():
    """The 16 best distinct teams by 49-battle label, per model (text: all fine-tuned arms pooled; scratch: base + cfg),
    re-battled fresh with 192 battles each against the top-50 schedule (cem_v3.final's protocol), then once against
    ranks 51-100 (heldout.evaluate). Reports top-16 fresh mean, best team, winner's curse."""
    import pool, activesearch as A, heldout as HO
    out = _load()
    if "final" in out and all(m in out["final"] for m in ("text", "scratch")): return report_final(out)
    LB._stack()
    out.setdefault("final", {})
    for model in ("text", "scratch"):
        if model in out["final"]: continue
        rows = []
        for tag, c in out["cells"].items():
            if c["model"] != model or "y" not in c: continue
            rows += [dict(paste=p, label=y, cell=tag) for p, y in zip(c["pastes"], c["y"]) if y is not None]
        rows.sort(key=lambda r: -r["label"])
        seen, top = set(), []
        for r in rows:
            k = tuple(team_fields(parse_team_text(r["paste"])))
            if k in seen: continue
            seen.add(k); top.append(r)
            if len(top) >= FINAL_N: break
        d = Path(f"/tmp/vgc-pilot/textdiff_final_{model}"); d.mkdir(parents=True, exist_ok=True); files = []
        for i, r in enumerate(top):
            f = d / f"{i:02d}.txt"; f.write_text(r["paste"]); files.append(str(f))
        res = pool.score([(f, A.opponents()) for f in files], battles=FINAL_BATTLES, conc=50)
        held = HO.evaluate(files, battles=FINAL_BATTLES, conc=50)
        out["final"][model] = [dict(paste=r["paste"], label=r["label"], cell=r["cell"], fresh=res.get(f, {}).get("win_rate"),
                                    heldout=held.get(f, {}).get("win_rate"), battles=res.get(f, {}).get("battles"),
                                    candidates=len(rows)) for r, f in zip(top, files)]
        _save(out)
    report_final(out)


def report_final(out):
    print(f"{'model':8s} {'cands':>6s} {'label mean':>10s} {'fresh top-16':>14s} {'best fresh':>10s} {'≥.531':>6s} {'vs 51-100':>10s} {'sets':>5s}")
    for model, rows in out["final"].items():
        fr = np.array([r["fresh"] for r in rows if r["fresh"] is not None]); he = np.array([r["heldout"] for r in rows if r["heldout"] is not None])
        teams = [parse_team_text(r["paste"]) for r in rows]
        sets = len({tuple(sorted(norm(s["species"]) for s in t)) for t in teams})
        print(f"{model:8s} {rows[0]['candidates']:6d} {np.mean([r['label'] for r in rows]):10.3f} "
              f"{fr.mean():.3f} ± {fr.std(ddof=1)/math.sqrt(len(fr)):.3f}  {fr.max():10.3f} {(fr >= 0.531).sum():3d}/{len(fr):<2d} "
              f"{he.mean():10.3f} {sets:5d}")
        best = rows[int(fr.argmax())]
        print("   best:", ", ".join(s["species"] for s in parse_team_text(best["paste"])), f"(label {best['label']:.3f}, {best['cell']})")


ONLY = set(sys.argv[2:])            # decode <arm...>: restrict to finished arms

if __name__ == "__main__":
    globals()[sys.argv[1]]()
