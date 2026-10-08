"""Paired test: does GENERATING the Stat Point spread change win rate?

Same model, same sampled teams. For each generated team, build a twin whose six
spreads are replaced by borrowed corpus spreads of the same species (exactly what
the earlier models did). Both twins are Showdown-validated and battled against the
same opponent pool. The paired difference removes team-composition variance, which
is the dominant noise source, so this resolves far smaller effects than two
independent arms can.
"""
import json, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, "/tmp/vgc-pilot/src")
from corpus import norm, STATS
import legaldiffusion as LD, spreaddiffusion as S
from propose import Validator, slot_to_text
import pool

def main():
    n_pairs = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    m, V, L, corpus, teams = LD.load_model(); C = S.SpreadConstraints(V, L)
    look, spreads = {}, defaultdict(list)
    for t in corpus:
        for s in t: spreads[norm(s["species"])].append(dict(s["evs"]))
    for t in corpus + teams:
        for s in t:
            look[norm(s["species"])] = s["species"]; look[norm(s["ability"])] = s["ability"]
            if s["item"]: look[norm(s["item"])] = s["item"]
            for mv in s["moves"]: look[norm(mv)] = mv
            if s["nature"]: look[norm(s["nature"])] = s["nature"]
    rng = np.random.default_rng(0); torch.manual_seed(0)
    out = Path("/tmp/vgc-pilot/paired_gen"); out.mkdir(exist_ok=True)
    for f in out.glob("*.txt"): f.unlink()
    val = Validator(); pairs = []; tries = 0
    while len(pairs) < n_pairs and tries < n_pairs * 4:
        X = LD.sample(m, C, 24, wbin=5, g_wr=2.0, leg=LD.LEG_NULL, g_leg=0.0, constrained=True).cpu().numpy(); tries += 24
        for row in X:
            gen = S.row_to_slots(V, row, look)
            bor = [dict(s) for s in gen]
            for s in bor:
                pool_ = spreads.get(norm(s["species"]))
                s["evs"] = dict(pool_[int(rng.integers(0, len(pool_)))]) if pool_ else s["evs"]
            tg = "\n\n".join(slot_to_text(s) for s in gen) + "\n"; tb = "\n\n".join(slot_to_text(s) for s in bor) + "\n"
            if len(pairs) < n_pairs and val(tg) is None and val(tb) is None:
                i = len(pairs); pg = out / f"gen_{i:04d}.txt"; pb = out / f"bor_{i:04d}.txt"
                pg.write_text(tg); pb.write_text(tb); pairs.append((str(pg), str(pb)))
    val.close()
    print(f"{len(pairs)} valid pairs from {tries} samples", flush=True)
    root = Path("/tmp/vgc-pilot/vgc-bench/teams/reg_mb"); opp = []
    for i in [t["id"] for t in json.load(open("/tmp/vgc-pilot/top50_evs.json"))]:
        for c in (root / f"{i}.txt", root / "featured" / f"{i}.txt"):
            if c.exists(): opp.append(str(c)); break
    files = [p for pr in pairs for p in pr]
    print(f"battling {len(files)} teams x 24 vs {len(opp)}", flush=True)
    res = pool.score([(f, opp) for f in files], battles=24, conc=50)
    g = np.array([res[a]["win_rate"] for a, b in pairs]); b = np.array([res[bb]["win_rate"] for a, bb in pairs])
    d = g - b
    json.dump({"pairs": pairs, "results": res}, open("/tmp/vgc-pilot/paired_eval.json", "w"), indent=1)
    print(f"\ngenerated spreads : {g.mean():.4f} ± {g.std(ddof=1)/np.sqrt(len(g)):.4f}")
    print(f"borrowed spreads  : {b.mean():.4f} ± {b.std(ddof=1)/np.sqrt(len(b)):.4f}")
    print(f"paired difference : {d.mean():+.4f} ± {d.std(ddof=1)/np.sqrt(len(d)):.4f}  ({d.mean()/(d.std(ddof=1)/np.sqrt(len(d))):+.1f} σ, n={len(d)} pairs)")
    print(f"pairs where generated > borrowed: {(d>0).sum()}   <: {(d<0).sum()}   =: {(d==0).sum()}")

if __name__ == "__main__":
    main()
