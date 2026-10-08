"""Battle-evaluate the HPS-trained diffusion model against the top-50 meta pool.

Arms (each 32 teams x 24 battles, BC policy both sides, same protocol as the
2026-08-27 status table):
  real      - 32 corpus teams drawn at random (re-measured baseline)
  uncond    - model samples, no conditioning
  bin5_gG   - model samples conditioned on the top win-rate bin, guidance G

The hps_random arm needs no battles: the 2,000-team labelling campaign already
measured the dataset itself.

Every generated team passes Showdown's validator before it battles; validity
rates are reported alongside win rates.
"""
import json, os, random, sys
from pathlib import Path
import numpy as np, torch
import hpsdiffusion as H
import diffusion as D
from corpus import norm, STATS
from hps_generate import Validator, slot_to_text
import top50

OUTDIR = Path("/tmp/vgc-pilot/hps_eval")
RESULTS = "/Users/ramiismael/Documents/code/vgc-team-generator-pilot/results/hps_eval.json"

def build_decode_tables(teams):
    look, spreads = {}, {}
    for t in teams:
        for s in t:
            sp = norm(s["species"])
            look[sp] = s["species"]; look[norm(s["ability"])] = s["ability"]
            if s["item"]: look[norm(s["item"])] = s["item"]
            for m in s["moves"]: look[norm(m)] = m
            if s["nature"]: look[norm(s["nature"])] = s["nature"]
            spreads.setdefault(sp, []).append(dict(s["evs"]))
    return look, spreads

def row_to_paste(V, row, look, spreads, rng):
    slots = []
    for i in range(H.NSLOT):
        b = i * H.NF
        g = lambda j: V.decode_field(b + j, int(row[b + j]))
        sp = g(0)
        mv = [look.get(g(3+j), g(3+j)) for j in range(4) if g(3+j) and g(3+j) != "[MASK]"]
        pool_ = spreads.get(sp) or [{s: 0 for s in STATS}]
        slots.append(dict(species=look.get(sp, sp), item=look.get(g(2), g(2)),
                          ability=look.get(g(1), g(1)), nature=look.get(g(7), g(7)),
                          moves=mv, evs=pool_[rng.integers(0, len(pool_))]))
    return "\n\n".join(slot_to_text(s) for s in slots) + "\n"

def make_arm(name, model, C, V, look, spreads, wbin, guidance, n, val, rng):
    d = OUTDIR / name
    d.mkdir(parents=True, exist_ok=True)
    for f in d.glob("*.txt"): f.unlink()
    kept, attempts = [], 0
    while len(kept) < n and attempts < n * 6:
        want = min(2 * (n - len(kept)), 64)
        x = H.sample_constrained(model, C, want, wbin, guidance).cpu().numpy()
        for i in range(want):
            attempts += 1
            txt = row_to_paste(V, x[i], look, spreads, rng)
            if val(txt) is None:
                p = d / f"{name}_{len(kept):03d}.txt"
                p.write_text(txt); kept.append(str(p))
                if len(kept) >= n: break
    print(f"  {name}: {len(kept)}/{n} teams, validity {len(kept)/max(attempts,1):.0%} "
          f"({attempts} sampled)", flush=True)
    return kept, len(kept) / max(attempts, 1)

def main():
    n = 32; battles = 24
    rng = np.random.default_rng(0); torch.manual_seed(0); random.seed(0)
    teams, W = H.load_hps()
    V, L = H.build_vocab(teams)
    look, spreads = build_decode_tables(teams)
    model = H.load_model(V)
    C = D.Constraints(V, L)
    val = Validator()

    opp = top50.files()                                  # all 50 incl. featured/; raises if one is missing

    corpus_files = sorted(Path("/tmp/vgc-pilot/vgc-bench/teams/reg_mb").glob("MB*.txt"))
    real = [str(p) for p in random.sample(corpus_files, n)]

    arms, validity = {"real": real}, {}
    for name, wbin, g in [("uncond", H.WNULL, 1.0), ("top_g1", H.TOPBIN, 1.0),
                          ("top_g2", H.TOPBIN, 2.0), ("top_g4", H.TOPBIN, 4.0)]:
        arms[name], validity[name] = make_arm(name, model, C, V, look, spreads,
                                              wbin, g, n, val, rng)
    val.close()

    sys.path.insert(0, "/tmp/vgc-pilot/src")
    import pool
    jobs = [(t, opp) for a in arms.values() for t in a]
    print(f"battling {len(jobs)} teams x {battles} vs {len(opp)} meta opponents", flush=True)
    res = pool.score(jobs, battles=battles, conc=50)

    lab = json.load(open(H.LABELS))
    hps_wr = [v["win_rate"] for v in lab.values()]
    summary = {"hps_random_labelled": {"mean_wr": float(np.mean(hps_wr)),
                                       "n_teams": len(hps_wr), "battles_each": 24}}
    for name, files in arms.items():
        rs = [res[f]["win_rate"] for f in files if f in res]
        ns = [res[f]["battles"] for f in files if f in res]
        mean = float(np.mean(rs)) if rs else None
        se = float(np.std(rs) / max(len(rs), 1) ** .5) if rs else None
        summary[name] = {"mean_wr": mean, "se_teams": se, "n_teams": len(rs),
                         "battles": int(np.sum(ns)) if ns else 0,
                         "validity": validity.get(name)}
        print(f"  {name:10s} mean win rate {mean if mean is None else round(mean,3)} "
              f"(se {se if se is None else round(se,3)}, {len(rs)} teams)", flush=True)
    json.dump({"summary": summary, "per_team": res}, open(RESULTS, "w"), indent=1)
    print("wrote", RESULTS)

if __name__ == "__main__":
    main()
