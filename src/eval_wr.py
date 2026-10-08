"""Does conditioning on win rate produce teams that WIN more?

For each (bin, guidance) cell: sample teams from the win-rate-conditioned model,
keep the ones Showdown accepts, battle them against the top-50 pool, and compare
the measured win rate with the unconditioned generator (invent, 0.128) and with
real teams (0.470). Errors are clustered by team.

This is the test of the research goal's §1.7.1 idea. Expectation, stated before
the run: DDOM-style conditioning loses to gradient ascent on its own small-data
benchmarks, and the top bin here holds 965 teams of which only 269 exceed 0.6, so
the honest prior is a modest lift over 0.128, not a jump to 0.47.
"""
import json, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, "/tmp/vgc-pilot/src")
from corpus import norm, STATS
from encode import NF, NSLOT
import diffusion as D, wrdiffusion as W
from propose import Validator, slot_to_text
import pool

def main():
    per = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    cells = [(5, 1.0), (5, 2.0), (5, 4.0), (0, 2.0)]     # top bin at three strengths, bottom bin as a control
    m, V, L, corpus, teams = W.load_model(); C = D.Constraints(V, L)
    look, spreads = {}, defaultdict(list)
    for t in corpus + teams:
        for s in t:
            look[norm(s["species"])] = s["species"]; look[norm(s["ability"])] = s["ability"]
            if s["item"]: look[norm(s["item"])] = s["item"]
            for mv in s["moves"]: look[norm(mv)] = mv
            if s["nature"]: look[norm(s["nature"])] = s["nature"]
            spreads[norm(s["species"])].append(dict(s["evs"]))
    rng = np.random.default_rng(0); torch.manual_seed(0)
    out = Path("/tmp/vgc-pilot/wr_gen"); out.mkdir(exist_ok=True)
    for f in out.glob("*.txt"): f.unlink()
    val = Validator(); meta = {}; acc = {}
    for wbin, g in cells:
        tag = f"bin{wbin}_g{g:g}"; kept = tries = 0
        while kept < per and tries < per * 3:
            tries += 1
            row = W.sample_constrained(m, C, 1, wbin, "none", g).cpu().numpy()[0]
            slots = []
            for i in range(NSLOT):
                b = i * NF; gf = lambda j: V.decode_field(b + j, int(row[b + j]))
                sp = gf(0); mv = [look.get(gf(3+j), gf(3+j)) for j in range(4) if gf(3+j) and gf(3+j) != "[MASK]"]
                pl = spreads.get(sp) or [{s: 0 for s in STATS}]
                slots.append(dict(species=look.get(sp, sp), item=look.get(gf(2), gf(2)), ability=look.get(gf(1), gf(1)),
                                  nature=look.get(gf(7), gf(7)), moves=mv, evs=pl[int(rng.integers(0, len(pl)))]))
            txt = "\n\n".join(slot_to_text(s) for s in slots) + "\n"
            if val(txt) is None:
                p = out / f"{tag}_{kept:04d}.txt"; p.write_text(txt); meta[str(p)] = tag; kept += 1
        acc[tag] = kept / max(tries, 1)
        print(f"  {tag}: {kept} Showdown-valid of {tries} ({acc[tag]:.0%})", flush=True)
    val.close()
    root = Path("/tmp/vgc-pilot/vgc-bench/teams/reg_mb"); opp = []
    for i in [t["id"] for t in json.load(open("/tmp/vgc-pilot/top50_evs.json"))]:
        for c in (root / f"{i}.txt", root / "featured" / f"{i}.txt"):
            if c.exists(): opp.append(str(c)); break
    print(f"\nbattling {len(meta)} teams x 24 vs {len(opp)} meta teams", flush=True)
    t0 = time.perf_counter()
    res = pool.score([(t, opp) for t in sorted(meta)], battles=24, conc=50)
    for t, v in res.items(): v["cell"] = meta[t]
    json.dump({"results": res, "acceptance": acc}, open("/tmp/vgc-pilot/wr_eval.json", "w"), indent=1)
    print(f"wall {time.perf_counter()-t0:.0f}s\n")
    print(f"{'cell':12s} {'n':>4s} {'win rate':>9s} {'clustered SE':>13s} {'p90':>6s} {'max':>6s}   reference")
    ref = {"real": 0.470, "invent (unconditioned)": 0.128, "random": 0.014}
    for wbin, g in cells:
        tag = f"bin{wbin}_g{g:g}"; w = np.array([v["win_rate"] for v in res.values() if v["cell"] == tag])
        if len(w): print(f"{tag:12s} {len(w):4d} {w.mean():9.4f} {w.std(ddof=1)/np.sqrt(len(w)):13.4f} "
                         f"{np.percentile(w,90):6.3f} {w.max():6.3f}")
    for k, v in ref.items(): print(f"  {k:24s} {v:.3f}")

if __name__ == "__main__":
    main()
