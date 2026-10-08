"""Do generated Stat Point spreads and legality-classifier guidance change anything?

Four arms, all conditioned on the top win-rate bin at guidance 2 so the only
things that vary are the two upgrades:
  A  constrained decoding, generated spreads, no classifier      (spread upgrade alone)
  B  constrained decoding + classifier guidance                  (both upgrades)
  C  classifier guidance only, NO rule constraints               (does the classifier alone reach legality?)
  D  neither                                                     (control: expected ~0% legal)
Reference from yesterday, same bin/guidance, spreads BORROWED: 0.168 win rate, 77% Showdown-valid.
Every kept team is Showdown-validated, then battled against the top-50 pool. Errors clustered by team.
"""
import json, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, "/tmp/vgc-pilot/src")
from corpus import norm
import spreaddiffusion as S, legalcls as LC
from propose import Validator, slot_to_text
import pool

def main():
    per = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    m, V, L, corpus, teams = S.load_model(); C = S.SpreadConstraints(V, L); cls = LC.load(V)
    look = {}
    for t in corpus + teams:
        for s in t:
            look[norm(s["species"])] = s["species"]; look[norm(s["ability"])] = s["ability"]
            if s["item"]: look[norm(s["item"])] = s["item"]
            for mv in s["moves"]: look[norm(mv)] = mv
            if s["nature"]: look[norm(s["nature"])] = s["nature"]
    ARMS = [("A_constrained", dict(constrained=True,  cls=None, cls_scale=0.0)),
            ("B_constr+cls",  dict(constrained=True,  cls=cls,  cls_scale=1.0)),
            ("C_cls_only",    dict(constrained=False, cls=cls,  cls_scale=1.0)),
            ("D_neither",     dict(constrained=False, cls=None, cls_scale=0.0))]
    torch.manual_seed(0)
    out = Path("/tmp/vgc-pilot/spread_gen"); out.mkdir(exist_ok=True)
    for f in out.glob("*.txt"): f.unlink()
    val = Validator(); meta = {}; stats = {}
    for tag, kw in ARMS:
        kept = tries = 0; novel_spread = 0; t0 = time.perf_counter()
        corpus_spreads = {tuple(s["evs"].get(k, 0) for k in S.STATS) for t in corpus for s in t}
        while kept < per and tries < per * 4:
            n = min(24, per * 4 - tries); tries += n
            X = S.sample(m, C, n, 5, "none", 2.0, **kw).cpu().numpy()
            for row in X:
                slots = S.row_to_slots(V, row, look)
                txt = "\n\n".join(slot_to_text(s) for s in slots) + "\n"
                if val(txt) is None and kept < per:
                    p = out / f"{tag}_{kept:04d}.txt"; p.write_text(txt); meta[str(p)] = tag; kept += 1
                    novel_spread += sum(tuple(s["evs"][k] for k in S.STATS) not in corpus_spreads for s in slots)
        stats[tag] = dict(valid=kept / max(tries, 1), novel_spread_frac=novel_spread / max(kept * 6, 1), gen_s=time.perf_counter() - t0)
        print(f"  {tag:14s} Showdown-valid {kept}/{tries} ({stats[tag]['valid']:.0%}) · novel spreads {stats[tag]['novel_spread_frac']:.0%} · {stats[tag]['gen_s']:.0f}s", flush=True)
    val.close()
    root = Path("/tmp/vgc-pilot/vgc-bench/teams/reg_mb"); opp = []
    for i in [t["id"] for t in json.load(open("/tmp/vgc-pilot/top50_evs.json"))]:
        for c in (root / f"{i}.txt", root / "featured" / f"{i}.txt"):
            if c.exists(): opp.append(str(c)); break
    jobs = [(t, opp) for t in sorted(meta)]
    print(f"\nbattling {len(jobs)} teams x 24 vs {len(opp)} meta teams", flush=True)
    res = pool.score(jobs, battles=24, conc=50)
    for t, v in res.items(): v["arm"] = meta[t]
    json.dump({"results": res, "stats": stats}, open("/tmp/vgc-pilot/spread_eval.json", "w"), indent=1)
    print(f"\n{'arm':14s} {'valid':>6s} {'n':>4s} {'win rate':>9s} {'clustered SE':>13s} {'p90':>6s} {'max':>6s}")
    for tag, _ in ARMS:
        w = np.array([v["win_rate"] for v in res.values() if v["arm"] == tag])
        if len(w): print(f"{tag:14s} {stats[tag]['valid']:6.0%} {len(w):4d} {w.mean():9.4f} {w.std(ddof=1)/np.sqrt(len(w)):13.4f} {np.percentile(w,90):6.3f} {w.max():6.3f}")
        else: print(f"{tag:14s} {stats[tag]['valid']:6.0%}    0   (nothing valid to battle)")
    print("reference  bin5 g2, borrowed spreads (2026-08-27): 77% valid, 0.168 ± 0.011 · real team 0.470 · unconditioned 0.128")

if __name__ == "__main__":
    main()
