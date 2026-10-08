"""Do generated Stat Point spreads and classifier-FREE legality guidance change anything?

All arms condition on win-rate bin 5 at guidance 2 (yesterday's reference cell:
0.168 win rate, 77% Showdown-valid, spreads borrowed). What varies:
  A  constrained decoding, spreads generated, no legality term       -> the spread upgrade alone
  B  constrained decoding + legality guidance (legal=1, g=2)         -> both upgrades
  C2 legality guidance g=2, NO rule constraints                      -> can CFG alone make legal teams?
  C4 legality guidance g=4, NO rule constraints
  D  neither (control; expected ~0% Showdown-valid)
Every kept team is Showdown-validated, then battled vs the top-50 pool. Errors clustered by team.
"""
import json, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, "/tmp/vgc-pilot/src")
from corpus import norm, STATS
import legaldiffusion as LD, spreaddiffusion as S
from propose import Validator, slot_to_text
import pool

def main():
    per = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    m, V, L, corpus, teams = LD.load_model(); C = S.SpreadConstraints(V, L)
    look = {}
    for t in corpus + teams:
        for s in t:
            look[norm(s["species"])] = s["species"]; look[norm(s["ability"])] = s["ability"]
            if s["item"]: look[norm(s["item"])] = s["item"]
            for mv in s["moves"]: look[norm(mv)] = mv
            if s["nature"]: look[norm(s["nature"])] = s["nature"]
    corpus_spreads = {tuple(s["evs"].get(k, 0) for k in STATS) for t in corpus for s in t}
    ARMS = [("A_constrained",   dict(constrained=True,  leg=LD.LEG_NULL,  g_leg=0.0)),
            ("B_constr+legCFG", dict(constrained=True,  leg=LD.LEG_LEGAL, g_leg=2.0)),
            ("C2_legCFG_only",  dict(constrained=False, leg=LD.LEG_LEGAL, g_leg=2.0)),
            ("C4_legCFG_only",  dict(constrained=False, leg=LD.LEG_LEGAL, g_leg=4.0)),
            ("D_neither",       dict(constrained=False, leg=LD.LEG_NULL,  g_leg=0.0))]
    torch.manual_seed(0)
    out = Path("/tmp/vgc-pilot/legal_gen"); out.mkdir(exist_ok=True)
    for f in out.glob("*.txt"): f.unlink()
    val = Validator(); meta = {}; stats = {}
    for tag, kw in ARMS:
        kept = tries = rule_ok = novel = 0; t0 = time.perf_counter()
        while kept < per and tries < per * 4:
            n = min(24, per * 4 - tries); tries += n
            X = LD.sample(m, C, n, wbin=5, g_wr=2.0, **kw).cpu().numpy()
            for row in X:
                slots = S.row_to_slots(V, row, look)
                cat = [V.decode_field(c, int(row[c])) for c in range(S.COLS2) if c % S.NF2 < 8]
                rule_ok += int(L.legal(cat))
                txt = "\n\n".join(slot_to_text(s) for s in slots) + "\n"
                if kept < per and val(txt) is None:
                    p = out / f"{tag}_{kept:04d}.txt"; p.write_text(txt); meta[str(p)] = tag; kept += 1
                    novel += sum(tuple(s["evs"][k] for k in STATS) not in corpus_spreads for s in slots)
        stats[tag] = dict(showdown_valid=kept / max(tries, 1), rule_valid=rule_ok / max(tries, 1),
                          novel_spread_frac=novel / max(kept * 6, 1), gen_s=time.perf_counter() - t0)
        print(f"  {tag:16s} rule-legal {stats[tag]['rule_valid']:4.0%} · Showdown-valid {kept}/{tries} ({stats[tag]['showdown_valid']:.0%}) · novel spreads {stats[tag]['novel_spread_frac']:.0%} · {stats[tag]['gen_s']:.0f}s", flush=True)
    val.close()
    root = Path("/tmp/vgc-pilot/vgc-bench/teams/reg_mb"); opp = []
    for i in [t["id"] for t in json.load(open("/tmp/vgc-pilot/top50_evs.json"))]:
        for c in (root / f"{i}.txt", root / "featured" / f"{i}.txt"):
            if c.exists(): opp.append(str(c)); break
    jobs = [(t, opp) for t in sorted(meta)]
    print(f"\nbattling {len(jobs)} teams x 24 vs {len(opp)} meta teams", flush=True)
    res = pool.score(jobs, battles=24, conc=50) if jobs else {}
    for t, v in res.items(): v["arm"] = meta[t]
    json.dump({"results": res, "stats": stats}, open("/tmp/vgc-pilot/legal_eval.json", "w"), indent=1)
    print(f"\n{'arm':16s} {'rule':>5s} {'valid':>6s} {'n':>4s} {'win rate':>9s} {'clustered SE':>13s} {'p90':>6s} {'max':>6s}")
    for tag, _ in ARMS:
        w = np.array([v["win_rate"] for v in res.values() if v["arm"] == tag]); st = stats[tag]
        if len(w): print(f"{tag:16s} {st['rule_valid']:5.0%} {st['showdown_valid']:6.0%} {len(w):4d} {w.mean():9.4f} {w.std(ddof=1)/np.sqrt(len(w)):13.4f} {np.percentile(w,90):6.3f} {w.max():6.3f}")
        else: print(f"{tag:16s} {st['rule_valid']:5.0%} {st['showdown_valid']:6.0%}    0   (nothing valid to battle)")
    print("reference  bin5 g2, borrowed spreads, 48-col model: 77% valid, 0.168 ± 0.011 · real team 0.470 · unconditioned 0.128")

if __name__ == "__main__":
    main()
