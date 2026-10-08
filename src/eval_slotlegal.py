"""Per-slot legality guidance: per-field diagnostic, then the five arms."""
import json, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, "/tmp/vgc-pilot/src")
from corpus import norm, STATS
from encode import NSLOT
import slotlegal as SLm, spreaddiffusion as S
from propose import Validator, slot_to_text
import pool

def diag(m, C, sleg, g_leg, n=64):
    X = SLm.sample(m, C, n, wbin=5, g_wr=2.0, sleg=sleg, g_leg=g_leg, constrained=False).cpu().numpy()
    mv_ok = ab_ok = mv_tot = ab_tot = 0; full = 0
    for row in X:
        g = row.reshape(NSLOT, S.NF2); ok_team = True
        bases = [C.base_of[int(s[0])] for s in g]; items = [int(s[2]) for s in g if int(s[2]) > 0]
        ok_team &= len(bases) == len(set(bases)) and len(items) == len(set(items))
        for s in g:
            si = int(s[0])
            if si in C.mv_ok:
                for j in range(3, 7):
                    v = int(s[j]); mv_tot += 1; o = bool(C.mv_ok[si][v]) if v > 0 else False; mv_ok += o; ok_team &= o
            if si in C.ab_ok:
                ab_tot += 1; o = bool(C.ab_ok[si][int(s[1])]) if int(s[1]) > 0 else False; ab_ok += o; ok_team &= o
            ok_team &= not (np.clip(s[8:] - 1, 0, None).sum() > S.BUDGET)
        full += ok_team
    return mv_ok / max(mv_tot, 1), ab_ok / max(ab_tot, 1), full / n

def main():
    per = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    m, V, L, corpus, teams = SLm.load_model(); C = S.SpreadConstraints(V, L); torch.manual_seed(1)
    print(f"{'condition (unconstrained)':28s} {'move legal':>11s} {'ability ok':>11s} {'whole team':>11s}")
    for name, sl, gl in [("no legality token", SLm.SL_NULL, 0.0), ("all slots legal=1, g=2", SLm.SL_LEGAL, 2.0),
                         ("all slots legal=1, g=4", SLm.SL_LEGAL, 4.0), ("all slots illegal=1, g=2", SLm.SL_ILLEGAL, 2.0)]:
        r = diag(m, C, sl, gl); print(f"{name:28s} {r[0]:11.3f} {r[1]:11.3f} {r[2]:11.3f}", flush=True)
    look = {}
    for t in corpus + teams:
        for s in t:
            look[norm(s["species"])] = s["species"]; look[norm(s["ability"])] = s["ability"]
            if s["item"]: look[norm(s["item"])] = s["item"]
            for mv in s["moves"]: look[norm(mv)] = mv
            if s["nature"]: look[norm(s["nature"])] = s["nature"]
    ARMS = [("A_constrained",  dict(constrained=True,  sleg=SLm.SL_NULL,  g_leg=0.0)),
            ("B_constr+slot",  dict(constrained=True,  sleg=SLm.SL_LEGAL, g_leg=2.0)),
            ("C2_slot_only",   dict(constrained=False, sleg=SLm.SL_LEGAL, g_leg=2.0)),
            ("C4_slot_only",   dict(constrained=False, sleg=SLm.SL_LEGAL, g_leg=4.0)),
            ("D_neither",      dict(constrained=False, sleg=SLm.SL_NULL,  g_leg=0.0))]
    torch.manual_seed(0); out = Path("/tmp/vgc-pilot/slotlegal_gen"); out.mkdir(exist_ok=True)
    for f in out.glob("*.txt"): f.unlink()
    val = Validator(); meta = {}; stats = {}
    for tag, kw in ARMS:
        kept = tries = rule_ok = 0; t0 = time.perf_counter()
        while kept < per and tries < per * 4:
            n = min(24, per * 4 - tries); tries += n
            X = SLm.sample(m, C, n, wbin=5, g_wr=2.0, **kw).cpu().numpy()
            for row in X:
                slots = S.row_to_slots(V, row, look)
                cat = [V.decode_field(c, int(row[c])) for c in range(S.COLS2) if c % S.NF2 < 8]; rule_ok += int(L.legal(cat))
                txt = "\n\n".join(slot_to_text(s) for s in slots) + "\n"
                if kept < per and val(txt) is None:
                    p = out / f"{tag}_{kept:04d}.txt"; p.write_text(txt); meta[str(p)] = tag; kept += 1
        stats[tag] = dict(showdown_valid=kept / max(tries, 1), rule_valid=rule_ok / max(tries, 1))
        print(f"  {tag:14s} rule-legal {stats[tag]['rule_valid']:4.0%} · Showdown-valid {kept}/{tries} ({stats[tag]['showdown_valid']:.0%}) · {time.perf_counter()-t0:.0f}s", flush=True)
    val.close()
    root = Path("/tmp/vgc-pilot/vgc-bench/teams/reg_mb"); opp = []
    for i in [t["id"] for t in json.load(open("/tmp/vgc-pilot/top50_evs.json"))]:
        for c in (root / f"{i}.txt", root / "featured" / f"{i}.txt"):
            if c.exists(): opp.append(str(c)); break
    jobs = [(t, opp) for t in sorted(meta)]
    print(f"\nbattling {len(jobs)} teams x 24 vs {len(opp)} meta teams", flush=True)
    res = pool.score(jobs, battles=24, conc=50) if jobs else {}
    for t, v in res.items(): v["arm"] = meta[t]
    json.dump({"results": res, "stats": stats}, open("/tmp/vgc-pilot/slotlegal_eval.json", "w"), indent=1)
    print(f"\n{'arm':14s} {'rule':>5s} {'valid':>6s} {'n':>4s} {'win rate':>9s} {'clustered SE':>13s} {'p90':>6s} {'max':>6s}")
    for tag, _ in ARMS:
        w = np.array([v["win_rate"] for v in res.values() if v["arm"] == tag]); st = stats[tag]
        if len(w): print(f"{tag:14s} {st['rule_valid']:5.0%} {st['showdown_valid']:6.0%} {len(w):4d} {w.mean():9.4f} {w.std(ddof=1)/np.sqrt(len(w)):13.4f} {np.percentile(w,90):6.3f} {w.max():6.3f}")
        else: print(f"{tag:14s} {st['rule_valid']:5.0%} {st['showdown_valid']:6.0%}    0   (nothing valid to battle)")

if __name__ == "__main__":
    main()
