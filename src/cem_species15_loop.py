"""Species guidance 1.5 inside the cross-entropy loop: summary and the pre-registered rule (2026-10-16).
Pre-registration: docs/species-guidance-loop.md. The runs themselves are cem_v3.py under
CEM_TAG=sp15loop_<arm>_s<seed> CEM_SEED=<seed> [CEM_G_SPECIES=1.5]  (scripts/run_sp15loop.sh).

    python cem_species15_loop.py report        -> results/cem_v3_species15_loop.json
(run with /tmp/vgc-pilot/.venv/bin/python)
"""
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, "/tmp/vgc-pilot/src")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lossdown as L
from corpus import parse_team_text, norm

ARMS, SEEDS, CUT, BOUND, DRAWS = ["plain", "species15"], [4041, 4042, 4043], 0.531, -0.03, 4000
OUT, REF = L.RES / "cem_v3_species15_loop.json", L.RES / "cem_v3_noheldout.json"


def lineup(paste): return tuple(sorted(norm(s["species"]) for s in parse_team_text(paste)))


def load(arm, seed):
    p = L.RES / f"sp15loop_{arm}_s{seed}.json"
    return json.load(open(p)) if p.exists() else None


def gens(out):
    return [c for _, c in sorted(out["gens"].items(), key=lambda kv: int(kv[0])) if "stats" in c]


def summarise(out, rng):
    rows = []
    for g, c in enumerate(gens(out)):
        y = np.array([v for v in c["y"] if v is not None], float)
        b = [y[rng.integers(0, len(y), len(y))].mean() for _ in range(DRAWS)]
        s = c["stats"]; e = c.get("elites", {})
        rows.append(dict(gen=g, sets=c["sets"], vendi=c.get("vendi"), battled_sets=s["battled_sets"], elite_sets=e.get("sets"),
                         elites=e.get("n"), mean=s["mean"], ci95=[float(v) for v in np.quantile(b, [.025, .975])], p90=s["p90"],
                         max=s["max"], above_cut=s["above_cut"], copies=c["copies"], valid=c["valid"], new_species=c["new_species"]))
    r = dict(gens=rows, stopped=out.get("stopped"), n_gens=len(rows),
             sets_g0=rows[0]["sets"] if rows else None,
             sets_ge1=float(np.mean([x["sets"] for x in rows[1:]])) if len(rows) > 1 else None)
    if out.get("final"):
        f = [x for x in out["final"] if x["fresh"] is not None]
        r["final"] = dict(n=len(f), fresh=[x["fresh"] for x in f], fresh_mean=float(np.mean([x["fresh"] for x in f])),
                          fresh_best=float(max(x["fresh"] for x in f)), above_cut=int(sum(x["fresh"] >= CUT for x in f)),
                          label_mean=float(np.mean([x["label"] for x in f])), sets=len({lineup(x["paste"]) for x in f}))
        if out.get("heldout"):
            h = [x["heldout"] for x in out["heldout"] if x["heldout"] is not None]
            r["final"]["heldout_mean"] = float(np.mean(h)); r["final"]["heldout_best"] = float(max(h))
    return r


def report():
    rng = np.random.default_rng(0)
    runs = {f"{a}_s{s}": summarise(o, rng) for a in ARMS for s in SEEDS if (o := load(a, s))}
    ref = summarise(json.load(open(REF)), rng) if REF.exists() else None
    print(f"{'run':20s} {'gen':>3s} {'sets':>5s} {'vendi':>6s} {'bsets':>5s} {'esets':>5s} {'mean [95%]':>22s} {'max':>5s} {'≥cut':>4s}")
    for k, r in list(runs.items()) + ([("ref cem_v3_noheldout", ref)] if ref else []):
        for x in r["gens"]:
            v = f"{x['vendi']:6.1f}" if x["vendi"] is not None else "     -"
            es = f"{x['elite_sets']:5d}" if x["elite_sets"] is not None else "    -"
            print(f"{k:20s} {x['gen']:3d} {x['sets']:5d} {v} {x['battled_sets']:5d} {es} "
                  f"{x['mean']:.3f} [{x['ci95'][0]:.3f},{x['ci95'][1]:.3f}] {x['max']:5.3f} {x['above_cut']:4d}")
        if r["stopped"]: print(f"{'':20s} stopped g{r['stopped']['generation']}: {r['stopped']['reason']}")
        if "final" in r:
            f = r["final"]
            print(f"{'':20s} final {f['n']}: fresh {f['fresh_mean']:.3f} (labels {f['label_mean']:.3f}) · best {f['fresh_best']:.3f} · "
                  f"≥cut {f['above_cut']} · line-ups {f['sets']}" + (f" · vs 51-100 {f['heldout_mean']:.3f}" if "heldout_mean" in f else ""))
    # the rule
    verdict = dict(rule="(a) mean sets/512 over generations >= 1 higher than plain in every seed AND "
                        "(b) bootstrap 95% lower bound of mean over seeds of (species15 - plain) finalists' fresh win rate > -0.03")
    pairs = [(runs.get(f"species15_s{s}"), runs.get(f"plain_s{s}")) for s in SEEDS]
    if all(a and b and a["sets_ge1"] is not None and b["sets_ge1"] is not None for a, b in pairs):
        d_sets = [a["sets_ge1"] - b["sets_ge1"] for a, b in pairs]
        verdict.update(sets_ge1_species15=[a["sets_ge1"] for a, _ in pairs], sets_ge1_plain=[b["sets_ge1"] for _, b in pairs],
                       sets_ge1_diff=d_sets, sets_g0_diff=[a["sets_g0"] - b["sets_g0"] for a, b in pairs], clause_a=bool(all(d > 0 for d in d_sets)))
        print(f"\n(a) sets/512 at g>=1, species15 - plain by seed: {[round(d, 1) for d in d_sets]} -> {'holds' if verdict['clause_a'] else 'fails'}"
              f"   (g0: {verdict['sets_g0_diff']})")
    else:
        print("\n(a) not every run has a generation >= 1 yet")
    if all(a and b and "final" in a and "final" in b for a, b in pairs):
        fa, fb = [np.array(a["final"]["fresh"]) for a, _ in pairs], [np.array(b["final"]["fresh"]) for _, b in pairs]
        boot = lambda y: y[rng.integers(0, len(y), len(y))].mean()
        point = float(np.mean([p.mean() - q.mean() for p, q in zip(fa, fb)]))
        b = [np.mean([boot(p) - boot(q) for p, q in zip(fa, fb)]) for _ in range(DRAWS)]
        lo, hi = (float(v) for v in np.quantile(b, [.025, .975]))
        verdict.update(fresh_species15=[float(p.mean()) for p in fa], fresh_plain=[float(q.mean()) for q in fb],
                       fresh_diff=point, fresh_ci95=[lo, hi], clause_b=bool(lo > BOUND))
        print(f"(b) finalists' fresh win rate, species15 - plain: {point:+.3f} [{lo:+.3f}, {hi:+.3f}] -> {'holds' if verdict['clause_b'] else 'fails'}")
        if "clause_a" in verdict:
            verdict["passes"] = bool(verdict["clause_a"] and verdict["clause_b"])
            print(f"species15 {'PASSES' if verdict['passes'] else 'FAILS'} the pre-registered rule")
    else:
        print("(b) not every run has its finalists yet")
    json.dump(dict(arms=ARMS, seeds=SEEDS, runs=runs, reference=ref, verdict=verdict), open(OUT, "w"))
    print(f"-> {OUT}")


if __name__ == "__main__":
    {"report": report}[sys.argv[1]]()
