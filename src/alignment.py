"""Does the Stat Alignment (nature) column explain part of the generator's deficit?

WHY THIS IS NOT A REPEAT OF THE SPREAD EXPERIMENTS
--------------------------------------------------
Every earlier spread arm (`eval_spread_paired.py`, the `shuffled` stratum, the
`scale`/`sampling`/`ridge` generators) COPIED a corpus EV spread onto the generated
candidate but kept the MODEL'S OWN nature. So the alignment column has never been
isolated, and the conclusion "copying a real spread costs nothing (-0.6 points,
-0.5 sigma)" was measured with a model-chosen nature still in place.

It matters because the column is the worst-learned one in the model. Measured
battle-free on 2026-08-27 over 692 real teams and 1,215 generated teams:

    contradictions per team   real 0.08   generated 2.46 - 2.90
    teams with >= 1           real  8%    generated 96 - 97%

where a contradiction is a nature that lowers the attacking stat the candidate
actually uses (Modest on a physical Garchomp) or raises one it never uses (+Atk with
zero physical moves). Nothing constrains this column: it is decoded LAST in
`diffusion.ORDER`, and all 21 alignments are legal on every candidate, so the
feasibility mask is a no-op there.

THE ARMS (paired - identical six candidates, items, abilities, moves and Stat Points;
only the alignment column differs)
    asgen   the model's own nature, corpus spread copied  <- today's default policy
    repair  rule-repaired alignment (see `repair_nature`)
    donor   nature AND spread copied together from ONE real corpus set for that
            species, so the pair is coherent instead of half-copied

PRE-REGISTERED EXPECTATION (written before the run)
    The 2026-08-26 landscape decomposition put ~32 of the 38-point deficit on
    "which six candidates it picked" and ~6 on Stat Points. Alignment is a subset of
    that 6, so the honest prior is a SMALL positive effect: repair and donor beat
    asgen by 0 - 3 points. A paired design on 150 teams resolves ~2 points. If the
    effect were large, the composition story would be wrong, which is worth knowing.
    A null result is a real result here: it says the incoherent natures are cosmetic.

    python alignment.py [--per 150] [--battles 24] [--smoke]
"""
import argparse, json, os, random, sys, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, "/tmp/vgc-pilot/src")
import diffusion as D, wrdiffusion as W
from corpus import load_corpus, norm, STATS
from encode import NF, NSLOT
from propose import Validator, slot_to_text
import pool

OUT = "/tmp/vgc-pilot/alignment_results.json"
GEN = Path("/tmp/vgc-pilot/align_gen")  # per-seed subdir set in main()

# nature -> (+stat, -stat); the 5 neutral ones raise and lower nothing
NATURE_EFFECT = {
    "adamant": ("Atk", "SpA"), "bold": ("Def", "Atk"), "brave": ("Atk", "Spe"),
    "calm": ("SpD", "Atk"), "careful": ("SpD", "SpA"), "gentle": ("SpD", "Def"),
    "hasty": ("Spe", "Def"), "impish": ("Def", "SpA"), "jolly": ("Spe", "SpA"),
    "lax": ("Def", "SpD"), "lonely": ("Atk", "Def"), "mild": ("SpA", "Def"),
    "modest": ("SpA", "Atk"), "naive": ("Spe", "SpD"), "naughty": ("Atk", "SpD"),
    "quiet": ("SpA", "Spe"), "rash": ("SpA", "SpD"), "relaxed": ("Def", "Spe"),
    "sassy": ("SpD", "Spe"), "timid": ("Spe", "Atk"),
    "hardy": (None, None), "docile": (None, None), "serious": (None, None),
    "bashful": (None, None), "quirky": (None, None),
}

def move_categories():
    mv = json.load(open("/tmp/vgc-pilot/data/moves.json"))
    cat = {}
    for k, v in mv.items():
        c = v.get("category")
        if c:
            cat[norm(v.get("name", k))] = c
            cat[norm(k)] = c
    return cat

CAT = move_categories()

def split_of(slot):
    """(physical attacking moves, special attacking moves) on this candidate."""
    ph = sp = 0
    for m in slot["moves"]:
        c = CAT.get(norm(m))
        if c == "Physical": ph += 1
        elif c == "Special": sp += 1
    return ph, sp

def contradictions(slot):
    """1 if the alignment fights the candidate's own moves, else 0 - the battle-free defect."""
    ph, sp = split_of(slot)
    up, dn = NATURE_EFFECT.get(norm(slot["nature"]), (None, None))
    drop_used = (dn == "Atk" and ph > 0 and ph >= sp) or (dn == "SpA" and sp > 0 and sp >= ph)
    boost_unused = (up == "Atk" and ph == 0) or (up == "SpA" and sp == 0)
    return int(drop_used or boost_unused)

def repair_nature(slot, species_modes, allowed):
    """Pick an alignment that does not fight the candidate's moves.

    The candidate set is decided by the move split; WHICH of the two admissible
    natures (the power one or the speed one) is chosen is not invented here - it is
    the mode among real corpus sets for that species, restricted to the admissible
    pair. Falls back to the first admissible nature the corpus actually uses.
    """
    ph, sp = split_of(slot)
    if ph > 0 and sp == 0:   cands = ["Adamant", "Jolly", "Brave"]
    elif sp > 0 and ph == 0: cands = ["Modest", "Timid", "Quiet"]
    elif ph > 0 and sp > 0:  cands = ["Hasty", "Naive", "Serious"]   # drops neither attack stat
    else:                    cands = ["Careful", "Impish", "Bold", "Calm"]
    cands = [c for c in cands if norm(c) in allowed]
    if not cands:
        return slot["nature"]
    counts = species_modes.get(norm(slot["species"]), Counter())
    return max(cands, key=lambda c: counts.get(norm(c), 0))

def build_tables(teams):
    look, spreads, sets, natures = {}, defaultdict(list), defaultdict(list), Counter()
    modes = defaultdict(Counter)
    for t in teams:
        for s in t:
            look[norm(s["species"])] = s["species"]; look[norm(s["ability"])] = s["ability"]
            if s["item"]: look[norm(s["item"])] = s["item"]
            for m in s["moves"]: look[norm(m)] = m
            if s["nature"]:
                look[norm(s["nature"])] = s["nature"]
                natures[norm(s["nature"])] += 1
                modes[norm(s["species"])][norm(s["nature"])] += 1
            spreads[norm(s["species"])].append(dict(s["evs"]))
            sets[norm(s["species"])].append((dict(s["evs"]), s["nature"]))
    return look, spreads, sets, modes, set(natures)

def opponents():
    root = Path("/tmp/vgc-pilot/vgc-bench/teams/reg_mb"); opp = []
    for i in [t["id"] for t in json.load(open("/tmp/vgc-pilot/top50_evs.json"))]:
        for c in (root / f"{i}.txt", root / "featured" / f"{i}.txt"):
            if c.exists(): opp.append(str(c)); break
    return opp

def clauses_ok(slots):
    b = [norm(s["species"]) for s in slots]
    it = [norm(s["item"]) for s in slots if s["item"]]
    return len(set(b)) == 6 and len(set(it)) == len(it)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=150)
    ap.add_argument("--battles", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke: a.per, a.battles = 4, 2

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    model, V, L, corpus, lab = W.load_model()
    C = D.Constraints(V, L)
    look, spreads, sets, modes, allowed = build_tables(corpus + lab)
    print(f"corpus alignments in use: {len(allowed)}", flush=True)

    gen = GEN.parent / f"align_gen_s{a.seed}"
    gen.mkdir(exist_ok=True)
    for f in gen.glob("*.txt"): f.unlink()
    val = Validator()
    meta, files = {}, []
    kept = tries = 0
    t0 = time.perf_counter()
    # each accepted sample yields a TRIPLE of teams that differ only in the alignment column
    while kept < a.per and tries < a.per * 6:
        tries += 1
        row = W.sample_constrained(model, C, 1, 5, "none", 2.0).cpu().numpy()[0]
        base = []
        for i in range(NSLOT):
            b = i * NF
            g = lambda j: V.decode_field(b + j, int(row[b + j]))
            sp = g(0)
            mv = [look.get(g(3 + j), g(3 + j)) for j in range(4)
                  if g(3 + j) and g(3 + j) != "[MASK]"]
            pool_sets = sets.get(sp) or [({s: 0 for s in STATS}, "Serious")]
            ev, nat_donor = pool_sets[int(rng.integers(0, len(pool_sets)))]
            base.append(dict(species=look.get(sp, sp), item=look.get(g(2), g(2)),
                             ability=look.get(g(1), g(1)),
                             nature=look.get(g(7), g(7)), moves=mv, evs=dict(ev),
                             _donor_nature=nat_donor))
        if not clauses_ok(base):
            continue
        variants = {}
        variants["asgen"] = [dict(s) for s in base]
        variants["repair"] = [dict(s, nature=repair_nature(s, modes, allowed)) for s in base]
        variants["donor"] = [dict(s, nature=s["_donor_nature"]) for s in base]
        texts, ok = {}, True
        for arm, slots in variants.items():
            clean = [{k: v for k, v in s.items() if not k.startswith("_")} for s in slots]
            txt = "\n\n".join(slot_to_text(s) for s in clean) + "\n"
            if val(txt) is not None:
                ok = False; break
            texts[arm] = (txt, clean)
        if not ok:
            continue
        for arm, (txt, clean) in texts.items():
            p = gen / f"{arm}_{kept:04d}.txt"
            p.write_text(txt)
            files.append(str(p))
            meta[str(p)] = {"arm": arm, "pair": kept,
                            "contradictions": sum(contradictions(s) for s in clean)}
        kept += 1
        if kept % 25 == 0:
            print(f"  {kept}/{a.per} triples ({kept/tries:.0%} of samples usable, "
                  f"{(time.perf_counter()-t0)/60:.1f} min)", flush=True)
    val.close()
    if kept < a.per:
        print(f"  CAP HIT: produced {kept} of {a.per} requested triples in {tries} attempts "
              f"- reporting on {kept}, not silently on {a.per}", flush=True)

    opp = opponents()
    print(f"\nbattling {len(files)} teams x {a.battles} vs {len(opp)} meta teams", flush=True)
    res = pool.score([(f, opp) for f in files], battles=a.battles, conc=50)
    for f, v in res.items():
        v.update(meta[f])

    arms = ["asgen", "repair", "donor"]
    by = {arm: {} for arm in arms}
    for f, v in res.items():
        by[v["arm"]][v["pair"]] = v["win_rate"]
    common = sorted(set.intersection(*[set(by[arm]) for arm in arms]))
    out = {"n_pairs": len(common), "battles_per_team": a.battles, "arms": {}, "pairs": {}}
    print(f"\n{'arm':8s} {'n':>4s} {'win rate':>9s} {'clustered SE':>13s} {'p90':>6s} {'max':>6s} "
          f"{'>=0.458':>8s} {'contradictions/team':>20s}")
    for arm in arms:
        w = np.array([by[arm][p] for p in common])
        cd = np.array([v["contradictions"] for v in res.values() if v["arm"] == arm])
        out["arms"][arm] = dict(n=len(w), win_rate=float(w.mean()),
                                se=float(w.std(ddof=1) / np.sqrt(len(w))),
                                p90=float(np.percentile(w, 90)), max=float(w.max()),
                                above_real_median=float((w >= .458).mean()),
                                contradictions=float(cd.mean()))
        print(f"{arm:8s} {len(w):4d} {w.mean():9.4f} {w.std(ddof=1)/np.sqrt(len(w)):13.4f} "
              f"{np.percentile(w,90):6.3f} {w.max():6.3f} {(w>=.458).mean():8.1%} {cd.mean():20.2f}")
    # the paired tests - this is what the design buys
    print()
    for arm in ["repair", "donor"]:
        d = np.array([by[arm][p] - by["asgen"][p] for p in common])
        se = d.std(ddof=1) / np.sqrt(len(d))
        out["arms"][arm]["paired_delta"] = float(d.mean())
        out["arms"][arm]["paired_se"] = float(se)
        out["arms"][arm]["paired_sigma"] = float(d.mean() / se) if se else None
        out["arms"][arm]["better"] = int((d > 0).sum())
        out["arms"][arm]["worse"] = int((d < 0).sum())
        print(f"paired {arm:6s} vs asgen: {d.mean():+.4f} +/- {se:.4f} "
              f"({d.mean()/se if se else 0:+.1f} sigma; {int((d>0).sum())} better, "
              f"{int((d<0).sum())} worse, {int((d==0).sum())} tied)", flush=True)
    out["pairs"] = {str(p): {arm: by[arm][p] for arm in arms} for p in common}
    out["reference"] = {"real_team": 0.470, "real_median": 0.458, "slotcopy": 0.436,
                        "generator_bin5_g2": 0.168, "generator_bin5_g4": 0.193,
                        "unconditioned": 0.128}
    out["seed"] = a.seed
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")
    print("reference: real 0.470 · slotcopy 0.436 · bin5 g2 0.168 · bin5 g4 0.193 · unconditioned 0.128")
    print("ALIGNMENT_DONE", flush=True)

if __name__ == "__main__":
    main()
