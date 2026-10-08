"""VGC side of the benchmark feature table: legality of uniform draws, domain size and single-edit
neighbourhood size of the full-team search unit.  Validator + counting only, no battles.
Pre-registration: docs/bench_features.md (metrics 4 and 5).

Draws, each judged by Showdown's TeamValidator (gen9championsvgc2026regmb):
  (a) unconditioned  every field uniform over its global vocabulary (species, ability, item, 4 moves,
                     alignment, each Stat Point stat 0..32 independently)
  (b) per-member     ability / item / moves / Stat Points drawn from the species' own legal sets
                     (mega forme holds its stone, SP total <= 66); roster clauses NOT enforced
  (c) sequential     as (b) plus Species Clause and Item Clause enforced while drawing
Run:  /tmp/vgc-pilot/.venv/bin/python scripts/vgc_domain_features.py
"""
import copy, json, math, re, sys
from collections import Counter
from itertools import product
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "results" / "vgc_domain_features.json"
N_A, N_B, N_C, SEED = 2000, 2000, 500, 3

g = {"__name__": "mut", "__file__": str(HERE / "mutation_closure_test.py")}
src = (HERE / "mutation_closure_test.py").read_text().split("# ---------------------------------------------------------------- run")[0]
sys.argv = [sys.argv[0], str(SEED)]
exec(compile(src, "mutation_closure_test.py", "exec"), g)
teams, L, TRUE, DISP = g["teams"], g["L"], g["TRUE"], g["DISP"]
SPECIES, ITEMS, ABILS, MOVES, ALIGN = g["SPECIES"], g["ITEMS"], g["ABILS"], g["MOVES"], g["ALIGNMENTS"]
STATS, MEGA_STONE_OF = g["STATS"], g["MEGA_STONE_OF"]
dexnum, is_mega, base_of, dex_entry, norm = g["dexnum"], g["is_mega"], g["base_of"], g["dex_entry"], g["norm"]
team_to_text, Validator = g["team_to_text"], g["Validator"]
rng = np.random.default_rng(SEED)
pick = lambda seq: seq[int(rng.integers(0, len(seq)))]
disp = lambda k: DISP.get(k, k)

def items_for(sp):                         # per-member item domain (no roster clause)
    if is_mega(sp):
        st = norm(dex_entry(sp).get("requiredItem", "")); return [st] if st else []
    return [it for it in ITEMS if not (it in MEGA_STONE_OF and MEGA_STONE_OF[it] != base_of(sp))]

def sp_uniform_legal():
    while True:
        v = rng.integers(0, 33, 6)
        if v.sum() <= 66: return dict(zip(STATS, map(int, v)))

def member(sp, conditioned, taken_items=()):
    if conditioned:
        ab = pick(sorted(L.abils_for(sp)))
        dom = [i for i in items_for(sp) if i not in taken_items]
        it = pick(dom) if dom else ""
        mv = list(rng.choice(TRUE[sp], size=min(4, len(TRUE[sp])), replace=False))
        evs = sp_uniform_legal()
    else:
        ab, it = pick(ABILS), pick(ITEMS)
        mv = list(rng.choice(MOVES, size=4, replace=False))
        evs = dict(zip(STATS, map(int, rng.integers(0, 33, 6))))
    al = pick(ALIGN)
    return {"species": disp(sp), "item": disp(it), "ability": disp(ab), "moves": [disp(m) for m in mv],
            "nature": disp(al) if al in DISP else al.capitalize(), "evs": evs}

def draw(kind):
    if kind == "c":
        team, nums, items = [], set(), set()
        while len(team) < 6:
            sp = pick(SPECIES)
            if dexnum(sp) in nums: continue
            m = member(sp, True, items)
            nums.add(dexnum(sp)); items.add(norm(m["item"])); team.append(m)
        return team
    return [member(pick(SPECIES), kind == "b") for _ in range(6)]

def category(err):
    e = err.lower()
    for key, pat in [("item clause", r"more than 1 |item clause"), ("move not learnable", r"can't learn|learn "),
                     ("ability", r"ability|can't have (?!.*\bmove)"), ("does not exist", r"does not exist"),
                     ("species clause", r"species clause|same species|more than one"), ("item clause", r"item clause"),
                     ("stat points", r"stat point|evs|ev "), ("item/mega", r"item|stone|mega"),
                     ("species not allowed", r"banned|not allowed|does not exist")]:
        if re.search(pat, e): return key
    return "other"

def legality():
    val = Validator(); res = {}
    for kind, n in (("a", N_A), ("b", N_B), ("c", N_C)):
        ok, cats = 0, Counter()
        for _ in range(n):
            err = val(team_to_text(draw(kind)))
            if err is None: ok += 1
            else: cats.update({category(x) for x in err.split("; ")})
        res[kind] = {"n": n, "legal": ok, "legal_frac": ok / n, "error_categories": dict(cats.most_common())}
        print(kind, ok, "/", n, dict(cats.most_common(6)), flush=True)
    val.close(); return res

# ---------------------------------------------------------------- counting
def n_sp_allocations():
    ways = np.zeros(67, dtype=object); ways[0] = 1
    for _ in range(6):
        nw = np.zeros(67, dtype=object)
        for s in range(67):
            if ways[s]:
                for k in range(33):
                    if s + k <= 66: nw[s + k] += ways[s]
        ways = nw
    return int(sum(ways))

def counting():
    NSP = n_sp_allocations()
    cfg = {sp: len(L.abils_for(sp)) * len(items_for(sp)) * math.comb(len(TRUE[sp]), 4) * len(ALIGN) * NSP
           for sp in SPECIES}
    zero = sorted(sp for sp, c in cfg.items() if c == 0); cfg = {sp: c for sp, c in cfg.items() if c}
    by_num = Counter()
    for sp, c in cfg.items(): by_num[dexnum(sp)] += c
    e = [1] + [0] * 6                                            # elementary symmetric polynomial e6
    for w in by_num.values():
        for k in range(6, 0, -1): e[k] += e[k - 1] * w
    def nbhd(team):
        tot_field, tot_species = 0, 0
        nums = [dexnum(s["species"]) for s in team]; items = [norm(s["item"]) for s in team if s["item"]]
        for i, s in enumerate(team):
            sp = norm(s["species"])
            if sp not in cfg: return None
            held = set(items) - {norm(s["item"])}
            tot_field += (4 * (len(TRUE[sp]) - 4) + len(L.abils_for(sp)) - 1
                          + len([x for x in items_for(sp) if x not in held]) - 1 + len(ALIGN) - 1 + NSP - 1)
            others = set(nums[:i] + nums[i + 1:])
            tot_species += sum(c for n_, c in by_num.items() if n_ not in others)   # whole member replaced
        return tot_field, tot_species
    nb = [x for x in (nbhd(t) for t in teams) if x]
    return {"species": len(SPECIES), "species_with_zero_configs": zero, "dex_numbers": len(by_num), "abilities_vocab": len(ABILS),
            "items_vocab": len(ITEMS), "moves_vocab": len(MOVES), "alignments": len(ALIGN),
            "sp_allocations_per_member": NSP,
            "log10_member_configs_median": float(np.median([math.log10(c) for c in cfg.values()])),
            "log10_teams_upper_bound": math.log10(e[6]),
            "log10_teams_note": "unordered, Species Clause applied, Item Clause ignored (upper bound)",
            "log10_neighbourhood_without_species_edit_median": float(np.median([math.log10(a) for a, _ in nb])),
            "log10_neighbourhood_with_species_edit_median": float(np.median([math.log10(a + b) for a, b in nb])),
            "neighbourhood_teams_counted": len(nb),
            "encoding_variables_per_team": 6 * (1 + 1 + 1 + 4 + 1 + 6)}

if __name__ == "__main__":
    out = {"counting": counting()}; print(json.dumps(out["counting"], indent=1), flush=True)
    out["legality"] = legality()
    json.dump(out, open(OUT, "w"), indent=1)
