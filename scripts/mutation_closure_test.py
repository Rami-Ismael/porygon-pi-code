"""Empirical closure test for legality-preserving mutation operators on Reg M-B teams.

Operators A-G sample their replacement from the CONDITIONAL legal domain given the
rest of the team; the NAIVE set draws from the global pool with no filtering.
Every child is judged by Showdown's own TeamValidator (propose.Validator), never by
a local rule check.  Legality only -- no battles, no training.

Run:  /tmp/vgc-pilot/.venv/bin/python scripts/mutation_closure_test.py [seed]
"""
import copy, json, os, sys, time
from collections import Counter, defaultdict

SRC = "/tmp/vgc-pilot/src"
LEARNSET = "/tmp/vgc-pilot/learnset_true.json"
OUT = "/Users/ramiismael/Documents/code/vgc-team-generator-pilot/results/mutation_closure_test.json"

blocking = []
for p in (SRC, LEARNSET, "/tmp/vgc-pilot/vgc-bench/pokemon-showdown/validate-teams-batch.js",
          "/tmp/vgc-pilot/vgc-bench/teams/reg_mb", "/tmp/vgc-pilot/data/pokedex.json"):
    if not os.path.exists(p):
        blocking.append(f"missing: {p}")
if blocking:
    print(json.dumps({"ran": False, "blocking_issues": blocking}, indent=1)); sys.exit(1)

sys.path.insert(0, SRC); os.chdir(SRC)
import numpy as np
from corpus import load_corpus, norm, dex_entry, MEGA_STONE_OF, NATURES, STATS
from encode import Legality
from propose import Validator, team_to_text

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
N_OP, N_CHAIN, CHAIN_LEN = 200, 100, 50
rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------- tables
teams, names = load_corpus()
L = Legality(teams)
TRUE = {k: sorted(m for m in v if m != "[MASK]") for k, v in json.load(open(LEARNSET)).items()}
DISP, ITEMS, ABILS, SPREADS, SLOTS_BY_SP = {}, set(), set(), defaultdict(list), defaultdict(list)
for t in teams:
    for s in t:
        sp = norm(s["species"])
        DISP[sp] = s["species"]; DISP[norm(s["ability"])] = s["ability"]; ABILS.add(norm(s["ability"]))
        if s["item"]: DISP[norm(s["item"])] = s["item"]; ITEMS.add(norm(s["item"]))
        for m in s["moves"]: DISP[norm(m)] = m
        if s["nature"]: DISP[norm(s["nature"])] = s["nature"]
        SPREADS[sp].append(dict(s["evs"]))
        SLOTS_BY_SP[sp].append(copy.deepcopy(s))
ALL_SLOTS = [s for v in SLOTS_BY_SP.values() for s in v]
ITEMS = sorted(ITEMS); ABILS = sorted(ABILS)
MOVES = sorted({m for v in TRUE.values() for m in v} | {norm(m) for t in teams for s in t for m in s["moves"]})
SPECIES = sorted(TRUE)                       # 189 species with a validator-probed learnset
NEUTRAL = {"bashful", "docile", "hardy", "quirky", "serious", ""}
ALIGNMENTS = sorted({norm(n) for n in NATURES if norm(n) not in NEUTRAL} | {"serious"})   # 21
assert len(ALIGNMENTS) == 21

def dexnum(species):
    e = dex_entry(species); return e["num"] if e else norm(species)
def is_mega(species):
    e = dex_entry(species); return bool(e and "mega" in norm(e.get("forme", "")))
def base_of(species):
    e = dex_entry(species); return norm(e.get("baseSpecies", e.get("name", species))) if e else norm(species)
def stone_for(species):
    """Mega stone(s) whose base is this (non-mega) species."""
    b = base_of(species); return [st for st, bs in MEGA_STONE_OF.items() if bs == b]
def choice(seq):
    return seq[int(rng.integers(0, len(seq)))]
def other_items(team, slot):
    return {norm(team[j]["item"]) for j in range(6) if j != slot and team[j]["item"]}

# ---------------------------------------------------------------- conditional item domain (C)
def item_domain(team, slot):
    sp = team[slot]["species"]; held = other_items(team, slot)
    if is_mega(sp):                                     # a mega forme must hold its own stone
        e = dex_entry(sp); st = norm(e.get("requiredItem", ""))
        return [st] if st and st not in held else []
    ok = []
    for it in ITEMS:
        if it in held: continue
        if it in MEGA_STONE_OF and MEGA_STONE_OF[it] != base_of(sp): continue   # stone for another species -> redraw
        ok.append(it)
    return ok

# ---------------------------------------------------------------- operators A-G (conditional)
def op_A(team):   # move-swap
    i = int(rng.integers(0, 6)); s = team[i]
    if not s["moves"]: return "noop:no-moves"
    j = int(rng.integers(0, len(s["moves"])))
    have = {norm(m) for m in s["moves"]}
    dom = [m for m in TRUE.get(norm(s["species"]), []) if m not in have]
    if not dom: return "noop:empty-domain"
    m = choice(dom); s["moves"][j] = DISP.get(m, m); return None
def op_B(team):   # ability-swap
    i = int(rng.integers(0, 6)); s = team[i]
    dom = sorted(L.abils_for(norm(s["species"])) - {norm(s["ability"])})
    if not dom: return "noop:single-ability"
    a = choice(dom); s["ability"] = DISP.get(a, a); return None
def op_C(team):   # item-swap
    i = int(rng.integers(0, 6)); s = team[i]
    dom = [it for it in item_domain(team, i) if it != norm(s["item"])]
    if not dom: return "noop:empty-domain"
    it = choice(dom); s["item"] = DISP.get(it, it); return None
def op_D(team):   # alignment-swap
    i = int(rng.integers(0, 6)); s = team[i]
    cur = norm(s["nature"]); cur = "serious" if cur in NEUTRAL else cur
    a = choice([x for x in ALIGNMENTS if x != cur]); s["nature"] = DISP.get(a, a.capitalize()); return None
def op_E(team):   # spread-copy from a real set of the same species
    i = int(rng.integers(0, 6)); s = team[i]
    pool = [p for p in SPREADS[norm(s["species"])] if p != s["evs"]]
    if not pool: return "noop:no-other-spread"
    s["evs"] = dict(choice(pool)); return None
def op_F(team):   # candidate-copy with Species Clause + Item Clause fix
    i = int(rng.integers(0, 6))
    nums = {dexnum(team[j]["species"]) for j in range(6) if j != i}
    dom = [s for s in ALL_SLOTS if dexnum(s["species"]) not in nums]
    if not dom: return "noop:empty-domain"
    new = copy.deepcopy(choice(dom)); team[i] = new
    if new["item"] and norm(new["item"]) in other_items(team, i):
        d = item_domain(team, i)
        if not d: return "noop:item-unfixable"
        it = choice(d); new["item"] = DISP.get(it, it)
    return None
def op_G(team):   # species-swap-with-refill; keep alignment + spread
    i = int(rng.integers(0, 6))
    nums = {dexnum(team[j]["species"]) for j in range(6) if j != i}
    dom = [sp for sp in SPECIES if dexnum(sp) not in nums and len(TRUE[sp]) >= 4 and L.abils_for(sp)]
    if not dom: return "noop:empty-domain"
    sp = choice(dom); s = team[i]
    s["species"] = DISP.get(sp, sp)
    a = choice(sorted(L.abils_for(sp))); s["ability"] = DISP.get(a, a)
    mv = [TRUE[sp][k] for k in rng.choice(len(TRUE[sp]), 4, replace=False)]
    s["moves"] = [DISP.get(m, m) for m in mv]
    d = item_domain(team, i)
    if not d: return "noop:no-item"
    it = choice(d); s["item"] = DISP.get(it, it)
    return None

# ---------------------------------------------------------------- NAIVE (global pool, no conditioning)
def nv_A(team):
    i = int(rng.integers(0, 6)); s = team[i]
    if not s["moves"]: return "noop:no-moves"
    m = choice(MOVES); s["moves"][int(rng.integers(0, len(s["moves"])))] = DISP.get(m, m); return None
def nv_B(team):
    i = int(rng.integers(0, 6)); a = choice(ABILS); team[i]["ability"] = DISP.get(a, a); return None
def nv_C(team):
    i = int(rng.integers(0, 6)); it = choice(ITEMS); team[i]["item"] = DISP.get(it, it); return None
def nv_D(team):
    i = int(rng.integers(0, 6)); team[i]["nature"] = choice(NATURES); return None
def nv_E(team):
    i = int(rng.integers(0, 6)); team[i]["evs"] = dict(choice(ALL_SLOTS)["evs"]); return None
def nv_F(team):
    i = int(rng.integers(0, 6)); team[i] = copy.deepcopy(choice(ALL_SLOTS)); return None
def nv_G(team):
    i = int(rng.integers(0, 6)); s = team[i]; sp = choice(SPECIES)
    s["species"] = DISP.get(sp, sp); a = choice(ABILS); s["ability"] = DISP.get(a, a)
    mv = [MOVES[k] for k in rng.choice(len(MOVES), 4, replace=False)]; s["moves"] = [DISP.get(m, m) for m in mv]
    it = choice(ITEMS); s["item"] = DISP.get(it, it); return None

COND = {"A_move_swap": op_A, "B_ability_swap": op_B, "C_item_swap": op_C, "D_alignment_swap": op_D,
        "E_spread_copy": op_E, "F_candidate_copy": op_F, "G_species_swap_refill": op_G}
NAIVE = {"naive_A_move": nv_A, "naive_B_ability": nv_B, "naive_C_item": nv_C, "naive_D_nature": nv_D,
         "naive_E_spread": nv_E, "naive_F_slot": nv_F, "naive_G_species": nv_G}

# ---------------------------------------------------------------- run
val = Validator()
t0 = time.time()
base_ok = sum(val(team_to_text(t)) is None for t in teams)
summary = {"seed": SEED, "corpus_teams": len(teams), "corpus_teams_valid": base_ok,
           "pools": {"species": len(SPECIES), "moves_global": len(MOVES), "items_global": len(ITEMS),
                     "abilities_global": len(ABILS), "alignments": len(ALIGNMENTS)},
           "single_step": {}, "chains": {}}
pick = rng.permutation(len(teams))[:N_OP]

for name, op in {**COND, **NAIVE}.items():
    valid = invalid = noop = 0; reasons = Counter()
    for k in pick:
        child = copy.deepcopy(teams[k]); r = op(child)
        if r: noop += 1
        err = val(team_to_text(child))
        if err is None: valid += 1
        else: invalid += 1; reasons[err] += 1
    summary["single_step"][name] = {"trials": int(N_OP), "valid": valid, "invalid": invalid, "noop": noop,
                                    "failure_reasons": dict(reasons.most_common())}
    print(f"{name:26s} valid {valid}/{N_OP}  noop {noop}", flush=True)

def chain(ops, label):
    keys = sorted(ops); chosen = rng.permutation(len(teams))[:N_CHAIN]
    alive = np.ones(N_CHAIN, bool); alive_at = []; first_fail = Counter(); fail_step = []
    for step in range(CHAIN_LEN):
        for c, k in enumerate(chosen):
            if step == 0: states[c] = copy.deepcopy(teams[k])
            if not alive[c]: continue
            opn = keys[int(rng.integers(0, len(keys)))]; ops[opn](states[c])
            err = val(team_to_text(states[c]))
            if err is not None:
                alive[c] = False; first_fail[f"{opn}: {err}"] += 1; fail_step.append(step + 1)
        alive_at.append(int(alive.sum()))
    summary["chains"][label] = {"chains": N_CHAIN, "steps": CHAIN_LEN, "valid_all_steps": int(alive.sum()),
                                "fraction_valid_all_steps": float(alive.sum()) / N_CHAIN,
                                "alive_after_each_step": alive_at,
                                "median_first_failure_step": float(np.median(fail_step)) if fail_step else None,
                                "first_failure_reasons": dict(first_fail.most_common())}
    print(f"chain {label}: {int(alive.sum())}/{N_CHAIN} valid at all {CHAIN_LEN} steps", flush=True)
states = [None] * N_CHAIN
chain(COND, "conditional_A_G")
states = [None] * N_CHAIN
chain(NAIVE, "naive")
val.close()
summary["runtime_s"] = round(time.time() - t0, 1)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(summary, open(OUT, "w"), indent=1)
print(json.dumps(summary))
