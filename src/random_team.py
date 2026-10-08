"""Generate random LEGAL Reg M-B teams, as many as you like.

Legality comes from the authoritative learnset table (built by querying Showdown's
own validator, 10,112 probes) plus the format's clauses. Every team is then checked
by Showdown itself, so nothing unverified reaches a battle.

This exists because the 695-team corpus cannot teach legality: it shows only 1,505
of 9,099 legal (species, move) pairs. The rule is not in the data — but we own it,
so we can manufacture as many legal examples as we want.
"""
import json, random, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from corpus import load_corpus, norm, STATS, dex_entry, MEGA_STONE_OF
from encode import Legality
from propose import Validator, slot_to_text

TRUE = {k: sorted(v) for k, v in json.load(open("/tmp/vgc-pilot/learnset_true.json")).items()}

def build_tables():
    teams, _ = load_corpus(); teams = [t for t in teams if len(t) == 6]
    L = Legality(teams)
    disp, abil, items, spreads, natures = {}, defaultdict(set), set(), defaultdict(list), set()
    for t in teams:
        for s in t:
            sp = norm(s["species"])
            disp[sp] = s["species"]; disp[norm(s["ability"])] = s["ability"]
            if s["item"]: disp[norm(s["item"])] = s["item"]; items.add(norm(s["item"]))
            for m in s["moves"]: disp[norm(m)] = m
            if s["nature"]: disp[norm(s["nature"])] = s["nature"]; natures.add(norm(s["nature"]))
            spreads[sp].append(dict(s["evs"]))
    for sp in TRUE:
        abil[sp] = {a for a in L.abils_for(sp)}
    base_of = {sp: norm((dex_entry(sp) or {}).get("baseSpecies", sp)) for sp in TRUE}
    is_mega = {sp: "mega" in sp for sp in TRUE}
    return dict(species=sorted(TRUE), disp=disp, abil=abil, items=sorted(items),
                spreads=spreads, natures=sorted(natures), base_of=base_of, is_mega=is_mega)

def one_team(T, rng):
    """One random legal team: 6 distinct base species, distinct items, legal sets."""
    slots, used_base, used_item = [], set(), set()
    tries = 0
    while len(slots) < 6 and tries < 400:
        tries += 1
        sp = T["species"][rng.integers(0, len(T["species"]))]
        if T["base_of"][sp] in used_base: continue
        legal_mv = TRUE[sp]
        if len(legal_mv) < 4 or not T["abil"][sp] or not T["spreads"].get(sp): continue
        # item: never a mega stone for another species, never one on a mega forme
        for _ in range(40):
            it = T["items"][rng.integers(0, len(T["items"]))]
            if it in used_item: continue
            if it in MEGA_STONE_OF and (T["is_mega"][sp] or MEGA_STONE_OF[it] != T["base_of"][sp]):
                continue
            break
        else:
            continue
        mv = list(rng.choice(len(legal_mv), 4, replace=False))
        ab = sorted(T["abil"][sp])[rng.integers(0, len(T["abil"][sp]))]
        pool = T["spreads"][sp]
        slots.append(dict(species=T["disp"].get(sp, sp), item=T["disp"].get(it, it),
                          ability=T["disp"].get(ab, ab),
                          nature=T["disp"].get(T["natures"][rng.integers(0, len(T["natures"]))]),
                          moves=[T["disp"].get(legal_mv[i], legal_mv[i]) for i in mv],
                          evs=pool[rng.integers(0, len(pool))]))
        used_base.add(T["base_of"][sp]); used_item.add(it)
    return slots if len(slots) == 6 else None

def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    out = Path(sys.argv[3] if len(sys.argv) > 3 else "/tmp/vgc-pilot/random_teams")
    rng = np.random.default_rng(seed); random.seed(seed)
    T = build_tables()
    out.mkdir(exist_ok=True)
    for f in out.glob("*.txt"): f.unlink()
    val = Validator()
    kept = tries = 0
    while kept < n and tries < n * 12:
        tries += 1
        sl = one_team(T, rng)
        if sl is None: continue
        txt = "\n\n".join(slot_to_text(s) for s in sl) + "\n"
        if val(txt) is None:
            (out / f"rnd_{kept:06d}.txt").write_text(txt); kept += 1
            if kept % 250 == 0: print(f"  {kept}/{n} ({kept/tries:.0%} accepted)", flush=True)
    val.close()
    print(f"generated {kept} Showdown-valid random teams from {tries} attempts "
          f"({kept/max(tries,1):.1%} accepted) -> {out}")

if __name__ == "__main__":
    main()
