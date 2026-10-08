"""Hierarchical product sampling of legal Reg M-B teams, at scale.

The sampler follows the vault note `hierarchical product sampling.md`: a team is
drawn field by field in a fixed order, later fields conditioned on earlier ones —
species first, then ability / item / moves from that species' own legal tables
(the authoritative `learnset_true.json`, 189 species), then nature and a Stat
Points spread under the 66-point budget. The team's probability is the product of
the per-field probabilities. Slot order is canonicalised (sorted on species) so
symmetric duplicates cannot appear.

Every sampled team is gated by Showdown's own TeamValidator
(gen9championsvgc2026regmb); rejected teams are resampled. Output is JSONL, one
accepted team per line.

Usage: hps_generate.py --n 12500 --seed 0 --out /path/shard_0.jsonl
"""
import argparse, hashlib, json, subprocess, sys, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from corpus import load_corpus, norm, STATS, NATURES, dex_entry, MEGA_STONE_OF
from encode import Legality

SHOWDOWN = Path("/tmp/vgc-pilot/vgc-bench/pokemon-showdown")
TRUE = {k: sorted(m for m in v if m != "[MASK]")
        for k, v in json.load(open("/tmp/vgc-pilot/learnset_true.json")).items()}


class Validator:
    def __init__(self):
        self.p = subprocess.Popen(["node", "validate-teams-batch.js"],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  text=True, cwd=str(SHOWDOWN))
    def __call__(self, text, fmt="gen9championsvgc2026regmb"):
        self.p.stdin.write(json.dumps({"format": fmt, "team": text}) + "\n")
        self.p.stdin.flush()
        r = json.loads(self.p.stdout.readline())
        return None if r["valid"] else "; ".join(r["errors"])
    def close(self):
        self.p.stdin.close(); self.p.wait()


def slot_to_text(sl):
    head = sl["species"] + (f" @ {sl['item']}" if sl["item"] else "")
    lines = [head, f"Ability: {sl['ability']}", "Level: 50"]
    ev = " / ".join(f"{sl['evs'][s]} {s}" for s in STATS if sl["evs"].get(s))
    if ev: lines.append(f"EVs: {ev}")
    if sl.get("nature"): lines.append(f"{sl['nature']} Nature")
    lines += [f"- {m}" for m in sl["moves"]]
    return "\n".join(lines)


def build_tables():
    teams, _ = load_corpus(); teams = [t for t in teams if len(t) == 6]
    L = Legality(teams)
    disp, items = {}, set()
    for t in teams:
        for s in t:
            disp[norm(s["species"])] = s["species"]; disp[norm(s["ability"])] = s["ability"]
            if s["item"]: disp[norm(s["item"])] = s["item"]; items.add(norm(s["item"]))
            for m in s["moves"]: disp[norm(m)] = m
    base_of, is_mega, stone_of = {}, {}, {}
    for sp in TRUE:
        e = dex_entry(sp) or {}
        disp.setdefault(sp, e.get("name", sp))
        base_of[sp] = norm(e.get("baseSpecies", e.get("name", sp)))
        is_mega[sp] = "mega" in norm(e.get("forme", ""))
        if is_mega[sp] and e.get("requiredItem"):
            stone_of[sp] = norm(e["requiredItem"])
            disp.setdefault(stone_of[sp], e["requiredItem"])
    abil = {sp: sorted(L.abils_for(sp)) for sp in TRUE}
    species = [sp for sp in sorted(TRUE)
               if len(TRUE[sp]) >= 4 and abil[sp] and (not is_mega[sp] or sp in stone_of)]
    return dict(species=species, disp=disp, abil=abil, items=sorted(items),
                base_of=base_of, is_mega=is_mega, stone_of=stone_of)


def sample_spread(rng):
    """66 Stat Points over 6 stats, each capped at 32."""
    ev = {s: 0 for s in STATS}
    for _ in range(66):
        open_stats = [s for s in STATS if ev[s] < 32]
        ev[open_stats[rng.integers(0, len(open_stats))]] += 1
    return ev


def sample_slot(T, rng, used_base, used_item):
    """One legal slot respecting Species/Item Clause vs what is already used.
    Mutates used_base/used_item on success; returns None on a dead draw."""
    sp = T["species"][rng.integers(0, len(T["species"]))]
    if T["base_of"][sp] in used_base: return None
    if T["is_mega"][sp]:
        it = T["stone_of"][sp]
        if it in used_item: return None
    else:
        for _ in range(40):
            it = T["items"][rng.integers(0, len(T["items"]))]
            if it in used_item: continue
            if it in MEGA_STONE_OF and MEGA_STONE_OF[it] != T["base_of"][sp]: continue
            break
        else:
            return None
    legal_mv = TRUE[sp]
    mv = [legal_mv[i] for i in rng.choice(len(legal_mv), 4, replace=False)]
    ab = T["abil"][sp][rng.integers(0, len(T["abil"][sp]))]
    d = T["disp"]
    used_base.add(T["base_of"][sp]); used_item.add(it)
    return dict(species=d.get(sp, sp), item=d.get(it, it), ability=d.get(ab, ab),
                nature=NATURES[rng.integers(0, len(NATURES))],
                moves=[d.get(m, m) for m in mv], evs=sample_spread(rng))

def one_team(T, rng):
    slots, used_base, used_item = [], set(), set()
    tries = 0
    while len(slots) < 6 and tries < 200:
        tries += 1
        s = sample_slot(T, rng, used_base, used_item)
        if s is not None: slots.append(s)
    if len(slots) < 6: return None
    return sorted(slots, key=lambda s: norm(s["species"]))


def team_key(slots):
    parts = []
    for s in slots:
        parts.append("|".join([norm(s["species"]), norm(s["ability"]), norm(s["item"]),
                               ",".join(sorted(norm(m) for m in s["moves"])), norm(s["nature"]),
                               ",".join(str(s["evs"][st]) for st in STATS)]))
    return hashlib.sha1(";".join(parts).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    T = build_tables()
    val = Validator()
    out = open(a.out, "w") if a.out else None
    seen = set()
    kept = attempts = 0
    errs = Counter()
    t0 = time.time()
    while kept < a.n and attempts < a.n * 20:
        attempts += 1
        slots = one_team(T, rng)
        if slots is None: continue
        key = team_key(slots)
        if key in seen: continue
        txt = "\n\n".join(slot_to_text(s) for s in slots) + "\n"
        e = val(txt)
        if e is None:
            seen.add(key); kept += 1
            if out: out.write(json.dumps({"key": key, "seed": a.seed, "team": txt}) + "\n")
            if kept % 1000 == 0:
                dt = time.time() - t0
                print(f"  seed {a.seed}: {kept}/{a.n} accepted "
                      f"({kept/attempts:.1%}, {kept/dt:.1f} teams/s)", flush=True)
        else:
            k = ("mega/item" if "mega" in e.lower() or "does not exist" in e.lower()
                 else "learnset" if "can't learn" in e.lower() or "cannot learn" in e.lower()
                 else "ability" if "ability" in e.lower()
                 else "statpoints" if "stat point" in e.lower() or "ev" in e.lower()
                 else "other")
            errs[k] += 1
            if errs[k] == 1: print(f"  example {k}: {e[:180]}", flush=True)
    val.close()
    if out: out.close()
    dt = time.time() - t0
    print(f"seed {a.seed}: {kept} accepted / {attempts} attempts "
          f"({kept/max(attempts,1):.1%}) in {dt:.0f}s ({kept/max(dt,1e-9):.1f} teams/s)")
    print(f"rejects: {dict(errs.most_common())}")

if __name__ == "__main__":
    main()
