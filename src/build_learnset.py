"""Build an AUTHORITATIVE per-species legal-move table by asking Showdown's own
TeamValidator, instead of guessing at forme/prevo merge rules.

Both hand-rolled rules are wrong: the strict rule wrongly forbids 7.1% of moves
that appear on real corpus teams, and the merged rule is simultaneously too
permissive (it let "Sneasler can't learn Reflect" through) and still misses 0.7%.

Trick: one validator call tests up to 24 (species, move) pairs, because a team is
six candidates x four moves and the validator reports every violation by name.
"""
import json, re, sys
from collections import defaultdict
import numpy as np
from corpus import load_corpus, norm, STATS
from encode import Vocab, NF, NSLOT
from propose import Validator, slot_to_text

CANT = re.compile(r"^(.*?) can'?t learn (.+?)\.?$")

def main():
    teams, _ = load_corpus(); teams = [t for t in teams if len(t) == 6]
    V = Vocab(teams)
    # a known-good template slot per species (from the corpus, so ability/item/spread are valid)
    tmpl = {}
    for t in teams:
        for s in t:
            tmpl.setdefault(norm(s["species"]), s)
    species = sorted(tmpl)
    moves_vocab = [m for m in V.itos["move"] if m]
    disp = {}
    for t in teams:
        for s in t:
            for m in s["moves"]: disp[norm(m)] = m
    moves_disp = [disp.get(m, m) for m in moves_vocab]
    print(f"probing {len(species)} species x {len(moves_vocab)} moves "
          f"= {len(species)*len(moves_vocab):,} pairs")

    val = Validator()
    illegal = defaultdict(set)
    # Showdown reports only the FIRST illegal move per candidate, so each slot
    # carries exactly ONE test move. Six species per call = six pairs per call.
    calls = 0
    for mi, mv in enumerate(moves_disp):
        for gi in range(0, len(species), NSLOT):
            group = species[gi:gi+NSLOT]
            slots = []
            for sp in group:
                s = dict(tmpl[sp]); s["moves"] = [mv]
                s["item"] = ""          # strip items: probe teams must not trip Item Clause
                slots.append(s)
            txt = "\n\n".join(slot_to_text(s) for s in slots) + "\n"
            err = val(txt); calls += 1
            if err:
                for part in err.split(";"):
                    m = CANT.match(part.strip())
                    if m:
                        illegal[norm(m.group(1))].add(norm(m.group(2)))
        if (mi+1) % 40 == 0:
            print(f"  {mi+1}/{len(moves_disp)} moves, {calls} calls", flush=True)
    val.close()
    table = {sp: sorted(set(moves_vocab) - illegal.get(sp, set())) for sp in species}
    json.dump(table, open("/tmp/vgc-pilot/learnset_true.json", "w"))
    sizes = [len(v) for v in table.values()]
    print(f"calls={calls}  mean legal moves/species {np.mean(sizes):.0f} "
          f"(min {min(sizes)}, max {max(sizes)})")
    # sanity: every corpus (species,move) pair must be legal in the table
    miss = 0; tot = 0
    for t in teams:
        for s in t:
            for mv in s["moves"]:
                tot += 1
                if norm(mv) not in table.get(norm(s["species"]), []): miss += 1
    print(f"corpus pairs wrongly forbidden by the true table: {miss}/{tot} ({miss/tot:.2%})")
    print("wrote /tmp/vgc-pilot/learnset_true.json")

if __name__ == "__main__":
    main()
