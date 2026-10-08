"""A corruption ladder: real teams with k of six candidates randomised, k = 0..6.

Purely random legal teams are useless as training labels — the pilot measured
median win rate 0.000 over 200 of them. Conditioning on win rate needs win rates
that vary. This ladder interpolates from a real team down to a random one, so the
labels span the range.

It is also the diffusion forward process by another name: replacing k of six slots
is corruption at noise level k/6, which is exactly what the model is trained to undo.
"""
import json, sys
from pathlib import Path
import numpy as np
from corpus import load_corpus, norm, STATS
from encode import canon
from propose import Validator, slot_to_text, team_to_text
from random_team import build_tables, one_team, TRUE
from corpus import MEGA_STONE_OF

def main():
    per = int(sys.argv[1]) if len(sys.argv) > 1 else 350
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    out = Path("/tmp/vgc-pilot/ladder"); out.mkdir(exist_ok=True)
    for f in out.glob("*.txt"): f.unlink()
    rng = np.random.default_rng(seed)
    teams, _ = load_corpus(); teams = [t for t in teams if len(t) == 6]
    T = build_tables()
    val = Validator()
    made = {k: 0 for k in range(7)}
    meta = {}
    for k in range(7):
        tries = 0
        while made[k] < per and tries < per * 25:
            tries += 1
            base = [dict(s) for s in canon(teams[int(rng.integers(0, len(teams)))])]
            if k > 0:
                rnd = one_team(T, rng)
                if rnd is None: continue
                idx = list(rng.choice(6, k, replace=False))
                for j, i in enumerate(idx): base[i] = rnd[j]
                # re-check the clauses the splice can break
                b = [norm(s["species"]) for s in base]
                it = [norm(s["item"]) for s in base if s["item"]]
                if len(set(b)) != 6 or len(set(it)) != len(it): continue
            txt = team_to_text(base)
            if val(txt) is not None: continue
            name = f"k{k}_{made[k]:05d}.txt"
            (out / name).write_text(txt)
            meta[str(out / name)] = {"k": k}
            made[k] += 1
        print(f"  k={k}: {made[k]} teams ({made[k]/max(tries,1):.0%} accepted)", flush=True)
    val.close()
    json.dump(meta, open("/tmp/vgc-pilot/ladder_meta.json", "w"))
    print(f"total {sum(made.values())} teams -> {out}")

if __name__ == "__main__":
    main()
