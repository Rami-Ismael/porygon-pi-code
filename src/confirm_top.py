"""Independent confirmation of the 8 best re-battled teams of the diversity run (2026-10-09).
Each team already has 4 battles per top-50 opponent in the matchup matrix (196; 49 of them picked it).
Raise every cell to 8 and score the NEW battles only: (wins_after - wins_before) / (battles_after - battles_before).
    python battle_supervisor.py confirm_top.py run
"""
import json, sys
from pathlib import Path
sys.path.insert(0, "/tmp/vgc-pilot/src"); sys.path.insert(0, str(Path(__file__).resolve().parent))
import lossdown as L, lossdown_battle as LB, matchup_db as MDB
OUT = L.RES / "confirm_top.json"

def run():
    LB._stack(); con = MDB.connect()
    o = json.load(open(L.RES / "diversity_methods_all.json")); rows = {}
    for k, v in o["rebattle"].items():
        for r in v: rows.setdefault(r["team_id"], (r["score"], k, r["label"], r["paste"]))
    top = sorted(rows.items(), key=lambda kv: -kv[1][0])[:8]
    cols = MDB.set_members(con, "top50"); ids = [t for t, _ in top]
    tot = lambda t: (lambda c: (sum(w for w, n in c.values()), sum(n for w, n in c.values())))(MDB.row_cells(con, LB.POLICY, t, [x for x in cols if x != t]))
    before = {t: tot(t) for t in ids}
    MDB.ensure(con, LB.POLICY, ids, cols, 8, note="confirm_top 2026-10-09")
    out = []
    for t, (s, k, l, p) in top:
        w1, n1 = tot(t); w0, n0 = before[t]
        out.append(dict(team_id=t, arm=k, label=l, rebattle=s, new_battles=n1 - n0, fresh=(w1 - w0) / (n1 - n0) if n1 > n0 else None, paste=p))
        print(t, k, f"label {l:.3f} rebattle {s:.3f} fresh {out[-1]['fresh']} ({n1 - n0} new battles)", flush=True)
    json.dump(out, open(OUT, "w"))

if __name__ == "__main__": run()
