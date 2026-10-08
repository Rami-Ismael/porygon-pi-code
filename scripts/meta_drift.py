"""Is the objective stationary over the top-50 window? Battle-free, from the matchup database.

Rows: every team with all 49 top-50 columns battled. Columns split by placement date (June vs August 2026;
July is held out so the two periods are separated). For each row, its weighted score against each period.
  between-period agreement : Spearman(score vs June teams, score vs August teams) across rows
  noise ceiling            : the same Spearman for random column splits of the same sizes, July excluded
  shift                    : mean(score vs August) - mean(score vs June)
Drift shows as between-period agreement below the noise ceiling. Bootstrap over rows (2,000).
Run: /tmp/vgc-pilot/.venv/bin/python scripts/meta_drift.py
"""
import json, sys
from datetime import datetime
from pathlib import Path
import numpy as np
sys.path.insert(0, "/tmp/vgc-pilot/src"); sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import matchup_db as MDB
import activesearch as A

RES = Path(__file__).resolve().parents[1] / "results"
con = MDB.connect(); pid = MDB.get_policy(con)
dates = {t["id"]: datetime.strptime(t["date"], "%d %b %Y") for t in json.load(open(RES / "top50_evs.json"))}
cols = {}
for tid, w in con.execute("SELECT team_id, weight FROM team_set WHERE set_name='top50'"):
    srcs = [s for (s,) in con.execute("SELECT source_id FROM team_source WHERE team_id=? AND source='vgcpastes'", (tid,)) if s in dates]
    cols[tid] = (w, min(dates[s] for s in srcs))
month = {c: d.month for c, (w, d) in cols.items()}
rows, M = [], []
for (r,) in con.execute("SELECT DISTINCT row_team FROM matchup WHERE policy_id=?", (pid,)):
    cells = MDB.row_cells(con, pid, r, list(cols))
    need = [c for c in cols if c != r]
    if all(c in cells for c in need):
        rows.append(r); M.append({c: cells[c][0] / cells[c][1] for c in need})
def score(m, cs):
    cs = [c for c in cs if c in m]; W = sum(cols[c][0] for c in cs)
    return sum(cols[c][0] * m[c] for c in cs) / W
june = [c for c in cols if month[c] == 6]; aug = [c for c in cols if month[c] == 8]
pool = june + aug
def agreement(idx, a, b):
    return A.spearman([score(M[i], a) for i in idx], [score(M[i], b) for i in idx])
rng = np.random.default_rng(0); idx = np.arange(len(rows))
between = agreement(idx, june, aug)
def split():
    p = rng.permutation(pool); return list(p[:len(june)]), list(p[len(june):])
ceiling = float(np.mean([agreement(idx, *split()) for _ in range(200)]))
boot = []
for _ in range(2000):
    b = rng.integers(0, len(rows), len(rows)); a1, a2 = split()
    boot.append(agreement(b, june, aug) - agreement(b, a1, a2))
shift = float(np.mean([score(m, aug) - score(m, june) for m in M]))
out = dict(rows=len(rows), june_columns=len(june), august_columns=len(aug), between_period_spearman=between,
           noise_ceiling_spearman=ceiling, drift_gap=between - ceiling,
           drift_gap_ci95=[float(x) for x in np.percentile(boot, [2.5, 97.5])], mean_shift_aug_minus_june=shift,
           note="rows are teams in the matchup database with every top-50 column battled (8+ battles per cell)")
json.dump(out, open(RES / "meta_drift.json", "w"), indent=1); print(json.dumps(out, indent=1))
