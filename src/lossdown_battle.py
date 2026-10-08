"""Battle the loss-down model's teams against the top 50 (2026-10-08).

Question: the F1 + G2 mixture (docs/loss-down.md) predicts held-out loop teams 40% better
than asked_vs_got.pt. Does a better density model WRITE better teams?

Same protocol as experiment A (asked_vs_got.py), so every cell compares 1:1 with the old model:
  cells     uncond, and asked y* in {0.3, 0.5, 0.7} at guidance 1 / 2 / 4
  teams     64 Showdown-valid teams per cell, temperature 1, corpus Stat Point spreads
  decoding  diffusion.ORDER (species -> ability/item -> moves -> nature), the legality masks the
            models were trained with, classifier-free guidance on the win-rate condition
  battles   one per (team, top-50 column), BC policy both sides (policy 1), all in the matchup
            matrix with origin 'lossdown:<cell>'
Win rate = wins / battles against the top-50 meta teams; a top-50 team vs the other 49 = 0.531.

    python lossdown_battle.py gen
    python battle_supervisor.py lossdown_battle.py battle --then report
(run with /tmp/vgc-pilot/.venv/bin/python)
"""
import json, math, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "/tmp/vgc-pilot/src")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lossdown as L
import asked_vs_got as X
import activesearch as A
import diffusion as D
import matchup_db as MDB
import pool
import hps_generate
from hps_generate import Validator
from corpus import parse_team_text
hps_generate.SHOWDOWN = Path.home() / ".local/share/vgc-pilot-runtime/validator-913da36"

OUT = L.RES / "lossdown_battle.json"
MEMBERS = ["F1_final_medium", "G2_family_matrix_noprtrain"]
ASKED, GUIDE = [0.3, 0.5, 0.7], [1.0, 2.0, 4.0]
N, SEED, POLICY, PER_CELL = 64, 2026, 1, 1
NAN = float("nan")


def cells():
    return [("uncond", None, 1.0)] + [(f"y{a:.1f}_g{g:g}", a, g) for g in GUIDE for a in ASKED]


def model():
    corpus, V, C, look, spreads = X.setup()
    tables = L.Tables(V, corpus)
    fs = []
    for m in MEMBERS:
        net, cfg, _ = L.load_net(m, V)
        fs.append(L.logprobs_fn(net, tables, tuple(cfg["rules"].split(","))))
    return V, tables, L.ensemble_fn(fs), look, spreads


@torch.no_grad()
def sample(f, tables, n, ystar, g):
    """Fixed-order constrained decode of the mixture; D-CFG on the win-rate condition."""
    w = torch.full((n,), NAN if ystar is None else ystar, device=L.DEV)
    wn = torch.full((n,), NAN, device=L.DEV)
    x = torch.zeros(n, L.COLS, dtype=torch.long, device=L.DEV)
    for step, c in enumerate(D.ORDER):
        t = torch.full((n,), 1.0 - step / L.COLS, device=L.DEV)
        k = L.KEY_OF_COL[c]; j = L.COLS_BY_KEY[k].index(c)
        lc = f(x, t, w)[k][:, j]
        if ystar is not None and g != 1.0:
            lc = g * lc + (1 - g) * f(x, t, wn)[k][:, j]
        ok = tables.allowed(x)[k][:, j]
        ok[~ok.any(1)] = True; ok[:, 0] = False                 # never strand a row; never emit [MASK]
        lc = lc.masked_fill(~ok, -float("inf"))
        x[:, c] = torch.multinomial(torch.softmax(lc, -1), 1).squeeze(1)
    return x


def gen():
    V, tables, f, look, spreads = model()
    out = json.load(open(OUT)) if OUT.exists() else {"members": MEMBERS, "cells": {}}
    val = Validator()
    try:
        for i, (tag, a, g) in enumerate(cells()):
            if tag in out["cells"]: continue
            torch.manual_seed(SEED + i); rng = np.random.default_rng(SEED + i)
            pastes, tried, t0 = [], 0, time.perf_counter()
            while len(pastes) < N and tried < N * 8:
                for r in sample(f, tables, 96, a, g).cpu().numpy():
                    tried += 1
                    p = A.row_to_paste(V, r, look, spreads, rng)
                    if val(p) is None: pastes.append(p)
                    if len(pastes) >= N: break
            out["cells"][tag] = dict(asked=a, guidance=g, pastes=pastes, tried=tried, valid=len(pastes) / tried)
            json.dump(out, open(OUT, "w"))
            print(f"  {tag:10s} {len(pastes)}/{tried} valid ({len(pastes)/tried:.0%}) {time.perf_counter()-t0:.0f}s", flush=True)
    finally:
        val.close()


def _stack():
    pool.REPO = str(Path.home() / ".local/share/vgc-pilot-runtime/unpacked/vgc-bench-d79f9532947ac114dce1dda2456a590afcd375b2")
    MDB.SHOWDOWN = hps_generate.SHOWDOWN


def battle():
    """asked_vs_got.battle(), origin 'lossdown:<cell>'. Only missing cells are battled."""
    _stack()
    con = MDB.connect(); out = json.load(open(OUT))
    todo = [t for t, c in out["cells"].items() if "y" not in c]
    for k in range(0, len(todo), 5):
        chunk = todo[k:k + 5]
        pastes = [p for t in chunk for p in out["cells"][t]["pastes"]]
        origins = [t for t in chunk for _ in out["cells"][t]["pastes"]]
        val = MDB.Validator()
        try:
            ids = [MDB.add_team(con, p, f"lossdown:{o}", val)[0] for p, o in zip(pastes, origins)]
        finally:
            val.close()
        legal = {t for (t,) in con.execute(f"SELECT team_id FROM team WHERE legal=1 AND team_id IN ({','.join('?' * len(ids))})", ids)}
        MDB.ensure(con, POLICY, sorted(legal), MDB.set_members(con, "top50"), PER_CELL, seed=SEED,
                   note=f"lossdown_battle {chunk[0]}..{chunk[-1]}")
        i = 0
        for t in chunk:
            n = len(out["cells"][t]["pastes"])
            sc = [MDB.score(con, POLICY, tid) if tid in legal else None for tid in ids[i:i + n]]
            out["cells"][t].update(team_id=ids[i:i + n], y=[s["score"] if s else None for s in sc],
                                   battles=[s["battles"] if s else 0 for s in sc])
            i += n
        json.dump(out, open(OUT, "w"))
        print(f"  battled {chunk}", flush=True)


def report():
    _stack()
    con = MDB.connect(); out = json.load(open(OUT))
    old = json.load(open(X.OUT))["summary"]
    known = {r.tobytes() for r in L.load_data()["X"]}
    corpus, V, *_ = X.setup()
    summ = {}
    for t, c in out["cells"].items():
        if "team_id" in c:                                  # re-read: a supervisor restart may have filled cells
            sc = [MDB.score(con, POLICY, tid) for tid in c["team_id"]]
            c["y"] = [s["score"] if s and s["score"] is not None else None for s in sc]
            c["battles"] = [s["battles"] if s else 0 for s in sc]
        y = np.array([v for v in c.get("y", []) if v is not None], float)
        teams = [parse_team_text(p) for p in c["pastes"]]
        sets = {tuple(sorted(s["species"] for s in tm)) for tm in teams}
        copies = sum(V.encode(tm).tobytes() in known for tm in teams)
        o = old.get(t, {})
        summ[t] = dict(asked=c["asked"], guidance=c["guidance"], n=len(y),
                       got=float(y.mean()) if len(y) else None,
                       se=float(y.std(ddof=1) / math.sqrt(len(y))) if len(y) > 1 else None,
                       battles=int(sum(c.get("battles", []))), valid=c["valid"], species_sets=len(sets), copies=copies,
                       old_got=o.get("got"), old_se=o.get("se"), old_valid=o.get("valid"), old_sets=o.get("species_sets"))
    out["summary"] = summ
    json.dump(out, open(OUT, "w"))
    print(f"{'cell':9s} {'new got':>13s} {'old got':>13s} {'diff':>15s} {'valid new/old':>14s} {'sets new/old':>13s} {'copies':>6s}")
    for t, s in summ.items():
        if s["got"] is None: continue
        d = s["got"] - s["old_got"]; se = math.sqrt(s["se"] ** 2 + s["old_se"] ** 2)
        print(f"{t:9s} {s['got']:.3f}±{s['se']:.3f}  {s['old_got']:.3f}±{s['old_se']:.3f}  {d:+.3f}±{se:.3f} "
              f"({d/se:+.1f}σ) {s['valid']:5.0%}/{s['old_valid']:4.0%} {s['species_sets']:6d}/{s['old_sets']:<4d} {s['copies']:6d}")
    print(f"battles: {sum(s['battles'] for s in summ.values()):,}")


if __name__ == "__main__":
    {"gen": gen, "battle": battle, "report": report}[sys.argv[1]]()
