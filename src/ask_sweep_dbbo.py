"""One-shot ask sweep: does training on all data with shrunk labels push the ask dial past 0.5? (2026-10-16)

Two models, same decoder, no loop:
  v3_original   the original v3 members (F1 + G2 noheldout), untouched
  dbbo_g3       the same members after cem_dbbo's generation-3 fine-tune (all labelled teams, shrunk labels)
Decoder = the loop's: guidance 2, species at temperature 1, every other field at 0.5, nature repair.
No ridge guidance, no taboo, no ridge pick. Asks 0.1 ... 0.9; 64 valid teams per ask; one battle per top-50
column (49) on the matchup matrix, behaviour-cloned policy (policy 1) on both sides.

    python battle_supervisor.py ask_sweep_dbbo.py run --then report
"""
import json, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "/tmp/vgc-pilot/src")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cem_dbbo as CB
import foundation as FD
import lossdown as L
import lossdown_battle as LB
import matchup_db as MDB
import heldout as HO
from hps_generate import Validator
from checkpoints import load_checkpoint

import os
# ASKSWEEP_TAG / ASKSWEEP_G / ASKSWEEP_MODELS ("name=prefix,..."; prefix "-" = original members) reuse this sweep
# for the asked-equals-got follow-ups: guidance 4, and the precise-label model (ask_dial.py).
OUT = L.RES / f"{os.environ.get('ASKSWEEP_TAG', 'ask_sweep_dbbo')}.json"
ASKS = [round(0.1 * i, 1) for i in range(1, 10)]
N, G_W, SEED = 64, float(os.environ.get("ASKSWEEP_G", 2.0)), 7171
MODELS = {"v3_original": None, "dbbo_g3": "cem_dbbo_g3"}
if os.environ.get("ASKSWEEP_MODELS"):
    MODELS = {k: (None if v == "-" else v) for k, v in (x.split("=") for x in os.environ["ASKSWEEP_MODELS"].split(","))}


def save(out): json.dump(out, open(OUT, "w"))


def model(G, prefix):
    nets = []
    for name in FD.V3_MEMBERS:
        net, cfg, _ = L.load_net(name, G.V)
        if prefix: net.load_state_dict(load_checkpoint(L.CK / f"{prefix}_{name}.pt", map_location=L.DEV)["sd"])
        net.eval(); nets.append(L.logprobs_fn(net, G.tables, tuple(cfg["rules"].split(","))))
    return L.ensemble_fn(nets)


def run():
    LB._stack(); con = MDB.connect()
    out = json.load(open(OUT)) if OUT.exists() else {"config": dict(asks=ASKS, n=N, guidance=G_W, seed=SEED, temp_sets=0.5,
                                                                    models={k: v for k, v in MODELS.items()}), "cells": {}}
    G = FD.Gen("v3")
    for mname, prefix in MODELS.items():
        G.f = model(G, prefix)
        for i, a in enumerate(ASKS):
            key = f"{mname}_y{a}"
            c = out["cells"].get(key)
            if c and "y" in c: continue
            if not c:
                rng = np.random.default_rng(SEED + i); torch.manual_seed(SEED + i)
                val = Validator(); keep, tried = [], 0; t0 = time.perf_counter()
                try:
                    while len(keep) < N and tried < 8 * N:
                        for r in CB.sample_asks(G, torch.full((96,), a), G_W, None, 0.0).cpu().numpy():
                            tried += 1
                            p, _ = G.to_paste(r, rng, repair=True)
                            if val(p) is None and not HO.is_heldout_text(p): keep.append(p)
                            if len(keep) >= N: break
                finally:
                    val.close()
                c = out["cells"][key] = dict(model=mname, asked=a, pastes=keep, tried=tried, valid=len(keep) / tried,
                                             decode_s=time.perf_counter() - t0)
                save(out)
            v = MDB.Validator()
            try: ids = [MDB.add_team(con, p, f"asksweep:{OUT.stem}:{key}", v)[0] for p in c["pastes"]]
            finally: v.close()
            legal = {t for (t,) in con.execute(f"SELECT team_id FROM team WHERE legal=1 AND team_id IN ({','.join('?' * len(ids))})", ids)}
            MDB.ensure(con, LB.POLICY, sorted(legal), MDB.set_members(con, "top50"), LB.PER_CELL, seed=SEED + i, note=f"asksweep {OUT.stem} {key}")
            sc = [MDB.score(con, LB.POLICY, t) if t in legal else None for t in ids]
            c.update(team_id=ids, y=[s["score"] if s else None for s in sc]); save(out)
            y = [x for x in c["y"] if x is not None]
            print(f"[{key}] {len(y)} teams: got {np.mean(y):.3f} ± {np.std(y, ddof=1) / np.sqrt(len(y)):.3f} · valid {c['valid']:.2f}", flush=True)


def report():
    out = json.load(open(OUT))
    print("got (mean win rate vs top-50 meta teams, 49 battles per team) by ask")
    print(f"{'ask':>5s} " + " ".join(f"{m:>16s}" for m in MODELS))
    summ = {}
    for a in ASKS:
        row = []
        for m in MODELS:
            c = out["cells"].get(f"{m}_y{a}", {}); y = [x for x in c.get("y", []) if x is not None]
            if y:
                mu, se = float(np.mean(y)), float(np.std(y, ddof=1) / np.sqrt(len(y))); summ[f"{m}_y{a}"] = dict(mean=mu, se=se, n=len(y))
                row.append(f"{mu:.3f} ± {se:.3f}")
            else: row.append("-")
        print(f"{a:5.1f} " + " ".join(f"{r:>16s}" for r in row))
    for m in MODELS:
        pts = [(a, summ[f"{m}_y{a}"]["mean"]) for a in ASKS if f"{m}_y{a}" in summ]
        if len(pts) > 2:
            lo = [p for p in pts if p[0] <= 0.5]; b = np.polyfit(*zip(*lo), 1)[0] if len(lo) > 1 else float("nan")
            print(f"{m}: peak {max(pts, key=lambda p: p[1])} · slope over asks 0.1-0.5 {b:+.3f}")
    out["summary"] = summ; save(out)


if __name__ == "__main__":
    {"run": run, "report": report}[sys.argv[1]]()
