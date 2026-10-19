"""Drill battle 2026-10-17: today vs first-species-uniform, ask 0.5, guidance 2, 64 valid teams x 49 battles, origin 'pinfirst'."""
import json, sys, time
import numpy as np, torch
sys.path.insert(0, "/tmp/vgc-pilot/src")
import lossdown as L, sampler_battle as SB, activesearch as A
from hps_generate import Validator
OUT = L.RES / "pin_first_drill.json"
SB.OUT, SB.ARMS, SB.ASKED, SB.SEED, SB.ORIGIN = OUT, ["today", "pinfirst"], [0.5], 8181, "pinfirst"
def gen():
    import importlib.util
    spec = importlib.util.spec_from_file_location("pfc", "/tmp/vgc-pilot/pin_first_check_lib.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    out = {"cells": {}}; val = Validator()
    try:
        for arm, pin in (("today", False), ("pinfirst", True)):
            torch.manual_seed(8181 + pin); rng = np.random.default_rng(8181 + pin)
            pastes, rows, tried, t0 = [], [], 0, time.perf_counter()
            while len(pastes) < 64:
                for r in m.sample(96, pin).cpu().numpy():
                    tried += 1; p = A.row_to_paste(m.V, r, m.SB.look, m.SB.spreads, rng)
                    if val(p) is None: pastes.append(p); rows.append(r.tolist())
                    if len(pastes) >= 64: break
            out["cells"][f"{arm}_y0.5"] = dict(arm=arm, asked=0.5, guidance=2.0, pastes=pastes, rows=rows, tried=tried,
                                                valid=len(pastes) / tried, decode_seconds=time.perf_counter() - t0, batches=1)
            print(arm, len(pastes), "/", tried, flush=True)
    finally: val.close()
    json.dump(out, open(OUT, "w"))
if __name__ == "__main__":
    {"gen": gen, "battle": SB.battle, "report": SB.report}[sys.argv[1]]()
