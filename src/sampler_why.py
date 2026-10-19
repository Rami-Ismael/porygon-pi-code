"""Why does Path Planning hold up where today's sampler falls? (2026-10-11, follow-up to sampler_battle.py)

Path Planning differs from today's sampler in three ways at once. Three control arms pull them apart:
  temp07, temp05   today's fixed order, temperature 0.7 / 0.5 on the guided logits
                   -> is it only sharpening (keeping high-probability draws)?
  conforder        48 passes; each pass draws every masked field and commits the single most
                   confident draw; nothing is ever re-masked
                   -> is it the confidence order, without the revision?
  settemp07/05     today's fixed order, temperature 1 on the six species fields and 0.7 / 0.5 on
                   every other field (added after the first three: the planner's trace showed it
                   rarely changes a species but redraws items and moves several times)
                   -> is it sharpening of the sets while the species stay as varied as today?
  (planning itself, from sampler_battle.json, = confidence order + re-masking fields that stop fitting)
Same protocol as sampler_battle.py: asked {0.1 .. 0.9}, guidance 2, 64 valid teams, one battle per top-50
column, policy 1. Results in results/sampler_why.json.

A diagnostic needs no battle: for every stored team, mask one field at a time and ask the model how
probable the true value is given the other 47 (a pseudo-log-likelihood), without the win-rate
condition, with it, and under the guided mix. Results in results/sampler_why_diag.json.

    python sampler_why.py diag
    python sampler_why.py gen
    python battle_supervisor.py sampler_why.py battle --then report
(run with /tmp/vgc-pilot/.venv/bin/python)
"""
import json, math, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "/tmp/vgc-pilot/src")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lossdown as L
import sampler_battle as SB

MAIN = L.RES / "sampler_battle.json"
OUT = L.RES / "sampler_why.json"
DIAG = L.RES / "sampler_why_diag.json"
DEV, COLS, NINF = SB.DEV, SB.COLS, SB.NINF
ARMS = ["temp07", "temp05", "conforder", "settemp07", "settemp05"]
KINDS = ["species", "ability", "item", "move", "nature"]


@torch.no_grad()
def arm_today_temp(n, tau, species_tau=None):
    """lossdown_battle.sample with the guided logits divided by tau before the draw (tau = 1 is today's).
    species_tau, if given, replaces tau on the six species fields."""
    f, tables = SB.f, SB.tables
    w = torch.full((n,), SB.YSTAR, device=DEV); wn = torch.full((n,), float("nan"), device=DEV)
    x = torch.zeros(n, COLS, dtype=torch.long, device=DEV)
    for step, c in enumerate(SB.ORDER):
        t = torch.full((n,), 1.0 - step / COLS, device=DEV)
        k = L.KEY_OF_COL[c]; j = L.COLS_BY_KEY[k].index(c)
        lc = SB.G * f(x, t, w)[k][:, j] + (1 - SB.G) * f(x, t, wn)[k][:, j]
        ok = tables.allowed(x)[k][:, j]
        ok[~ok.any(1)] = True; ok[:, 0] = False
        lc = lc.masked_fill(~ok, NINF)
        tc = species_tau if (species_tau is not None and k == "species") else tau
        x[:, c] = torch.multinomial(torch.softmax(lc / tc, -1), 1).squeeze(1)
    return x


@torch.no_grad()
def arm_conforder(n):
    x = torch.zeros(n, COLS, dtype=torch.long, device=DEV)
    ar = torch.arange(n, device=DEV)
    for _ in range(COLS):
        masked = x == 0
        lg = SB.guided(x, SB.mfrac(x))
        logp = torch.log_softmax(lg.masked_fill(~SB.mask_T(x), NINF), -1)
        z = torch.multinomial(logp.exp().view(-1, SB.VMAX), 1).view(n, COLS)
        score = logp.gather(2, z.unsqueeze(-1)).squeeze(-1).masked_fill(~masked, NINF)
        rev = torch.zeros(n, COLS, dtype=torch.bool, device=DEV)
        rev[ar, score.argmax(1)] = True
        SB.commit_set(x, lg, rev & masked, pre=z)
    return x


def _route():
    SB.OUT, SB.ARMS, SB.SEED = OUT, ARMS, 5252
    SB.FN.update(temp07=lambda n: arm_today_temp(n, 0.7), temp05=lambda n: arm_today_temp(n, 0.5),
                 conforder=arm_conforder, settemp07=lambda n: arm_today_temp(n, 0.7, 1.0),
                 settemp05=lambda n: arm_today_temp(n, 0.5, 1.0))


@torch.no_grad()
def pll(rows, ystar):
    """-> dict of [n,48] arrays: log p(true value | other 47 fields) without the condition (u), with it (c),
    and under the guided mix the samplers draw from (g). Illegal-under-the-mask values are floored at -30."""
    x = torch.tensor(np.array(rows), dtype=torch.long, device=DEV); n = len(x)
    t = torch.full((n,), 1.0 / COLS, device=DEV)
    w = torch.full((n,), ystar, device=DEV); wn = torch.full((n,), float("nan"), device=DEV)
    out = {k: torch.zeros(n, COLS) for k in "ucg"}
    for c in range(COLS):
        xc = x.clone(); xc[:, c] = 0
        k = L.KEY_OF_COL[c]; j = L.COLS_BY_KEY[k].index(c)
        ok = SB.mask_T(xc)[:, c, :SB.SZ[c]]
        lc, lu = SB.f(xc, t, w)[k][:, j], SB.f(xc, t, wn)[k][:, j]
        for name, lg in (("u", lu), ("c", lc), ("g", SB.G * lc + (1 - SB.G) * lu)):
            lp = torch.log_softmax(lg.masked_fill(~ok, NINF), -1).gather(1, x[:, c:c + 1]).squeeze(1)
            out[name][:, c] = lp.clamp_min(-30.0).cpu()
    return out


def diag():
    SB._setup()
    kind_cols = {k: [c for c in range(COLS) if L.KEY_OF_COL[c] == k] for k in KINDS}
    res = json.load(open(DIAG)) if DIAG.exists() else {}
    for path in (MAIN, OUT):
        if not path.exists(): continue
        for tag, cell in json.load(open(path))["cells"].items():
            if tag in res or "y" not in cell: continue
            p = pll(cell["rows"], cell["asked"])
            res[tag] = dict(arm=cell["arm"], asked=cell["asked"], y=cell["y"],
                            **{f"{name}_{k}": p[name][:, cols].mean(1).tolist() for name in "ucg" for k, cols in kind_cols.items()},
                            **{f"{name}_all": p[name].mean(1).tolist() for name in "ucg"},
                            floor_hits=int((p["u"] <= -30).sum()))
            json.dump(res, open(DIAG, "w"))
            m = lambda key: float(np.mean(res[tag][key]))
            print(f"  {tag:16s} win {np.mean(cell['y']):.3f}  u {m('u_all'):6.3f}  c {m('c_all'):6.3f}  g {m('g_all'):6.3f}  "
                  + "  ".join(f"{k[:3]} {m('u_' + k):6.2f}" for k in KINDS) + f"  floor {res[tag]['floor_hits']}", flush=True)


def report():
    _route(); SB.report()


if __name__ == "__main__":
    st = sys.argv[1]
    if st == "diag": diag()
    elif st == "gen": _route(); SB.gen("--smoke" in sys.argv)
    elif st == "battle": _route(); SB.battle()
    else: report()
