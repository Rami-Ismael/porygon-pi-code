"""Does the model still condition on the ask? (2026-10-16, docs/ask-dial.md)

For a set of teams with well-measured win rates, ask each model which win rate best explains each team:
the ask y on a grid 0.1 ... 0.8 that minimises the team's negative ELBO under the model (common random numbers,
so every y sees the same masks). Flat prior, so this is the model's maximum-likelihood label for the team.

  sensitivity   mean over teams of (max - min) negative ELBO across the grid, in nats per team: about 0 = the model
                ignores the ask (a forward model of "teams like my training set").
  implied vs true   correlation, and the mean implied label for teams whose true win rate is low / mid / high.

Evaluation teams: the 1,105 precise-label teams (ask_dial.py relabel, about 196 battles each, true win rates 0.0-0.8).
They were training data for the precise model (so its row is in-sample), and many sit in the original v3 training
pool; the comparison between models is the point, not the absolute fit.

    python implied_label.py        # writes results/implied_label.json
"""
import json, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "/tmp/vgc-pilot/src")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import foundation as FD
import lossdown as L
from corpus import parse_team_text
from checkpoints import load_checkpoint

GRID = [round(0.1 * i, 1) for i in range(1, 9)]
MODELS = {"v3 as trained": None, "fixed-ask loop g3 (elite re-steer)": "cem_fixed_elite_g3",
          "Diffusion-BBO fine-tune g3": "cem_dbbo_g3", "precise labels": "dial_precise"}
REPS = 8


def members(G, prefix):
    nets = []
    for name in FD.V3_MEMBERS:
        net, cfg, _ = L.load_net(name, G.V)
        if prefix: net.load_state_dict(load_checkpoint(L.CK / f"{prefix}_{name}.pt", map_location=L.DEV)["sd"])
        net.eval(); nets.append(L.logprobs_fn(net, G.tables, tuple(cfg["rules"].split(","))))
    return L.ensemble_fn(nets)


def scratch(G, ck):
    z = load_checkpoint(ck, map_location=L.DEV); cfg = z["cfg"]
    net = L.Net(G.V, d=cfg["d"], nhead=cfg["nhead"], nlayer=cfg["nlayer"], dropout=cfg["dropout"], time_cond=bool(cfg["time_cond"]),
                ntag=0, prime=cfg["prime"], mrow=0).to(L.DEV)
    net.load_state_dict(z["sd"]); net.eval()
    return L.logprobs_fn(net, G.tables, tuple(cfg["rules"].split(",")))


def main():
    G = FD.Gen("v3"); V = G.V
    d = json.load(open(L.RES / "ask_dial.json"))["teams"]
    X = np.stack([V.encode(parse_team_text(r["paste"])) for r in d]); true = np.array([r["new"] for r in d])
    fs = {k: members(G, p) for k, p in MODELS.items()}
    sc = sorted(L.CK.glob("cem_scratch_g*_scratch.pt"))
    if sc: fs[f"from scratch ({sc[-1].stem.split('_')[2]})"] = scratch(G, sc[-1])
    out = {}
    for name, f in fs.items():
        nll = []
        for y in GRID:
            w = np.full(len(X), y, np.float32)
            with torch.no_grad():
                _, pt = L.evaluate(f, X, w, reps=REPS, batch=256)
            nll.append(np.asarray(pt, float))
        nll = np.stack(nll, 1)                                   # teams x grid
        implied = np.array(GRID)[nll.argmin(1)]
        sens = float((nll.max(1) - nll.min(1)).mean())
        bins = {"true < 0.3": true < 0.3, "0.3-0.5": (true >= 0.3) & (true < 0.5), "true >= 0.5": true >= 0.5}
        res = dict(sensitivity_nats=sens, corr=float(np.corrcoef(implied, true)[0, 1]),
                   implied_mean_by_true={k: float(implied[m].mean()) for k, m in bins.items()},
                   implied_hist={str(y): int((implied == y).sum()) for y in GRID})
        out[name] = res
        print(f"{name:40s} sensitivity {sens:6.2f} nats · implied vs true r {res['corr']:+.2f} · implied mean by true win rate: "
              + " · ".join(f"{k} → {v:.2f}" for k, v in res["implied_mean_by_true"].items()), flush=True)
    json.dump(dict(grid=GRID, n=len(X), models=out), open(L.RES / "implied_label.json", "w"), indent=1)


if __name__ == "__main__":
    main()
