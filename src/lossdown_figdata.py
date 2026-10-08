"""Data for the loss-down explainer page (2026-10-08): one test team field by field, and legal-option
counts at prediction time in the deployed order vs a random order. -> results/lossdown_figdata.json"""
import json, math, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, "/tmp/vgc-pilot/src"); sys.path.insert(0, str(Path(__file__).resolve().parent))
import lossdown as L, diffusion as D, asked_vs_got as X
from encode import NF, NSLOT

corpus, V, *_ = X.setup(); d = L.load_data(); T = L.Tables(V, corpus)
look = {}
for t in corpus:
    for s in t:
        for k, v in (("species", s["species"]), ("ability", s["ability"]), ("item", s["item"]), ("nature", s["nature"])):
            from corpus import norm; look[norm(v)] = v
        for m in s["moves"]: look[norm(m)] = m
name = lambda c, v: look.get(V.decode_field(c, v), V.decode_field(c, v)) or "(none)"

old = X.ContinuousWR(V).to(L.DEV); old.load_state_dict(torch.load(L.RES / "asked_vs_got.pt", map_location=L.DEV)["sd"]); old.eval()
F1 = L.load_net("F1_final_medium", V)[0]; G2 = L.load_net("G2_family_matrix_noprtrain", V)[0]
models = {"old": L.logprobs_fn(old, None), "old_masked": L.logprobs_fn(old, T),
          "new": L.ensemble_fn([L.logprobs_fn(F1, T), L.logprobs_fn(G2, T)])}
R = json.load(open(L.OUT))
pt_new = np.array(R["ENS_F1_G2"]["per_team_test_exact"]); pt_old = np.array(R["OLD asked_vs_got.pt (2026-10-04, best so far)"]["per_team_test_exact"])
spr = {p.stem for p in Path("/Users/ramiismael/Documents/yakumsi-vault/Personal Project/Using Advance method search in the whole search of possible vgc pokemon format to find the right counter to a pokemon team/Blog/visuals/search-loop/sprites").glob("*.png")}
# a representative team: every species has a sprite, and both models near their median
ok = [i for i in range(500) if all(V.decode_field(s * NF, d["xte"][i, s * NF]) in spr for s in range(NSLOT))]
med_new, med_old = np.median(pt_new), np.median(pt_old)
i = min(ok, key=lambda i: abs(pt_new[i] - med_new) / pt_new.std() + abs(pt_old[i] - med_old) / pt_old.std())
x = torch.as_tensor(d["xte"][i:i + 1], device=L.DEV); w = torch.as_tensor(d["wte"][i:i + 1], device=L.DEV)

@torch.no_grad()
def walk(f):
    cur = torch.zeros_like(x); out = []
    for step, c in enumerate(D.ORDER):
        t = torch.full((1,), 1.0 - step / L.COLS, device=L.DEV); k = L.KEY_OF_COL[c]; j = L.COLS_BY_KEY[k].index(c)
        lp = f(cur, t, w)[k][0, j]; p = float(lp[x[0, c]].exp())
        top = torch.topk(lp, 3); n_ok = int(T.allowed(cur)[k][0, j].sum())
        out.append(dict(step=step, col=c, slot=c // NF, field=D.FIELDS[c % NF], key=k, truth=name(c, int(x[0, c])),
                        p=round(p, 5), nats=round(-math.log(max(p, 1e-12)), 4), legal=n_ok, vocab=V.sizes[k] - 1,
                        top=[[name(c, int(v)), round(float(s.exp()), 4)] for s, v in zip(top.values, top.indices)]))
        cur[0, c] = x[0, c]
    return out
team = dict(index=i, win_rate=float(d["wte"][i]), species=[name(s * NF, int(x[0, s * NF])) for s in range(NSLOT)],
            species_id=[V.decode_field(s * NF, int(x[0, s * NF])) for s in range(NSLOT)],
            walks={m: walk(f) for m, f in models.items()})
for m, wk in team["walks"].items(): print(m, "team total nats", round(sum(s["nats"] for s in wk), 2))

# legal options at prediction time: deployed order vs random orders, over the 500 test teams
@torch.no_grad()
def options(order_fn, reps):
    acc = {k: [] for k in L.KEYS}
    xa = torch.as_tensor(d["xte"], device=L.DEV); g = torch.Generator().manual_seed(0)
    for _ in range(reps):
        order = order_fn(g); cur = torch.zeros_like(xa)
        for c in order:
            k = L.KEY_OF_COL[c]; j = L.COLS_BY_KEY[k].index(c)
            acc[k].append(T.allowed(cur)[k][:, j].sum(1).float().mean().item()); cur[:, c] = xa[:, c]
    return {k: round(float(np.mean(v)), 1) for k, v in acc.items()}
opts = dict(deployed=options(lambda g: list(D.ORDER), 1),
            random=options(lambda g: torch.randperm(L.COLS, generator=g).tolist(), 8),
            vocab={k: V.sizes[k] - 1 for k in L.KEYS})
print(opts)
json.dump(dict(team=team, options=opts), open(L.RES / "lossdown_figdata.json", "w"), indent=1)
