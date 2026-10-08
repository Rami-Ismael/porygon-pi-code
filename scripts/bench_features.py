"""Landscape features (smoothness rho, noise share, evaluation cost, naive-encoding validity) of the
collected combinatorial / latent-space BO benchmarks, with the same statistics as ruggedness2 so
the VGC numbers compare cell by cell.  Pre-registration: docs/bench_features.md.

No battles.  Needs rdkit, selfies, PyTDC, networkx (not in the battle venv):
  uv run --no-project --python 3.11 --with numpy --with rdkit --with selfies --with networkx \
     --with PyTDC --with "setuptools<81" python scripts/bench_features.py [bench ...]
MaxSAT-60 instance: data/frb10-6-4.wcnf (COMBO repo, MaxSAT Evaluation 2018).
"""
import itertools, json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"; PARTS = RES / "bench_features_parts"; PARTS.mkdir(parents=True, exist_ok=True)
N_RAND, N_SEARCH, N_EDIT, N_REP, CLIMB, BOOT = 24, 16, 15, 3, 100, 2000

# ---------------------------------------------------------------- binary / categorical tasks
def contamination_task():
    """COMBO _contamination + lambda*|x|, dynamics drawn once (seed 42, as Bounce) -> deterministic."""
    n, sims, lam = 25, 100, 1e-2
    init_Z = np.random.RandomState(42).beta(1.0, 30.0, size=(sims,))
    lambdas = np.random.RandomState(42).beta(1.0, 17.0 / 3.0, size=(n, sims))
    gammas = np.random.RandomState(42).beta(1.0, 3.0 / 7.0, size=(n, sims))
    def f(x, rng=None):
        Z = np.zeros((n, sims))
        Z[0] = lambdas[0] * (1 - x[0]) * (1 - init_Z) + (1 - gammas[0] * x[0]) * init_Z
        for i in range(1, n):
            Z[i] = lambdas[i] * (1 - x[i]) * (1 - Z[i - 1]) + (1 - gammas[i] * x[i]) * Z[i - 1]
        cons = np.mean(Z < 0.1, axis=1) - (1 - 0.05)
        return float(np.sum(x * 1.0 - cons) + lam * x.sum())
    return f, [2] * n, False

def ising_task():
    """COMBO Ising sparsification: 4x4 grid, 24 edges, KL(p||q_x) + lambda*|x|, instance seed 0."""
    r = np.random.RandomState(0)      # COMBO's generator distribution; numpy RNG instead of torch's
    h = ((r.randint(0, 2, 12) * 2 - 1) * (r.rand(12) * 4.95 + 0.05)).reshape(4, 3)
    v = ((r.randint(0, 2, 12) * 2 - 1) * (r.rand(12) * 4.95 + 0.05)).reshape(3, 4)
    spins = np.array(list(itertools.product([-1, 1], repeat=16))).reshape(-1, 4, 4)
    hp = (spins[:, :, :-1] * spins[:, :, 1:]).reshape(len(spins), -1)          # 12 horizontal products
    vp = (spins[:, :-1, :] * spins[:, 1:, :]).reshape(len(spins), -1)          # 12 vertical products
    J = np.concatenate([h.ravel(), v.ravel()]); P = np.concatenate([hp, vp], axis=1).astype(float)
    def logZ(w):
        e = 2 * P @ w; m = e.max(); return m + np.log(np.exp(e - m).sum())
    e0 = 2 * P @ J; p = np.exp(e0 - e0.max()); p /= p.sum(); EP = p @ P       # E_p[z_i z_j] per edge
    lz0 = logZ(J)
    hi = [0, 2, 4, 7, 9, 11, 14, 16, 18, 21, 22, 23]; vi = sorted(set(range(24)) - set(hi))   # BOCS mapping
    order = np.array(hi + vi)
    def f(x, rng=None):
        mask = x[order].astype(float); Jq = J * mask
        return float(2 * np.sum((J - Jq) * EP) + logZ(Jq) - lz0 + 1e-2 * x.sum())
    return f, [2] * 24, False

def pest_score(x, rng):
    U, n_stages, n_sim = 0.1, x.size, 100
    disc = {1: 0.2, 2: 0.3, 3: 0.3, 4: 0.0}; tol = {1: 1 / 7, 2: 2.5 / 7, 3: 2 / 7, 4: 0.5 / 7}
    price_ = {1: 1.0, 2: 0.8, 3: 0.7, 4: 0.5}; beta = {1: 2 / 7, 2: 3 / 7, 3: 3 / 7, 4: 5 / 7}
    payed = above = 0.0; curr = rng.beta(1.0, 30.0, size=n_sim)
    for i in range(n_stages):
        spread = rng.beta(1.0, 17.0 / 3.0, size=n_sim)
        if x[i] > 0:
            ctrl = rng.beta(1.0, beta[x[i]], size=n_sim); nxt = (1 - ctrl) * curr
            beta[x[i]] += tol[x[i]] / n_stages
            price = price_[x[i]] * (1 - disc[x[i]] / n_stages * float(np.sum(x == x[i])))
        else:
            nxt = spread * (1 - curr) + curr; price = 0
        payed += price; above += np.mean(curr > U); curr = nxt
    return float(payed + above)

def pest_combo_task():      # COMBO: global np.random, never reseeded -> a fresh draw on every call
    return (lambda x, rng: pest_score(x, rng)), [5] * 25, True
def pest_bounce_task():     # Bounce: _pest_control_score(x, seed=self.seed) -> same draw on every call
    return (lambda x, rng=None: pest_score(x, np.random.default_rng(0))), [5] * 25, False

def maxsat_task():
    lines = [l.split() for l in open(HERE.parent / "data" / "frb10-6-4.wcnf") if l[0] not in "cp"]
    w = np.array([float(l[0]) for l in lines]); w = (w - w.mean()) / w.std()
    cl = [(np.abs(np.array(l[1:-1], int)) - 1, np.array(l[1:-1], int) > 0) for l in lines]
    def f(x, rng=None):
        xb = x.astype(bool)
        return float(-np.sum(w * np.array([np.any(xb[i] == s) for i, s in cl])))
    return f, [2] * 60, False

def labs_task():
    n = 50
    def f(x, rng=None):
        s = 2 * x - 1; e = sum(float(np.dot(s[:-k], s[k:])) ** 2 for k in range(1, n))
        return -n * n / (2 * e)
    return f, [2] * n, False

def ackley53_task():
    """Bounce Ackley53: 50 binary + 3 continuous on [0,1] -> [-1,1]; lambda=1e-6 uniform noise."""
    def f(x, rng):
        z = np.array(x, float); z[50:] = -1 + 2 * z[50:]
        r = 20 + np.e - 20 * np.exp(-0.2 * np.sqrt(np.sum(z * z) / 53)) - np.exp(np.sum(np.cos(2 * np.pi * z)) / 53)
        return float(r + 1e-6 * rng.random())
    return f, [2] * 50 + [0] * 3, True        # card 0 = continuous coordinate

# ---------------------------------------------------------------- molecule tasks
_MOL = {}
def mol_env():
    if not _MOL:
        import selfies as sf
        from rdkit import Chem, RDLogger; RDLogger.DisableLog("rdApp.*")
        from tdc.generation import MolGen
        zinc = MolGen(name="ZINC", path=str(PARTS / "tdc")).get_data()["smiles"].tolist()
        _MOL.update(sf=sf, Chem=Chem, zinc=zinc, alphabet=sorted(sf.get_semantic_robust_alphabet()))
    return _MOL

def canon(smi):
    m = mol_env()["Chem"].MolFromSmiles(smi); return mol_env()["Chem"].MolToSmiles(m) if m else None

def pen_logp_task():
    import networkx as nx
    from rdkit.Chem import Crippen, rdmolops, RDConfig
    import os; sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score")); import sascorer
    Chem = mol_env()["Chem"]
    def f(smi, rng=None):                       # LOL-BO smile_to_penalized_logP, negated to minimise
        m = Chem.MolFromSmiles(smi)
        cyc = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(m)))
        c = max([len(j) for j in cyc], default=0); c = c - 6 if c > 6 else 0
        v = ((Crippen.MolLogP(m) - 2.45777691) / 1.43341767 + (-sascorer.calculateScore(m) + 3.05352042) / 0.83460587
             + (-c + 0.04861121) / 0.28746695)
        return -float(v)
    return f, "mol", False

def tdc_task(name):
    def make():
        from tdc import Oracle
        o = Oracle(name=name)
        return (lambda smi, rng=None: -float(o(smi))), "mol", False
    return make

TASKS = {"contamination": contamination_task, "ising": ising_task, "pest_combo": pest_combo_task,
         "pest_bounce": pest_bounce_task, "maxsat60": maxsat_task, "labs50": labs_task,
         "ackley53": ackley53_task, "pen_logp": pen_logp_task,
         "perindopril_mpo": tdc_task("Perindopril_MPO"), "zaleplon_mpo": tdc_task("Zaleplon_MPO"),
         "drd2": tdc_task("DRD2")}

# ---------------------------------------------------------------- neighbourhood + design
def rand_point(card, rng):
    if card == "mol":
        while True:
            smi = mol_env()["zinc"][int(rng.integers(0, len(mol_env()["zinc"])))]
            try: mol_env()["sf"].encoder(smi); return smi
            except Exception: continue
    return np.array([rng.integers(0, c) if c else rng.random() for c in card], dtype=float if 0 in card else int)

def edit(x, card, rng):
    if card == "mol":
        sf = mol_env()["sf"]; toks = list(sf.split_selfies(sf.encoder(x))); c0 = canon(x)
        for _ in range(50):
            t = toks[:]; j = int(rng.integers(0, len(t))); t[j] = mol_env()["alphabet"][int(rng.integers(0, len(mol_env()["alphabet"])))]
            y = sf.decoder("".join(t)); cy = canon(y) if y else None
            if cy and cy != c0: return y      # decoder SMILES (kekulé): re-encodable by SELFIES
        raise RuntimeError("no distinct SELFIES neighbour")
    y = x.copy(); j = int(rng.integers(0, len(card))); c = card[j]
    if c == 0: y[j] = rng.random()
    else: y[j] = rng.choice([v for v in range(c) if v != x[j]])
    return y

def run_task(name):
    f, card, noisy = TASKS[name]()
    rng = np.random.default_rng(1); sim = np.random.default_rng(5000)
    n_eval = [0]; t_eval = [0.0]
    def ev(x):
        t = time.perf_counter(); v = f(x, np.random.default_rng(int(sim.integers(0, 2**31))))
        t_eval[0] += time.perf_counter() - t; n_eval[0] += 1; return v
    out = {}
    for start, n in (("random", N_RAND), ("search", N_SEARCH)):
        anchors = []
        for _ in range(n):
            x = rand_point(card, rng)
            if start == "search":
                fx = ev(x)
                for _ in range(CLIMB):
                    y = edit(x, card, rng); fy = ev(y)
                    if fy < fx: x, fx = y, fy
            f0 = ev(x)
            reps = [ev(x) for _ in range(N_REP)] if noisy else [f0] * N_REP
            anchors.append({"f": f0, "reps": reps,
                            "edits": [{"op": "one_edit", "f": ev(edit(x, card, rng))} for _ in range(N_EDIT)]})
        out[start] = anchors
    out["_cost_s_per_eval"] = t_eval[0] / n_eval[0]; out["_n_eval"] = n_eval[0]
    json.dump(out, open(PARTS / f"{name}.json", "w"))
    print(f"{name}: {n_eval[0]} evals, {out['_cost_s_per_eval']*1e3:.2f} ms/eval", flush=True)

# ---------------------------------------------------------------- statistics (as ruggedness2)
def moments(groups, start, op=None):
    A = groups[start]
    d2 = [(e["f"] - a["f"]) ** 2 for a in A for e in a["edits"] if op is None or e["op"] == op]
    r2 = np.mean([(r - a["f"]) ** 2 for a in A for r in a["reps"]])
    allr2 = np.mean([(r - a["f"]) ** 2 for G in groups.values() for a in G for r in a["reps"]])
    fa = np.array([a["f"] for G in groups.values() for a in G])
    return np.mean(d2), r2, fa.var(ddof=1) - allr2 / 2, allr2 / 2

def stat(groups, start, op=None):
    d2, r2, var_f, sig2 = moments(groups, start, op)
    return {"rho": float(1 - max(d2 - r2, 0.0) / (2 * var_f)), "E_d2": float(d2), "E_r2": float(r2),
            "var_f": float(var_f), "noise_var": float(sig2), "noise_share": float(sig2 / (sig2 + var_f))}

def summarise(groups, op=None, rng=None):
    out = {}
    for start in groups:
        s = stat(groups, start, op)
        bs = [stat({k: [G[i] for i in rng.integers(0, len(G), len(G))] for k, G in groups.items()}, start, op)["rho"]
              for _ in range(BOOT)]
        s["rho_ci95"] = [float(x) for x in np.percentile(bs, [2.5, 97.5])]
        out[start] = s
    return out

def vgc_groups():
    g = {"__name__": "rug2", "__file__": str(HERE / "ruggedness2.py")}
    src = (HERE / "ruggedness2.py").read_text().split('if __name__ == "__main__":')[0]
    exec(compile(src, "ruggedness2.py", "exec"), g)
    return g["vgc_anchors"]()

def analyse():
    rng = np.random.default_rng(7); res = {"benchmarks": {}, "vgc": {}}
    for name in TASKS:
        p = PARTS / f"{name}.json"
        if not p.exists(): continue
        d = json.load(open(p)); groups = {k: v for k, v in d.items() if not k.startswith("_")}
        res["benchmarks"][name] = {"cost_s_per_eval": d["_cost_s_per_eval"], "n_eval": d["_n_eval"],
                                   **summarise(groups, None, rng)}
    V = vgc_groups()
    res["vgc"]["all_edit_types"] = summarise(V, None, rng)
    for op in ["A_move_swap", "B_ability_swap", "C_item_swap", "D_alignment_swap", "E_spread_copy", "F_candidate_copy"]:
        res["vgc"][op] = summarise(V, op, rng)
    s = res["vgc"]["all_edit_types"]["meta"]
    sig24 = s["noise_var"] * 384 / 24
    res["vgc"]["noise_share_at_24_battles_derived"] = sig24 / (sig24 + s["var_f"])
    res["vgc"]["anchor_mean_win_rate"] = {k: float(np.mean([a["f"] for a in G])) for k, G in V.items()}
    if (PARTS / "mol_validity.json").exists(): res["mol_naive_validity"] = json.load(open(PARTS / "mol_validity.json"))
    json.dump(res, open(RES / "bench_features.json", "w"), indent=1)
    for name, r in res["benchmarks"].items():
        print(f"{name:16s} rho rand {r['random']['rho']:.3f} {r['random']['rho_ci95']} search {r['search']['rho']:.3f} "
              f"noise {r['random']['noise_share']:.3f} cost {r['cost_s_per_eval']*1e3:.2f} ms")
    for k, r in res["vgc"].items():
        if isinstance(r, dict) and isinstance(r.get("meta"), dict):
            print(f"VGC {k:16s} rho meta {r['meta']['rho']:.3f} {r['meta']['rho_ci95']} search {r['search']['rho']:.3f} "
                  f"noise {r['meta']['noise_share']:.3f}")
    print("VGC noise share at 24 battles (derived):", round(res["vgc"]["noise_share_at_24_battles_derived"], 3))

def mol_validity(n=2000):
    """Metric 4, molecule side: uniform draws from the naive string encodings, ZINC length distribution."""
    E = mol_env(); rng = np.random.default_rng(11); sf, Chem = E["sf"], E["Chem"]
    zinc = E["zinc"]; chars = sorted({c for smi in zinc[:20000] for c in smi})
    lens = [len(zinc[int(i)]) for i in rng.integers(0, len(zinc), n)]
    smi_ok = sum(Chem.MolFromSmiles("".join(rng.choice(chars, L))) is not None for L in lens)
    toks = [len(list(sf.split_selfies(sf.encoder(zinc[int(i)])))) for i in rng.integers(0, len(zinc), 200)]
    sel_ok = 0
    for _ in range(n):
        t = "".join(rng.choice(E["alphabet"], int(rng.choice(toks)))); d = sf.decoder(t)
        sel_ok += bool(d) and Chem.MolFromSmiles(d) is not None
    out = {"n": n, "smiles_uniform_chars_valid_frac": smi_ok / n, "selfies_uniform_tokens_valid_frac": sel_ok / n,
           "smiles_alphabet": len(chars), "selfies_alphabet": len(E["alphabet"])}
    json.dump(out, open(PARTS / "mol_validity.json", "w"), indent=1); print(out)

if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["analyse"]: analyse()
    elif args == ["validity"]: mol_validity()
    else:
        for name in (args or list(TASKS)): run_task(name)
