"""Masked diffusion tree search vs direct sampling, same generator / scorer / battle budget.
Pre-registration: docs/mdts-experiment.md.

    /tmp/vgc-pilot/.venv/bin/python scripts/mdts_experiment.py smoke   # tiny budget, no battles
    /tmp/vgc-pilot/.venv/bin/python scripts/mdts_experiment.py run     # resumable, per replicate
    /tmp/vgc-pilot/.venv/bin/python scripts/mdts_experiment.py report
"""
import json, math, os, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "/tmp/vgc-pilot/src")
import diffusion as D
import hpsdiffusion as H
import activesearch as A
import gradguide as GG
import matchup_db as MDB
from corpus import parse_team_text
from hps_generate import Validator

REPO = Path(__file__).resolve().parents[1]
CKPT = REPO / "results" / "temperature_p0.pt"
OUT = REPO / "results" / "mdts_experiment.json"
LEVELS = [0, 6, 18, 42, 48]           # decode steps revealed at tree depth 0..4 (D.ORDER stages)
BR, J, OMEGA = 5, 5, 1.0
BUDGET, TOPK, VERIFY = 400, 16, 4
PER_CELL, VERIFY_CELL = 8, 24
SEEDS = [1101, 1202, 1303, 1404, 1505, 1606]
NORM = lambda z: 0.5 * (1 + math.erf(z / math.sqrt(2)))

# ---------------------------------------------------------------- scorer: ridge + posterior variance -> EI
class EIScorer:
    def __init__(self, lab_teams, lab_y, corpus_teams):
        self.F, self.w = GG.fit_ridge(lab_teams, lab_y, corpus_teams)
        X = self.F.mat(lab_teams); y = np.asarray(lab_y, float)
        self.Ainv = np.linalg.inv(X.T @ X + np.diag(self.F.lam))
        r = y - X @ self.w
        self.s2 = float(r @ r / max(len(y) - 1, 1))
        # incumbent = best posterior mean among labelled teams (noisy-EI convention). Amended before any battle:
        # the best raw label (0.833 at 24 battles) made every EI ~1e-17, a degenerate ranking (smoke, 2026-09-30)
        self.best = float((X @ self.w).max())
        self.best_label = float(y.max())
    def score(self, teams):
        P = self.F.mat(teams)
        mu = P @ self.w
        sd = np.sqrt(np.maximum(self.s2 * np.einsum("ij,jk,ik->i", P, self.Ainv, P), 1e-12))
        z = (mu - self.best) / sd
        ei = (mu - self.best) * np.vectorize(NORM)(z) + sd * np.exp(-z * z / 2) / math.sqrt(2 * math.pi)
        return mu, sd, ei

# ---------------------------------------------------------------- generator: constrained decode of a step range
@torch.no_grad()
def decode(model, C, x, start, end, gen):
    """Fill decode steps [start, end) of D.ORDER for every row of x (legality projected at each step).
    Returns x and the log-probability of what was sampled."""
    n = x.shape[0]
    w = torch.full((n,), H.WNULL, device=D.DEV, dtype=torch.long)
    lp = torch.zeros(n)
    seq = list(D.ORDER)
    for step in range(start, end):
        c = seq[step]
        tt = torch.full((n,), 1.0 - step / len(seq), device=D.DEV)
        lg = model.logits(model(x, tt, w), c)
        for b in range(n):
            ok = C.mask_for(c, x[b], D.DEV)
            l = lg[b].clone(); l[~ok] = -1e9
            p = torch.softmax(l, -1).cpu()
            v = int(torch.multinomial(p, 1, generator=gen).item())
            x[b, c] = v; lp[b] += math.log(max(float(p[v]), 1e-12))
    return x, lp

class Ctx:
    def __init__(self):
        cf = A.load_corpus_files(); self.corpus = [t for _, t in cf]
        self.V = A.Vocab(self.corpus); L = A.Legality(self.corpus); self.C = D.Constraints(self.V, L)
        self.grid = np.stack([self.V.encode(t) for t in self.corpus])
        self.look, self.spreads = A.decode_tables(self.corpus)
        lab_t, lab_y, _ = GG.all_labels()
        self.scorer = EIScorer(lab_t, lab_y, self.corpus)
        self.n_labels = len(lab_t)
        self.model = H.TeamDiffusionHPS(self.V).to(D.DEV)
        self.model.load_state_dict(torch.load(CKPT, map_location=D.DEV)["sd"]); self.model.eval()

def to_designs(ctx, rows, rng):
    """Complete grid rows -> pastes (Stat Points from corpus spreads) -> EI."""
    pastes = [A.row_to_paste(ctx.V, r, ctx.look, ctx.spreads, rng) for r in rows]
    teams = [parse_team_text(p) for p in pastes]
    mu, sd, ei = ctx.scorer.score(teams)
    return [dict(row=r.tolist(), paste=p, mu=float(m), sd=float(s), ei=float(e))
            for r, p, m, s, e in zip(rows, pastes, mu, sd, ei)]

# ---------------------------------------------------------------- the two arms
def run_direct(ctx, n, seed):
    gen = torch.Generator().manual_seed(seed); rng = np.random.default_rng(seed)
    t0 = time.perf_counter(); out = []
    for s in range(0, n, 48):
        k = min(48, n - s)
        x, _ = decode(ctx.model, ctx.C, torch.zeros(k, D.COLS, dtype=torch.long, device=D.DEV), 0, 48, gen)
        out += to_designs(ctx, x.cpu().numpy(), rng)
    return out, time.perf_counter() - t0, {}

def run_tree(ctx, budget, seed):
    gen = torch.Generator().manual_seed(seed); rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    root = dict(x=torch.zeros(1, D.COLS, dtype=torch.long, device=D.DEV), d=0, kids=[], N=0, V=0.0, prior=1.0, dead=False)
    designs, iters, expansions = [], 0, [0] * 4
    uct = lambda p, c: c["V"] + OMEGA * c["prior"] * math.sqrt(math.log(max(p["N"], 1)) / max(c["N"], 1e-9))
    while len(designs) < budget:
        iters += 1
        path, n = [root], root
        while n["kids"]:
            live = [k for k in n["kids"] if not k["dead"]]
            if not live: n["dead"] = True; break
            n = max(live, key=lambda k: uct(path[-1], k)); path.append(n)
        if root["dead"]: break
        if n["dead"] or n["d"] == 4:
            n["dead"] = True
            for a in reversed(path[:-1]):                       # a node whose children are all spent is spent
                if all(k["dead"] for k in a["kids"]): a["dead"] = True
            continue
        d = n["d"]; expansions[d] += 1
        xs, lp = decode(ctx.model, ctx.C, n["x"].repeat(BR, 1), LEVELS[d], LEVELS[d + 1], gen)
        prior = torch.softmax(lp, 0).tolist()
        kids = []
        for i in range(BR):
            k = dict(x=xs[i:i + 1].clone(), d=d + 1, kids=[], N=1, V=0.0, prior=prior[i], dead=d + 1 == 4)
            kids.append(k)
        if d + 1 == 4:                                          # children are complete teams
            ds = to_designs(ctx, xs.cpu().numpy(), rng)
            for k, dsg in zip(kids, ds): k["V"] = dsg["ei"]
            designs += ds
        else:                                                   # J completions per child, scored together
            roll, _ = decode(ctx.model, ctx.C, xs.repeat_interleave(J, 0), LEVELS[d + 1], 48, gen)
            ds = to_designs(ctx, roll.cpu().numpy(), rng)
            for i, k in enumerate(kids): k["V"] = float(np.mean([z["ei"] for z in ds[i * J:(i + 1) * J]]))
            designs += ds
        n["kids"] = kids
        v = float(np.mean([k["V"] for k in kids]))
        for a in path: a["V"] = (a["V"] * a["N"] + v) / (a["N"] + 1); a["N"] += 1
    return designs, time.perf_counter() - t0, dict(iterations=iters, expansions_by_depth=expansions)

# ---------------------------------------------------------------- selection, battles, measurements
def select(designs, val, k=TOPK):
    seen, picked = set(), []
    for dsg in sorted(designs, key=lambda z: -z["ei"]):
        h = MDB.canon_hash(MDB.canonical_paste(dsg["paste"]))
        if h in seen: continue
        seen.add(h)
        if val(dsg["paste"]) is None: picked.append(dsg)
        if len(picked) == k: break
    return picked

def diversity(ctx, picked):
    teams = [parse_team_text(d["paste"]) for d in picked]
    G = np.stack([ctx.V.encode(t) for t in teams])
    pair = [(G[i] != G[j]).sum() for i in range(len(G)) for j in range(i + 1, len(G))]
    return dict(**A.memorisation(ctx.V, teams, ctx.grid), mean_pairwise_hamming=float(np.mean(pair)) if pair else None,
                distinct_designs_scored=None)

def fresh_score(con, pid, team, note):
    cols = dict(con.execute("SELECT team_id, weight FROM team_set WHERE set_name='top50'"))
    cols.pop(team, None)
    rows = con.execute("SELECT c.lo, c.hi, SUM(c.lo_wins), SUM(c.hi_wins), SUM(c.battles) FROM contribution c "
                       "JOIN batch b USING (batch_id) WHERE b.policy_id=? AND b.note=? AND (c.lo=? OR c.hi=?) "
                       "GROUP BY c.lo, c.hi", (pid, note, team, team)).fetchall()
    cells = {}
    for lo, hi, lw, hw, n in rows:
        c = hi if lo == team else lo
        if c in cols and n: cells[c] = ((lw if lo == team else hw), n)
    if not cells: return None, 0
    W = sum(cols[c] for c in cells)
    return sum(cols[c] * w / n for c, (w, n) in cells.items()) / W, sum(n for _, n in cells.values())

def replicate(ctx, con, pid, val, seed, battles=True):
    rec = {}
    tree, t_tree, info = run_tree(ctx, BUDGET, seed)
    direct, t_direct, _ = run_direct(ctx, len(tree), seed + 50_000)
    for arm, designs, secs, extra in (("tree", tree, t_tree, info), ("direct", direct, t_direct, {})):
        picked = select(designs, val)
        r = dict(scored=len(designs), proposal_seconds=secs, **extra,
                 distinct_scored=len({MDB.canon_hash(MDB.canonical_paste(d["paste"])) for d in designs}),
                 selected_mu=float(np.mean([d["mu"] for d in picked])), selected_ei=float(np.mean([d["ei"] for d in picked])),
                 diversity=diversity(ctx, picked), pastes=[d["paste"] for d in picked])
        if battles:
            t0 = time.perf_counter()
            sc = MDB.score_pastes(con, r["pastes"], per_cell=PER_CELL, seed=seed, origin=f"mdts:{arm}:{seed}", policy_id=pid)
            r["first"] = sc
            ok = [s for s in sc if s.get("score") is not None]
            top = sorted(ok, key=lambda s: -s["score"])[:VERIFY]
            note = f"mdts verify {arm} {seed}"
            MDB.ensure(con, pid, [s["team_id"] for s in top], MDB.set_members(con, "top50"), VERIFY_CELL,
                       seed=seed + 7, note=note)
            r["verified"] = []
            for s in top:
                fs, n = fresh_score(con, pid, s["team_id"], note)
                if fs is None:                                   # already topped up in an earlier replicate
                    fs, n = fresh_score_any(con, pid, s["team_id"])
                r["verified"].append(dict(team_id=s["team_id"], first=s["score"], fresh=fs, fresh_battles=n))
            r["battle_seconds"] = time.perf_counter() - t0
        rec[arm] = r
    return rec

def fresh_score_any(con, pid, team):
    for (note,) in con.execute("SELECT DISTINCT b.note FROM batch b JOIN contribution c USING (batch_id) "
                               "WHERE b.note LIKE 'mdts verify%' AND (c.lo=? OR c.hi=?)", (team, team)):
        fs, n = fresh_score(con, pid, team, note)
        if fs is not None: return fs, n
    return None, 0

# ---------------------------------------------------------------- report
def report(out):
    reps = [k for k in out["replicates"] if all("verified" in out["replicates"][k][a] for a in ("tree", "direct"))]
    prim = {a: [np.mean([v["fresh"] for v in out["replicates"][k][a]["verified"] if v["fresh"] is not None]) for k in reps]
            for a in ("tree", "direct")}
    first = {a: [np.mean([s["score"] for s in out["replicates"][k][a]["first"] if s.get("score") is not None]) for k in reps]
             for a in ("tree", "direct")}
    d = np.array(prim["tree"]) - np.array(prim["direct"])
    rng = np.random.default_rng(0)
    boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(10_000)] if len(d) else [0]
    ci = [float(x) for x in np.percentile(boot, [2.5, 97.5])]
    verdict = ("tree search raised verified win rate at this budget" if ci[0] > 0 else
               "tree search lowered verified win rate at this budget" if ci[1] < 0 else
               "no detectable difference at this budget")
    agg = lambda a, f: float(np.mean([f(out["replicates"][k][a]) for k in reps]))
    summ = {"replicates": len(reps), "primary_delta_mean": float(d.mean()) if len(d) else None, "primary_delta_ci95": ci,
            "verdict": verdict,
            **{f"{a}_{m}": v for a in ("tree", "direct") for m, v in {
                "verified_top4_mean": float(np.mean(prim[a])), "first16_mean": float(np.mean(first[a])),
                "best_verified": float(max(v["fresh"] for k in reps for v in out["replicates"][k][a]["verified"] if v["fresh"] is not None)),
                "selected_mu": agg(a, lambda r: r["selected_mu"]), "selected_ei": agg(a, lambda r: r["selected_ei"]),
                "distinct_species_sets": agg(a, lambda r: r["diversity"]["distinct_species_sets"]),
                "pairwise_hamming": agg(a, lambda r: r["diversity"]["mean_pairwise_hamming"]),
                "copy_rate": agg(a, lambda r: r["diversity"]["copy_rate"]), "nn_hamming": agg(a, lambda r: r["diversity"]["nn_hamming"]),
                "proposal_seconds": agg(a, lambda r: r["proposal_seconds"]), "battle_seconds": agg(a, lambda r: r["battle_seconds"]),
                "scored": agg(a, lambda r: r["scored"]), "distinct_scored": agg(a, lambda r: r["distinct_scored"])}.items()},
            "per_replicate_primary": {a: [float(x) for x in prim[a]] for a in prim}}
    out["summary"] = summ
    return summ

def main(stage):
    ctx = Ctx(); val = Validator()
    print(f"labels {ctx.n_labels} · best label {ctx.scorer.best_label:.3f} · incumbent (best fitted mean) {ctx.scorer.best:.3f} · σ² {ctx.scorer.s2:.4f} · device {D.DEV}", flush=True)
    if stage == "smoke":
        global BUDGET, TOPK
        BUDGET, TOPK = 60, 4
        rec = replicate(ctx, None, None, val, 99, battles=False)
        for a in rec:
            r = rec[a]; print(a, {k: r[k] for k in ("scored", "distinct_scored", "proposal_seconds", "selected_mu", "selected_ei")},
                              r.get("iterations"), r.get("expansions_by_depth"), r["diversity"], flush=True)
        val.close(); return
    con = MDB.connect(); pid = MDB.get_policy(con)
    out = json.load(open(OUT)) if OUT.exists() else {"config": {}, "replicates": {}}
    out["config"] = dict(ckpt=str(CKPT), levels=LEVELS, branching=BR, completions=J, omega=OMEGA, budget=BUDGET,
                         topk=TOPK, verify=VERIFY, per_cell=PER_CELL, verify_cell=VERIFY_CELL, seeds=SEEDS,
                         n_labels=ctx.n_labels, best_label=ctx.scorer.best_label, incumbent=ctx.scorer.best)
    if stage == "run":
        for s in SEEDS:
            if str(s) in out["replicates"]: print(f"seed {s} cached", flush=True); continue
            t0 = time.perf_counter()
            out["replicates"][str(s)] = replicate(ctx, con, pid, val, s)
            json.dump(out, open(str(OUT) + ".tmp", "w"), indent=1); os.replace(str(OUT) + ".tmp", OUT)
            r = out["replicates"][str(s)]
            print(f"seed {s}: " + " | ".join(
                f"{a} first {np.mean([x['score'] for x in r[a]['first'] if x.get('score') is not None]):.3f} "
                f"verified {np.mean([v['fresh'] for v in r[a]['verified'] if v['fresh'] is not None]):.3f} "
                f"prop {r[a]['proposal_seconds']:.0f}s sets {r[a]['diversity']['distinct_species_sets']}"
                for a in ("tree", "direct")) + f" · {time.perf_counter() - t0:.0f}s", flush=True)
    summ = report(out)
    json.dump(out, open(str(OUT) + ".tmp", "w"), indent=1); os.replace(str(OUT) + ".tmp", OUT)
    print(json.dumps(summ, indent=1)); val.close()

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "run")
