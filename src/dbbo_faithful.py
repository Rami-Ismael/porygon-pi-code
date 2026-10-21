"""Diffusion-BBO, best-faith port (2026-10-16). Pre-registration: docs/dbbo-faithful.md.

Wu, Kuang, Niu, Ma & Yu, "Diffusion-BBO: Diffusion-Based Inverse Modeling for Online Black-Box Optimization"
(arXiv 2407.00610), Algorithm 1, with no pretrained model and none of this project's loop machinery:

  D      starts as the labelled teams between the 25th and 50th percentile of win rate (the paper's initial dataset
         rule), drawn from the matchup matrix (full top-50 coverage, policy 1, held-out ranks never present).
  round  train a conditional masked-diffusion model FROM RANDOM WEIGHTS on D: label dropout 0.15 (classifier-free
         guidance), learning rate 1e-3, batch 256 (the paper's values), 10% of D held out as validation each round,
         the checkpoint with the best validation loss kept. Ask y = w * (best raw label in D) with w = 1.
         Sample N = 128 legal teams at guidance 2, temperature 1 on every field, battle all of them, append to D.
  rounds 4 x 128 = 512 battled teams, the same budget as cem_v3_noheldout and the other arms.

Departures forced by this problem, not choices: the network is the project's masked transformer over 48 fields
(the paper's tasks are continuous vectors); decoding keeps the regulation's legality tables (illegal teams cannot
be battled); labels are 49-battle win rates (the paper's oracles are noiseless or nearly). Dropped on purpose: the
ensemble and its uncertainty term (UaE) that picks w. This project does not use uncertainty acquisition. Without
that term the paper's argmax over w in {0.6 ... 1.0} reduces to w = 1. No ridge, no taboo, no elite re-steer.

Finalists as every other arm: the 16 best distinct battled teams by label, 192 fresh battles vs the top-50 pool,
then once vs ranks 51-100.

    python battle_supervisor.py dbbo_faithful.py run --then final --then report
    DBBO_SMOKE=1 python dbbo_faithful.py run
"""
import copy, json, math, os, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "/tmp/vgc-pilot/src")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from showdown_validator import VGC_BENCH
import foundation as FD
import lossdown as L
import lossdown_battle as LB
import matchup_db as MDB
import heldout as HO
import activesearch as A
import diversity_loops as DL
import pool
from corpus import parse_team_text
from encode import team_fields
from hps_generate import Validator
from checkpoints import load_checkpoint

# DBBO_PRECISE=1 (2026-10-16, docs/dbbo-faithful.md "Precise-label variant"): D starts from ALL the ask-dial precise-label
# teams (about 196 battles each), and every battled team gets PER_CELL_PRECISE battles per top-50 column before its label
# enters D. Everything else unchanged.
PRECISE = bool(os.environ.get("DBBO_PRECISE")); PER_CELL_PRECISE = 4
MIN_STEPS = 1100 if PRECISE else 0     # the raw-label port's step count (15,372 teams x 0.9 / 256 x 20 epochs): the precise D
                                       # is 14x smaller, so equal epochs would mean ~80 steps from random weights
# DBBO_FIX=data,ask,pick (2026-10-17, docs/dbbo-faithful.md "Fixing the likely causes"): one switch per suspected cause of the
# w = 1 port's low win rate, on top of the raw-label port.
#   data  D0 = every labelled matrix team (full win-rate range), labels shrunk toward the mean (beta-binomial); appended teams
#         are shrunk with the same prior
#   ask   ask = 90th percentile of D's labels (reachable) instead of the best label in D
#   pick  decode 4 x 128 legal teams and battle the ridge surrogate's top 128 (the loop's pick, no ridge push)
FIX = sorted(x for x in os.environ.get("DBBO_FIX", "").split(",") if x)
assert set(FIX) <= {"data", "ask", "pick"}, FIX
TAG = os.environ.get("DBBO_TAG", ("dbbo_fix_all" if len(FIX) == 3 else "dbbo_fix_" + "_".join(FIX)) if FIX
                     else "dbbo_faithful_precise" if PRECISE else "dbbo_faithful")
ROUNDS, N, W, G_W, LO_Q, HI_Q = 4, 128, 1.0, 2.0, 0.25, 0.50
EPOCHS, BATCH, LR, VAL_FRAC, P_UNCOND, SEED = 20, 256, 1e-3, 0.10, 0.15, 9191
FINAL_N, FINAL_BATTLES = 16, 192
SMOKE = bool(os.environ.get("DBBO_SMOKE"))
if SMOKE: TAG += "_smoke"; ROUNDS, N, EPOCHS, FINAL_N, FINAL_BATTLES = 2, 8, 1, 2, 8
OUT = L.RES / f"{TAG}.json"
DEV = L.DEV
REPO = str(VGC_BENCH)


def save(out): json.dump(out, open(OUT, "w"))


def initial_D_precise(V):
    """All the ask-dial precise-label teams, with their re-battled labels (not a percentile slice: that set was built to
    cover every win-rate band, so its percentiles do not mean what the paper's percentile of an offline dataset means)."""
    teams = json.load(open(L.RES / "ask_dial.json"))["teams"]; rows = []
    for r in teams:
        t = parse_team_text(r["paste"])
        if len(t) != 6: continue
        try: V.encode(t)
        except ValueError: continue
        rows.append(dict(paste=r["paste"], y=float(r["new"]), team_id=r["team_id"], src="initial"))
    ys = np.array([x["y"] for x in rows])
    return rows, dict(source="ask_dial precise-label teams (all)", n=len(rows), min=float(ys.min()), median=float(np.median(ys)),
                      max=float(ys.max()), mean_battles=float(np.mean([r.get("new_battles", 0) for r in teams])))


def initial_D_all(con, V):
    """Every labelled matrix team with full top-50 coverage, labels shrunk toward the global mean (method-of-moments beta prior)."""
    labs = MDB.labels(con, LB.POLICY); rows, ys, ns = [], [], []
    for s in labs:
        (p,) = con.execute("SELECT paste FROM team WHERE team_id=?", (s["team_id"],)).fetchone()
        t = parse_team_text(p)
        if len(t) != 6: continue
        try: V.encode(t)
        except ValueError: continue
        rows.append(dict(paste=p, raw=float(s["score"]), n=int(s["battles"]), team_id=s["team_id"], src="initial"))
    y = np.array([r["raw"] for r in rows]); n = np.array([r["n"] for r in rows], float)
    m = float(y.mean()); tau2 = max(float(y.var(ddof=1)) - float(np.mean(y * (1 - y) / n)), 1e-4); k = max(m * (1 - m) / tau2 - 1, 0.0)
    for r in rows: r["y"] = (r["raw"] * r["n"] + m * k) / (r["n"] + k)
    return rows, dict(source="all labelled teams", n=len(rows), prior_mean=m, prior_k=k, raw_min=float(y.min()), raw_max=float(y.max()),
                      shrunk_p90=float(np.quantile([r["y"] for r in rows], 0.9)))


def initial_D(con, V):
    """Labelled matrix teams between the 25th and 50th percentile of win rate, encodable in the M-B vocabulary."""
    labs = MDB.labels(con, LB.POLICY)
    ys = np.array([s["score"] for s in labs]); lo, hi = np.quantile(ys, [LO_Q, HI_Q])
    rows = []
    for s in labs:
        if not lo <= s["score"] <= hi: continue
        (p,) = con.execute("SELECT paste FROM team WHERE team_id=?", (s["team_id"],)).fetchone()
        t = parse_team_text(p)
        if len(t) != 6: continue
        try: V.encode(t)
        except ValueError: continue
        rows.append(dict(paste=p, y=float(s["score"]), team_id=s["team_id"], src="initial"))
    return rows, dict(n_labelled=len(labs), q25=float(lo), q50=float(hi), n=len(rows))


def train(V, tables, rows, r):
    """A fresh network (the F1 member's architecture, scalar condition) trained on D; best validation checkpoint."""
    ck = L.CK / f"{TAG}_r{r}.pt"
    _, cfg, _ = L.load_net(FD.V3_MEMBERS[0], V); cfg = dict(cfg, mrow=0, p_uncond=P_UNCOND)
    rules = tuple(cfg["rules"].split(","))
    net = L.Net(V, d=cfg["d"], nhead=cfg["nhead"], nlayer=cfg["nlayer"], dropout=cfg["dropout"],
                time_cond=bool(cfg["time_cond"]), ntag=0, prime=cfg["prime"], mrow=0).to(DEV)
    if ck.exists():
        net.load_state_dict(load_checkpoint(ck, map_location=DEV)["sd"]); net.eval(); return L.logprobs_fn(net, tables, rules), None
    rng = np.random.default_rng(SEED + r); torch.manual_seed(SEED + r)
    enc, ys, skipped = [], [], 0
    for x in rows:                                  # parse_team_text merges an item-less slot into its neighbour (known): such a
        t = parse_team_text(x["paste"])             # paste reads as 5 slots and encodes short, so it is skipped and counted
        e = V.encode(t) if len(t) == 6 else None
        if e is None or len(e) != L.COLS: skipped += 1; continue
        enc.append(e); ys.append(x["y"])
    if skipped: print(f"[r{r}] skipped {skipped} of {len(rows)} teams that do not parse to 6 slots", flush=True)
    X = np.stack(enc); Y = np.array(ys, np.float32)
    perm = rng.permutation(len(X)); nv = max(1, int(VAL_FRAC * len(X)))
    xv, wv = torch.as_tensor(X[perm[:nv]], device=DEV), torch.as_tensor(Y[perm[:nv]], device=DEV)
    xt, wt = torch.as_tensor(X[perm[nv:]], device=DEV), torch.as_tensor(Y[perm[nv:]], device=DEV)
    sched = L.sched_tensor(cfg["sched"]).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.01)
    best, best_sd, log, t0 = float("inf"), None, [], time.perf_counter()
    epochs = max(EPOCHS, math.ceil(MIN_STEPS / math.ceil(len(xt) / BATCH))) if not SMOKE else EPOCHS
    for ep in range(epochs):
        net.train(); p = torch.randperm(len(xt), device=DEV); tr = 0.0; nb = 0
        for i in range(0, len(xt), BATCH):
            b = p[i:i + BATCH]
            loss = L.batch_loss(net, tables, rules, xt[b], wt[b], None, sched, cfg)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            tr += float(loss.detach()); nb += 1
        net.eval(); torch.manual_seed(0)
        with torch.no_grad():
            vl = float(np.mean([float(L.batch_loss(net, tables, rules, xv[i:i + BATCH], wv[i:i + BATCH], None, sched, cfg))
                                for i in range(0, len(xv), BATCH)]))
        log.append(dict(epoch=ep + 1, train=tr / nb, val=vl))
        if vl < best: best, best_sd = vl, copy.deepcopy(net.state_dict())
        print(f"[r{r}] epoch {ep + 1}/{epochs}: train {tr / nb:.3f} · val {vl:.3f} ({(time.perf_counter() - t0) / 60:.1f} min)", flush=True)
    net.load_state_dict(best_sd); net.eval()
    torch.save({"sd": best_sd, "cfg": cfg, "n_train": len(xt)}, ck)
    return L.logprobs_fn(net, tables, rules), dict(best_val=best, log=log, n_train=len(xt), n_val=nv, epochs=epochs, skipped=skipped)


def sample(G, y, rng, n):
    val = Validator(); keep, tried = [], 0
    try:
        while len(keep) < n and tried < 8 * n:
            for r in G.sample(96, y, G_W, 1.0, 1.0).cpu().numpy():
                tried += 1; p, _ = G.to_paste(r, rng, repair=True)
                if val(p) is None and not HO.is_heldout_text(p): keep.append(p)
                if len(keep) >= n: break
    finally: val.close()
    return keep, tried


def run():
    LB._stack(); con = MDB.connect()
    out = json.load(open(OUT)) if OUT.exists() else {"config": dict(rounds=ROUNDS, n=N, w=W, guidance=G_W, init_quantiles=[LO_Q, HI_Q],
        epochs=EPOCHS, batch=BATCH, lr=LR, val_frac=VAL_FRAC, p_uncond=P_UNCOND, seed=SEED, uncertainty_term="dropped (w = 1)",
        precise=PRECISE, fix=FIX, min_steps=MIN_STEPS, battles_per_column=PER_CELL_PRECISE if PRECISE else LB.PER_CELL),
        "rounds": {}}
    G = FD.Gen("v3"); V, tables = G.V, G.tables
    if "D0" not in out:
        out["D0"], out["D0_info"] = (initial_D_all(con, V) if "data" in FIX else initial_D_precise(V) if PRECISE else initial_D(con, V)); save(out)
        print(f"initial D: {out['D0_info']}", flush=True)
    D = list(out["D0"])
    for r in range(1, ROUNDS + 1):
        c = out["rounds"].get(str(r))
        if c and "y" in c:
            if "data" in FIX:
                m_, k_ = out["D0_info"]["prior_mean"], out["D0_info"]["prior_k"]
                D += [dict(paste=p, raw=v, y=(v * 49 + m_ * k_) / (49 + k_), src=f"r{r}") for p, v in zip(c["battled"], c["y"]) if v is not None]
            else:
                D += [dict(paste=p, y=v, src=f"r{r}") for p, v in zip(c["battled"], c["y"]) if v is not None]
            continue
        t0 = time.perf_counter(); rng = np.random.default_rng(SEED + 100 * r); torch.manual_seed(SEED + 100 * r)
        best = max(x["y"] for x in D); y = float(np.quantile([x["y"] for x in D], 0.9)) if "ask" in FIX else W * best
        print(f"[r{r}] |D| = {len(D)} · best label {best:.3f} · ask {y:.3f}", flush=True)
        G.f, info = train(V, tables, D, r)
        if "pick" in FIX:                               # decode 4N legal teams, battle the ridge's top N
            import cem_v3 as CV
            DL.CV_V[0] = V
            battled_so_far = [dict(paste=x["paste"], y=x.get("raw", x["y"]), src=x["src"]) for x in D if x["src"] != "initial"]
            sur = DL.ridge_for(CV.Ridge(), battled_so_far, L.RES / f"{TAG}_ridge_r{r}.npz")
            pool_, tried = sample(G, y, rng, 4 * N)
            mu = np.array([float(sur.score(parse_team_text(p))) for p in pool_])
            pastes = [pool_[i] for i in np.argsort(-mu)[:N]]
        else:
            pastes, tried = sample(G, y, rng, N)
        v = MDB.Validator()
        try: ids = [MDB.add_team(con, p, f"{TAG}:r{r}", v)[0] for p in pastes]
        finally: v.close()
        legal = {t for (t,) in con.execute(f"SELECT team_id FROM team WHERE legal=1 AND team_id IN ({','.join('?' * len(ids))})", ids)}
        MDB.ensure(con, LB.POLICY, sorted(legal), MDB.set_members(con, "top50"), PER_CELL_PRECISE if PRECISE else LB.PER_CELL,
                   seed=SEED + r, note=f"{TAG} r{r}")
        ys = [MDB.score(con, LB.POLICY, t)["score"] if t in legal else None for t in ids]
        c = dict(ask=y, best_in_D=best, n_D=len(D), train=info, battled=pastes, team_id=ids, y=ys, tried=tried,
                 valid=len(pastes) / max(tried, 1), stats=DL.bstats(ys, pastes),
                 compositions=len({DL.comp_paste(p) for p in pastes if DL.comp_paste(p)}), minutes=(time.perf_counter() - t0) / 60)
        out["rounds"][str(r)] = c; save(out)
        if "data" in FIX:                               # appended teams shrunk with the same prior as D0
            m_, k_ = out["D0_info"]["prior_mean"], out["D0_info"]["prior_k"]
            D += [dict(paste=p, raw=v, y=(v * 49 + m_ * k_) / (49 + k_), src=f"r{r}") for p, v in zip(pastes, ys) if v is not None]
        else:
            D += [dict(paste=p, y=v, src=f"r{r}") for p, v in zip(pastes, ys) if v is not None]
        s = c["stats"]
        print(f"[r{r}] battled {s['n']}: win rate {s['mean']:.3f} ± {s['se']:.3f} · median {np.median([v for v in ys if v is not None]):.3f} · "
              f"max {s['max']:.3f} · ≥0.531 {s['above_cut']} · line-ups {c['compositions']} · valid {c['valid']:.2f} · {c['minutes']:.1f} min", flush=True)


def final():
    out = json.load(open(OUT)); pool.REPO = REPO
    if "final" not in out:
        rows = sorted([dict(paste=p, y=v, src=f"r{r}") for r, c in out["rounds"].items() for p, v in zip(c["battled"], c["y"]) if v is not None],
                      key=lambda x: -x["y"])
        seen, top = set(), []
        for x in rows:
            k = tuple(team_fields(parse_team_text(x["paste"])))
            if k in seen: continue
            seen.add(k); top.append(x)
            if len(top) >= FINAL_N: break
        d = Path(f"/tmp/vgc-pilot/{TAG}_final"); d.mkdir(parents=True, exist_ok=True); files = []
        for i, x in enumerate(top):
            f = d / f"{i:02d}.txt"; f.write_text(x["paste"]); files.append(str(f))
        res = pool.score([(f, A.opponents()) for f in files], battles=FINAL_BATTLES, conc=50)
        out["final"] = [dict(paste=x["paste"], label=x["y"], src=x["src"], fresh=res.get(f, {}).get("win_rate")) for x, f in zip(top, files)]
        save(out)
    if "heldout" not in out:
        d = Path(f"/tmp/vgc-pilot/{TAG}_final"); files = [str(d / f"{i:02d}.txt") for i in range(len(out["final"]))]
        res = HO.evaluate(files, battles=FINAL_BATTLES, conc=50)
        out["heldout"] = [dict(fresh_top50=x["fresh"], heldout=res.get(f, {}).get("win_rate")) for x, f in zip(out["final"], files)]
        save(out)


def report():
    out = json.load(open(OUT)); base = json.load(open(L.RES / "cem_v3_noheldout.json"))
    print("win rate = wins / battles vs the top-50 meta teams, behaviour-cloned policy (policy 1) on both sides")
    print(f"initial D: {out['D0_info']}")
    for r, c in sorted(out["rounds"].items(), key=lambda kv: int(kv[0])):
        s = c["stats"]
        print(f"round {r}: ask {c['ask']:.3f} (best in D {c['best_in_D']:.3f}, |D| {c['n_D']}) · battled {s['mean']:.3f} ± {s['se']:.3f} · "
              f"max {s['max']:.3f} · line-ups {c['compositions']} · valid {c['valid']:.2f}")
    if out.get("heldout"):
        rng = np.random.default_rng(0)
        f = np.array([x["fresh"] for x in out["final"]]); h = np.array([x["heldout"] for x in out["heldout"]])
        bf = np.array([x["fresh"] for x in base["final"]]); bh = np.array([x["heldout"] for x in base["heldout"]])
        def bd(a, b):
            bs = [a[rng.integers(0, len(a), len(a))].mean() - b[rng.integers(0, len(b), len(b))].mean() for _ in range(10000)]
            return a.mean() - b.mean(), np.quantile(bs, [.025, .975])
        d1, c1 = bd(f, bf); d2, c2 = bd(h, bh)
        print(f"finalists ({len(f)} x {FINAL_BATTLES}): vs top-50 {f.mean():.3f} (best {f.max():.3f}) · vs ranks 51-100 {h.mean():.3f}")
        print(f"minus cem_v3_noheldout (0.504 / 0.509): {d1:+.3f} [{c1[0]:+.3f}, {c1[1]:+.3f}] · {d2:+.3f} [{c2[0]:+.3f}, {c2[1]:+.3f}]")
        out["summary"] = dict(final_top50=float(f.mean()), final_heldout=float(h.mean()), diff_top50=float(d1), ci_top50=[float(x) for x in c1],
                              diff_heldout=float(d2), ci_heldout=[float(x) for x in c2]); save(out)


if __name__ == "__main__":
    {"run": run, "final": final, "report": report}[sys.argv[1]]()
