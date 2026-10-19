"""Making asked = got: precise labels and calibration (2026-10-16). Notes: docs/ask-dial.md.

relabel    ~1,000 matrix teams, stratified by their current label (bins of 0.05, up to 70 each), re-battled to
           4 battles per top-50 column (196 battles; label SE ~0.035 instead of ~0.07). Battles go into the
           matchup matrix, so every later read of these teams also gets the better label.
train      both v3 members fine-tuned from their ORIGINAL checkpoints, 600 steps, batch 64: half the member's
           original training rows with the label hidden (unconditional, keeps the prior), half the precise teams
           with their 196-battle label. The win-rate condition is learned only from precise labels.
           Saved as dial_precise_<member>.pt; swept by ask_sweep_dbbo.py (ASKSWEEP_MODELS=v3_precise=dial_precise).
calibrate  reads every ask sweep, picks the curve with the widest rising range, inverts it (got -> ask) by
           linear interpolation over its rising part, then asks for 4 target win rates inside that range:
           64 teams x 49 battles each. Reports target vs got.

    python battle_supervisor.py ask_dial.py relabel --then train
    python battle_supervisor.py ask_dial.py calibrate --then report
"""
import json, math, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "/tmp/vgc-pilot/src")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cem_dbbo as CB
import ask_sweep_dbbo as AS
import foundation as FD
import lossdown as L
import lossdown_battle as LB
import matchup_db as MDB
import heldout as HO
from corpus import parse_team_text
from hps_generate import Validator

OUT = L.RES / "ask_dial.json"
BIN, PER_BIN, PER_CELL_PRECISE = 0.05, 70, 4
STEPS, BATCH, LR, SEED = 600, 64, 1e-4, 8181
SWEEPS = ["ask_sweep_dbbo", "ask_sweep_precise", "ask_sweep_g4"]
N_TARGETS, N_TEAMS = 4, 64
PREFIX = {"v3_original": None, "dbbo_g3": "cem_dbbo_g3", "v3_precise": "dial_precise"}


def save(out): json.dump(out, open(OUT, "w"))
def load(): return json.load(open(OUT)) if OUT.exists() else {}


def relabel():
    LB._stack(); con = MDB.connect(); out = load(); rng = np.random.default_rng(SEED)
    if "teams" not in out:
        G = FD.Gen("v3"); V = G.V
        top = set(MDB.set_members(con, "top50"))
        labs = [s for s in MDB.labels(con, LB.POLICY) if s["team_id"] not in top]
        print(f"{len(labs)} labelled teams with full coverage", flush=True)
        bins = {}
        for s in labs: bins.setdefault(min(int(s["score"] / BIN), int(0.8 / BIN)), []).append(s)
        pick = []
        for b in sorted(bins):
            cand = bins[b]; rng.shuffle(cand); n = 0
            for s in cand:
                if n >= PER_BIN: break
                (p,) = con.execute("SELECT paste FROM team WHERE team_id=?", (s["team_id"],)).fetchone()
                t = parse_team_text(p)
                if len(t) != 6 or HO.is_heldout_text(p): continue
                try: V.encode(t)
                except ValueError: continue
                pick.append(dict(team_id=s["team_id"], paste=p, old=s["score"], old_battles=s["battles"])); n += 1
            print(f"bin {b * BIN:.2f}-{(b + 1) * BIN:.2f}: {n} of {len(bins[b])}", flush=True)
        out["teams"] = pick; save(out)
    ids = [r["team_id"] for r in out["teams"]]
    for i in range(0, len(ids), 100):                                    # chunks, so a stall loses little
        MDB.ensure(con, LB.POLICY, ids[i:i + 100], MDB.set_members(con, "top50"), PER_CELL_PRECISE, seed=SEED + i,
                   note=f"ask_dial relabel {i}")
        print(f"relabelled {min(i + 100, len(ids))}/{len(ids)}", flush=True)
    for r in out["teams"]:
        s = MDB.score(con, LB.POLICY, r["team_id"]); r.update(new=s["score"], new_battles=s["battles"], new_se=s["se"])
    o, n = np.array([r["old"] for r in out["teams"]]), np.array([r["new"] for r in out["teams"]])
    out["relabel_summary"] = dict(n=len(o), corr=float(np.corrcoef(o, n)[0, 1]), old_sd=float(o.std()), new_sd=float(n.std()),
                                  mean_new_se=float(np.mean([r["new_se"] for r in out["teams"]])))
    save(out); print("relabel:", out["relabel_summary"], flush=True)


def train():
    out = load(); G = FD.Gen("v3"); V, tables = G.V, G.tables
    lab = [(V.encode(parse_team_text(r["paste"])), r["new"]) for r in out["teams"]]
    for name in FD.V3_MEMBERS:
        ck = L.CK / f"dial_precise_{name}.pt"
        if ck.exists(): print("exists", ck.name); continue
        net, cfg, _ = L.load_net(name, V); torch.manual_seed(SEED)
        xo, wo = CB.original_rows(name, cfg); wo = torch.full_like(wo, float("nan"))   # prior only: label hidden
        Xl = torch.as_tensor(np.stack([e[0] for e in lab]), device=L.DEV)
        Wl = torch.as_tensor(np.array([e[1] for e in lab], np.float32), device=L.DEV)
        if cfg.get("mrow"): Wl = torch.cat([Wl[:, None], torch.full((len(Wl), 49), float("nan"), device=L.DEV)], 1)
        sched = L.sched_tensor(cfg["sched"]).to(L.DEV); rules = tuple(cfg["rules"].split(","))
        opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.01); net.train(); half = BATCH // 2; run = 0.0
        t0 = time.perf_counter()
        for step in range(STEPS):
            bo = torch.randint(len(xo), (half,), device=L.DEV); bl = torch.randint(len(Xl), (half,), device=L.DEV)
            loss = L.batch_loss(net, tables, rules, torch.cat([xo[bo], Xl[bl]]), torch.cat([wo[bo], Wl[bl]]), None, sched, cfg)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            run += float(loss.detach())
        net.eval(); torch.save({"sd": net.state_dict(), "cfg": ck.name}, ck)
        print(f"trained {ck.name}: {STEPS} steps, mean loss {run / STEPS:.3f}, {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    out["train"] = dict(steps=STEPS, batch=BATCH, lr=LR, n_precise=len(lab), original_rows_label="hidden"); save(out)


def curves():
    """{(sweep, model, guidance): [(ask, got)]} from every sweep file that exists."""
    res = {}
    for tag in SWEEPS:
        f = L.RES / f"{tag}.json"
        if not f.exists(): continue
        o = json.load(open(f)); g = o["config"]["guidance"]
        for c in o["cells"].values():
            if c.get("y"):
                res.setdefault((tag, c["model"], g), []).append((c["asked"], float(np.mean([v for v in c["y"] if v is not None]))))
    return {k: sorted(v) for k, v in res.items()}


def rising(pts):
    """The rising part of a curve: from the lowest ask up to its peak, made monotone (running max)."""
    a, y = zip(*pts); k = int(np.argmax(y)); a, y = np.array(a[:k + 1]), np.maximum.accumulate(np.array(y[:k + 1]))
    return a, y


def calibrate():
    LB._stack(); con = MDB.connect(); out = load()
    cs = curves(); best = max(cs, key=lambda k: np.ptp(rising(cs[k])[1]))
    a, y = rising(cs[best]); lo, hi = y[0] + 0.02, y[-1] - 0.02
    targets = [float(t) for t in np.linspace(lo, hi, N_TARGETS)]
    asks = [float(np.interp(t, y, a)) for t in targets]
    cal = out.setdefault("calibration", dict(curve=dict(sweep=best[0], model=best[1], guidance=best[2], points=cs[best]),
                                             targets=targets, asks=asks, cells={}))
    G = FD.Gen("v3"); G.f = AS.model(G, PREFIX[best[1]])
    print(f"calibration curve: {best} · targets {[round(t, 3) for t in targets]} -> asks {[round(x, 3) for x in asks]}", flush=True)
    for i, (t, q) in enumerate(zip(cal["targets"], cal["asks"])):
        key = f"t{t:.3f}"; c = cal["cells"].get(key)
        if c and "y" in c: continue
        if not c:
            rng = np.random.default_rng(SEED + i); torch.manual_seed(SEED + i); val = Validator(); keep, tried = [], 0
            try:
                while len(keep) < N_TEAMS and tried < 8 * N_TEAMS:
                    for r in CB.sample_asks(G, torch.full((96,), q), best[2], None, 0.0).cpu().numpy():
                        tried += 1; p, _ = G.to_paste(r, rng, repair=True)
                        if val(p) is None and not HO.is_heldout_text(p): keep.append(p)
                        if len(keep) >= N_TEAMS: break
            finally: val.close()
            c = cal["cells"][key] = dict(target=t, ask=q, pastes=keep, valid=len(keep) / tried); save(out)
        v = MDB.Validator()
        try: ids = [MDB.add_team(con, p, f"askdial:{key}", v)[0] for p in c["pastes"]]
        finally: v.close()
        legal = {x for (x,) in con.execute(f"SELECT team_id FROM team WHERE legal=1 AND team_id IN ({','.join('?' * len(ids))})", ids)}
        MDB.ensure(con, LB.POLICY, sorted(legal), MDB.set_members(con, "top50"), LB.PER_CELL, seed=SEED + 50 + i, note=f"ask_dial {key}")
        c["y"] = [MDB.score(con, LB.POLICY, x)["score"] if x in legal else None for x in ids]; save(out)
        yy = [v for v in c["y"] if v is not None]
        print(f"target {t:.3f} (ask {q:.3f}): got {np.mean(yy):.3f} ± {np.std(yy, ddof=1) / math.sqrt(len(yy)):.3f}", flush=True)


def report():
    out = load()
    if "relabel_summary" in out: print("relabel:", out["relabel_summary"])
    cs = curves()
    for k, pts in sorted(cs.items()):
        a, y = rising(pts)
        print(f"{k[0]:18s} {k[1]:12s} guidance {k[2]}: " + " ".join(f"{x:.1f}:{v:.3f}" for x, v in pts)
              + f" · rising range {y[0]:.3f} -> {y[-1]:.3f} (peak at ask {a[-1]:.1f})")
    cal = out.get("calibration")
    if cal:
        print(f"\ncalibration on {cal['curve']['sweep']} / {cal['curve']['model']} / guidance {cal['curve']['guidance']}")
        errs = []
        for c in cal["cells"].values():
            yy = [v for v in c.get("y", []) if v is not None]
            if not yy: continue
            m, se = float(np.mean(yy)), float(np.std(yy, ddof=1) / math.sqrt(len(yy))); errs.append(m - c["target"])
            print(f"  target {c['target']:.3f} -> ask {c['ask']:.3f} -> got {m:.3f} ± {se:.3f} (miss {m - c['target']:+.3f})")
        if errs: print(f"  mean absolute miss {np.mean(np.abs(errs)):.3f}"); cal["mean_abs_miss"] = float(np.mean(np.abs(errs)))
    save(out)


if __name__ == "__main__":
    {"relabel": relabel, "train": train, "calibrate": calibrate, "report": report}[sys.argv[1]]()
