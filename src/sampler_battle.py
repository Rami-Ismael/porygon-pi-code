"""Battle test of four samplers on the F1 + G2 mixture: asked vs got, and diversity (2026-10-11).

Question: the reveal-order legality run (2026-10-10, decode + validator only) left three samplers that
keep teams legal. Does any of them WRITE stronger or more varied teams than today's sampler?

Protocol = lossdown_battle.py / experiment A, so cells compare 1:1 with results/lossdown_battle.json:
  arms      today      lossdown_battle.sample, one field per pass in diffusion.ORDER (48 calls)
            planning   Path Planning, self-planned: every pass redraws all 48 fields and keeps the
                       s most confident (48 calls)
            remask     remasking loop, one call per loop pass, species never re-masked (about 95 calls)
            steering   Feynman-Kac steering, 8 particles per returned team, paper-style many-sample
                       reward built from the conditional and unconditional halves (384 calls)
  asked     y* in {0.1, 0.3, 0.5, 0.7, 0.9}, guidance 2, the current legality mask, temperature 1
  teams     the first 64 Showdown-valid teams per cell, corpus Stat Point spreads
  battles   one per (team, top-50 column), BC policy both sides (policy 1), all in the matchup
            matrix with origin 'sampler:<cell>'
Win rate = wins / battles against the top-50 meta teams; a top-50 team vs the other 49 = 0.531.

Decision rule, written before any battle: per arm, mean win rate over the five asked values minus
today's, with the standard error over teams. Beyond 2 standard errors counts as a difference; inside
it the arm is "no detectable difference". Diversity = distinct species line-ups among the 64 teams.

    python sampler_battle.py gen
    python battle_supervisor.py sampler_battle.py battle --then report
(run with /tmp/vgc-pilot/.venv/bin/python)
"""
import json, math, sys, time
from collections import Counter
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "/tmp/vgc-pilot/src")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lossdown as L
import lossdown_battle as LB
import activesearch as A
import diffusion as D
import matchup_db as MDB
from hps_generate import Validator
from corpus import parse_team_text

OUT = L.RES / "sampler_battle.json"
ARMS = ["today", "planning", "remask", "steering"]
ASKED, G = [0.1, 0.3, 0.5, 0.7, 0.9], 2.0
N, SEED, POLICY, PER_CELL, NB = 64, 4242, 1, 1, 96
DEV, COLS, NF, NSLOT = L.DEV, L.COLS, L.NF, L.NSLOT
ORDER = list(D.ORDER)
NINF = -float("inf")
YSTAR = None                                   # set per cell by gen()
ORIGIN = "sampler"                             # matrix origin prefix; reruns set their own


def cells():
    return [(f"{arm}_y{a:.1f}", arm, a) for arm in ARMS for a in ASKED]


# ---------------------------------------------------------------- shared sampler pieces
# Copied unchanged in logic from the 2026-10-10 legality scripts (reveal_legality.py, gaps_check.py,
# fk_paper_reward.py), which lived in a temporary folder.
def _setup():
    global V, tables, f, look, spreads, SIZES, VMAX, SZ, SIZEMASK, SPCOLS
    V, tables, f, look, spreads = LB.model()
    SIZES = tables.sizes
    VMAX = max(SIZES.values())
    SZ = [SIZES[L.KEY_OF_COL[c]] for c in range(COLS)]
    SIZEMASK = torch.zeros(COLS, VMAX, dtype=torch.bool, device=DEV)
    for c in range(COLS): SIZEMASK[c, :SZ[c]] = True
    SPCOLS = L.COLS_BY_KEY["species"]


def guided2(x, t):
    """One guided call -> guided log-scores [n,48,VMAX] and cond - uncond log-probs [n,48,VMAX]."""
    n = x.shape[0]
    w = torch.full((n,), YSTAR, device=DEV); wn = torch.full((n,), float("nan"), device=DEV)
    lc, lu = f(x, t, w), f(x, t, wn)
    out = torch.full((n, COLS, VMAX), NINF, device=DEV)
    lr = torch.zeros(n, COLS, VMAX, device=DEV)
    for k, cols in L.COLS_BY_KEY.items():
        out[:, cols, :SIZES[k]] = G * lc[k] + (1 - G) * lu[k]
        lr[:, cols, :SIZES[k]] = lc[k] - lu[k]
    return out, lr


def guided(x, t):
    return guided2(x, t)[0]


def mask_T(x):
    """The current legality mask after lossdown_battle.sample's two fix-ups -> [n,48,VMAX]."""
    n = x.shape[0]
    al = tables.allowed(x)
    ok = torch.zeros(n, COLS, VMAX, dtype=torch.bool, device=DEV)
    for k, cols in L.COLS_BY_KEY.items(): ok[:, cols, :SIZES[k]] = al[k]
    ok = torch.where((~ok.any(2)).unsqueeze(-1), SIZEMASK.unsqueeze(0), ok)     # never strand a row
    ok[:, :, 0] = False                                                          # never emit [MASK]
    return ok


def commit(x, lg, c, sel, pre=None):
    """Commit column c for rows `sel` from pass logits `lg` under the mask recomputed on the current x."""
    okc = mask_T(x)[:, c, :SZ[c]]
    v = torch.multinomial(torch.softmax(lg[:, c, :SZ[c]].masked_fill(~okc, NINF), -1), 1).squeeze(1)
    if pre is not None:
        keep = okc.gather(1, pre.view(-1, 1)).squeeze(1)                         # pre-drawn value still allowed?
        v = torch.where(keep, pre, v)
    x[:, c] = torch.where(sel, v, x[:, c])


def commit_set(x, lg, rev, pre=None):
    """Commit this pass's chosen fields one column at a time in ORDER, mask recomputed before each."""
    anyc = rev.any(0).cpu().tolist()
    for c in ORDER:
        if anyc[c]: commit(x, lg, c, rev[:, c], None if pre is None else pre[:, c])


def topk_mask(score_cpu, k):
    out = torch.zeros(score_cpu.shape, dtype=torch.bool)
    if k > 0: out.scatter_(1, score_cpu.topk(k, dim=1).indices, True)
    return out


def mfrac(x):
    return (x == 0).sum(1).float() / COLS


# ---------------------------------------------------------------- arms
@torch.no_grad()
def arm_today(n):
    return LB.sample(f, tables, n, YSTAR, G)


@torch.no_grad()
def arm_planning(n):
    x = torch.zeros(n, COLS, dtype=torch.long, device=DEV)
    for s in range(1, COLS + 1):
        masked = x == 0
        lg = guided(x, mfrac(x))
        ok = torch.zeros(n, COLS, VMAX, dtype=torch.bool, device=DEV)
        for c in range(COLS):                                            # mask of c with c itself treated as masked
            xc = x.clone(); xc[:, c] = 0
            ok[:, c] = mask_T(xc)[:, c]
        logp = torch.log_softmax(lg.masked_fill(~ok, NINF), -1)
        z = torch.multinomial(logp.exp().view(-1, VMAX), 1).view(n, COLS)
        score = logp.gather(2, z.unsqueeze(-1)).squeeze(-1)
        keep = topk_mask(score.cpu(), s).to(DEV)
        x = torch.where(keep & ~masked, x, torch.zeros_like(x))          # un-kept fields go back to mask first
        commit_set(x, lg, keep & masked, pre=z)
    return x


N_REVEAL, SIGMA, LOOPS = 43, 0.02, 48
ALPHA = N_REVEAL / COLS
P_REVEAL = SIGMA * ALPHA / (1 - ALPHA)
T_LOOP = 1.0 - N_REVEAL / COLS


@torch.no_grad()
def arm_remask(n):
    x = torch.zeros(n, COLS, dtype=torch.long, device=DEV)
    ALL = torch.ones(n, dtype=torch.bool, device=DEV)
    for step in range(N_REVEAL):                                         # phase 1: today's sampler
        lg = guided(x, torch.full((n,), 1.0 - step / COLS, device=DEV))
        commit(x, lg, ORDER[step], ALL)
    for _ in range(LOOPS):                                               # phase 2: one call, then both moves
        masked = x == 0
        lg = guided(x, torch.full((n,), T_LOOP, device=DEV))
        rm = (~masked) & (torch.rand(n, COLS, device=DEV) < SIGMA)
        rm[:, SPCOLS] = False                                            # species never re-masked
        rev = masked & (torch.rand(n, COLS, device=DEV) < P_REVEAL)
        x = x.masked_fill(rm, 0)
        commit_set(x, lg, rev)
    od = torch.tensor(ORDER, device=DEV)
    while bool((x == 0).any()):                                          # phase 3: finish in ORDER, one per call
        masked = x == 0
        lg = guided(x, mfrac(x))
        first = od[masked[:, od].long().argmax(1)]
        rev = torch.zeros(n, COLS, dtype=torch.bool, device=DEV)
        rev[torch.arange(n, device=DEV), first] = True
        commit_set(x, lg, rev & masked)
    return x


K, LAM, EVERY, NS = 8, 10.0, 6, 4


def many_sample_reward(x, lg, lr):
    n = x.shape[0]
    masked = x == 0
    p = torch.softmax(lg.masked_fill(~mask_T(x), NINF), -1)
    safe = torch.zeros_like(p); safe[..., 1] = 1.0                       # revealed fields: dummy draw, weight 0
    p = torch.where(masked.unsqueeze(-1), p, safe)
    z = torch.multinomial(p.view(-1, VMAX), NS, replacement=True).view(n, COLS, NS)
    term = torch.where(masked.unsqueeze(-1), lr.gather(2, z), torch.zeros(1, device=DEV))
    return torch.logsumexp(term.sum(1), 1) - math.log(NS)


@torch.no_grad()
def arm_steering(n):
    """n particles -> n // K returned teams (the highest-reward particle of each group of K)."""
    assert n % K == 0
    g = n // K
    x = torch.zeros(n, COLS, dtype=torch.long, device=DEV)
    ALL = torch.ones(n, dtype=torch.bool, device=DEV)
    rprev = torch.zeros(n, device=DEV)
    off = (torch.arange(g, device=DEV) * K).view(g, 1)
    for step, c in enumerate(ORDER):
        lg, lr = guided2(x, torch.full((n,), 1.0 - step / COLS, device=DEV))
        if step % EVERY == EVERY - 1:
            r = many_sample_reward(x, lg, lr)
            w = torch.softmax((LAM * (r - rprev)).view(g, K), 1)
            idx = (torch.multinomial(w, K, replacement=True) + off).view(-1)
            x, lg, rprev = x[idx], lg[idx], r[idx]
        commit(x, lg, c, ALL)
    best = rprev.view(g, K).argmax(1) + off.view(-1)
    return x[best]


FN = dict(today=arm_today, planning=arm_planning, remask=arm_remask, steering=arm_steering)
CALLS = dict(today=48, planning=48, remask=None, steering=384)


# ---------------------------------------------------------------- stages
def gen(smoke=False):
    global YSTAR
    _setup()
    path = OUT.with_name("sampler_battle_smoke.json") if smoke else OUT
    out = json.load(open(path)) if path.exists() else {"members": LB.MEMBERS, "guidance": G, "cells": {}}
    val = Validator()
    try:
        for i, (tag, arm, a) in enumerate(cells()):
            if tag in out["cells"]: continue
            if smoke and a != 0.5: continue
            YSTAR = a
            rng = np.random.default_rng(SEED + i)
            pastes, rows, tried, b, t0 = [], [], 0, 0, time.perf_counter()
            want = 4 if smoke else N
            while len(pastes) < want and tried < N * 8:
                torch.manual_seed(SEED + 1000 * i + b); b += 1
                x = FN[arm](NB).cpu().numpy()
                assert (x != 0).all(), "a [MASK] left the sampler"
                if DEV == "mps": torch.mps.empty_cache()
                for r in x:
                    tried += 1
                    p = A.row_to_paste(V, r, look, spreads, rng)
                    if val(p) is None: pastes.append(p); rows.append(r.tolist())
                    if len(pastes) >= want: break
            out["cells"][tag] = dict(arm=arm, asked=a, guidance=G, pastes=pastes, rows=rows, tried=tried,
                                     valid=len(pastes) / tried, decode_seconds=time.perf_counter() - t0, batches=b)
            json.dump(out, open(path, "w"))
            print(f"  {tag:16s} {len(pastes)}/{tried} valid ({len(pastes)/tried:.0%}) {b} batches "
                  f"{time.perf_counter()-t0:.0f}s", flush=True)
    finally:
        val.close()


def battle():
    """lossdown_battle.battle(), origin 'sampler:<cell>'. Only missing cells are battled."""
    LB._stack()
    con = MDB.connect(); out = json.load(open(OUT))
    todo = [t for t, c in out["cells"].items() if "y" not in c]
    for k in range(0, len(todo), 5):
        chunk = todo[k:k + 5]
        pastes = [p for t in chunk for p in out["cells"][t]["pastes"]]
        origins = [t for t in chunk for _ in out["cells"][t]["pastes"]]
        val = MDB.Validator()
        try:
            ids = [MDB.add_team(con, p, f"{ORIGIN}:{o}", val)[0] for p, o in zip(pastes, origins)]
        finally:
            val.close()
        legal = {t for (t,) in con.execute(f"SELECT team_id FROM team WHERE legal=1 AND team_id IN ({','.join('?' * len(ids))})", ids)}
        MDB.ensure(con, POLICY, sorted(legal), MDB.set_members(con, "top50"), PER_CELL, seed=SEED,
                   note=f"{ORIGIN} {chunk[0]}..{chunk[-1]}")
        i = 0
        for t in chunk:
            n = len(out["cells"][t]["pastes"])
            sc = [MDB.score(con, POLICY, tid) if tid in legal else None for tid in ids[i:i + n]]
            out["cells"][t].update(team_id=ids[i:i + n], y=[s["score"] if s else None for s in sc],
                                   battles=[s["battles"] if s else 0 for s in sc])
            i += n
        json.dump(out, open(OUT, "w"))
        print(f"  battled {chunk}", flush=True)


def diversity(teams):
    """teams = parsed pastes. Line-ups are sets of six species names."""
    lines = [tuple(sorted(s["species"] for s in tm)) for tm in teams]
    cnt = Counter(lines); n = len(lines)
    shared = [len(set(a) & set(b)) for i, a in enumerate(lines) for b in lines[i + 1:]]
    return dict(lineups=len(cnt), largest_share=max(cnt.values()) / n,
                species_used=len({s for l in lines for s in l}),
                mean_shared_species=float(np.mean(shared)) if shared else None)


def report():
    LB._stack()
    con = MDB.connect(); out = json.load(open(OUT))
    known = {r.tobytes() for r in L.load_data()["X"]}
    Vr = LB.X.setup()[1]
    summ = {}
    for t, c in out["cells"].items():
        if "team_id" in c:                                  # re-read: a supervisor restart may have filled cells
            sc = [MDB.score(con, POLICY, tid) for tid in c["team_id"]]
            c["y"] = [s["score"] if s and s["score"] is not None else None for s in sc]
            c["battles"] = [s["battles"] if s else 0 for s in sc]
        y = np.array([v for v in c.get("y", []) if v is not None], float)
        teams = [parse_team_text(p) for p in c["pastes"]]
        copies = sum(Vr.encode(tm).tobytes() in known for tm in teams)
        summ[t] = dict(arm=c["arm"], asked=c["asked"], n=len(y), got=float(y.mean()) if len(y) else None,
                       se=float(y.std(ddof=1) / math.sqrt(len(y))) if len(y) > 1 else None,
                       best=float(y.max()) if len(y) else None, at_or_above_real=int((y >= 0.531).sum()),
                       battles=int(sum(c.get("battles", []))), valid=c["valid"], tried=c["tried"],
                       decode_seconds=c["decode_seconds"], distinct_teams=len({tuple(r) for r in c["rows"]}),
                       copies=copies, **diversity(teams))
    arms = {}
    for arm in ARMS:
        ys = [np.array([v for v in c["y"] if v is not None], float) for c in out["cells"].values()
              if c["arm"] == arm and "y" in c]
        if len(ys) < len(ASKED): continue
        m = float(np.mean([y.mean() for y in ys]))
        se = math.sqrt(sum(y.var(ddof=1) / len(y) for y in ys)) / len(ys)
        arms[arm] = dict(mean=m, se=se, lineups_mean=float(np.mean([s["lineups"] for s in summ.values() if s["arm"] == arm])))
    for arm, s in arms.items():
        if arm == "today" or "today" not in arms: continue
        d = s["mean"] - arms["today"]["mean"]; se = math.hypot(s["se"], arms["today"]["se"])
        s.update(diff=d, diff_se=se, sigma=d / se)
    out["summary"], out["arms"] = summ, arms
    json.dump(out, open(OUT, "w"))
    print(f"{'cell':16s} {'got':>12s} {'best':>5s} {'valid':>6s} {'line-ups':>8s} {'top share':>9s} {'species':>7s} {'shared':>6s} {'copies':>6s}")
    for t, s in summ.items():
        if s["got"] is None: continue
        print(f"{t:16s} {s['got']:.3f}±{s['se']:.3f} {s['best']:5.2f} {s['valid']:6.0%} {s['lineups']:8d} "
              f"{s['largest_share']:9.2f} {s['species_used']:7d} {s['mean_shared_species']:6.2f} {s['copies']!s:>6s}")
    for arm, s in arms.items():
        extra = f"  vs today {s['diff']:+.3f}±{s['diff_se']:.3f} ({s['sigma']:+.1f}σ)" if "diff" in s else ""
        print(f"{arm:9s} mean over asked {s['mean']:.3f}±{s['se']:.3f}  line-ups {s['lineups_mean']:.1f}{extra}")
    print(f"battles: {sum(s['battles'] for s in summ.values()):,}")


if __name__ == "__main__":
    st = sys.argv[1]
    if st == "gen": gen("--smoke" in sys.argv)
    else: {"battle": battle, "report": report}[st]()
