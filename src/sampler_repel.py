"""Species repulsion inside the batch: a port of "Free Lunch for Pass@k" to the team sampler (2026-10-15).

Owner's pick after the 2026-10-14 diversity sweep: before touching guidance timing, test batch-coupled
sampling. Lamont et al. (arXiv 2603.04893, preprint) make the samples of one batch repel each other
during denoising: sample i's feature vector is projected onto the span of samples 1..i-1 and the
redundant component is penalised with one gradient step on the logits. Here the feature is the
SPECIES-PRESENCE vector of a team (sum over the six slots of the one-hot of each decoded species, plus
the current species column's probabilities), so the repulsion acts only on the six species fields and
leaves abilities, items, moves and natures to the model. Everything else is today's sampler.

Arms (same protocol as sampler_qd_run.py: asked {0.1 .. 0.9}, guidance 2, 64 Showdown-valid teams per
cell, one battle per top-50 column = 49 battles per team, policy 1, origin 'repel:<cell>'):
  today            lossdown_battle.sample, one field per pass in diffusion.ORDER
  settemp05        "free species, focused sets": species at temperature 1, every other field at 0.5
  repel            today + species repulsion at strength ALPHA (fixed by the battle-free `check` stage)
  repel_settemp05  both
Win rate = wins / battles against the top-50 meta teams, behaviour-cloned policy both sides.

Decision rule (docs/species-repulsion.md, written before any battle):
  diversity   distinct species line-ups among the 64 teams: repel above today at every asked value
  QD          winning line-ups (best team >= 0.531) and QD-score, repel minus today, jackknife SE;
              beyond 2 SE is a difference (sampler_qd.py)
  strength    mean win rate over the five asked values, repel minus today, lower bound of a 95%
              interval above -0.03 (the diversity-methods guard)
  confirm     best team of every winning line-up re-battled on 49 fresh battles; "confirmed" = fresh >= 0.531
PASS = diversity AND strength AND confirmed winners not below today's. A pass sends the sampler into the
cross-entropy loop (next experiment); a fail closes batch repulsion as a lever.

    python sampler_repel.py check              # battle-free alpha sweep, picks ALPHA
    python sampler_repel.py gen
    python battle_supervisor.py sampler_repel.py battle --then confirm --then report
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
import sampler_battle as SB
import sampler_why as SW
import sampler_qd as Q
import matchup_db as MDB
import activesearch as A
from hps_generate import Validator
from corpus import parse_team_text

OUT = L.RES / "sampler_repel.json"
CHECK = L.RES / "sampler_repel_check.json"
ARMS = ["today", "settemp05", "repel", "repel_settemp05"]
SWEEP = [0.0, 64.0, 128.0, 256.0, 384.0, 512.0, 768.0, 1024.0]
ALPHA = None                                      # set by check(); read back from CHECK by _route()
DEV, COLS, NINF = SB.DEV, SB.COLS, SB.NINF
CONFIRM_NOTE = "repel confirm"


def repel(z, pres, alpha):
    """One gradient step of species repulsion on the tempered species logits z [n, V].

    f_i = pres_i + softmax(z_i) is team i's species-presence vector. Teams are visited in batch order;
    the penalty for team i is the squared length of f_i's projection onto the orthonormal basis of
    f_1 .. f_{i-1} (the redundant part). The gradient flows through the softmax only; the basis is held
    fixed (stop-gradient), as in Lamont et al. Runs on the CPU: 96 tiny matmuls per species column."""
    z = z.detach().cpu().clone().float(); pres = pres.cpu().float()
    n = z.shape[0]
    basis = []
    for i in range(n):
        p = torch.softmax(z[i], -1)
        if basis:
            Bm = torch.stack(basis)                            # [m, V]
            f = pres[i] + p
            g = 2.0 * (Bm.t() @ (Bm @ f))                      # d/df ||B^T f||^2
            gz = p * (g - (p * g).sum())                       # softmax Jacobian
            z[i] = z[i] - alpha * gz
            p = torch.softmax(z[i], -1)
        v = pres[i] + p
        if basis:
            v = v - Bm.t() @ (Bm @ v)
        nv = v.norm()
        if nv > 1e-6: basis.append(v / nv)
    return z


@torch.no_grad()
def arm_repel(n, alpha, tau=1.0, species_tau=1.0):
    """sampler_why.arm_today_temp with species repulsion on the six species columns."""
    f, tables = SB.f, SB.tables
    w = torch.full((n,), SB.YSTAR, device=DEV); wn = torch.full((n,), float("nan"), device=DEV)
    x = torch.zeros(n, COLS, dtype=torch.long, device=DEV)
    sp_cols = L.COLS_BY_KEY["species"]
    pres = torch.zeros(n, SB.SIZES["species"], device=DEV)
    for step, c in enumerate(SB.ORDER):
        t = torch.full((n,), 1.0 - step / COLS, device=DEV)
        k = L.KEY_OF_COL[c]; j = L.COLS_BY_KEY[k].index(c)
        lc = SB.G * f(x, t, w)[k][:, j] + (1 - SB.G) * f(x, t, wn)[k][:, j]
        ok = tables.allowed(x)[k][:, j]
        ok[~ok.any(1)] = True; ok[:, 0] = False
        lc = lc.masked_fill(~ok, NINF)
        tc = species_tau if k == "species" else tau
        z = lc / tc
        if k == "species" and alpha > 0:
            s = sp_cols.index(c)
            z = repel(z, pres, alpha * (1.0 - s / len(sp_cols))).to(DEV)   # strength fades slot by slot
            z = z.masked_fill(~ok, NINF)
        x[:, c] = torch.multinomial(torch.softmax(z, -1), 1).squeeze(1)
        if k == "species":
            pres[torch.arange(n, device=DEV), x[:, c]] += 1.0
    return x


def _alpha():
    global ALPHA
    if ALPHA is None:
        ALPHA = json.load(open(CHECK))["alpha"]
    return ALPHA


def _route():
    SB.OUT, SB.ARMS, SB.SEED, SB.ORIGIN = OUT, ARMS, 6161, "repel"
    a = _alpha()
    SB.FN.update(settemp05=lambda n: SW.arm_today_temp(n, 0.5, species_tau=1.0),
                 repel=lambda n: arm_repel(n, a),
                 repel_settemp05=lambda n: arm_repel(n, a, tau=0.5, species_tau=1.0))


# ---------------------------------------------------------------- battle-free alpha sweep
def _lineups(rows):
    return len({tuple(sorted(int(r[c]) for c in L.COLS_BY_KEY["species"])) for r in rows})


def check(batches=3):
    """asked 0.5, guidance 2, `batches` x 96 teams per alpha: validity, distinct line-ups among the valid
    teams, species entropy, and the model's own log-probability of the drawn species (off-model cost).
    ALPHA = the alpha with the most distinct line-ups per valid team among alphas with validity >= 90%
    (ties -> the smaller alpha). A first sweep (0/4/16/64/256/1024, 2 batches) peaked at 256 and collapsed
    at 1024 (53 line-ups, 29 species: every team pushed onto the same unpopular species)."""
    global ALPHA
    SB._setup(); SB.YSTAR = 0.5
    val = Validator()
    res = {}
    try:
        for a in SWEEP:
            rows, valid, logp, t0 = [], [], [], time.perf_counter()
            for b in range(batches):
                torch.manual_seed(7000 + b)
                x = arm_repel(96, a)
                rng = np.random.default_rng(7000 + b)
                for r in x.cpu().numpy():
                    p = A.row_to_paste(SB.V, r, SB.look, SB.spreads, rng)
                    ok = val(p) is None
                    valid.append(ok)
                    if ok: rows.append(r.tolist())
                logp.append(float(species_logp(x).mean()))
            sp = Counter(int(r[c]) for r in rows for c in L.COLS_BY_KEY["species"])
            tot = sum(sp.values()); H = -sum(v / tot * math.log(v / tot) for v in sp.values()) if tot else 0.0
            res[str(a)] = dict(alpha=a, tried=len(valid), valid=float(np.mean(valid)), n_valid=len(rows),
                               lineups=_lineups(rows), lineups_per_valid=_lineups(rows) / max(len(rows), 1),
                               species_used=len(sp), species_exp_entropy=math.exp(H),
                               top_species_share=max(sp.values()) / tot if tot else None,
                               species_logp=float(np.mean(logp)), seconds=time.perf_counter() - t0)
            r = res[str(a)]
            print(f"  alpha {a:7.1f}  valid {r['valid']:.0%}  line-ups {r['lineups']:3d}/{r['n_valid']:3d}  "
                  f"species {r['species_used']:3d} (exp H {r['species_exp_entropy']:5.1f}, top {r['top_species_share']:.2f})  "
                  f"species log p {r['species_logp']:6.2f}  {r['seconds']:.0f}s", flush=True)
    finally:
        val.close()
    ok = [r for r in res.values() if r["valid"] >= 0.9]
    pick = max(ok, key=lambda r: (r["lineups_per_valid"], -r["alpha"]))
    ALPHA = pick["alpha"]
    json.dump(dict(asked=0.5, guidance=SB.G, batches=batches, sweep=res, alpha=ALPHA,
                   rule="alpha with the most distinct line-ups per valid team among validity >= 0.9; ties -> smaller alpha"),
              open(CHECK, "w"))
    print(f"ALPHA = {ALPHA}", flush=True)


@torch.no_grad()
def species_logp(x):
    """Mean over the six species of log p(drawn species | the other 47 fields) under the unguided mixture:
    how far the repelled draws sit from what the model itself would say (sampler_why.pll, 'u')."""
    n = x.shape[0]
    t = torch.full((n,), 1.0 / COLS, device=DEV); wn = torch.full((n,), float("nan"), device=DEV)
    out = torch.zeros(n, device=DEV)
    for c in L.COLS_BY_KEY["species"]:
        xc = x.clone(); xc[:, c] = 0
        j = L.COLS_BY_KEY["species"].index(c)
        lu = SB.f(xc, t, wn)["species"][:, j]
        out += torch.log_softmax(lu, -1).gather(1, x[:, c:c + 1]).squeeze(1).clamp_min(-30.0)
    return out / len(L.COLS_BY_KEY["species"])


# ---------------------------------------------------------------- confirm + report (sampler_qd_run.py, own OUT)
def winners(out):
    res = {}
    for c in out["cells"].values():
        if "y" not in c: continue
        for p, t, y in zip(c["pastes"], c["team_id"], c["y"]):
            if y is None: continue
            l = Q.describe(p)[0]
            cur = res.setdefault(c["arm"], {}).get(l)
            if cur is None or y > cur[1]: res[c["arm"]][l] = (t, y)
    return {a: [(l, t, y) for l, (t, y) in d.items() if y >= Q.BAR] for a, d in res.items()}


def fresh(con, team, note):
    cols = dict(con.execute("SELECT team_id, weight FROM team_set WHERE set_name='top50'")); cols.pop(team, None)
    rows = con.execute("SELECT c.lo, c.hi, SUM(c.lo_wins), SUM(c.hi_wins), SUM(c.battles) FROM contribution c JOIN batch b "
                       "USING (batch_id) WHERE b.policy_id=? AND b.note=? AND (c.lo=? OR c.hi=?) GROUP BY c.lo, c.hi",
                       (SB.POLICY, note, team, team)).fetchall()
    w = n = 0
    for lo, hi, lw, hw, k in rows:
        c = hi if lo == team else lo
        if c in cols and k: w += lw if lo == team else hw; n += k
    return (w / n if n else None), n


def confirm():
    LB._stack()
    con = MDB.connect(); out = json.load(open(OUT))
    win = winners(out)
    ids = sorted({t for v in win.values() for _, t, _ in v})
    cols = MDB.set_members(con, "top50")
    by_target = {}
    for t in ids:
        have = max([MDB.battles_done(con, SB.POLICY, t, c) for c in cols if c != t] or [0])
        if out.get("confirm", {}).get(str(t)) is None:
            by_target.setdefault(have + 1, []).append(t)
    for target, ts in sorted(by_target.items()):
        MDB.ensure(con, SB.POLICY, ts, cols, target, seed=9393, note=CONFIRM_NOTE)
    out["confirm"] = {str(t): dict(zip(("fresh", "fresh_battles"), fresh(con, t, CONFIRM_NOTE))) for t in ids}
    json.dump(out, open(OUT, "w"))
    print(f"confirmed {len(ids)} teams", flush=True)


def report():
    SB.report()                                                   # per-cell table, line-ups, strength vs today
    out = json.load(open(OUT))
    arms = {}
    for c in out["cells"].values():
        if "y" not in c: continue
        arms.setdefault(c["arm"], []).extend((*Q.describe(p), y) for p, y in zip(c["pastes"], c["y"]) if y is not None)
    summ = {}
    Q.table(f"SPECIES REPULSION alpha {_alpha()} (64 teams x asked 0.1/0.3/0.5/0.7/0.9 per arm; 49 battles each)", arms, "today", summ)
    conf = out.get("confirm", {})
    if conf:
        print("\nconfirmation: best team of each winning line-up, 49 fresh battles")
        for a, v in winners(out).items():
            f = [conf[str(t)]["fresh"] for _, t, _ in v if conf.get(str(t), {}).get("fresh") is not None]
            summ[a]["confirmed"] = sum(x >= Q.BAR for x in f); summ[a]["confirm_fresh_mean"] = float(np.mean(f)) if f else None
            print(f"{a:16s} winning {len(v):3d}  confirmed {summ[a]['confirmed']:3d}  fresh mean {np.mean(f) if f else float('nan'):.3f}")
    # decision rule
    cells = out["summary"]; s_arms = out["arms"]
    verdict = {}
    for arm in ("repel", "repel_settemp05"):
        if arm not in s_arms: continue
        div = all(cells[f"{arm}_y{a:.1f}"]["lineups"] > cells[f"today_y{a:.1f}"]["lineups"] for a in SB.ASKED)
        d, se = s_arms[arm]["diff"], s_arms[arm]["diff_se"]
        strength = d - 1.96 * se > -0.03
        conf_ok = summ.get(arm, {}).get("confirmed", 0) >= summ.get("today", {}).get("confirmed", 0) if conf else None
        verdict[arm] = dict(diversity=div, strength=strength, strength_lower=d - 1.96 * se, confirmed_not_below_today=conf_ok,
                            pass_=bool(div and strength and (conf_ok is not False)))
        print(f"\n{arm}: diversity {div}  strength lower bound {d - 1.96 * se:+.3f} ({'ok' if strength else 'FAIL'})  "
              f"confirmed vs today {conf_ok}  -> {'PASS' if verdict[arm]['pass_'] else 'FAIL'}")
    out["summary_qd"], out["verdict"], out["alpha"] = summ, verdict, _alpha()
    json.dump(out, open(OUT, "w"))


if __name__ == "__main__":
    st = sys.argv[1]
    if st == "check": check()
    else:
        _route()
        if st == "gen": SB.gen("--smoke" in sys.argv)
        elif st == "battle": SB.battle()
        elif st == "confirm": confirm()
        elif st == "report": report()
