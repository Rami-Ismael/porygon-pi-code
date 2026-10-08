"""Guidance + acquisition + re-steer, composed - the talk's Algorithm 2 on the
activesearch protocol.

gradguide.py (2026-09-01) showed decode-time gradient guidance on a FROZEN p0
reaches 0.230, matching the activesearch re-steer arms without spending a battle
on search. Mengdi Wang's talk says the full method is guidance + adaptive
fine-tuning on the newly generated samples (45:14: guidance alone converges only
to a regularized optimum; iterating the fine-tune moves the regularizer). That
composed loop is exactly activesearch + gradguide, and this runs it.

Protocol is activesearch.py's, unchanged: 512 valid proposals per generation,
ridge fit on the arm's own label history, battle the 128 the ridge ranks highest
(value acquisition), re-steer = finetune p0 on the top-RHO elites, 4 generations.
Generation 0 and the 200-team anchor are REUSED from results/activesearch.json,
so at generation 1 each new arm is exactly paired with its 2026-09-01
counterpart - same label set, same elite cut, same proposal seed - and differs
in ONE thing, the guided decode:

    combined     : gloss guidance + value acquisition + re-steer
                     vs yesterday's `active`  (0.245, 0.220, 0.224, 0.218)
    guidedfrozen : gloss guidance + value acquisition + frozen p0
                     vs yesterday's `frozen`  (0.204, 0.201, 0.218, 0.219)

Guidance is the look-ahead form at lambda 108 (gradguide's best validity at the
strong effective strength; win rate tied naive@36). The ridge - and therefore
the tilt - is refit each generation from the arm's own labels only (no gradguide
or activesearch2 labels), so guidance knows exactly what the acquisition knows:
the label flow is the self-contained adaptive loop of the talk, and early
generations steer weaker simply because ridge shrinkage is stronger at n=328.
The tilt-spread diagnostic per generation records that growth.

    python gradloop.py smoke                       # plumbing, tiny, nothing persisted
    python gradloop.py                             # the run; resumable, flushes every arm-gen
    python gradloop.py --arms combined --gens 16   # extend one arm until saturation

Saturation rule (owner request 2026-09-01, "run more generations until it
saturates"): an arm stops when its running-best generation mean has gone 3
consecutive generations without improving by at least 0.01 (~1 SE at n=128).
The g2-g3 flat spot of the first run (0.274, 0.272) followed by the g4 jump to
0.334 would NOT have tripped this - two stalls, then a reset. Guardrails stop
earlier if novelty breaks: proposal copy rate above 0.10, or fewer than 200
distinct species sets in the 512 proposals (the 2026-08-27 filter-rate sweep put
copy-paste collapse at 0.540; these trip long before that).
"""
import json, os, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/tmp/vgc-pilot/src")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from corpus import parse_team, parse_team_text
import diffusion as D
import hpsdiffusion as H
import activesearch as A
import gradguide as GG
from hps_generate import Validator

ROOT = Path("/tmp/vgc-pilot/gradloop")
RESULTS = "/Users/ramiismael/Documents/code/vgc-team-generator-pilot/results/gradloop.json"

MODE, LAM = "gloss", 108.0      # gradguide's strong cell: win 0.231, validity 0.82
PROPOSE = 512
BATTLE = 128
BATTLES = 24
GENS = 4
RHO, ELITE_MIN = A.RHO, A.ELITE_MIN
REAL_MEAN = 0.4625              # the 2026-09-01 anchor mean, reporting only


SAT_EPS = 0.01          # a generation must beat the running best by this to count
SAT_PATIENCE = 3        # consecutive non-improving generations before stopping
COPY_MAX = 0.10         # guardrail: proposal copy rate
SETS_MIN = 200          # guardrail: distinct species sets per 512 proposals


def saturated(means, stats_by_gen):
    """(stop?, reason) after the most recent generation."""
    s = stats_by_gen[len(means)]
    p = s.get("proposal") or {}
    if (p.get("copy_rate") or 0) > COPY_MAX:
        return True, f"copy rate {p['copy_rate']:.2f} > {COPY_MAX}"
    if (p.get("distinct_species_sets") or 512) < SETS_MIN:
        return True, f"{p['distinct_species_sets']} distinct sets < {SETS_MIN}"
    best, streak = -1.0, 0
    for m in means:
        if m >= best + SAT_EPS:
            best, streak = m, 0
        else:
            streak += 1
    if streak >= SAT_PATIENCE:
        return True, (f"no +{SAT_EPS} improvement on best {best:.4f} "
                      f"for {streak} generations")
    return False, f"best {best:.4f}, stall streak {streak}/{SAT_PATIENCE}"


def main(smoke=False, gens=None, only_arms=None, mode=None, lam=None):
    global PROPOSE, BATTLE, BATTLES, GENS, MODE, LAM
    if gens:
        GENS = gens
    if mode:
        MODE = mode
    if lam:
        LAM = lam
    # the gloss@108 run keeps its bare tags; any other guidance gets a suffix so
    # both loops sit side by side in the same results file
    suffix = "" if (MODE, LAM) == ("gloss", 108.0) else f"_{MODE}{LAM:g}"
    if smoke:
        PROPOSE, BATTLE, BATTLES, GENS = 8, 4, 2, 1

    print("loading corpus + the 2026-09-01 anchor/gen0 labels …", flush=True)
    cf = A.load_corpus_files()
    corpus_teams = [t for _, t in cf]
    corpus_sets = {A.species_set(t) for t in corpus_teams}
    V = A.Vocab(corpus_teams); L = A.Legality(corpus_teams); C = D.Constraints(V, L)
    corpus_grid = np.stack([V.encode(t) for t in corpus_teams])
    look, spreads = A.decode_tables(corpus_teams)
    opp = A.opponents()

    as1 = json.load(open(A.RESULTS))
    anchor_teams = [parse_team(f) for f in as1["anchor"]]
    anchor_y = [float(y) for y in as1["anchor"].values()]
    gen0 = as1["gens"]["gen0"]
    gen0_teams = [parse_team_text(p) for p in gen0["pastes"]]
    gen0_y = [float(y) for y in gen0["y"]]
    print(f"  {len(corpus_teams)} corpus teams · anchor {len(anchor_y)} "
          f"(mean {np.mean(anchor_y):.4f}) · gen0 {len(gen0_y)} "
          f"(mean {np.mean(gen0_y):.4f}) · {len(opp)} meta opponents", flush=True)

    p0 = H.TeamDiffusionHPS(V).to(D.DEV)
    p0.load_state_dict(torch.load(A.P0_CKPT, map_location=D.DEV)["sd"]); p0.eval()
    val = Validator()

    out = (json.load(open(RESULTS)) if os.path.exists(RESULTS) and not smoke
           else {"config": {}, "gens": {}})
    out["config"] = dict(mode=MODE, lam=LAM, propose=PROPOSE, battle=BATTLE,
                         battles=BATTLES, gens=GENS, rho=RHO, elite_min=ELITE_MIN,
                         target_y=GG.TARGET_Y, p0=A.P0_CKPT,
                         gen0="reused from activesearch.json 2026-09-01")

    def flush():
        if smoke:
            return
        tmp = RESULTS + ".tmp"
        json.dump(out, open(tmp, "w"), indent=1); os.replace(tmp, RESULTS)

    arms = {"combined": True, "guidedfrozen": False}
    if only_arms:
        arms = {a: r for a, r in arms.items() if a in only_arms}
    for arm0, do_resteer in arms.items():
        arm = arm0 + suffix
        lab_teams = list(anchor_teams) + list(gen0_teams)
        lab_y = list(anchor_y) + list(gen0_y)
        means, stats_by_gen = [], {}
        for g in range(1, GENS + 1):
            tag = f"{arm}_g{g}"
            if tag in out["gens"]:
                r = out["gens"][tag]
                lab_teams += [parse_team_text(p) for p in r["pastes"]]
                lab_y += list(r["y"])
                means.append(r["stats"]["mean"]); stats_by_gen[g] = r["stats"]
                print(f"[{tag}] cached", flush=True)
                continue
            t0 = time.perf_counter()

            # the arm's own label history -> this generation's f (and its tilt)
            F, wv = GG.fit_ridge(lab_teams, lab_y, corpus_teams)
            G = GG.Guide(F, wv, V, D.DEV)

            extra = {}
            if do_resteer:
                k = max(ELITE_MIN, int(RHO * len(lab_y)))
                el = np.argsort(-np.asarray(lab_y))[:k]
                elite = [lab_teams[i] for i in el]
                n_real = int(sum(1 for i in el if i < len(anchor_teams)))
                extra = dict(elite_n=len(elite),
                             elite_cut=float(np.asarray(lab_y)[el].min()),
                             elite_real=n_real, elite_proposals=len(elite) - n_real)
                print(f"\n=== {tag} · re-steer on {len(elite)} elites "
                      f"(cut {extra['elite_cut']:.3f}, {n_real} real, "
                      f"{len(elite)-n_real} proposals) · ridge on {len(lab_y)} ===",
                      flush=True)
                model = A.resteer(V, elite)
            else:
                print(f"\n=== {tag} · p0 frozen · ridge on {len(lab_y)} ===",
                      flush=True)
                model = p0

            files, teams, pastes, validity, diag = GG.propose_guided(
                model, C, G, V, look, spreads, val, MODE, LAM, PROPOSE,
                ROOT / tag, 1000 + g)
            mu = F.mat(teams) @ wv
            sel = [int(i) for i in np.argsort(-mu)[:BATTLE]]

            if not smoke:
                GG.wait_for_quiet()
            res = A.pool.score([(files[i], opp) for i in sel],
                               battles=BATTLES, conc=50)
            keep = [i for i in sel if files[i] in res and res[files[i]]["battles"] > 0]
            y = [res[files[i]]["win_rate"] for i in keep]
            st = A.batch_stats(y, [teams[i] for i in keep], corpus_sets, REAL_MEAN)
            st.update(validity=validity, acquisition="value", mode=MODE, lam=LAM,
                      n_labels_fit=len(lab_y),
                      surrogate_spearman=A.spearman([mu[i] for i in keep], y),
                      mu_batch=float(np.mean([mu[i] for i in keep])),
                      mu_all=float(np.mean(mu)),
                      proposal=A.memorisation(V, teams, corpus_grid),
                      n_proposed=len(teams), diag=diag, **extra)
            st["minutes"] = (time.perf_counter() - t0) / 60
            out["gens"][tag] = dict(stats=st, pastes=[pastes[i] for i in keep], y=y)
            flush()
            print(A.fmt(tag, st), flush=True)
            lab_teams += [teams[i] for i in keep]; lab_y += y
            means.append(st["mean"]); stats_by_gen[g] = st
            stop, why = saturated(means, stats_by_gen)
            print(f"    saturation: {why}", flush=True)
            if stop:
                out.setdefault("saturation", {})[arm] = dict(gen=g, reason=why)
                flush()
                print(f"[{arm}] SATURATED at generation {g}: {why}", flush=True)
                break

    val.close()

    # ---- headline: the new arms beside their 2026-09-01 no-guidance twins ----
    print(f"\n{'arm-gen':16s} {'mean':>8s} {'se':>7s} {'p90':>7s} {'max':>7s} "
          f"{'valid':>6s} {'mu':>7s} {'rho':>7s} {'tilt0':>7s}", flush=True)
    rows = []
    for tag in sorted(out["gens"], key=lambda t: (t.rsplit("_g", 1)[0],
                                                  int(t.rsplit("_g", 1)[1]))):
        rows.append((tag, out["gens"][tag]["stats"]))
    for ref in ["active", "frozen"]:
        for g in range(1, 5):
            s = as1["gens"].get(f"{ref}_g{g}")
            if s:
                rows.append((f"{ref}_g{g} (as1)", s["stats"]))
    for tag, s in rows:
        d0 = (s.get("diag") or [{}])[0]
        r = s.get("surrogate_spearman")
        print(f"{tag:16s} {s['mean']:8.4f} {s['se'] or 0:7.4f} {s['p90']:7.3f} "
              f"{s['max']:7.3f} {s['validity']:6.2f} "
              f"{s.get('mu_batch', float('nan')):7.3f} "
              f"{'  n/a' if r is None else f'{r:+7.3f}'} "
              f"{d0.get('tilt_spread', 0) * d0.get('scale', 0):7.2f}", flush=True)
    flush()
    print("\nGRADLOOP_DONE", flush=True)


if __name__ == "__main__":
    def opt(flag, cast):
        return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else None
    main(smoke=("smoke" in sys.argv), gens=opt("--gens", int),
         only_arms=opt("--arms", lambda s: s.split(",")),
         mode=opt("--mode", str), lam=opt("--lam", float))
