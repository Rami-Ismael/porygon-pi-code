"""The two levers the 2026-09-01 active-search run left on the table, as a 2x2.

Run 1 (`activesearch.py`) found the value acquisition worth +0.055 and durable, and
the re-steer worth +0.052 at generation 1 decaying to +0.008 by generation 4. It
named two causes and two fixes:

  lever 1 — the elite CUT, not the elite FRACTION.  Elites were max(100, 0.25|D|),
            so as labels piled up the cut loosened 0.458 -> 0.375 and the tilt
            weakened.  Fix: hold the cut at the anchor's own median and let the
            elite set stay as selective as it was at generation 1.

  lever 2 — score the MARGIN, not the binary win.  At 24 battles most of a weak
            team's label variance is coin noise.  The faint differential is the
            same battles at finer resolution (Fontaine et al., GECCO 2019), so it
            should sharpen both the surrogate and the elite set at zero extra cost.

Four arms, one factor each way, everything else identical to run 1:

              score      elite rule
    active    win rate   top 0.25 fraction     <- run 1's configuration, the reference cell
    cut       win rate   fixed cut
    margin    margin     top 0.25 fraction
    both      margin     fixed cut

The margin is a SURROGATE and ELITE-SELECTION signal only.  The objective f stays
expected win rate against the meta, and every table below reports win rate.

p0 and the 200-team anchor set are the same as run 1 (same checkpoint, same seed-7
draw); the anchor is re-battled because run 1's labels carry no margin, which also
gives an independent re-measurement of the real-team baseline.

    python activesearch2.py smoke
    python activesearch2.py
"""
import json, os, random, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/tmp/vgc-pilot/src")
from corpus import parse_team, parse_team_text
from encode import Vocab, Legality
import diffusion as D
import hpsdiffusion as H
from hps_generate import Validator
import pool
import activesearch as A

ROOT = Path("/tmp/vgc-pilot/activesearch2")
RESULTS = "/Users/ramiismael/Documents/code/vgc-team-generator-pilot/results/activesearch2.json"

ANCHOR_N = 200
PROPOSE = 512
BATTLE = 128
BATTLES = 24
GENS = 4
RHO = 0.25
ELITE_MIN = 100

# score key -> how a labelled team is ranked for the surrogate and the elite set
ARMS = {"active": ("win_rate", "frac"),
        "cut":    ("win_rate", "cut"),
        "margin": ("margin",   "frac"),
        "both":   ("margin",   "cut")}


def elite_index(lab_score, rule, cut):
    """Indices of the elite set under one of the two rules."""
    s = np.asarray(lab_score, dtype=float)
    if rule == "cut":
        idx = np.flatnonzero(s >= cut)
        if len(idx) < ELITE_MIN:                       # never finetune on too few
            idx = np.argsort(-s)[:ELITE_MIN]
        return idx[np.argsort(-s[idx])]
    return np.argsort(-s)[:max(ELITE_MIN, int(RHO * len(s)))]


def main(smoke=False):
    global ANCHOR_N, PROPOSE, BATTLE, BATTLES, GENS, ELITE_MIN
    if smoke:
        ANCHOR_N, PROPOSE, BATTLE, BATTLES, GENS, ELITE_MIN = 8, 8, 4, 2, 1, 4

    random.seed(0); torch.manual_seed(0)
    ROOT.mkdir(parents=True, exist_ok=True)
    out = (json.load(open(RESULTS)) if os.path.exists(RESULTS) and not smoke
           else {"config": {}, "anchor": None, "gens": {}})
    out["config"] = dict(anchor_n=ANCHOR_N, propose=PROPOSE, battle=BATTLE,
                         battles=BATTLES, gens=GENS, rho=RHO, elite_min=ELITE_MIN,
                         arms={k: list(v) for k, v in ARMS.items()})

    def flush():
        if smoke:
            return
        tmp = RESULTS + ".tmp"
        json.dump(out, open(tmp, "w"), indent=1); os.replace(tmp, RESULTS)

    print("loading corpus …", flush=True)
    cf = A.load_corpus_files()
    corpus_files = [f for f, _ in cf]
    corpus_teams = [t for _, t in cf]
    corpus_sets = {A.species_set(t) for t in corpus_teams}
    V = Vocab(corpus_teams); L = Legality(corpus_teams); C = D.Constraints(V, L)
    corpus_grid = np.stack([V.encode(t) for t in corpus_teams])
    look, spreads = A.decode_tables(corpus_teams)
    opp = A.opponents()
    print(f"  {len(corpus_teams)} corpus teams, {len(opp)} meta opponents, {D.DEV}",
          flush=True)

    # ---- anchor: same 200 real teams as run 1, re-battled for the margin ----
    if out.get("anchor") is None:
        pk = np.random.default_rng(7).permutation(len(corpus_files))[:ANCHOR_N]
        pick = [corpus_files[i] for i in pk]
        print(f"labelling {len(pick)} real corpus teams "
              f"({len(pick)*BATTLES} battles) …", flush=True)
        res = pool.score([(f, opp) for f in pick], battles=BATTLES, conc=50)
        out["anchor"] = {f: {"win_rate": res[f]["win_rate"], "margin": res[f]["margin"]}
                         for f in pick if f in res}
        flush()
    anchor = out["anchor"]
    anchor_files = list(anchor)
    anchor_teams = [parse_team(f) for f in anchor_files]
    anchor_lab = [anchor[f] for f in anchor_files]
    real_mean = float(np.mean([v["win_rate"] for v in anchor_lab]))
    # The fixed cut is the anchor's OWN median under whichever score the arm uses,
    # so the two score types are equally selective by construction.
    CUT = {k: float(np.median([v[k] for v in anchor_lab])) for k in ("win_rate", "margin")}
    print(f"  real baseline: win rate {real_mean:.4f} "
          f"(median {CUT['win_rate']:.3f}) · margin "
          f"{np.mean([v['margin'] for v in anchor_lab]):.4f} (median {CUT['margin']:.3f})",
          flush=True)

    if not os.path.exists(A.P0_CKPT):
        print("training p0 …", flush=True)
        A.train_p0(V, corpus_teams, 3 if smoke else A.P0_EPOCHS)
    p0 = H.TeamDiffusionHPS(V).to(D.DEV)
    p0.load_state_dict(torch.load(A.P0_CKPT, map_location=D.DEV)["sd"]); p0.eval()
    print(f"  p0 loaded from {A.P0_CKPT}", flush=True)

    val = Validator()

    def run_batch(tag, model, lab_teams, lab_score, seed):
        files, teams, pastes, validity = A.propose(model, V, C, look, spreads, val,
                                                   PROPOSE, ROOT / tag, seed)
        mu = A.ridge_predict(lab_teams, lab_score, teams)
        sel = [int(i) for i in np.argsort(-mu)[:BATTLE]]      # value acquisition, always
        res = pool.score([(files[i], opp) for i in sel], battles=BATTLES, conc=50)
        keep = [i for i in sel if files[i] in res and res[files[i]]["battles"] > 0]
        lab = [{"win_rate": res[files[i]]["win_rate"], "margin": res[files[i]]["margin"]}
               for i in keep]
        wr = [v["win_rate"] for v in lab]
        mg = [v["margin"] for v in lab]
        st = A.batch_stats(wr, [teams[i] for i in keep], corpus_sets, real_mean)
        st.update(validity=validity, mean_margin=float(np.mean(mg)),
                  surrogate_spearman=A.spearman([mu[i] for i in keep], wr),
                  surrogate_spearman_margin=A.spearman([mu[i] for i in keep], mg),
                  margin_vs_win=A.spearman(mg, wr),
                  proposal=A.memorisation(V, teams, corpus_grid), n_proposed=len(teams))
        return st, [teams[i] for i in keep], [pastes[i] for i in keep], lab

    # ---- generation 0: one shared batch from p0, random acquisition ----
    if "gen0" not in out["gens"]:
        print("\n=== generation 0 (shared, random acquisition) ===", flush=True)
        files, teams, pastes, validity = A.propose(p0, V, C, look, spreads, val,
                                                   PROPOSE, ROOT / "gen0", 2000)
        sel = [int(i) for i in
               np.random.default_rng(2_500_000).permutation(len(files))[:BATTLE]]
        res = pool.score([(files[i], opp) for i in sel], battles=BATTLES, conc=50)
        keep = [i for i in sel if files[i] in res and res[files[i]]["battles"] > 0]
        lab = [{"win_rate": res[files[i]]["win_rate"], "margin": res[files[i]]["margin"]}
               for i in keep]
        wr = [v["win_rate"] for v in lab]
        st = A.batch_stats(wr, [teams[i] for i in keep], corpus_sets, real_mean)
        st.update(validity=validity, mean_margin=float(np.mean([v["margin"] for v in lab])),
                  surrogate_spearman=None, margin_vs_win=A.spearman(
                      [v["margin"] for v in lab], wr),
                  proposal=A.memorisation(V, teams, corpus_grid), n_proposed=len(teams))
        out["gens"]["gen0"] = dict(stats=st, pastes=[pastes[i] for i in keep], lab=lab)
        flush()
        print(A.fmt("gen0", st), flush=True)
    gen0 = out["gens"]["gen0"]
    gen0_teams = [parse_team_text(p) for p in gen0["pastes"]]
    gen0_lab = list(gen0["lab"])

    # ---- the 2x2 ----
    for arm, (key, rule) in ARMS.items():
        lab_teams = list(anchor_teams) + list(gen0_teams)
        lab = list(anchor_lab) + list(gen0_lab)
        for g in range(1, GENS + 1):
            tag = f"{arm}_g{g}"
            if tag in out["gens"]:
                r = out["gens"][tag]
                lab_teams += [parse_team_text(p) for p in r["pastes"]]
                lab += list(r["lab"])
                print(f"[{tag}] cached", flush=True)
                continue
            t0 = time.perf_counter()
            score = [v[key] for v in lab]
            el = elite_index(score, rule, CUT[key])
            elite = [lab_teams[i] for i in el]
            n_real = int(sum(1 for i in el if i < len(anchor_teams)))
            print(f"\n=== {tag} · score={key} elite={rule} · {len(elite)} elites "
                  f"(cut {min(score[i] for i in el):.3f}, {n_real} real, "
                  f"{len(elite)-n_real} proposals) ===", flush=True)
            model = A.resteer(V, elite)
            st, tms, pas, nl = run_batch(tag, model, lab_teams, score, 2000 + g)
            st.update(arm=arm, score_key=key, elite_rule=rule, elite_n=len(elite),
                      elite_cut=float(min(score[i] for i in el)), elite_real=n_real,
                      elite_proposals=len(elite) - n_real,
                      minutes=(time.perf_counter() - t0) / 60)
            out["gens"][tag] = dict(stats=st, pastes=pas, lab=nl)
            flush()
            print(A.fmt(tag, st), flush=True)
            lab_teams += tms; lab += nl

    val.close()
    print("\nACTIVESEARCH2_DONE", flush=True)


if __name__ == "__main__":
    main(smoke=("smoke" in sys.argv))
