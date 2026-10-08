"""Active search with the diffusion model as the learned proposal distribution.

The loop the 2026-08-29 proposer sweep said was missing a component:

    propose de novo  ->  spend the battle budget on the proposals a surrogate
    ranks HIGHEST  ->  re-steer the proposal on the labels  ->  repeat

The acquisition is greedy expected value -- Garnett's active search utility
u(D) = sum_i y_i over the teams actually labelled. It is NOT expected information
gain, and it is NOT classifier-free guidance (a tilt on the proposer, not an
acquisition, and measured ~1x mass lift on 2026-08-29).

This is a controlled replacement of the acquisition in `loop.py` (2026-08-27),
which battled the 200 MOST PMI-SURPRISING of 400 proposals -- a novelty acquisition,
i.e. the information-gain family -- and folded all 200 results back undifferentiated.
Its diffusion arm went 0.165 / 0.172 / 0.153 over three rounds: flat.

Three budget-matched arms, identical p0, identical battles per generation,
differing only in the two things under test:

    active  : surrogate-ranked acquisition  +  re-steer      (the method)
    random  : random acquisition            +  re-steer      (isolates acquisition)
    frozen  : surrogate-ranked acquisition  +  no re-steer   (isolates re-steer)

Generation 0 is a single shared random batch (no surrogate exists yet); the arms
diverge from generation 1.  Proposal draws use common random numbers per
generation, so at generation 1 `active` and `random` hold the same model and the
same 512 proposals and differ ONLY in which 128 are battled -- an exactly paired
test of the acquisition.

Re-steering is cross-entropy-method style and always finetunes p0, never
p_{g-1}: p_g = finetune(p0, elites(D_g)).  Re-initialising keeps generations
comparable and avoids the self-consuming chain that stalled on 2026-08-29
(`selftrain.py`).  Elites are the top RHO of everything labelled so far, real
anchor teams included -- so a proposal only moves the distribution by actually
outscoring real teams, which is the bar that matters.

    python activesearch.py smoke     # tiny sizes, proves the plumbing end to end
    python activesearch.py           # the run; resumable, flushes every arm-gen
"""
import itertools, json, os, random, sys, time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/tmp/vgc-pilot/src")
from corpus import parse_team, parse_team_text, norm, STATS
from encode import Vocab, Legality, NF, NSLOT
import diffusion as D
import hpsdiffusion as H
from hps_generate import Validator, slot_to_text
import pool

ROOT = Path("/tmp/vgc-pilot/activesearch")
RESULTS = "/Users/ramiismael/Documents/code/vgc-team-generator-pilot/results/activesearch.json"
CORPUS_DIR = Path("/tmp/vgc-pilot/vgc-bench/teams/reg_mb")
P0_CKPT = "/tmp/vgc-pilot/activesearch_p0.pt"

# ---- budget ---------------------------------------------------------------
ANCHOR_N = 200          # real corpus teams labelled once: this run's baseline + elite floor
PROPOSE = 512           # valid de novo proposals drawn per arm per generation
BATTLE = 128            # of those, how many get the battle budget
BATTLES = 24            # battles per labelled team, vs the top-50 meta pool
GENS = 4                # generations after the shared generation 0
RHO = 0.25              # elite fraction for the re-steer
ELITE_MIN = 100         # never finetune on fewer than this many teams
P0_EPOCHS = 600         # p0: trained on the corpus from scratch
FT_EPOCHS = 40          # re-steer: finetune p0 on the elites
FT_LR = 1e-4
REAL_MEDIAN = 0.458     # measured 2026-08-27; a reporting threshold only


# --------------------------------------------------------------- corpus + tables
def load_corpus_files():
    """(path, parsed team) for every 6-slot real team in the Reg M-B corpus."""
    out = []
    for f in sorted(CORPUS_DIR.rglob("*.txt")):
        t = parse_team(f)
        if len(t) == 6:
            out.append((str(f), t))
    return out


def decode_tables(teams):
    """Showdown display names, and the corpus Stat Point spreads per species.

    Stat Points are not in the 48-column diffusion grid, so a proposal inherits a
    spread drawn from the corpus sets of that species -- the hps_eval.py convention.
    Isolating the spread is worth ~1.3 points (2026-08-27) and is not what this
    experiment tests, so every arm carries the same handicap.
    """
    look, spreads = {}, defaultdict(list)
    for t in teams:
        for s in t:
            sp = norm(s["species"])
            look[sp] = s["species"]
            look[norm(s["ability"])] = s["ability"]
            if s["item"]:
                look[norm(s["item"])] = s["item"]
            for m in s["moves"]:
                look[norm(m)] = m
            if s["nature"]:
                look[norm(s["nature"])] = s["nature"]
            spreads[sp].append(dict(s["evs"]))
    return look, spreads


def row_to_paste(V, row, look, spreads, rng):
    slots = []
    for i in range(NSLOT):
        b = i * NF
        g = lambda j: V.decode_field(b + j, int(row[b + j]))
        sp = g(0)
        mv = [look.get(g(3 + j), g(3 + j)) for j in range(4)
              if g(3 + j) and g(3 + j) != "[MASK]"]
        pl = spreads.get(sp) or [{s: 0 for s in STATS}]
        slots.append(dict(species=look.get(sp, sp), item=look.get(g(2), g(2)),
                          ability=look.get(g(1), g(1)), nature=look.get(g(7), g(7)),
                          moves=mv, evs=pl[int(rng.integers(0, len(pl)))]))
    return "\n\n".join(slot_to_text(s) for s in slots) + "\n"


def opponents():
    """The top-50 meta pool: the same opponents every campaign in this repo used."""
    opp = []
    for i in [t["id"] for t in json.load(open("/tmp/vgc-pilot/top50_evs.json"))]:
        for c in (CORPUS_DIR / f"{i}.txt", CORPUS_DIR / "featured" / f"{i}.txt"):
            if c.exists():
                opp.append(str(c)); break
    return opp


# --------------------------------------------------------------- the proposal model
def _epoch(m, opt, xt, wt, batch):
    perm = torch.randperm(xt.shape[0], device=D.DEV)
    tot = nb = 0
    for i in range(0, xt.shape[0], batch):
        b = perm[i:i + batch]
        xb = xt[b].view(-1, NSLOT, NF)
        ps = torch.rand(xb.shape[0], NSLOT, device=xb.device).argsort(1)
        xb = torch.gather(xb, 1, ps.unsqueeze(-1).expand(-1, -1, NF)).reshape(-1, D.COLS)
        loss = m.loss(xb, wt[b])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.); opt.step()
        tot += float(loss.detach()); nb += 1
    return tot / max(nb, 1)


def _tensors(V, teams):
    X = np.stack([V.encode(t) for t in teams])
    return (torch.tensor(X, device=D.DEV),
            torch.full((len(X),), H.WNULL, device=D.DEV, dtype=torch.long))


def train_p0(V, corpus_teams, epochs, seed=0, batch=128):
    """p0: unconditional masked diffusion over the real Reg M-B corpus.

    TeamDiffusionHPS's win-rate-bin token is pinned to WNULL everywhere, so its
    embedding is a constant bias and the model is unconditional.  Steering happens
    in the re-steer, not through a conditioning token: guidance measured ~1x mass
    lift on 2026-08-29 and is a tilt on the proposer, not an acquisition.
    """
    torch.manual_seed(seed)
    xt, wt = _tensors(V, corpus_teams)
    m = H.TeamDiffusionHPS(V).to(D.DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    t0 = time.perf_counter()
    for ep in range(epochs):
        m.train()
        ls = _epoch(m, opt, xt, wt, batch)
        sch.step()
        if (ep + 1) % 100 == 0 or ep == 0:
            print(f"    p0 ep{ep+1:4d} loss {ls:7.3f} "
                  f"({(time.perf_counter()-t0)/60:.1f} min)", flush=True)
    m.eval()
    torch.save({"sd": m.state_dict()}, P0_CKPT)
    return m


def resteer(V, elite_teams, epochs=FT_EPOCHS, lr=FT_LR, seed=0, batch=64):
    """p_g = finetune(p0, elites).  Always from p0, never from p_{g-1}."""
    torch.manual_seed(seed)
    m = H.TeamDiffusionHPS(V).to(D.DEV)
    m.load_state_dict(torch.load(P0_CKPT, map_location=D.DEV)["sd"])
    xt, wt = _tensors(V, elite_teams)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for _ in range(epochs):
        m.train(); _epoch(m, opt, xt, wt, batch); sch.step()
    m.eval()
    return m


def propose(model, V, C, look, spreads, val, n_valid, outdir, seed, chunk=48):
    """Draw `n_valid` Showdown-valid de novo teams.

    Unbiased decoding: temperature 1.0, no guidance, no top-p.  Biased decoding
    accelerates recall collapse (Alemohammad, ICLR 2024) and would confound the
    re-steer with a decode-time tilt.  Seeded per generation so arms holding the
    same model draw the same proposals (common random numbers).
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    outdir.mkdir(parents=True, exist_ok=True)
    for f in outdir.glob("*.txt"):
        f.unlink()
    files, teams, pastes, attempts = [], [], [], 0
    t0 = time.perf_counter()
    while len(files) < n_valid and attempts < n_valid * 6:
        want = max(2, min(chunk, 2 * (n_valid - len(files))))
        x = H.sample_constrained(model, C, want, H.WNULL, 1.0).cpu().numpy()
        for i in range(want):
            attempts += 1
            txt = row_to_paste(V, x[i], look, spreads, rng)
            if val(txt) is None:
                p = outdir / f"{len(files):04d}.txt"
                p.write_text(txt)
                files.append(str(p)); pastes.append(txt); teams.append(parse_team_text(txt))
                if len(files) >= n_valid:
                    break
    print(f"    proposed {len(files)} valid / {attempts} sampled "
          f"({len(files)/max(attempts,1):.0%}) in {(time.perf_counter()-t0)/60:.1f} min",
          flush=True)
    return files, teams, pastes, len(files) / max(attempts, 1)


# --------------------------------------------------------------- the surrogate
LAM_S, LAM_P, MIN_PAIR = 10.0, 30.0, 8


class Feats:
    """Field indicators + species-pair indicators -- the family that scored
    Spearman 0.83 / R2 0.68 from 500 labels on 2026-08-29 (`al_experiment.py`).
    The map is unsupervised, so rebuilding it over labelled + candidate teams
    leaks no label."""

    def __init__(self, teams):
        vals = {k: set() for k in ["species", "ability", "item", "move", "nature"]}
        pc = Counter()
        for t in teams:
            sps = sorted({norm(s["species"]) for s in t})
            for a, b in itertools.combinations(sps, 2):
                pc[(a, b)] += 1
            for s in t:
                vals["species"].add(norm(s["species"]))
                vals["ability"].add(norm(s["ability"]))
                if s["item"]:
                    vals["item"].add(norm(s["item"]))
                if s["nature"]:
                    vals["nature"].add(norm(s["nature"]))
                for m in s["moves"]:
                    vals["move"].add(norm(m))
        self.idx, off = {}, 0
        for k in vals:
            for v in sorted(vals[k]):
                self.idx[(k, v)] = off; off += 1
        self.n_single = off
        self.pairs = {p: off + i for i, p in enumerate(
            sorted(p for p, c in pc.items() if c >= MIN_PAIR))}
        self.dim = off + len(self.pairs) + 1
        self.lam = np.concatenate([np.full(self.n_single, LAM_S),
                                   np.full(len(self.pairs), LAM_P), [0.0]])

    def vec(self, t):
        x = np.zeros(self.dim); x[-1] = 1.0
        sps = sorted({norm(s["species"]) for s in t})
        for a, b in itertools.combinations(sps, 2):
            if (a, b) in self.pairs:
                x[self.pairs[(a, b)]] = 1.0
        for s in t:
            for k, v in [("species", s["species"]), ("ability", s["ability"]),
                         ("item", s["item"]), ("nature", s["nature"])]:
                if v and (k, norm(v)) in self.idx:
                    x[self.idx[(k, norm(v))]] = 1.0
            for m in s["moves"]:
                if ("move", norm(m)) in self.idx:
                    x[self.idx[("move", norm(m))]] = 1.0
        return x

    def mat(self, teams):
        return np.stack([self.vec(t) for t in teams]) if teams else np.zeros((0, self.dim))


def ridge_predict(lab_teams, lab_y, cand_teams):
    """Fit on everything labelled so far, predict the fresh proposals."""
    F = Feats(lab_teams + cand_teams)
    Xl, Xc = F.mat(lab_teams), F.mat(cand_teams)
    w = np.linalg.solve(Xl.T @ Xl + np.diag(F.lam), Xl.T @ np.asarray(lab_y, dtype=float))
    return Xc @ w


def spearman(a, b):
    if len(a) < 3:
        return None
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d else None


# --------------------------------------------------------------- reporting
def species_set(t):
    return tuple(sorted(norm(s["species"]) for s in t))


def memorisation(V, teams, corpus_grid):
    """Nearest-neighbour Hamming distance to the real corpus, over the 48-field grid.

    The re-steer finetunes on ~100-200 elites, so copying is the thing that would
    fake a win: a proposal that IS a corpus team inherits a corpus team's win rate.
    Calibration from 2026-08-26: held-out real teams sit at mean distance 21.0,
    uniform-random teams at 44.7, and the copy-paste collapse threshold that the
    2026-08-27 filter-rate sweep hit is a copy rate of 0.540.
    """
    if not teams:
        return dict(copy_rate=None, nn_hamming=None, distinct_species_sets=None)
    P = np.stack([V.encode(t) for t in teams])
    d = (P[:, None, :] != corpus_grid[None, :, :]).sum(2).min(1)
    return dict(copy_rate=float((d == 0).mean()),
                nn_hamming=float(d.mean()),
                distinct_species_sets=int(len({species_set(t) for t in teams})))


def batch_stats(y, teams, corpus_sets, real_mean):
    y = np.asarray(y, dtype=float)
    return dict(
        n=int(len(y)),
        mean=float(y.mean()),
        se=float(y.std(ddof=1) / np.sqrt(len(y))) if len(y) > 1 else None,
        total=float(y.sum()),                     # the active-search utility itself
        p90=float(np.percentile(y, 90)),
        max=float(y.max()),
        frac_ge_real_median=float((y >= REAL_MEDIAN).mean()),
        frac_ge_real_mean=float((y >= real_mean).mean()),
        novel_species_set=float(np.mean([species_set(t) not in corpus_sets for t in teams])),
    )


def fmt(tag, st):
    se = "" if st["se"] is None else f" ± {st['se']:.4f}"
    rho = st.get("surrogate_spearman")
    rho = "n/a" if rho is None else f"{rho:+.3f}"
    pr = st.get("proposal") or {}
    return (f"[{tag}] n={st['n']} mean {st['mean']:.4f}{se}"
            f" · p90 {st['p90']:.3f} · max {st['max']:.3f}"
            f" · ≥median {st['frac_ge_real_median']:.1%}"
            f" · valid {st['validity']:.0%}"
            f" · surrogate ρ {rho}"
            f" · copy {pr.get('copy_rate', float('nan')):.1%}"
            f" · NN-Hamming {pr.get('nn_hamming', float('nan')):.1f}"
            f" · {pr.get('distinct_species_sets')} distinct sets/{st.get('n_proposed')}")


# --------------------------------------------------------------- the run
def main(smoke=False):
    global ANCHOR_N, PROPOSE, BATTLE, BATTLES, GENS, P0_EPOCHS, FT_EPOCHS, ELITE_MIN
    if smoke:
        ANCHOR_N, PROPOSE, BATTLE, BATTLES, GENS = 8, 8, 4, 2, 1
        P0_EPOCHS, FT_EPOCHS, ELITE_MIN = 3, 2, 4

    random.seed(0); torch.manual_seed(0)
    ROOT.mkdir(parents=True, exist_ok=True)
    out = (json.load(open(RESULTS)) if os.path.exists(RESULTS) and not smoke
           else {"config": {}, "anchor": None, "gens": {}})
    out["config"] = dict(anchor_n=ANCHOR_N, propose=PROPOSE, battle=BATTLE,
                         battles=BATTLES, gens=GENS, rho=RHO, elite_min=ELITE_MIN,
                         p0_epochs=P0_EPOCHS, ft_epochs=FT_EPOCHS, ft_lr=FT_LR)

    def flush():
        if smoke:
            return
        tmp = RESULTS + ".tmp"
        json.dump(out, open(tmp, "w"), indent=1); os.replace(tmp, RESULTS)

    print("loading corpus …", flush=True)
    cf = load_corpus_files()
    corpus_files = [f for f, _ in cf]
    corpus_teams = [t for _, t in cf]
    corpus_sets = {species_set(t) for t in corpus_teams}
    V = Vocab(corpus_teams); L = Legality(corpus_teams); C = D.Constraints(V, L)
    corpus_grid = np.stack([V.encode(t) for t in corpus_teams])
    look, spreads = decode_tables(corpus_teams)
    opp = opponents()
    print(f"  {len(corpus_teams)} corpus teams, {len(corpus_sets)} distinct species sets, "
          f"{len(opp)} meta opponents, device {D.DEV}", flush=True)

    # ---- anchor: label real teams. This run's baseline AND the elite floor. ----
    if out.get("anchor") is None:
        pk = np.random.default_rng(7).permutation(len(corpus_files))[:ANCHOR_N]
        pick = [corpus_files[i] for i in pk]
        print(f"labelling {len(pick)} real corpus teams "
              f"({len(pick)*BATTLES} battles) …", flush=True)
        res = pool.score([(f, opp) for f in pick], battles=BATTLES, conc=50)
        out["anchor"] = {f: res[f]["win_rate"] for f in pick if f in res}
        flush()
    anchor = out["anchor"]
    anchor_files = list(anchor)
    anchor_teams = [parse_team(f) for f in anchor_files]
    anchor_y = [anchor[f] for f in anchor_files]
    real_mean = float(np.mean(anchor_y))
    print(f"  real corpus baseline: {real_mean:.4f} "
          f"(median {np.median(anchor_y):.3f}, max {max(anchor_y):.3f}, n={len(anchor_y)})",
          flush=True)

    # ---- p0 ----
    if os.path.exists(P0_CKPT) and not smoke:
        p0 = H.TeamDiffusionHPS(V).to(D.DEV)
        p0.load_state_dict(torch.load(P0_CKPT, map_location=D.DEV)["sd"]); p0.eval()
        print(f"  p0 loaded from {P0_CKPT}", flush=True)
    else:
        print(f"training p0 on the corpus, {P0_EPOCHS} epochs …", flush=True)
        p0 = train_p0(V, corpus_teams, P0_EPOCHS)

    val = Validator()

    def run_batch(tag, model, mode, lab_teams, lab_y, seed):
        """One arm-generation: propose -> acquire -> label."""
        files, teams, pastes, validity = propose(model, V, C, look, spreads, val,
                                                 PROPOSE, ROOT / tag, seed)
        mu = ridge_predict(lab_teams, lab_y, teams)      # always fit: free diagnostic
        if mode == "value":
            sel = [int(i) for i in np.argsort(-mu)[:BATTLE]]
        else:
            sel = [int(i) for i in
                   np.random.default_rng(seed + 500_000).permutation(len(files))[:BATTLE]]
        res = pool.score([(files[i], opp) for i in sel], battles=BATTLES, conc=50)
        keep = [i for i in sel if files[i] in res and res[files[i]]["battles"] > 0]
        y = [res[files[i]]["win_rate"] for i in keep]
        st = batch_stats(y, [teams[i] for i in keep], corpus_sets, real_mean)
        st.update(validity=validity, acquisition=mode,
                  # how well the surrogate ranked WITHIN this batch of proposals
                  surrogate_spearman=spearman([mu[i] for i in keep], y),
                  mu_batch=float(np.mean([mu[i] for i in keep])),
                  mu_all=float(np.mean(mu)),
                  # measured over the WHOLE proposal set: a property of p_g, not of
                  # the acquisition, so `active` and `random` share it when paired
                  proposal=memorisation(V, teams, corpus_grid),
                  n_proposed=len(teams))
        return st, [teams[i] for i in keep], [pastes[i] for i in keep], y

    # ---- generation 0: one shared random batch, no surrogate exists yet ----
    if "gen0" not in out["gens"]:
        print("\n=== generation 0 (shared, random acquisition) ===", flush=True)
        st, tms, pas, y = run_batch("gen0", p0, "random", anchor_teams, anchor_y, 1000)
        out["gens"]["gen0"] = dict(stats=st, pastes=pas, y=y)
        flush()
        print(fmt("gen0", st), flush=True)
    gen0 = out["gens"]["gen0"]
    gen0_teams = [parse_team_text(p) for p in gen0["pastes"]]
    gen0_y = list(gen0["y"])

    # ---- generations 1..G, three arms ----
    arms = {"active": ("value", True), "random": ("random", True), "frozen": ("value", False)}
    for arm, (mode, do_resteer) in arms.items():
        lab_teams = list(anchor_teams) + list(gen0_teams)
        lab_y = list(anchor_y) + list(gen0_y)
        for g in range(1, GENS + 1):
            tag = f"{arm}_g{g}"
            if tag in out["gens"]:
                r = out["gens"][tag]
                lab_teams += [parse_team_text(p) for p in r["pastes"]]
                lab_y += list(r["y"])
                print(f"[{tag}] cached", flush=True)
                continue
            t0 = time.perf_counter()
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
                      f"{len(elite)-n_real} proposals) ===", flush=True)
                model = resteer(V, elite)
            else:
                print(f"\n=== {tag} · proposal frozen at p0 ===", flush=True)
                model = p0
            st, tms, pas, y = run_batch(tag, model, mode, lab_teams, lab_y, 1000 + g)
            st.update(extra); st["minutes"] = (time.perf_counter() - t0) / 60
            out["gens"][tag] = dict(stats=st, pastes=pas, y=y)
            flush()
            print(fmt(tag, st), flush=True)
            lab_teams += tms; lab_y += y

    val.close()

    # ---- headline table ----
    print(f"\nreal corpus teams (n={len(anchor_y)}): {real_mean:.4f}", flush=True)
    print(f"{'arm-gen':12s} {'mean':>8s} {'se':>7s} {'p90':>7s} {'max':>7s} "
          f"{'≥median':>8s} {'Σy':>8s} {'ρ':>7s} {'copy':>6s} {'NN':>5s}", flush=True)
    for tag in ["gen0"] + [f"{a}_g{g}" for a in arms for g in range(1, GENS + 1)]:
        if tag not in out["gens"]:
            continue
        s = out["gens"][tag]["stats"]
        r = s.get("surrogate_spearman"); p = s.get("proposal") or {}
        print(f"{tag:12s} {s['mean']:8.4f} {s['se'] or 0:7.4f} {s['p90']:7.3f} "
              f"{s['max']:7.3f} {s['frac_ge_real_median']:8.1%} {s['total']:8.1f} "
              f"{'  n/a' if r is None else f'{r:+7.3f}'} "
              f"{p.get('copy_rate', 0):6.1%} {p.get('nn_hamming', 0):5.1f}", flush=True)
    flush()
    print("\nACTIVESEARCH_DONE", flush=True)


if __name__ == "__main__":
    main(smoke=("smoke" in sys.argv))
