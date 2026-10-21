"""Cross-entropy loop with Diffusion-BBO's learning rule (2026-10-16). Pre-registration: docs/cem-dbbo.md.

Diffusion-BBO (arXiv 2407.00610) retrains its conditional diffusion model on ALL data gathered so far and
asks for a target score tied to what the last round delivered. The cross-entropy loop (cem_v3.py) instead
re-steers on the elites only, and inside it the ask stopped mattering (cem_ask_range, correlation -0.03),
plausibly because elite-only training never shows the model a low label. This arm swaps the learning rule:

  1. Model.  Each generation both v3 members are fine-tuned from their ORIGINAL checkpoints on every
     labelled team so far (the 49 top-50 anchors + every battled loop team) mixed half-and-half, batch
     by batch, with a replay of each member's own original training rows (corpus, labelled pool,
     synthetic). No elite cut.
  2. Labels. Before training, each battled batch's win rates are shrunk toward that batch's mean with a
     beta-binomial prior fitted by the method of moments (anchors form their own batch).
  3. Delivered. The 10 best battled teams of a generation (by loop label) are re-battled fresh, 98
     battles each against the top-50 pool; delivered_k = their mean fresh win rate.
  4. Ask.    y_{k+1} = min(delivered_k + 0.1, 90th percentile of the shrunk labels, training max).
     To read got against ask inside a generation, 3/4 of the proposals ask y and 1/4 ask y - 0.2
     (a contrast stratum); the ridge picks the top 96 and the top 32 within each stratum.

Kept from the current base (cem_ask_range's sets_taboo): v3 members, sets-only temperature 0.5 (species at
temperature 1, every other field 0.5), taboo list (k = 2), ridge refit + ridge guidance + ridge top-k,
guidance 2, nature repair. Shared generation 0 = cem_v3_noheldout's generation 0; then generations 1-3, so
the battle budget (4 x 128 teams x 49 battles) and generation count match cem_v3_noheldout. Finalists as
cem_v3.final: the 16 best distinct battled teams by loop label, 192 fresh battles vs the top-50 pool,
then once vs ranks 51-100. No uncertainty or disagreement term anywhere.

    python battle_supervisor.py cem_dbbo.py run --then final --then report
    CEM_SMOKE=1 python cem_dbbo.py run           # tiny plumbing check, own file
(run with /tmp/vgc-pilot/.venv/bin/python)
"""
import copy, json, math, os, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "/tmp/vgc-pilot/src")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from showdown_validator import VGC_BENCH
import cem_v3 as CV
import diversity_loops as DL
import foundation as FD
import lossdown as L
import lossdown_battle as LB
import activesearch as A
import asked_vs_got as X
import matchup_db as MDB
import heldout as HO
import diffusion as D
import pool
from diversity_metrics import Reference
from hps_generate import Validator
from corpus import parse_team_text, norm
from encode import team_fields
from checkpoints import load_checkpoint

TAG = os.environ.get("CEM_TAG", "cem_dbbo")
OUT = L.RES / f"{TAG}.json"
BASE_RUN = L.RES / "cem_v3_noheldout.json"
GENS, PROPOSE, BATTLE, CONTRAST_FRAC, CONTRAST_GAP = 3, 512, 128, 0.25, 0.2
STEP, Q_CAP, TOP_DELIVER, DELIVER_BATTLES = 0.1, 0.9, 10, 98
FT_STEPS, FT_BATCH, FT_LR = 300, 64, 1e-4
FINAL_N, FINAL_BATTLES, SEED = 16, 192, 6060
# Ablation switches (2026-10-16). Defaults = the Diffusion-BBO arm above.
#   CEM_TRAIN=elite  re-steer on the elites only (cem_v3.members: >= 0.531, min 64, raw labels, 40 epochs)
#   CEM_ASK=fixed    ask FIXED_ASK every generation (contrast FIXED_ASK - 0.2), instead of the moving ask
# Arms: cem_fixed_elite (elite, fixed) = the fixed loop with this decoder; cem_moving_elite (elite, moving)
# isolates the moving ask; cem_dbbo (all, moving) adds all-data training and shrinkage on top.
TRAIN = os.environ.get("CEM_TRAIN", "all"); ASK = os.environ.get("CEM_ASK", "moving")
FIXED_ASK = float(os.environ.get("CEM_FIXED_ASK", 0.5))
#   CEM_ASK=bestw     Diffusion-BBO's own target (2026-10-16, docs/cem-dbbo.md "Ask = w x best label so far"): ask y = CEM_W x (best
#                     raw label among the teams the arm trains on: anchors + battled loop teams, or the matrix's labelled teams
#                     when CEM_TRAIN=scratch), from the first generation this arm proposes. Contrast stratum y - 0.2 as before.
W = float(os.environ.get("CEM_W", 1.0))
NO_TILT = bool(os.environ.get("CEM_NO_TILT"))                  # ridge guidance off in the decode (the ridge still picks the 128)
RANDOM_PICK = bool(os.environ.get("CEM_RANDOM_PICK"))          # battle random proposals within each ask group, not the ridge's top       # CEM_FIXED_ASK=0.6/0.7/0.8: the fixed-ask sweep (2026-10-16)
# Ablations of the fixed-ask loop (2026-10-16, docs/cem-ablations.md). Unset, every arm above runs unchanged.
#   CEM_ASK=none      no ask: elites trained with the label hidden, every proposal decoded unconditionally
#   CEM_FROZEN=1      never fine-tune: the original members propose every generation; the ridge still refits and tilts
#   CEM_CUT=0.50      elite cut (cem_v3.CUT, default 0.531);  CEM_NO_ANCHORS=1  no top-50 anchors among the elites
#   CEM_FT_EPOCHS=10  elite fine-tune length (read by cem_v3, default 40)
#   CEM_BATTLE=512    battle every proposal;  CEM_SEED  search seed
#   CEM_OWN_G0=1      generation 0 drawn by this sampler from the original members and battled in full, once per seed,
#                     shared by every arm with that seed (cem_abl_g0_s<seed>.json), instead of cem_v3_noheldout's
FROZEN, NO_ANCHORS, OWN_G0 = (bool(os.environ.get(k)) for k in ("CEM_FROZEN", "CEM_NO_ANCHORS", "CEM_OWN_G0"))
if "CEM_CUT" in os.environ: CV.CUT = float(os.environ["CEM_CUT"])
BATTLE, SEED = int(os.environ.get("CEM_BATTLE", BATTLE)), int(os.environ.get("CEM_SEED", SEED))
assert TRAIN in ("all", "elite", "scratch") and ASK in ("moving", "fixed", "none", "bestw")
ARMS = {"cem_fixed_elite": "fixed ask 0.5, elite re-steer", "cem_moving_elite": "moving ask, elite re-steer",
        "cem_dbbo": "moving ask, all-data training + shrinkage (Diffusion-BBO)",
        "cem_fixed06_elite": "fixed ask 0.6, elite re-steer", "cem_fixed07_elite": "fixed ask 0.7, elite re-steer",
        "cem_fixed08_elite": "fixed ask 0.8, elite re-steer",
        "cem_fixed_elite_notilt": "fixed ask 0.5, elite re-steer, ridge guidance off",
        "cem_fixed_elite_random": "fixed ask 0.5, elite re-steer, ridge guidance off, random pick",
        "cem_dbbo_bestw10": "ask = 1.0 x best label so far, all-data training + shrinkage",
        "cem_scratch_bestw10": "ask = 1.0 x best label so far, trained from scratch on the labelled teams",
        "cem_dbbo_bestw08": "ask = 0.8 x best label so far, all-data training + shrinkage",
        "cem_scratch_bestw08": "ask = 0.8 x best label so far, trained from scratch on the labelled teams",
        "cem_dbbo_bestw06": "ask = 0.6 x best label so far, all-data training + shrinkage",
        "cem_scratch_bestw06": "ask = 0.6 x best label so far, trained from scratch on the labelled teams"}
SMOKE = bool(os.environ.get("CEM_SMOKE"))
if SMOKE:
    TAG += "_smoke"; OUT = L.RES / f"{TAG}.json"
    GENS, PROPOSE, BATTLE, TOP_DELIVER, DELIVER_BATTLES, FT_STEPS, FINAL_N, FINAL_BATTLES = 1, 16, 8, 2, 8, 20, 2, 8
    CV.FT_EPOCHS = 1
REPO = str(VGC_BENCH)
DEV, COLS, NINF, NAN = L.DEV, L.COLS, -float("inf"), float("nan")
ORDER = list(D.ORDER)
DL.PROPOSE = PROPOSE
V3DIR = Path.home() / ".local/share/vgc-pilot-runtime/regmb-v3"


def save(out): json.dump(out, open(OUT, "w"))


# ---------------------------------------------------------------- 2. beta-binomial shrinkage toward the batch mean
def shrink(y, n):
    """Method-of-moments beta prior on one batch: mean m, between-team variance tau2 = var(y) - mean binomial
    noise. Returns the posterior means (wins + m k) / (n + k) and the prior strength k."""
    y, n = np.asarray(y, float), np.asarray(n, float)
    m = float(y.mean()); noise = float(np.mean(y * (1 - y) / n))
    tau2 = max(float(y.var(ddof=1)) - noise, 1e-4)
    k = max(m * (1 - m) / tau2 - 1, 0.0)
    return (y * n + m * k) / (n + k), dict(mean=m, var=float(y.var(ddof=1)), noise=noise, tau2=tau2, k=k,
                                           weight_on_own=float(np.mean(n / (n + k))))


def shrunk_pool(out, upto):
    """Anchors + every battled batch of generations < upto, each shrunk within its own batch."""
    rows = []
    a = out["anchors"]; s, _ = shrink([r["y"] for r in a], [r["n"] for r in a])
    rows += [dict(paste=r["paste"], y=float(v), raw=r["y"], src="anchor") for r, v in zip(a, s)]
    for g in range(upto):
        c = out["gens"][str(g)]
        rows += [dict(paste=p, y=float(v), raw=r, src=f"g{g}") for p, v, r in zip(c["lab_paste"], c["shrunk"], c["lab_raw"])]
    return rows


def label_batch(c):
    ok = [(p, y, n) for p, y, n in zip(c["battled"], c["y"], c["n"]) if y is not None]
    s, info = shrink([y for _, y, _ in ok], [n for _, _, n in ok])
    c.update(lab_paste=[p for p, _, _ in ok], lab_raw=[y for _, y, _ in ok], shrunk=[float(v) for v in s], shrink=info)


# ---------------------------------------------------------------- 1. fine-tune on all data (from the original members)
_ORIG = {}
def original_rows(name, cfg):
    """The member's own training rows, rebuilt exactly as scripts/retrain_regmb.py chose them."""
    if name in _ORIG: return _ORIG[name]
    d = L.load_data(include_heldout=True); mapped = np.load(V3DIR / "remapped.npz"); syn = np.load(V3DIR / "synthetic.npy")
    keep = HO.row_mask(d["X"], X.setup()[1]); tr, _ = L.splits(d); tr = tr[keep[tr]]; tag = L.tag_of(d["S"])
    idx = tr[(tag[tr] == L.TEST_TAG) | (tag[tr] == 0) | ((d["M"][tr, :, 1].sum(1) > 0) if cfg.get("mrow") else False)]
    if cfg.get("mrow"): idx = np.setdiff1d(idx, L.matrix_holdout(d))
    idx = idx[mapped["valid"][idx]]
    w = L.cond_matrix(d, idx) if cfg.get("mrow") else d["W"][idx]
    sw = np.full((len(syn), 50) if cfg.get("mrow") else (len(syn),), np.nan, np.float32)
    xx = np.concatenate([mapped["X"][idx], syn]); ww = np.concatenate([w, sw]).astype(np.float32)
    _ORIG[name] = (torch.as_tensor(xx, device=DEV), torch.as_tensor(ww, device=DEV))
    return _ORIG[name]


def corpus_max(V):
    mx = 0.0
    for name in FD.V3_MEMBERS:
        _, cfg, _ = L.load_net(name, V); _, ww = original_rows(name, cfg)
        w0 = (ww[:, 0] if ww.dim() == 2 else ww).cpu().numpy(); mx = max(mx, float(np.nanmax(w0)))
    return mx


def members(V, tables, lab, g):
    """Each v3 member fine-tuned from its ORIGINAL checkpoint for FT_STEPS steps; every batch is half
    original training rows, half the labelled pool (shrunk labels). Cached per generation."""
    nets = []
    for name in FD.V3_MEMBERS:
        ck = L.CK / f"{TAG}_g{g}_{name}.pt"
        net, cfg, _ = L.load_net(name, V)
        if ck.exists():
            net.load_state_dict(load_checkpoint(ck, map_location=DEV)["sd"])
        else:
            torch.manual_seed(SEED + g)
            xo, wo = original_rows(name, cfg)
            Xl = torch.as_tensor(np.stack([e[0] for e in lab]), device=DEV)
            Wl = torch.as_tensor(np.array([e[1] for e in lab], np.float32), device=DEV)
            if cfg.get("mrow"): Wl = torch.cat([Wl[:, None], torch.full((len(Wl), 49), NAN, device=DEV)], 1)
            sched = L.sched_tensor(cfg["sched"]).to(DEV); rules = tuple(cfg["rules"].split(","))
            opt = torch.optim.AdamW(net.parameters(), lr=FT_LR, weight_decay=0.01)
            net.train(); half = FT_BATCH // 2; t0 = time.perf_counter(); run = 0.0
            for step in range(FT_STEPS):
                bo = torch.randint(len(xo), (half,), device=DEV); bl = torch.randint(len(Xl), (half,), device=DEV)
                loss = L.batch_loss(net, tables, rules, torch.cat([xo[bo], Xl[bl]]), torch.cat([wo[bo], Wl[bl]]), None, sched, cfg)
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); run += float(loss.detach())
            net.eval()
            print(f"[g{g}] fine-tuned {name}: {FT_STEPS} steps, mean loss {run / FT_STEPS:.3f}, "
                  f"{(time.perf_counter() - t0) / 60:.1f} min", flush=True)
            torch.save({"sd": net.state_dict(), "cfg": ck.name}, ck)
        nets.append(L.logprobs_fn(net, tables, tuple(cfg["rules"].split(","))))
    return L.ensemble_fn(nets)


# ---------------------------------------------------------------- CEM_START=precise / CEM_PRECISE_ELITES=1 (2026-10-16, docs/ask-dial.md)
#   CEM_START=precise       generation 0 proposes from the precise-label members (dial_precise_*, ask_dial.py), and every
#                           later fine-tune starts from them instead of the original v3 members
#   CEM_PRECISE_ELITES=1    before each fine-tune, the elites are re-battled to PRECISE_CELL battles per top-50 column
#                           (about 196) and trained on that label
START, PRECISE_ELITES, PRECISE_CELL = os.environ.get("CEM_START", "v3"), bool(os.environ.get("CEM_PRECISE_ELITES")), 4
assert START in ("v3", "precise")


def precise_base(V, tables):
    nets = []
    for name in FD.V3_MEMBERS:
        net, cfg, _ = L.load_net(name, V)
        net.load_state_dict(load_checkpoint(L.CK / f"dial_precise_{name}.pt", map_location=DEV)["sd"]); net.eval()
        nets.append(L.logprobs_fn(net, tables, tuple(cfg["rules"].split(","))))
    return L.ensemble_fn(nets)


def elites_precise(con, rows, V, g):
    """cem_v3's elite choice (label >= cut, else the top ELITE_MIN), then each elite re-battled to PRECISE_CELL per
    column; returns (encoded, precise label, src) and a summary of how the labels moved."""
    ok = [r for r in rows if r["y"] >= CV.CUT]
    if len(ok) < CV.ELITE_MIN: ok = sorted(rows, key=lambda r: -r["y"])[:CV.ELITE_MIN]
    v = MDB.Validator()
    try: ids = [MDB.add_team(con, r["paste"], f"{TAG}:elite", v)[0] for r in ok]
    finally: v.close()
    MDB.ensure(con, LB.POLICY, sorted(set(ids)), MDB.set_members(con, "top50"), PRECISE_CELL, seed=SEED + 900 + g,
               note=f"{TAG} g{g} precise elites")
    out, old, new = [], [], []
    for r, t in zip(ok, ids):
        sc = MDB.score(con, LB.POLICY, t)
        tm = parse_team_text(r["paste"])
        if sc["score"] is None or len(tm) != 6: continue
        try: out.append((V.encode(tm), float(sc["score"]), r["src"]))
        except ValueError: continue
        old.append(r["y"]); new.append(sc["score"])
    return out, dict(n=len(out), old_mean=float(np.mean(old)), new_mean=float(np.mean(new)),
                     below_cut_after=int(sum(x < CV.CUT for x in new)))


def members_precise(V, tables, el, g):
    """cem_v3.members, but every fine-tune starts from the precise-label members."""
    nets = []
    for name in FD.V3_MEMBERS:
        ck = L.CK / f"{TAG}_g{g}_{name}.pt"
        net, cfg, _ = L.load_net(name, V)
        if ck.exists(): net.load_state_dict(load_checkpoint(ck, map_location=DEV)["sd"])
        else:
            net.load_state_dict(load_checkpoint(L.CK / f"dial_precise_{name}.pt", map_location=DEV)["sd"])
            torch.manual_seed(SEED + g)
            Xe = torch.as_tensor(np.stack([e[0] for e in el]), device=DEV)
            We = torch.as_tensor(np.array([e[1] for e in el], np.float32), device=DEV)
            if cfg.get("mrow"): We = torch.cat([We[:, None], torch.full((len(We), 49), NAN, device=DEV)], 1)
            sched = L.sched_tensor(cfg["sched"]).to(DEV); rules = tuple(cfg["rules"].split(","))
            opt = torch.optim.AdamW(net.parameters(), lr=CV.FT_LR, weight_decay=0.01); net.train()
            for _ in range(CV.FT_EPOCHS):
                perm = torch.randperm(len(Xe), device=DEV)
                for i in range(0, len(Xe), CV.FT_BATCH):
                    b = perm[i:i + CV.FT_BATCH]
                    loss = L.batch_loss(net, tables, rules, Xe[b], We[b], None, sched, cfg)
                    opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            net.eval(); torch.save({"sd": net.state_dict(), "cfg": ck.name}, ck)
        nets.append(L.logprobs_fn(net, tables, tuple(cfg["rules"].split(","))))
    return L.ensemble_fn(nets)


# ---------------------------------------------------------------- CEM_TRAIN=scratch: Diffusion-BBO's training, paper-faithful (2026-10-16)
SCRATCH_EPOCHS, SCRATCH_BATCH, SCRATCH_LR = int(os.environ.get("CEM_SCRATCH_EPOCHS", 20)), 256, 5e-4
if SMOKE: SCRATCH_EPOCHS = 1


def scratch_labels(con, V):
    """Every labelled team in the matchup matrix (full top-50 coverage, policy 1; held-out ranks never appear), its win
    rate shrunk toward the global mean with a method-of-moments beta prior (n = its battle count). Labelled teams only:
    no corpus, no synthetic rows. Returns encoded rows, shrunk labels, and the shrinkage summary."""
    labs = MDB.labels(con, LB.POLICY)
    xs, ys, ns = [], [], []
    for s_ in labs:
        (p,) = con.execute("SELECT paste FROM team WHERE team_id=?", (s_["team_id"],)).fetchone()
        t = parse_team_text(p)
        if len(t) != 6: continue
        try: xs.append(V.encode(t))
        except ValueError: continue
        ys.append(s_["score"]); ns.append(s_["battles"])
    sh, info = shrink(ys, ns)
    return np.stack(xs), sh.astype(np.float32), dict(info, n_teams=len(ys), raw_max=float(max(ys)))


def train_scratch(V, tables, X, W, g):
    """One fresh model per generation, random initialisation (the F1 member's architecture, scalar condition), trained
    on the labelled teams only; the label is hidden 15% of the time (classifier-free guidance, as in the paper).
    No ensemble: the paper's ensemble only feeds its uncertainty term, which this project does not use."""
    ck = L.CK / f"{TAG}_g{g}_scratch.pt"
    _, cfg, _ = L.load_net(FD.V3_MEMBERS[0], V); cfg = dict(cfg, mrow=0, p_uncond=0.15)
    rules = tuple(cfg["rules"].split(","))
    net = L.Net(V, d=cfg["d"], nhead=cfg["nhead"], nlayer=cfg["nlayer"], dropout=cfg["dropout"], time_cond=bool(cfg["time_cond"]),
                ntag=0, prime=cfg["prime"], mrow=0).to(DEV)
    if ck.exists():
        net.load_state_dict(load_checkpoint(ck, map_location=DEV)["sd"]); net.eval()
        return L.logprobs_fn(net, tables, rules)
    torch.manual_seed(SEED + 500 + g)
    xt, wt = torch.as_tensor(X, device=DEV), torch.as_tensor(W, device=DEV)
    ema = copy.deepcopy(net).eval(); sched = L.sched_tensor(cfg["sched"]).to(DEV)
    per = math.ceil(len(xt) / SCRATCH_BATCH); total = per * SCRATCH_EPOCHS; warm = min(500, total // 10 + 1)
    opt = torch.optim.AdamW(net.parameters(), lr=SCRATCH_LR, weight_decay=0.01, betas=(0.9, 0.98))
    t0 = time.perf_counter(); step = 0
    for ep in range(SCRATCH_EPOCHS):
        net.train(); perm = torch.randperm(len(xt), device=DEV); run_ = 0.0
        for i in range(0, len(xt), SCRATCH_BATCH):
            b = perm[i:i + SCRATCH_BATCH]
            for gp in opt.param_groups: gp["lr"] = SCRATCH_LR * min(1.0, (step + 1) / warm) * 0.5 * (1 + math.cos(math.pi * step / total))
            loss = L.batch_loss(net, tables, rules, xt[b], wt[b], None, sched, cfg)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); step += 1
            with torch.no_grad():
                dec = min(0.999, (1 + step) / (10 + step))
                for pe, p_ in zip(ema.parameters(), net.parameters()): pe.lerp_(p_, 1 - dec)
            run_ += float(loss.detach())
        print(f"[g{g}] scratch epoch {ep + 1}/{SCRATCH_EPOCHS}: loss {run_ / per:.3f} ({(time.perf_counter() - t0) / 60:.1f} min)", flush=True)
    torch.save({"sd": ema.state_dict(), "cfg": cfg, "n_train": len(xt)}, ck)
    return L.logprobs_fn(ema, tables, rules)


def encode_rows(rows, V):
    out = []
    for r in rows:
        t = parse_team_text(r["paste"])
        if len(t) != 6: continue                                 # parse_team_text merges item-less slots (known): skip
        try: out.append((V.encode(t), float(r["y"])))
        except ValueError: pass                                  # outside the regulation vocabulary
    return out


# ---------------------------------------------------------------- proposing: sets-only temperature, one ask per row
@torch.no_grad()
def sample_asks(G, asks, g, sur, lam, temp_sets=0.5, uncond=False):
    n = len(asks); w = asks.to(DEV); wn = torch.full((n,), NAN, device=DEV)
    x = torch.zeros(n, COLS, dtype=torch.long, device=DEV)
    for step, c in enumerate(ORDER):
        t = torch.full((n,), 1.0 - step / COLS, device=DEV)
        lc = G.logp(x, t, wn, c) if uncond else g * G.logp(x, t, w, c) + (1 - g) * G.logp(x, t, wn, c)
        if sur is not None and lam > 0: lc = lc + lam * sur.tilt(G.V, x, c)
        tc = 1.0 if c in FD.SPECIES_COLS else temp_sets
        x[:, c] = torch.multinomial(torch.softmax(lc.masked_fill(~G.ok(x, c), NINF) / tc, -1), 1).squeeze(1)
    return x


def propose(G, sur, rng, taboo, y_main, y_con, batch=96, uncond=False):
    """Accept PROPOSE valid, non-taboo teams: 3/4 asked y_main, 1/4 asked y_con. Returns (paste, slots, ask).
    uncond: decode every row unconditionally; the two asks are then only labels for two random strata."""
    need = {y_main: PROPOSE - int(PROPOSE * CONTRAST_FRAC), y_con: int(PROPOSE * CONTRAST_FRAC)}
    have = {y_main: 0, y_con: 0}
    val = Validator(); raw, kept, tried, cap = 0, [], 0, 8 * PROPOSE
    try:
        while sum(have.values()) < PROPOSE and tried < cap:
            open_ = [a for a in need if have[a] < need[a]]
            p_con = CONTRAST_FRAC if len(open_) == 2 else float(open_ == [y_con])
            asks = torch.tensor([y_con if rng.random() < p_con else y_main for _ in range(batch)], dtype=torch.float32)
            for r, a in zip(sample_asks(G, asks, 2.0, sur, 0.0 if NO_TILT else sur.lam, uncond=uncond).cpu().numpy(), asks.tolist()):
                a = y_con if abs(a - y_con) < 1e-6 else y_main
                tried += 1
                if have[a] >= need[a]: continue
                p, slots = G.to_paste(r, rng, repair=True)
                if val(p) is not None or HO.is_heldout_text(p): continue
                raw += 1
                if taboo.accept(DL.comp(slots)): kept.append((p, slots, a)); have[a] += 1
    finally:
        val.close()
    return kept, raw, tried


# ---------------------------------------------------------------- 3. delivered: fresh battles for the top 10
def deliver(out, g):
    c = out["gens"][str(g)]
    if "delivered" in c: return c["delivered"]
    pool.REPO = REPO
    ok = sorted([(y, p) for p, y in zip(c["battled"], c["y"]) if y is not None], key=lambda r: -r[0])
    seen, top = set(), []
    for y, p in ok:
        k = tuple(team_fields(parse_team_text(p)))
        if k in seen: continue
        seen.add(k); top.append((y, p))
        if len(top) >= TOP_DELIVER: break
    d = Path(f"/tmp/vgc-pilot/{TAG}_deliver_g{g}"); d.mkdir(parents=True, exist_ok=True); files = []
    for i, (_, p) in enumerate(top):
        f = d / f"{i:02d}.txt"; f.write_text(p); files.append(str(f))
    res = pool.score([(f, A.opponents()) for f in files], battles=DELIVER_BATTLES, conc=50)
    fr = [res.get(f, {}).get("win_rate") for f in files]
    c["deliver_teams"] = [dict(label=y, fresh=v, battles=res.get(f, {}).get("battles")) for (y, _), v, f in zip(top, fr, files)]
    c["delivered"] = float(np.mean([v for v in fr if v is not None])); save(out)
    print(f"[g{g}] delivered (top {len(top)} fresh x {DELIVER_BATTLES}): {c['delivered']:.3f} "
          f"vs their loop labels {np.mean([y for y, _ in top]):.3f}", flush=True)
    return c["delivered"]


def battle(con, pastes, g):
    v = MDB.Validator()
    try: ids = [MDB.add_team(con, p, f"{TAG}:g{g}", v)[0] for p in pastes]
    finally: v.close()
    legal = {t for (t,) in con.execute(f"SELECT team_id FROM team WHERE legal=1 AND team_id IN ({','.join('?' * len(ids))})", ids)}
    MDB.ensure(con, LB.POLICY, sorted(legal), MDB.set_members(con, "top50"), LB.PER_CELL, seed=SEED + g, note=f"{TAG} g{g}")
    sc = [MDB.score(con, LB.POLICY, t) if t in legal else None for t in ids]
    return ids, [s["score"] if s else None for s in sc], [s["battles"] if s else None for s in sc]


def own_g0(con, G, ref):
    """Generation 0 for the seeded ablations: the original members asked FIXED_ASK (contrast FIXED_ASK - 0.2), this
    sampler and taboo, the default ridge's tilt, BATTLE of the proposals battled. Drawn once per seed and shared by
    every arm with that seed, so the arms differ only from generation 1."""
    f = L.RES / f"cem_abl_g0_s{SEED}_b{BATTLE}.json"
    if f.exists(): return json.load(open(f))
    rng = np.random.default_rng(SEED); torch.manual_seed(SEED)
    sur = FD.Surrogate(G.V); y, yc = FIXED_ASK, FIXED_ASK - CONTRAST_GAP
    kept, raw, tried = propose(G, sur, rng, DL.Taboo(set()), y, yc)
    mu = np.array([float(sur.score(s)) for _, s, _ in kept]); asks = np.array([a for _, _, a in kept]); sel = []
    for a, k in ((y, BATTLE - int(BATTLE * CONTRAST_FRAC)), (yc, int(BATTLE * CONTRAST_FRAC))):
        ix = np.where(np.isclose(asks, a))[0]; sel += [int(i) for i in ix[np.argsort(-mu[ix])][:k]]
    c = dict(shared_from=f.name, ask=y, ask_contrast=yc, valid=raw / max(tried, 1), tried=tried,
             div=DL.diversity([(p, s) for p, s, _ in kept], ref, tried), sets=len({DL.comp(s) for _, s, _ in kept}),
             stream_comps=[list(DL.comp(s)) for _, s, _ in kept], asks_battled=[float(asks[i]) for i in sel],
             battled=[kept[i][0] for i in sel])
    ids, ys, ns = battle(con, c["battled"], 0)
    c.update(team_id=ids, y=ys, n=ns, stats=DL.bstats(ys, c["battled"]))
    label_batch(c); json.dump(c, open(f, "w"))
    return c


def corr_ci(a, y, rng, B=4000):
    a, y = np.asarray(a, float), np.asarray(y, float)
    if len(set(a)) < 2: return None, None, None
    r = float(np.corrcoef(a, y)[0, 1]); bs = []
    for _ in range(B):
        i = rng.integers(0, len(a), len(a))
        if len(set(a[i])) > 1: bs.append(np.corrcoef(a[i], y[i])[0, 1])
    lo, hi = np.quantile(bs, [.025, .975])
    return r, float(lo), float(hi)


# ---------------------------------------------------------------- the run
def run():
    LB._stack(); con = MDB.connect(); DL.CV_V[0] = None
    base = json.load(open(BASE_RUN)); g0 = base["gens"]["0"]
    out = json.load(open(OUT)) if OUT.exists() else {"config": dict(base=BASE_RUN.name, gens=GENS, propose=PROPOSE, battle=BATTLE,
        contrast_frac=CONTRAST_FRAC, contrast_gap=CONTRAST_GAP, step=STEP, q_cap=Q_CAP, top_deliver=TOP_DELIVER,
        deliver_battles=DELIVER_BATTLES, train=TRAIN, ask_rule=ASK, no_tilt=NO_TILT, random_pick=RANDOM_PICK, fixed_ask=FIXED_ASK, w=W, ft_steps=FT_STEPS, ft_batch=FT_BATCH, ft_lr=FT_LR, taboo_k=DL.TABOO_K, temp_sets=0.5,
        seed=SEED, frozen=FROZEN, start=START, precise_elites=PRECISE_ELITES, cut=CV.CUT, no_anchors=NO_ANCHORS, ft_epochs_elite=CV.FT_EPOCHS, own_g0=OWN_G0), "gens": {}}
    if "anchors" not in out:                                     # anchors with their battle counts, for the shrinkage
        out["anchors"] = []
        for tid in MDB.set_members(con, "top50"):
            (p,) = con.execute("SELECT paste FROM team WHERE team_id=?", (tid,)).fetchone()
            s = MDB.score(con, LB.POLICY, tid)
            if s["score"] is not None: out["anchors"].append(dict(paste=p, y=s["score"], n=s["battles"], team_id=tid))
        save(out)
    if "0" not in out["gens"] and not OWN_G0 and TRAIN != "scratch" and START != "precise":   # shared generation 0 = cem_v3_noheldout's
        n0 = [MDB.score(con, LB.POLICY, t)["battles"] if y is not None else None for t, y in zip(g0["team_id"], g0["y"])]
        y0 = [MDB.score(con, LB.POLICY, t)["score"] if y is not None else None for t, y in zip(g0["team_id"], g0["y"])]
        c = dict(shared_from=BASE_RUN.name, battled=g0["battled"], team_id=g0["team_id"], y=y0, n=n0, ask=0.5,
                 asks_battled=[0.5] * len(g0["battled"]), stats=DL.bstats(y0, g0["battled"]),
                 stream_comps=[list(DL.comp_paste(p)) for p in g0["proposals"] if DL.comp_paste(p)], sets=g0["sets"])
        label_batch(c); out["gens"]["0"] = c; save(out)
    G = FD.Gen("v3"); V, tables = G.V, G.tables; DL.CV_V[0] = V
    ref = Reference(G.corpus, scope="corpus", format_id="reg_mb")
    ridge = CV.Ridge()
    known = {tuple(team_fields(parse_team_text(r["paste"]))) for r in out["anchors"]}
    if "corpus_max" not in out: out["corpus_max"] = corpus_max(V); save(out)
    if "0" not in out["gens"] and TRAIN != "scratch" and START != "precise": out["gens"]["0"] = own_g0(con, G, ref); save(out)
    if START == "precise": G.f = precise_base(V, tables)
    for g in range(0, GENS + 1):
        c = out["gens"][str(g)] if str(g) in out["gens"] else None
        if (g > 0 or TRAIN == "scratch" or START == "precise") and (c is None or "y" not in c):
            t0 = time.perf_counter(); rng = np.random.default_rng(SEED + g); torch.manual_seed(SEED + g)
            c = out["gens"].setdefault(str(g), {})
            lab_rows = shrunk_pool(out, g)
            lab = encode_rows(lab_rows, V)
            sh = np.array([r["y"] for r in lab_rows])
            tmax = max(float(sh.max()), out["corpus_max"])
            if TRAIN == "scratch":                               # the training labels ARE the matrix's labelled teams
                Xs, Ws, c["scratch_labels"] = scratch_labels(con, V); sh = Ws; tmax = float(Ws.max()); save(out)
            p90 = float(np.quantile(sh, Q_CAP)); dlv = deliver(out, g - 1) if g > 0 else None
            best_raw = None
            if ASK == "bestw":
                best_raw = c["scratch_labels"]["raw_max"] if TRAIN == "scratch" else max(r["raw"] for r in lab_rows)
                y = W * best_raw
            elif g == 0: y = FIXED_ASK                           # no delivered rate yet
            else: y = min(dlv + STEP, p90, tmax) if ASK == "moving" else FIXED_ASK
            yc = y - CONTRAST_GAP
            c.update(ask=y, ask_contrast=yc, ask_parts=dict(delivered_prev=dlv, p90_shrunk=p90, training_max=tmax,
                                                                        **(dict(best_raw=best_raw, w=W) if ASK == "bestw" else {})),
                     n_labelled=len(lab_rows), n_labelled_encoded=len(lab))
            save(out)
            print(f"[g{g}] ask {y:.3f} (delivered {dlv if dlv is None else round(dlv, 3)} + {STEP}, p90 {p90:.3f}, max {tmax:.3f}); contrast {yc:.3f}; "
                  + (f"training from scratch on {c['scratch_labels']['n_teams']} labelled teams" if TRAIN == "scratch"
                     else f"training on {len(lab)} labelled + original rows"), flush=True)
            raw_rows = [dict(paste=r["paste"], y=r["raw"], src=r["src"]) for r in lab_rows]   # the ridge keeps raw labels, as in the base
            if TRAIN == "scratch":
                G.f = train_scratch(V, tables, Xs, Ws, g)
            elif g == 0 and START == "precise":                  # generation 0: the precise-label members as they are
                c["elites"] = None
            elif PRECISE_ELITES or START == "precise":
                CV.TAG, CV.SEED = TAG, SEED
                pool_ = [r for r in raw_rows if not (NO_ANCHORS and r["src"] == "anchor")]
                if PRECISE_ELITES: el, c["precise_elites"] = elites_precise(con, pool_, V, g)
                else: el = CV.elites_from(pool_, V)
                c["elites"] = dict(n=len(el), anchors=sum(e[2] == "anchor" for e in el), cut=min(e[1] for e in el)); save(out)
                G.f = members_precise(V, tables, el, g) if START == "precise" else CV.members(V, tables, el, g)
            elif FROZEN:                                         # G.f stays the original members
                c["elites"] = None
            elif TRAIN == "all":
                G.f = members(V, tables, lab, g)
            else:                                                # cem_v3's re-steer: elites only, raw labels
                CV.TAG, CV.SEED = TAG, SEED
                el = CV.elites_from([r for r in raw_rows if not (NO_ANCHORS and r["src"] == "anchor")], V)
                c["elites"] = dict(n=len(el), anchors=sum(e[2] == "anchor" for e in el), cut=min(e[1] for e in el))
                if ASK == "none": el = [(x, NAN, s) for x, _, s in el]    # label hidden: unconditional fine-tune
                G.f = CV.members(V, tables, el, g)
            sur = DL.ridge_for(ridge, raw_rows, L.RES / f"{TAG}_ridge_g{g}.npz")
            seen = {tuple(x) for k in range(g) for x in out["gens"][str(k)]["stream_comps"]}
            kept, raw, tried = propose(G, sur, rng, DL.Taboo(seen), y, yc, uncond=ASK == "none")
            mu = np.array([float(sur.score(s)) for _, s, _ in kept]); asks = np.array([a for _, _, a in kept])
            sel = []
            for a, k in ((y, BATTLE - int(BATTLE * CONTRAST_FRAC)), (yc, int(BATTLE * CONTRAST_FRAC))):
                ix = np.where(np.isclose(asks, a))[0]
                sel += [int(i) for i in (rng.permutation(ix) if RANDOM_PICK else ix[np.argsort(-mu[ix])])[:k]]
            battled = [kept[i][0] for i in sel]
            teams = [s for _, s, _ in kept]
            c.update(div=DL.diversity([(p, s) for p, s, _ in kept], ref, tried), valid=raw / max(tried, 1), tried=tried,
                     sets=len({DL.comp(s) for s in teams}),
                     copies=sum(tuple(team_fields(t)) in known for t in teams) / max(len(teams), 1),
                     stream_comps=[list(DL.comp(s)) for _, s, _ in kept], asks_battled=[float(asks[i]) for i in sel],
                     mu_battled=[float(mu[i]) for i in sel])
            save(out)
            ids, ys, ns = battle(con, battled, g)
            c.update(battled=battled, team_id=ids, y=ys, n=ns, stats=DL.bstats(ys, battled), minutes=(time.perf_counter() - t0) / 60)
            ok = [(a, v) for a, v in zip(c["asks_battled"], ys) if v is not None]
            r, lo, hi = corr_ci([a for a, _ in ok], [v for _, v in ok], np.random.default_rng(g))
            main = [v for a, v in ok if np.isclose(a, y)]; con_ = [v for a, v in ok if np.isclose(a, yc)]
            c["got_vs_ask"] = dict(corr=r, ci=[lo, hi], main_mean=float(np.mean(main)), main_n=len(main),
                                   contrast_mean=float(np.mean(con_)), contrast_n=len(con_),
                                   diff=float(np.mean(main) - np.mean(con_)),
                                   diff_se=float(math.sqrt(np.var(main, ddof=1) / len(main) + np.var(con_, ddof=1) / len(con_))))
            label_batch(c); save(out)
        c = out["gens"][str(g)]
        known |= {tuple(team_fields(parse_team_text(p))) for p in c["battled"]}
        s = c["stats"]; gv = c.get("got_vs_ask")
        print(f"[g{g}] battled {s['n']}: win rate {s['mean']:.3f} ± {s['se']:.3f} · max {s['max']:.3f} · ≥0.531 {s['above_cut']} · "
              f"compositions {c.get('div', {}).get('distinct_compositions', c.get('sets'))} · battled compositions {s['battled_compositions']}"
              + (f" · got-vs-ask r {gv['corr']:+.2f} [{gv['ci'][0]:+.2f}, {gv['ci'][1]:+.2f}] · asked {c['ask']:.2f}: {gv['main_mean']:.3f}"
                 f" vs asked {c['ask_contrast']:.2f}: {gv['contrast_mean']:.3f}" if gv else ""), flush=True)
    deliver(out, GENS)                                           # reported, not used for an ask
    out["done"] = True; save(out)


# ---------------------------------------------------------------- finalists (cem_v3.final's protocol)
def final():
    out = json.load(open(OUT)); pool.REPO = REPO
    if "final" not in out:
        rows = [dict(paste=p, y=y, src=f"g{g}") for g, c in out["gens"].items() for p, y in zip(c["battled"], c["y"]) if y is not None]
        rows.sort(key=lambda r: -r["y"])
        seen, top = set(), []
        for r in rows:
            k = tuple(team_fields(parse_team_text(r["paste"])))
            if k in seen: continue
            seen.add(k); top.append(r)
            if len(top) >= FINAL_N: break
        d = Path(f"/tmp/vgc-pilot/{TAG}_final"); d.mkdir(parents=True, exist_ok=True); files = []
        for i, r in enumerate(top):
            f = d / f"{i:02d}.txt"; f.write_text(r["paste"]); files.append(str(f))
        res = pool.score([(f, A.opponents()) for f in files], battles=FINAL_BATTLES, conc=50)
        out["final"] = [dict(paste=r["paste"], label=r["y"], src=r["src"], fresh=res.get(f, {}).get("win_rate"),
                             battles=res.get(f, {}).get("battles")) for r, f in zip(top, files)]
        save(out)
    if "heldout" not in out:
        d = Path(f"/tmp/vgc-pilot/{TAG}_final"); d.mkdir(parents=True, exist_ok=True); files = []
        for i, r in enumerate(out["final"]):
            f = d / f"{i:02d}.txt"; f.write_text(r["paste"]); files.append(str(f))
        res = HO.evaluate(files, battles=FINAL_BATTLES, conc=50)
        out["heldout"] = [dict(fresh_top50=r["fresh"], heldout=res.get(f, {}).get("win_rate"), battles=res.get(f, {}).get("battles"))
                          for r, f in zip(out["final"], files)]
        save(out)


def max_fresh(c):
    """Fresh win rate of the generation's highest-label team (from the delivered re-battle), as text."""
    d = c.get("deliver_teams")
    return f"{d[0]['fresh']:.3f}" if d and d[0].get("fresh") is not None else "-"


def boot_diff(a, b, rng, B=10000):
    a, b = np.asarray(a, float), np.asarray(b, float)
    bs = [a[rng.integers(0, len(a), len(a))].mean() - b[rng.integers(0, len(b), len(b))].mean() for _ in range(B)]
    return float(a.mean() - b.mean()), [float(x) for x in np.quantile(bs, [.025, .975])]


def report():
    out = json.load(open(OUT)); base = json.load(open(BASE_RUN)); rng = np.random.default_rng(0)
    print("win rate = wins / battles against the top-50 meta teams, behaviour-cloned policy (policy 1) on both sides")
    print(f"{'gen':>3s} {'ask':>6s} {'battled win rate':>17s} {'median':>6s} {'top10%':>6s} {'max':>5s} {'max fresh':>9s} {'asked y':>8s} {'asked y-0.2':>11s} {'got-vs-ask r [95%]':>22s} "
          f"{'compositions':>13s} {'battled comps':>13s} {'delivered':>9s} {'base win rate':>13s}")
    summ = []
    for g in range(GENS + 1):
        c = out["gens"].get(str(g)); b = base["gens"].get(str(g))
        if not c or "stats" not in c: continue
        s = c["stats"]; gv = c.get("got_vs_ask")
        comps = c.get("div", {}).get("distinct_compositions", c.get("sets"))
        yv = np.array([v for v in c["y"] if v is not None])
        print(f"{g:>3d} {c['ask']:6.3f} {s['mean']:.3f} ± {s['se']:.3f}      {np.median(yv):6.3f} {np.quantile(yv, .9):6.3f} {yv.max():5.3f} {max_fresh(c):>9s} "
              + (f"{gv['main_mean']:8.3f} {gv['contrast_mean']:11.3f} {gv['corr']:+.2f} [{gv['ci'][0]:+.2f}, {gv['ci'][1]:+.2f}]     " if gv else f"{'-':>8s} {'-':>11s} {'-':>22s} ")
              + f"{comps:13d} {s['battled_compositions']:13d} {c.get('delivered', float('nan')):9.3f} "
              + (f"{b['stats']['mean']:.3f} ({b['sets']} sets)" if b else "-"))
        summ.append(dict(gen=g, ask=c["ask"], win=s["mean"], se=s["se"], median=float(np.median(yv)), p90=float(np.quantile(yv, .9)),
                         max=float(yv.max()), max_fresh=max_fresh(c), compositions=comps, battled_compositions=s["battled_compositions"],
                         delivered=c.get("delivered"), got_vs_ask=gv, shrink=c.get("shrink"), ask_parts=c.get("ask_parts"),
                         base_win=b["stats"]["mean"] if b else None, base_sets=b["sets"] if b else None))
    res = dict(gens=summ)
    if out.get("final") and out.get("heldout"):
        f = np.array([r["fresh"] for r in out["final"]]); h = np.array([r["heldout"] for r in out["heldout"]])
        bf = np.array([r["fresh"] for r in base["final"]]); bh = np.array([r["heldout"] for r in base["heldout"]])
        d1, ci1 = boot_diff(f, bf, rng); d2, ci2 = boot_diff(h, bh, rng)
        own = h - f; own_se = float(own.std(ddof=1) / math.sqrt(len(own)))
        print(f"\nfinalists ({len(f)} x {FINAL_BATTLES} fresh battles): vs top-50 {f.mean():.3f} (best {f.max():.3f}, "
              f"{int((np.round(f, 4) >= 0.531).sum())} ≥ 0.531) · vs ranks 51-100 {h.mean():.3f} (best {h.max():.3f}) · "
              f"held-out minus top-50 {own.mean():+.3f} ± {own_se:.3f}")
        print(f"cem_v3_noheldout:  vs top-50 {bf.mean():.3f} · vs ranks 51-100 {bh.mean():.3f}")
        print(f"difference vs top-50 {d1:+.3f} [{ci1[0]:+.3f}, {ci1[1]:+.3f}] · vs ranks 51-100 {d2:+.3f} [{ci2[0]:+.3f}, {ci2[1]:+.3f}] "
              f"(two-sample bootstrap over finalist teams)")
        res.update(final_top50=float(f.mean()), final_heldout=float(h.mean()), best_top50=float(f.max()), best_heldout=float(h.max()),
                   diff_top50=d1, ci_top50=ci1, diff_heldout=d2, ci_heldout=ci2, base_top50=float(bf.mean()), base_heldout=float(bh.mean()),
                   final_srcs=[r["src"] for r in out["final"]], final_labels=[r["label"] for r in out["final"]])
    out["summary"] = res; save(out)


def compare():
    """The three arms side by side, plus the paired-by-design contrasts: fixed vs moving ask (elite re-steer),
    and elite vs all-data training (moving ask)."""
    rng = np.random.default_rng(0); res = {}
    for tag, name in ARMS.items():
        f = L.RES / f"{tag}.json"
        if not f.exists(): print(f"{tag}: not run"); continue
        o = json.load(open(f)); res[tag] = o
        print(f"\n{name} ({tag})")
        for g in range(1, GENS + 1):
            c = o["gens"].get(str(g))
            if not c or "stats" not in c: continue
            gv = c["got_vs_ask"]
            yv = np.array([v for v in c["y"] if v is not None])
            print(f"  g{g}: ask {c['ask']:.3f} · battled win rate mean {c['stats']['mean']:.3f} ± {c['stats']['se']:.3f} · median "
                  f"{np.median(yv):.3f} · top 10% {np.quantile(yv, .9):.3f} · max {yv.max():.3f} (fresh {max_fresh(c)}) · "
                  f"got-vs-ask r {gv['corr']:+.2f} [{gv['ci'][0]:+.2f}, {gv['ci'][1]:+.2f}] · asked y {gv['main_mean']:.3f} vs y-0.2 "
                  f"{gv['contrast_mean']:.3f} · compositions {c['div']['distinct_compositions']} · delivered {c.get('delivered', float('nan')):.3f}")
        if o.get("heldout"):
            print(f"  finalists: vs top-50 {np.mean([r['fresh'] for r in o['final']]):.3f} · vs ranks 51-100 "
                  f"{np.mean([r['heldout'] for r in o['heldout']]):.3f}")
    for a, b, what in (("cem_fixed_elite_notilt", "cem_fixed_elite", "ridge guidance off minus on (fixed ask 0.5)"),
                       ("cem_fixed_elite_random", "cem_fixed_elite_notilt", "random pick minus ridge pick (both ridge guidance off)"),
                       ("cem_fixed06_elite", "cem_fixed_elite", "fixed ask 0.6 minus fixed ask 0.5"),
                       ("cem_fixed07_elite", "cem_fixed_elite", "fixed ask 0.7 minus fixed ask 0.5"),
                       ("cem_fixed08_elite", "cem_fixed_elite", "fixed ask 0.8 minus fixed ask 0.5"),
                       ("cem_moving_elite", "cem_fixed_elite", "moving ask minus fixed ask (both elite re-steer)"),
                       ("cem_dbbo", "cem_moving_elite", "all-data training minus elite re-steer (both moving ask)")):
        if a in res and b in res and res[a].get("heldout") and res[b].get("heldout"):
            fa = [r["fresh"] for r in res[a]["final"]]; fb = [r["fresh"] for r in res[b]["final"]]
            ha = [r["heldout"] for r in res[a]["heldout"]]; hb = [r["heldout"] for r in res[b]["heldout"]]
            d1, c1 = boot_diff(fa, fb, rng); d2, c2 = boot_diff(ha, hb, rng)
            ba = np.mean([res[a]["gens"][str(g)]["stats"]["mean"] for g in range(1, GENS + 1)])
            bb = np.mean([res[b]["gens"][str(g)]["stats"]["mean"] for g in range(1, GENS + 1)])
            print(f"\n{what}: finalists vs top-50 {d1:+.3f} [{c1[0]:+.3f}, {c1[1]:+.3f}] · vs ranks 51-100 {d2:+.3f} "
                  f"[{c2[0]:+.3f}, {c2[1]:+.3f}] · battled win rate g1-g3 {ba - bb:+.3f}")


ABL_SEEDS = (6060, 7070, 8080)
ABL_ARMS = {"noask": "no ask (unconditional decode, label-hidden fine-tune)", "frozen": "frozen generator (no fine-tune)",
            "cut050": "elite cut 0.50", "cut056": "elite cut 0.56", "ep10": "fine-tune 10 epochs, not 40",
            "noanch": "no top-50 anchors among the elites"}


def ablate():
    """Each ablation minus the base arm (fixed ask 0.5, elite re-steer, all 512 battled), paired by seed.
    Pooled difference = mean of the per-seed differences; its 95% interval resamples finalist teams within each
    seed and arm (team noise only), so the per-seed differences are printed beside it for the seed-to-seed spread."""
    rng = np.random.default_rng(0); load = lambda a, s: L.RES / f"abl_{a}_s{s}.json"
    def fin(o): return np.array([r["fresh"] for r in o["final"]]), np.array([r["heldout"] for r in o["heldout"]])
    def gwin(o): return float(np.mean([o["gens"][str(g)]["stats"]["mean"] for g in range(1, GENS + 1)]))
    def comps(o): return int(np.mean([o["gens"][str(g)]["div"]["distinct_compositions"] for g in range(1, GENS + 1)]))
    base = {s: json.load(open(load("base", s))) for s in ABL_SEEDS if load("base", s).exists()}
    base = {s: o for s, o in base.items() if o.get("heldout")}
    print("win rate = wins / battles against the top-50 meta teams, behaviour-cloned policy (policy 1) on both sides; "
          f"finalists = 16 best distinct teams by loop label, {FINAL_BATTLES} fresh battles each")
    for s, o in base.items():
        f, h = fin(o); print(f"base s{s}: finalists vs top-50 {f.mean():.3f} · vs ranks 51-100 {h.mean():.3f} · "
                             f"battled g1-g3 {gwin(o):.3f} · compositions {comps(o)}")
    res = {}
    for a, name in ABL_ARMS.items():
        per = []
        for s, bo in base.items():
            if not load(a, s).exists(): continue
            o = json.load(open(load(a, s)))
            if o.get("heldout"): per.append((s, o, bo))
        if not per: print(f"\n{name}: not run"); continue
        d = {k: [] for k in ("top50", "heldout", "battled", "comps")}; bs = np.zeros((4000, 2))
        for s, o, bo in per:
            (fa, ha), (fb, hb) = fin(o), fin(bo)
            d["top50"].append(fa.mean() - fb.mean()); d["heldout"].append(ha.mean() - hb.mean())
            d["battled"].append(gwin(o) - gwin(bo)); d["comps"].append(comps(o) - comps(bo))
            for i in range(len(bs)):
                bs[i, 0] += (fa[rng.integers(0, len(fa), len(fa))].mean() - fb[rng.integers(0, len(fb), len(fb))].mean()) / len(per)
                bs[i, 1] += (ha[rng.integers(0, len(ha), len(ha))].mean() - hb[rng.integers(0, len(hb), len(hb))].mean()) / len(per)
        ci = np.quantile(bs, [.025, .975], axis=0)
        same_sign = len(per) == len(ABL_SEEDS) and (all(x > 0 for x in d["top50"]) or all(x < 0 for x in d["top50"]))
        moved = bool((ci[0, 0] > 0 or ci[1, 0] < 0) and same_sign)
        print(f"\n{name} ({len(per)} seeds) minus base: finalists vs top-50 {np.mean(d['top50']):+.3f} [{ci[0, 0]:+.3f}, {ci[1, 0]:+.3f}] "
              f"per seed {' '.join(f'{x:+.3f}' for x in d['top50'])} · vs ranks 51-100 {np.mean(d['heldout']):+.3f} "
              f"[{ci[0, 1]:+.3f}, {ci[1, 1]:+.3f}] · battled g1-g3 {np.mean(d['battled']):+.3f} · compositions "
              f"{np.mean(d['comps']):+.0f} · {'MATTERS' if moved else 'no detectable difference'}")
        res[a] = dict(name=name, seeds=[s for s, _, _ in per], diff_top50=float(np.mean(d["top50"])), ci_top50=ci[:, 0].tolist(),
                      per_seed_top50=[float(x) for x in d["top50"]], diff_heldout=float(np.mean(d["heldout"])),
                      ci_heldout=ci[:, 1].tolist(), diff_battled=float(np.mean(d["battled"])),
                      diff_compositions=float(np.mean(d["comps"])), matters=moved)
    json.dump(res, open(L.RES / "cem_ablations.json", "w"), indent=1)


if __name__ == "__main__":
    {"run": run, "final": final, "report": report, "compare": compare, "ablate": ablate}[sys.argv[1]]()
