"""Entropy-bounded unmasking on the F1 + G2 mixture: strength and quality-diversity (2026-10-14).

The one sampler on the owner's list that sampler_battle.py (2026-10-11) did not battle. Entropy-bounded
unmasking (the EB-Sampler) reveals SEVERAL fields per model call: rank the masked fields by the entropy
of their guided, legality-masked distribution and reveal the largest group whose summed entropy stays
under a budget gamma (always at least one). It is the "parallel decoding" of the diffusion literature.
  eb_conf    paper form: lowest-entropy masked fields first, any position
  eb_order   today's fixed order (six species, then the sets), revealing the next fields in that order
             while their summed entropy stays under gamma
  gamma = 0.5 nats for both here. A team carries ~42 nats in all (median, clean mixture, asked 0.5), so
  0.5 is near-sequential (~47 passes per row); gamma 1 / 4 give ~45 / ~20 passes (eb_conf) and
  gamma 4 ~15 passes (eb_order). sampler_qd_run.py battles gamma 1 and 4. Fields revealed in one pass are committed one column at a time with the
  legality mask recomputed before each (sampler_battle.commit_set), so teams stay legal.
Protocol = sampler_battle.py, unchanged: asked {0.1 .. 0.9}, guidance 2, temperature 1, the first 64
Showdown-valid teams per cell, one battle per top-50 column, policy 1, origin 'sampler:<cell>'.
Win rate = wins / battles against the top-50 meta teams, behaviour-cloned policy both sides.

Decision rule, written before any battle: compared with today's sampler (sampler_battle.json) on
winning line-ups (distinct line-ups whose best team scores >= 0.531) and QD-score (sum over line-ups of
the best win rate), each with a jackknife standard error (sampler_qd.py). Beyond 2 standard errors is
a difference; inside it is "no detectable difference".

    python sampler_eb.py gen
    python battle_supervisor.py sampler_eb.py battle --then report
(run with /tmp/vgc-pilot/.venv/bin/python)
"""
import sys
from pathlib import Path
import torch

sys.path.insert(0, "/tmp/vgc-pilot/src")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lossdown as L
import sampler_battle as SB

OUT = L.RES / "sampler_eb.json"
ARMS = ["eb_conf", "eb_order"]
GAMMA = 0.5
DEV, COLS, NINF = SB.DEV, SB.COLS, SB.NINF
PASSES = {}                                     # model calls per batch, per arm, for the report


def entropy(x, lg):
    """Entropy of each field's guided distribution under the current legality mask -> [n,48]."""
    logp = torch.log_softmax(lg.masked_fill(~SB.mask_T(x), NINF), -1)
    p = logp.exp()
    return -torch.where(p > 0, p * logp, torch.zeros_like(p)).sum(-1)


def bounded(h_sorted, gamma=None):
    """h_sorted [n,m] ascending per row -> bool [n,m]: the longest prefix with sum <= gamma, at least one."""
    cum = h_sorted.cumsum(1)
    keep = cum <= (GAMMA if gamma is None else gamma)
    keep[:, 0] = True
    return keep


@torch.no_grad()
def arm_eb(n, by_order, gamma=None):
    x = torch.zeros(n, COLS, dtype=torch.long, device=DEV)
    od = torch.tensor(SB.ORDER, device=DEV)
    rank = torch.empty(COLS, dtype=torch.long, device=DEV); rank[od] = torch.arange(COLS, device=DEV)
    passes = 0
    while bool((x == 0).any()):
        masked = x == 0
        lg = SB.guided(x, SB.mfrac(x))
        h = entropy(x, lg).masked_fill(~masked, float("inf"))
        if by_order:                            # walk today's order; already revealed fields sort last
            key = torch.where(masked, rank.expand(n, -1).float(), torch.full_like(h, 1e9))
            idx = key.argsort(1)
        else:
            idx = h.argsort(1)
        hs = h.gather(1, idx)
        keep = bounded(torch.where(torch.isinf(hs), torch.full_like(hs, 1e9), hs), gamma)
        rev = torch.zeros(n, COLS, dtype=torch.bool, device=DEV)
        rev.scatter_(1, idx, keep)
        SB.commit_set(x, lg, rev & masked)
        passes += 1
    PASSES.setdefault(f"eb_{'order' if by_order else 'conf'} gamma {GAMMA if gamma is None else gamma}", []).append(passes)
    return x


def _route():
    SB.OUT, SB.ARMS, SB.SEED = OUT, ARMS, 6262
    SB.FN.update(eb_conf=lambda n: arm_eb(n, False), eb_order=lambda n: arm_eb(n, True))


if __name__ == "__main__":
    st = sys.argv[1]
    _route()
    if st == "gen":
        SB.gen("--smoke" in sys.argv)
        for a, v in PASSES.items(): print(f"{a}: model calls per batch mean {sum(v)/len(v):.1f}, min {min(v)}, max {max(v)} (today: 48)")
    elif st == "battle": SB.battle()
    elif st == "report": SB.report()
