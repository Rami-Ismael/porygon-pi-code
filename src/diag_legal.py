"""Per-field legality of UNCONSTRAINED samples under each legality condition —
the test of whether the legality token learned anything at all."""
import sys, numpy as np, torch
sys.path.insert(0, "/tmp/vgc-pilot/src")
import legaldiffusion as LD, spreaddiffusion as S
from encode import NSLOT
m, V, L, corpus, teams = LD.load_model(); C = S.SpreadConstraints(V, L); torch.manual_seed(1)
def diag(leg, g_leg, n=64):
    X = LD.sample(m, C, n, wbin=5, g_wr=2.0, leg=leg, g_leg=g_leg, constrained=False).cpu().numpy()
    mv_ok = ab_ok = mv_tot = ab_tot = 0; sp_dup = it_dup = bud = 0; full = 0
    for row in X:
        g = row.reshape(NSLOT, S.NF2); ok_team = True
        bases = [C.base_of[int(s[0])] for s in g]; items = [int(s[2]) for s in g if int(s[2]) > 0]
        sp_dup += len(bases) != len(set(bases)); it_dup += len(items) != len(set(items))
        ok_team &= len(bases) == len(set(bases)) and len(items) == len(set(items))
        for s in g:
            si = int(s[0])
            if si in C.mv_ok:
                for j in range(3, 7):
                    v = int(s[j]); mv_tot += 1; o = bool(C.mv_ok[si][v]) if v > 0 else False; mv_ok += o; ok_team &= o
            if si in C.ab_ok:
                ab_tot += 1; o = bool(C.ab_ok[si][int(s[1])]) if int(s[1]) > 0 else False; ab_ok += o; ok_team &= o
            b = np.clip(s[8:] - 1, 0, None).sum() > S.BUDGET; bud += b; ok_team &= not b
        full += ok_team
    return mv_ok/max(mv_tot,1), ab_ok/max(ab_tot,1), sp_dup/n, it_dup/n, bud/(n*6), full/n
print(f"{'condition':24s} {'move legal':>11s} {'ability ok':>11s} {'species dup':>12s} {'item dup':>9s} {'over-budget':>12s} {'whole team':>11s}")
for name, leg, gl in [("no legality token", LD.LEG_NULL, 0.0), ("legal=1, g=2", LD.LEG_LEGAL, 2.0), ("legal=1, g=4", LD.LEG_LEGAL, 4.0), ("legal=0 (illegal), g=2", LD.LEG_ILLEGAL, 2.0)]:
    r = diag(leg, gl); print(f"{name:24s} {r[0]:11.3f} {r[1]:11.3f} {r[2]:12.3f} {r[3]:9.3f} {r[4]:12.3f} {r[5]:11.3f}")
