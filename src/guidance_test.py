"""Does classifier-free guidance actually steer generation toward a playstyle?

This is property (2) of the research goal, and it is the half no earlier experiment
touched. The test is direct: condition on a style, sample, and measure how often the
sampled team actually satisfies that style's rule. The rule is the same one that
produced the training labels, so adherence is objectively checkable.

Reported against two baselines:
  base rate   - how often the style appears in the corpus (what you'd get by luck)
  guidance 1  - conditioning with no amplification
Higher guidance should trade adherence against sample quality; that trade is the
whole point of the guidance weight, so legality is reported alongside.
"""
import json
import numpy as np, torch
from corpus import load_corpus, norm
from encode import Vocab, Legality, NF, NSLOT
import diffusion as D

def team_from_row(V, row):
    """Reconstruct the minimal structure the style rules need."""
    slots = []
    for i in range(NSLOT):
        b = i * NF
        g = lambda j: V.decode_field(b + j, int(row[b + j]))
        slots.append({"species": g(0), "ability": g(1), "item": g(2),
                      "moves": [g(3+j) for j in range(4) if g(3+j) and g(3+j) != "[MASK]"],
                      "nature": g(7), "evs": {}})
    return slots

def main():
    teams, V, L, X, Y = D.load_all()
    model = D._load(V); C = D.Constraints(V, L)
    n = 48
    base = {s: float(np.mean([D.style_of(t) == i for t in teams]))
            for i, s in enumerate(D.STYLES)}
    print(f"n={n} per cell | corpus base rates: "
          + ", ".join(f"{k} {v:.2f}" for k, v in base.items() if k != "none"))
    print(f"\n{'style':11s} {'guid':>5s} {'adherence':>10s} {'base':>6s} {'lift':>6s} {'legal':>7s}")
    out = {}
    for style in ["trickroom", "tailwind", "rain", "sand"]:
        for g in [1.0, 2.0, 4.0]:
            x = D.sample_constrained(model, C, n, style=style, guidance=g, temp=1.0).cpu().numpy()
            hit = leg = 0
            for r in x:
                t = team_from_row(V, r)
                if D.RULES[style](t): hit += 1
                if L.legal([V.decode_field(c, int(r[c])) for c in range(D.COLS)]): leg += 1
            a = hit / n; b = base[style]
            print(f"{style:11s} {g:5.1f} {a:10.3f} {b:6.3f} {a/max(b,1e-9):6.2f}x {leg/n:7.3f}")
            out[f"{style}@{g}"] = dict(adherence=a, base=b, lift=a/max(b,1e-9), legal=leg/n)
    # unconditional control: what does the model produce with no style at all?
    x = D.sample_constrained(model, C, n, style="none", guidance=1.0).cpu().numpy()
    for style in ["trickroom", "tailwind", "rain", "sand"]:
        hit = sum(1 for r in x if D.RULES[style](team_from_row(V, r)))
        out[f"uncond_{style}"] = hit / n
        print(f"{'(uncond)':11s} {'-':>5s} {hit/n:10.3f} {base[style]:6.3f} "
              f"{(hit/n)/max(base[style],1e-9):6.2f}x")
    json.dump(out, open("/tmp/vgc-pilot/guidance.json", "w"), indent=2)
    print("\nwrote /tmp/vgc-pilot/guidance.json")

if __name__ == "__main__":
    main()
