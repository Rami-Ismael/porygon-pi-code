"""Read results/activesearch.json and answer the four questions the run was built for.

  1 acquisition  : does ranking by the surrogate beat labelling at random?
                   Generation 1 is exactly paired -- same model, same 512 proposals,
                   different 128 battled -- so this is a clean two-sample test.
  2 re-steer     : does refitting on the elites move the PROPOSAL, i.e. does the
                   random-acquisition arm's own mean climb across generations, and
                   does `active` pull away from `frozen`?
  3 level        : does anything reach the real-team bar this run measured?
  4 provenance   : is any gain generation, or is it copying? (copy rate, NN-Hamming)

Bootstrap intervals are over teams (the unit that was randomised), 20,000 resamples.
"""
import json, sys
import numpy as np

RESULTS = "/Users/ramiismael/Documents/code/vgc-team-generator-pilot/results/activesearch.json"
B = 20_000


def boot_diff(a, b, rng):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = np.array([rng.choice(a, len(a), True).mean() - rng.choice(b, len(b), True).mean()
                  for _ in range(B)])
    return a.mean() - b.mean(), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    out = json.load(open(RESULTS))
    G = out["gens"]
    cfg = out["config"]
    anchor = np.array(list(out["anchor"].values()), float)
    rng = np.random.default_rng(0)
    gens = cfg["gens"]

    print(f"budget: {cfg['anchor_n']} anchor + "
          f"{(1 + 3 * gens) * cfg['battle']} labelled proposals "
          f"× {cfg['battles']} battles = "
          f"{(cfg['anchor_n'] + (1 + 3 * gens) * cfg['battle']) * cfg['battles']:,} battles")
    print(f"real corpus teams (n={len(anchor)}): mean {anchor.mean():.4f} "
          f"median {np.median(anchor):.3f}  max {anchor.max():.3f}\n")

    print(f"{'arm-gen':12s} {'mean':>8s} {'se':>7s} {'p90':>7s} {'max':>7s} "
          f"{'≥.458':>7s} {'Σy':>7s} {'ρ':>7s} {'copy':>6s} {'NN':>5s} {'sets':>9s}")
    order = ["gen0"] + [f"{a}_g{g}" for a in ("active", "random", "frozen")
                        for g in range(1, gens + 1)]
    for tag in order:
        if tag not in G:
            continue
        s = G[tag]["stats"]; p = s.get("proposal") or {}
        r = s.get("surrogate_spearman")
        print(f"{tag:12s} {s['mean']:8.4f} {s['se'] or 0:7.4f} {s['p90']:7.3f} "
              f"{s['max']:7.3f} {s['frac_ge_real_median']:7.1%} {s['total']:7.1f} "
              f"{'    n/a' if r is None else f'{r:+7.3f}'} "
              f"{p.get('copy_rate', 0):6.1%} {p.get('nn_hamming', 0):5.1f} "
              f"{p.get('distinct_species_sets', 0):4d}/{s.get('n_proposed', 0):<4d}")

    def y(tag):
        return G[tag]["y"] if tag in G else []

    print("\n1 ACQUISITION — surrogate-ranked vs random, same model, same proposals")
    for g in range(1, gens + 1):
        a, b = y(f"active_g{g}"), y(f"random_g{g}")
        if not a or not b:
            continue
        d, lo, hi = boot_diff(a, b, rng)
        pa = (G[f"active_g{g}"]["stats"].get("proposal") or {}).get("nn_hamming")
        pb = (G[f"random_g{g}"]["stats"].get("proposal") or {}).get("nn_hamming")
        paired = "paired" if pa is not None and pa == pb else "unpaired (models diverged)"
        star = "" if lo <= 0 <= hi else "  <-- interval excludes 0"
        print(f"  g{g} {paired:28s} active − random = {d:+.4f} "
              f"[{lo:+.4f}, {hi:+.4f}]{star}")
    a = sum((y(f"active_g{g}") for g in range(1, gens + 1)), [])
    b = sum((y(f"random_g{g}") for g in range(1, gens + 1)), [])
    if a and b:
        d, lo, hi = boot_diff(a, b, rng)
        print(f"  all generations pooled              active − random = {d:+.4f} "
              f"[{lo:+.4f}, {hi:+.4f}]")

    print("\n2 RE-STEER — does the proposal itself move?")
    print("  random-acquisition arm is the clean read on p_g (its labels are an "
          "unbiased sample of the proposal):")
    base = y("gen0")
    for g in range(1, gens + 1):
        r = y(f"random_g{g}")
        if not r or not base:
            continue
        d, lo, hi = boot_diff(r, base, rng)
        print(f"    gen0 → random_g{g}: {np.mean(base):.4f} → {np.mean(r):.4f}  "
              f"({d:+.4f} [{lo:+.4f}, {hi:+.4f}])")
    a = sum((y(f"active_g{g}") for g in range(1, gens + 1)), [])
    f = sum((y(f"frozen_g{g}") for g in range(1, gens + 1)), [])
    if a and f:
        d, lo, hi = boot_diff(a, f, rng)
        print(f"  same acquisition, re-steer on vs off: active − frozen = {d:+.4f} "
              f"[{lo:+.4f}, {hi:+.4f}]")

    print("\n3 LEVEL — against the bar")
    allp = [(t, v) for t in order if t in G for v in G[t]["y"]]
    vals = np.array([v for _, v in allp], float)
    print(f"  {len(vals)} labelled proposals · best {vals.max():.3f} · "
          f"{(vals >= 0.458).sum()} ≥ real median 0.458 · "
          f"{(vals >= anchor.mean()).sum()} ≥ real mean {anchor.mean():.3f}")
    bestarm = max(order, key=lambda t: G[t]["stats"]["mean"] if t in G else -1)
    print(f"  best arm-generation: {bestarm} at {G[bestarm]['stats']['mean']:.4f} "
          f"vs real {anchor.mean():.4f}")
    print(f"  reference — loop.py 2026-08-27, same proposer family, PMI-surprise "
          f"(information-gain) acquisition: 0.165 / 0.172 / 0.153")

    best_tag, best_i, best_v = None, None, -1.0
    for t in order:
        if t not in G:
            continue
        for i, v in enumerate(G[t]["y"]):
            if v > best_v:
                best_tag, best_i, best_v = t, i, v
    if best_tag:
        print(f"\n  best single proposal: {best_v:.3f} from {best_tag}\n")
        for ln in G[best_tag]["pastes"][best_i].strip().splitlines():
            print(f"    {ln}")

    print("\n4 PROVENANCE — generation or copying?")
    print("  calibration 2026-08-26: held-out real teams NN-Hamming 21.0, "
          "uniform-random 44.7; copy-paste collapse threshold 0.540")
    for tag in order:
        if tag not in G:
            continue
        p = G[tag]["stats"].get("proposal") or {}
        if p.get("copy_rate") is None:
            continue
        print(f"    {tag:12s} copy {p['copy_rate']:.1%}  NN-Hamming {p['nn_hamming']:.1f}  "
              f"{p['distinct_species_sets']} distinct species sets of "
              f"{G[tag]['stats'].get('n_proposed')}")


if __name__ == "__main__":
    main()
