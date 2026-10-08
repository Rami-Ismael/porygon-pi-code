"""Random-fill baseline figure -> results/random_fill_baseline.png."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RES = "/Users/ramiismael/Documents/code/vgc-team-generator-pilot/results/"
S = json.load(open(RES + "random_fill_baseline.json"))["summary"]
ORIG = 0.450
C = {"pokemon": "#2a78d6", "item": "#eb6834", "stats": "#1baf7a", "alignment": "#eda100", "ability": "#e87ba4",
     "move": "#2a78d6", "mixed": "#eb6834"}
NAME = {"pokemon": "whole Pokémon", "item": "items", "stats": "Stat Points", "alignment": "Stat Alignment",
        "ability": "abilities", "move": "moves", "mixed": "mixed parts"}
INK, INK2, GRID = "#1f1f1e", "#5f5e5a", "#e6e5e0"
plt.rcParams.update({"font.family": "Helvetica Neue", "font.size": 11, "text.color": INK,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), gridspec_kw={"width_ratios": [1, 1.5]})
for ax, tasks, kmax in ((axes[0], ["pokemon", "item", "stats", "alignment", "ability"], 6), (axes[1], ["move", "mixed"], 40)):
    for s in ax.spines.values(): s.set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True); ax.tick_params(length=0)
    ax.axhline(ORIG, color=INK2, lw=1, ls=":")
    ax.text(1 if kmax == 6 else kmax, ORIG + (0.065 if kmax == 6 else 0.012), f"original teams, {ORIG:.2f}", color=INK2, fontsize=10, ha="left" if kmax == 6 else "right")
    for t in tasks:
        ks = sorted(int(k[len(t):]) for k in S if k.startswith(t) and k[len(t):].isdigit())
        y = np.array([S[f"{t}{k}"]["filled"] for k in ks])
        lo = np.array([ORIG + S[f"{t}{k}"]["ci"][0] for k in ks]); hi = np.array([ORIG + S[f"{t}{k}"]["ci"][1] for k in ks])
        ax.fill_between(ks, lo, hi, color=C[t], alpha=0.15, lw=0)
        ax.plot(ks, y, color=C[t], lw=2, marker="o" if kmax == 6 else None, ms=6, mec="white", mew=1.5)
        ax.text(ks[-1] + (0.12 if kmax == 6 else 0.6), y[-1] + {"ability": 0.014, "alignment": -0.02}.get(t, 0), NAME[t], color=INK, fontsize=10, va="center")
    ax.set_ylim(0, 0.55); ax.set_xlabel("k = parts left blank and filled at random")
    ax.set_xlim(0.7, kmax + (1.6 if kmax == 6 else 7))
axes[0].set_ylabel("win rate vs the top-50 meta teams")
axes[0].set_xticks(range(1, 7))
axes[0].set_title("Random Stat Points, abilities and alignments barely hurt", loc="left", fontsize=13, pad=12)
axes[1].set_title("Random moves and mixed parts cost steadily", loc="left", fontsize=13, pad=12)
fig.text(0.01, 0.005, "50 real VGCPastes teams outside the top 50; one random legal fill per team per k; 49 battles per team "
         "(one per top-50 opponent), behaviour-cloning policy both sides. Bands: 95% intervals over teams.", color=INK2, fontsize=9)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(RES + "random_fill_baseline.png", dpi=160, facecolor="white")
print("ok")
