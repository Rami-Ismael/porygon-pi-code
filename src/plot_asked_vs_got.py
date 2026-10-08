"""Asked vs got figure for experiment A -> results/asked_vs_got.png (light) and _dark.png."""
import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RES = "/Users/ramiismael/Documents/code/vgc-team-generator-pilot/results/"
d = json.load(open(RES + "asked_vs_got.json"))
S = d["summary"]
REAL = 0.531          # top-50 meta teams vs the other 49, same matrix, same policy
UNC = S["uncond"]["got"]
TRAIN_MAX = d["training"]["max"]
THEMES = {
    "light": dict(bg="#ffffff", ink="#1f1f1e", ink2="#5f5e5a", grid="#e6e5e0",
                  ramp={1.0: "#86b6ef", 2.0: "#2a78d6", 4.0: "#104281"}, shade="#f3f2ee"),
    "dark": dict(bg="#1a1a19", ink="#ffffff", ink2="#c3c2b7", grid="#33332f",
                 ramp={1.0: "#3f6fa6", 2.0: "#3987e5", 4.0: "#b7d3f6"}, shade="#242422"),
}

def draw(theme, path):
    T = THEMES[theme]
    plt.rcParams.update({"font.family": "Helvetica Neue", "font.size": 11, "text.color": T["ink"],
                         "axes.labelcolor": T["ink2"], "xtick.color": T["ink2"], "ytick.color": T["ink2"]})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={"width_ratios": [1.6, 1]}, facecolor=T["bg"])
    for a in (ax, bx):
        a.set_facecolor(T["bg"])
        for s in a.spines.values(): s.set_visible(False)
        a.grid(axis="y", color=T["grid"], lw=0.8); a.set_axisbelow(True)
        a.axvspan(TRAIN_MAX, 0.95, color=T["shade"], zorder=0)
        a.set_xlim(-0.05, 0.95); a.set_xticks(np.arange(0, 1.0, 0.1))
        a.tick_params(length=0)
    ax.plot([0, 0.9], [0, 0.9], ls=(0, (4, 4)), color=T["ink2"], lw=1)
    ax.text(0.5, 0.55, "got = asked", color=T["ink2"], rotation=33, fontsize=10)
    ax.axhline(REAL, color=T["ink2"], lw=1, ls=":")
    ax.text(-0.04, REAL + 0.012, f"top-50 meta team, {REAL:.2f}", color=T["ink2"], fontsize=10)
    ax.axhline(UNC, color=T["ink2"], lw=1, ls=":")
    ax.text(0.42, UNC - 0.045, f"no win-rate condition, {UNC:.2f}", color=T["ink2"], fontsize=10)
    for g in (1.0, 2.0, 4.0):
        cs = sorted([s for s in S.values() if s["asked"] is not None and s["guidance"] == g], key=lambda s: s["asked"])
        x = [s["asked"] for s in cs]; y = [s["got"] for s in cs]; e = [1.96 * s["se"] for s in cs]
        c = T["ramp"][g]
        ax.fill_between(x, np.array(y) - e, np.array(y) + e, color=c, alpha=0.18, lw=0)
        ax.plot(x, y, color=c, lw=2, marker="o", ms=6, mec=T["bg"], mew=1.5, label=f"guidance {g:g}")
        bx.plot(x, [s["species_sets"] for s in cs], color=c, lw=2, marker="o", ms=6, mec=T["bg"], mew=1.5)
        i = int(np.argmax(y))
        if g == 4.0: ax.annotate(f"{y[i]:.2f}", (x[i], y[i]), xytext=(0, 9), textcoords="offset points", ha="center",
                    color=T["ink"], fontsize=10)
    ax.text(TRAIN_MAX + 0.01, 0.45, "above every\ntraining label", color=T["ink2"], fontsize=9, va="top")
    ax.set_ylim(0, 0.9); ax.set_xlabel("win rate asked for"); ax.set_ylabel("win rate got (49 battles per team, 64 teams per point)")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.93), labelcolor=T["ink"])
    ax.set_title("Asking for a higher win rate works up to ~0.5, then stops", loc="left", color=T["ink"], fontsize=13, pad=12)
    bx.set_ylim(0, 68); bx.set_xlabel("win rate asked for"); bx.set_ylabel("different species sets among 64 teams")
    bx.set_title("…and the teams it gets there with are the same few", loc="left", color=T["ink"], fontsize=13, pad=12)
    fig.text(0.01, 0.005, "Win rate = wins vs the top-50 meta teams (one battle per opponent), VGC-Bench behaviour-cloning policy both sides. "
             "Bands: 95% intervals over teams.", color=T["ink2"], fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=160, facecolor=T["bg"]); plt.close(fig)

draw("light", RES + "asked_vs_got.png")
draw("dark", RES + "asked_vs_got_dark.png")
print("ok")
