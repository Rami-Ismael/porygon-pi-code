"""Plot the frozen matched-budget comparison without changing the running driver."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("directory", type=Path)
    args = ap.parse_args()
    root = args.directory.resolve()
    proto = json.loads((root / "protocol.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    data = {arm: json.loads((root / f"{arm}.json").read_text()) for arm in ("cem", "dibo")}
    colors = dict(cem="#7c4d94", dibo="#087e8b")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "savefig.facecolor": "white"})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained")
    for arm, out in data.items():
        rounds = [r for _, r in sorted(out["rounds"].items(), key=lambda kv: int(kv[0])) if "evaluated" in r]
        battles = np.cumsum([r["physical_battles"] for r in rounds])
        means = np.array([r["mean"] for r in rounds])
        errors = np.array([r["se"] for r in rounds]) * 1.96
        best = np.maximum.accumulate([r["best"] for r in rounds])
        axes[0].plot(battles, means, marker="o", label=arm.upper() if arm == "cem" else "DiBO", color=colors[arm])
        axes[0].fill_between(battles, means - errors, means + errors, color=colors[arm], alpha=.13)
        axes[1].plot(battles, best, marker="o", label=arm.upper() if arm == "cem" else "DiBO", color=colors[arm])
    axes[0].set_title("Mean of each selected battle batch")
    axes[1].set_title("Best new candidate's noisy search label")
    for ax in axes:
        ax.set_xlabel("Fresh search battles spent")
        ax.set_ylabel("Win rate against the top50 meta")
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=.2)
        ax.legend(frameon=False)
    fig.suptitle("DiBO versus CEM: equal starting labels and fresh-battle budgets")
    fig.savefig(root / "learning-curves.png", dpi=180)
    fig.savefig(root / "learning-curves.svg")
    plt.close(fig)
    lines = ["# Matched-budget comparison: plots and interpretation", "",
             "![Learning curves](learning-curves.png)", "",
             "The left band is ±1.96 standard errors over the selected teams in that round; it does not estimate variability across training seeds. The right plot shows noisy search labels, so selecting maxima can inflate apparent performance. Neither plot uses heldout outcomes.", ""]
    if summary["complete"]:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), layout="constrained")
        for ax, stage in zip(axes, ("confirmation", "heldout")):
            values = [summary[stage]["cem_first"], summary[stage]["dibo_first"]]
            ax.bar(["CEM", "DiBO"], values, color=[colors["cem"], colors["dibo"]], width=.55)
            for i, value in enumerate(values):
                ax.text(i, value + .025, f"{value:.1%}", ha="center")
            ax.set_title("Fresh top50 confirmation" if stage == "confirmation" else "Fresh heldout ranks 51–100")
            ax.set_ylim(0, 1)
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.set_ylabel("Win rate of first search-selected finalist")
        fig.suptitle(f"Same preselected finalist rule · {proto['config']['final_battles']} new battles per team per stage")
        fig.savefig(root / "final-comparison.png", dpi=180)
        fig.savefig(root / "final-comparison.svg")
        plt.close(fig)
        lines += ["![Independent evaluations](final-comparison.png)", "",
                  "These are the highest-search-score finalists, locked before independent evaluations. They are not picked by the highest heldout or confirmation result.", "",
                  f"Descriptive primary winner in this single matched run: **{summary['descriptive_primary_winner']}**.", "",
                  "| Independent stage | CEM finalist-set mean | DiBO finalist-set mean | DiBO − CEM | Team-cluster bootstrap 95% interval |",
                  "| --- | ---: | ---: | ---: | --- |"]
        for stage in ("confirmation", "heldout"):
            row = summary[stage]
            lo, hi = row["team_cluster_bootstrap_95"]
            lines.append(f"| {stage} | {row['cem_mean']:.1%} | {row['dibo_mean']:.1%} | {row['mean_difference']:+.1%} | [{lo:+.1%}, {hi:+.1%}] |")
        times = {arm: sum(r.get("model_and_proposal_seconds", 0) for r in out["rounds"].values()) for arm, out in data.items()}
        lines += ["", f"Recorded model/proposal wall time: CEM {times['cem']/60:.1f} minutes; DiBO {times['dibo']/60:.1f} minutes. These wall timings span an overnight run and omit interrupted training; active compute time was not monitored, so they are not a clean compute-cost comparison. Battle expenditure was matched.", "",
                  "One seed and the project's BC policy limit the conclusion. These results compare this categorical DiBO adaptation with a fixed-budget CEM repeat using the same initial observations. Historical CEM used different initial labels and stopped early, so its archived score is context only."]
        for checkpoint in (3,):
            lines += ["", f"## {checkpoint}-round checkpoint", "",
                      "| Method | Fresh battles | Mean in round 3 | Best new search label so far |", "| --- | ---: | ---: | ---: |"]
            for arm, out in data.items():
                rounds = [r for _, r in sorted(out["rounds"].items(), key=lambda kv: int(kv[0])) if "evaluated" in r][:checkpoint]
                if len(rounds) == checkpoint:
                    lines.append(f"| {arm} | {sum(r['physical_battles'] for r in rounds)} | {rounds[-1]['mean']:.1%} | {max(r['best'] for r in rounds):.1%} |")
    else:
        lines += ["Full run is still in progress. Final independent comparison is not available yet."]
    lines += ["", "[Budget ledger and raw report](report.md) · [Frozen protocol](protocol.json) · [Machine-readable summary](summary.json)"]
    (root / "analysis.md").write_text("\n".join(lines) + "\n")
    print(root / "analysis.md")


if __name__ == "__main__":
    main()
