"""Verify and compare the no-fine-tuning baseline with saved campaign metrics."""
import argparse
import json
import math
from pathlib import Path
import statistics

from capture_diversity_baseline import verify_baseline
from temperature_experiment import REPO, digest


def compare(baseline, history):
    from corpus import parse_team_text
    if baseline.get("status") != "complete":
        raise ValueError("baseline is not complete")
    verify_baseline(baseline, lambda pastes: [parse_team_text(p) for p in pastes])
    manifest = baseline["manifest"]
    if history["source_sha256"] != manifest["source_campaign_sha256"]:
        raise ValueError("historical campaign identity mismatch")
    if digest(history["source"]) != history["source_sha256"]:
        raise ValueError("historical campaign file changed")
    if digest(REPO / "src/diversity_metrics.py") != history["metric_source_sha256"]:
        raise ValueError("historical report uses a different metric implementation")
    if history["reference"]["content_sha256"] != manifest["reference"]["metadata"]["content_sha256"]:
        raise ValueError("baseline and history use different novelty references")
    seeds = manifest["config"]["seeds"]
    original = json.loads(Path(history["source"]).read_text())
    arms = sorted(original["manifest"]["arms"])
    generations = manifest["config"]["generations"]
    expected = {(seed, arm, g, "generation") for seed in seeds for arm in arms for g in range(1, generations + 1)}
    expected |= {(seed, arm, generations, "holdout") for seed in seeds for arm in arms}
    keys = [(r["seed"], r["arm"], r["generation"], r["phase"]) for r in history["rows"]]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("historical cells incomplete or duplicated")
    for row in history["rows"]:
        expected_n = manifest["config"]["propose" if row["phase"] == "generation" else "final_sample"]
        if row["metrics"]["sampling"]["accepted"] != expected_n:
            raise ValueError("historical and baseline sample sizes differ")
    rows = []
    for seed in seeds:
        row = baseline["runs"][str(seed)]
        scores = row["scores"]
        rows.append(dict(arm="before fine-tuning", seed=seed,
            proposal=row["proposal"]["diversity"]["team_diversity"],
            holdout=row["holdout"]["diversity"]["team_diversity"],
            win_rate=sum(s["wins"] for s in scores) / sum(s["battles"] for s in scores)))
        for arm in arms:
            proposal = next(r for r in history["rows"] if (r["seed"], r["arm"], r["generation"], r["phase"]) == (seed, arm, generations, "generation"))
            holdout = next(r for r in history["rows"] if (r["seed"], r["arm"], r["phase"]) == (seed, arm, "holdout"))
            rows.append(dict(arm=arm, seed=seed, proposal=proposal["metrics"],
                             holdout=holdout["metrics"], win_rate=holdout["win_rate"]["mean"]))
    return rows


def plot(baseline, history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    config = baseline["manifest"]["config"]
    seeds = config["seeds"]
    arms = sorted({r["arm"] for r in history["rows"]})
    palette = dict(adaptive="#A05A00", annealed="#007A78", fixed_010="#A72D59",
                   fixed_020="#7250A1", fixed_030="#2965AF")
    labels = dict(adaptive="Adaptive", annealed="Annealed", fixed_010="Fixed 0.10",
                  fixed_020="Fixed 0.20", fixed_030="Fixed 0.30")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
    for ax, key, metric, scale in ((axes[0], "composition", "unique_count", 1),
                                    (axes[1], "with_spreads", "unique_fraction", 100)):
        start = statistics.mean(baseline["runs"][str(seed)]["proposal"]["diversity"]["team_diversity"][key][metric] for seed in seeds)
        for arm in arms:
            values = [start]
            for generation in range(1, config["generations"] + 1):
                values.append(statistics.mean(r["metrics"][key][metric] for r in history["rows"]
                    if r["arm"] == arm and r["generation"] == generation and r["phase"] == "generation"))
            ax.plot(range(len(values)), [v * scale for v in values], color=palette[arm],
                    label=labels[arm], linewidth=2, marker="o", markersize=3)
        ax.scatter([0], [start * scale], color="#222222", zorder=5, s=35)
        ax.set_xlabel("CEM generation (0 = before fine-tuning)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.18)
        ax.set_xticks([0, 1, 3, 5, 7, 9, 11])
    axes[0].set(title="Distinct six-species/form compositions", ylabel="Count per 512 accepted draws", ylim=(0, 530))
    axes[0].legend(fontsize=8, frameon=False, loc="lower left", ncol=2)
    axes[1].set(title="Unique teams including stat spreads", ylabel="Percent of 512 accepted draws", ylim=(0, 105))
    if all(r["metrics"]["with_spreads"]["unique_fraction"] == 1 for r in history["rows"] if r["phase"] == "generation"):
        axes[1].text(.5, .65, "All five arms remain at 100%", transform=axes[1].transAxes,
                     ha="center", color="#40516a", fontsize=11)
    fig.suptitle("Diversity before and across fine-tuning", fontsize=15, fontweight="bold")
    fig.supxlabel("Means across three seeds; repeated draws retained. No baseline generator fine-tuning.", fontsize=9)
    fig.savefig(path, dpi=180)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--baseline", type=Path, required=True)
    cli.add_argument("--history", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    baseline = json.loads(args.baseline.read_text())
    history = json.loads(args.history.read_text())
    rows = compare(baseline, history)
    config = baseline["manifest"]["config"]
    image = args.output.with_suffix(".png")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plot(baseline, history, image)
    lines = ["# Diversity before and across fine-tuning", "",
        "The original checkpoint was evaluated without generator fine-tuning, using the completed campaign's "
        "decoding/guidance settings, initial surrogate-label pool, opponents, battle policy and package versions. "
        "The baseline was captured retrospectively and is saved separately; its seeds and budgets are matched.", "",
        f"Each of {len(config['seeds'])} seeds supplies {config['propose']} accepted proposals and an independent "
        f"{config['final_sample']}-team unranked sample evaluated for {config['final_battles']} battles per team. "
        f"All {baseline['manifest']['expected_battles']:,} baseline battles completed. "
        "The initial training reference contains 692 teams verified against the checkpoint and source files.", "",
        f"![Diversity trajectories]({image.resolve()})", "",
        "| Model | Distinct compositions / 512 | Unique teams / 512 | Novel teams / 512 | Generator win rate |",
        "| --- | ---: | ---: | ---: | ---: |"]
    order = ["before fine-tuning"] + sorted({r["arm"] for r in rows if r["arm"] != "before fine-tuning"})
    for arm in order:
        selected = [r for r in rows if r["arm"] == arm]
        comp = statistics.mean(r["proposal"]["composition"]["unique_count"] for r in selected)
        unique = statistics.mean(r["proposal"]["with_spreads"]["unique_fraction"] for r in selected)
        novel = statistics.mean(r["proposal"]["with_spreads"]["novel_draw_fraction"] for r in selected)
        win = statistics.mean(r["win_rate"] for r in selected)
        lines.append(f"| {arm} | {comp:.1f} | {unique:.1%} | {novel:.1%} | {win:.2%} |")
    lines += ["", "## Where concentration occurs", "",
        "These additional measurements use the same 512-proposal streams. The effective composition count "
        "is exp(−Σ p ln p), where p is each composition's observed fraction of draws. It decreases when draws "
        "concentrate on a few rosters even if many rare rosters remain.", "",
        "| Model | Draws with unseen compositions | Largest roster share | Effective compositions |",
        "| --- | ---: | ---: | ---: |"]
    for arm in order:
        group = [r["proposal"]["composition"] for r in rows if r["arm"] == arm]
        novelty = statistics.mean(m["novel_draw_fraction"] for m in group)
        largest = statistics.mean(m["largest_share"] for m in group)
        effective = []
        for m in group:
            probabilities = [entry["count"] / m["n"] for entry in m["frequencies"]]
            effective.append(math.exp(-sum(p * math.log(p) for p in probabilities)))
        lines.append(f"| {arm} | {novelty:.1%} | {largest:.1%} | {statistics.mean(effective):.1f} |")
    lines += ["", "Values are means over the same three seeds. Post-fine-tuning diversity is measured at generation 11; "
        "win rates come from independent, unranked 128-team samples before and after fine-tuning. "
        "CEM also refits the guidance surrogate, so changes describe the whole proposal procedure rather than isolating generator training. "
        "These descriptive means do not establish statistical significance or new strategic behavior. "
        "The plot tracks composition counts because full-team uniqueness can remain high while roster variety declines.", "",
        f"[Baseline draws and battle counts]({args.baseline.resolve()}) · "
        f"[All historical metric records]({args.history.resolve()})", ""]
    args.output.write_text("\n".join(lines))
    args.output.with_suffix(".json").write_text(json.dumps(dict(
        baseline_sha256=digest(args.baseline), history_sha256=digest(args.history),
        report_script_sha256=digest(__file__), rows=rows), indent=2, allow_nan=False) + "\n")
    print(f"Verified baseline and wrote {args.output}")


if __name__ == "__main__":
    main()
