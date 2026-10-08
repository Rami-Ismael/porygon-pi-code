"""Seed-level estimates and diversity trajectories for a completed comparison.

Three-seed paired t intervals are exploratory (normality cannot be checked with
three observations). Four-comparator intervals use a Bonferroni adjustment.
"""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics


def paired_estimate(values):
    n = len(values)
    mean = statistics.mean(values)
    out = dict(n=n, differences=values, mean=mean)
    if n > 1:
        se = statistics.stdev(values) / math.sqrt(n)
        out["standard_error_across_seeds"] = se
        if n == 3:
            # Exact df=2 inverse t CDF: F(t)=1/2+t/(2*sqrt(t*t+2)).
            for label, probability in (("ci95", .975), ("ci95_four_comparisons", 1 - .05 / 8)):
                q = 2 * probability - 1
                critical = math.sqrt(2) * q / math.sqrt(1 - q * q)
                out[label] = [mean - critical * se, mean + critical * se]
        # Exact paired sign-flip test under a symmetric zero-effect null.
        if n <= 16:
            observed = abs(sum(values))
            sums = [abs(sum(sign * v for sign, v in zip(signs, values)))
                    for signs in itertools.product((-1, 1), repeat=n)]
            out["two_sided_sign_flip_p"] = sum(s >= observed - 1e-12 for s in sums) / len(sums)
    return out


def analyze(data):
    if data["status"] != "complete":
        raise ValueError("final analysis requires every arm and seed to complete")
    c = data["manifest"]["config"]
    seeds, arms = c["seeds"], data["manifest"]["arms"]
    rows = {}
    for key, rec in data["runs"].items():
        hd = rec["holdout"]["proposal"]["diversity"]
        final_generation = rec["generations"][str(c["generations"])]
        pd = final_generation["proposal"]["diversity"]
        # Historical entropyloop.py guardrails, retained as diagnostics. They
        # never stop this matched-budget study. The 200-set rule was defined
        # for 512 draws, so do not apply it to a tiny plumbing run.
        guardrail_generations = []
        for generation, cell in rec["generations"].items():
            proposal = cell["proposal"]["diversity"]
            if (proposal["memorisation"]["copy_rate"] > .10 or
                (c["propose"] == 512 and proposal["distinct_species_sets"] < 200)):
                guardrail_generations.append(int(generation))
        guardrail_generations.sort()
        rows[key] = dict(seed=rec["seed"], arm=rec["arm"],
            generator_win_rate=statistics.mean(s["win_rate"] for s in rec["holdout"]["scores"]),
            finalist_win_rate=statistics.mean(s["win_rate"] for s in rec["final"]["scores"]),
            search_rank1_fresh_win_rate=rec["final"]["scores"][0]["win_rate"],
            final_proposal_distinct_sets=pd["distinct_species_sets"],
            final_proposal_effective_sets=pd["effective_species_sets"],
            final_proposal_copy_rate=pd["memorisation"]["copy_rate"],
            holdout_effective_sets=hd["effective_species_sets"],
            holdout_species_distance=hd["mean_pairwise_species_jaccard_distance"],
            holdout_copy_rate=hd["memorisation"]["copy_rate"],
            diversity_guardrail_generations=guardrail_generations,
            distinct_set_guardrail_applicable=c["propose"] == 512)
    comparisons = {}
    for arm in arms:
        if arm == "annealed":
            continue
        comparisons[arm] = {}
        for metric in ("generator_win_rate", "finalist_win_rate", "final_proposal_effective_sets",
                       "holdout_effective_sets", "holdout_species_distance", "holdout_copy_rate"):
            differences = [rows[f"seed{s}/annealed"][metric] - rows[f"seed{s}/{arm}"][metric] for s in seeds]
            comparisons[arm][metric] = paired_estimate(differences)
    return dict(seed_results=rows, annealed_minus_comparator=comparisons,
                inference_unit="training seed; conditional on one shared rebuilt checkpoint and initial pool",
                interval_caveat="Three-seed paired t intervals require a normal-effects assumption. "
                "Four-comparator correction is within each metric, not simultaneous across all diversity metrics.")


def markdown(summary):
    rows = summary["seed_results"]
    lines = ["# Temperature comparison results", "", summary["inference_unit"] + ".", "",
             "All win rates below use fresh battles. Diversity uses equal-sized final proposal batches.", "",
             "## Means across training seeds", "",
             "| Arm | Generator win rate | Finalist mean | Distinct sets | Effective sets | Seeds crossing a guardrail |",
             "|---|---:|---:|---:|---:|---:|"]
    for arm in ("annealed", "fixed_010", "fixed_020", "fixed_030", "adaptive"):
        arm_rows = [r for r in rows.values() if r["arm"] == arm]
        means = {metric: statistics.mean(r[metric] for r in arm_rows) for metric in (
            "generator_win_rate", "finalist_win_rate", "final_proposal_distinct_sets",
            "final_proposal_effective_sets")}
        failures = sum(bool(r["diversity_guardrail_generations"]) for r in arm_rows)
        lines.append(f"| {arm} | {means['generator_win_rate']:.2%} | {means['finalist_win_rate']:.2%} | "
                     f"{means['final_proposal_distinct_sets']:.1f} | {means['final_proposal_effective_sets']:.1f} | "
                     f"{failures}/{len(arm_rows)} |")
    lines += ["", "## Individual seeds", "",
             "| Seed | Arm | Generator win rate | Finalist mean | Distinct sets | Effective sets | Copy rate |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for key in sorted(rows):
        r = rows[key]
        lines.append(f"| {r['seed']} | {r['arm']} | {r['generator_win_rate']:.2%} | "
                     f"{r['finalist_win_rate']:.2%} | {r['final_proposal_distinct_sets']} | "
                     f"{r['final_proposal_effective_sets']:.1f} | {r['final_proposal_copy_rate']:.1%} |")
    lines += ["", "## Paired primary endpoint", "",
              "Annealed minus comparator in final-generator mean win rate, in percentage points. "
              "Intervals adjust for four comparators.", "",
              "| Comparator | Difference (pp) | Adjusted 95% paired t interval (pp) | Seed differences (pp) |",
              "|---|---:|---|---|"]
    for arm, metrics in summary["annealed_minus_comparator"].items():
        r = metrics["generator_win_rate"]
        interval = r.get("ci95_four_comparisons")
        text = "not computed" if interval is None else f"[{interval[0] * 100:+.2f}, {interval[1] * 100:+.2f}]"
        lines.append(f"| {arm} | {r['mean'] * 100:+.2f} | {text} | "
                     + ", ".join(f"{v * 100:+.2f}" for v in r["differences"]) + " |")
    lines += ["", summary["interval_caveat"], "",
              "## Existing diversity guardrails", "",
              "The original loop flagged 48-field corpus copy rate >0.10 or fewer than 200 distinct species "
              "sets in 512 proposals. This study records those failures while completing the same budget "
              "for every arm. The 200-set rule is not applied to smaller smoke-test batches.", "",
              "| Seed | Arm | Generations crossing either guardrail |",
              "|---|---|---|"]
    for key in sorted(rows):
        r = rows[key]
        hits = ", ".join(map(str, r["diversity_guardrail_generations"])) or "none"
        lines.append(f"| {r['seed']} | {r['arm']} | {hits} |")
    lines += ["",
              "Diversity differences and individual seed effects are in the companion JSON and trajectory plot. "
              "These absolute guardrails are separate from noninferiority to a fixed-temperature arm: "
              "no relative noninferiority margin was specified, so that comparison remains descriptive. "
              "A positive point estimate alone does not establish improvement. "
              "The original search-rank-1 finalist's fresh score is retained without picking a winner "
              "from the fresh validation results. These results do not test generalization to another battle policy "
              "or a different opponent population."]
    return "\n".join(lines) + "\n"


def plot(data, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c = data["manifest"]["config"]
    colors = dict(zip(data["manifest"]["arms"], ("#173c68", "#bd4c3c", "#c1892b", "#59864d", "#80539c")))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    metrics = (("Search batch win rate", lambda x: statistics.mean(r["win_rate"] for r in x["scores"])),
               ("Effective species sets / proposal batch", lambda x: x["proposal"]["diversity"]["effective_species_sets"]),
               ("Selection temperature", lambda x: x["training"]["temperature"]),
               ("Final generator: fresh win rate", None))
    for ax, (title, metric) in zip(axes.flat, metrics):
        if metric:
            for arm, color in colors.items():
                trajectories = [[metric(data["runs"][f"seed{s}/{arm}"]["generations"][str(g)])
                                  for g in range(1, c["generations"] + 1)] for s in c["seeds"]]
                for vals in trajectories:
                    ax.plot(range(1, len(vals) + 1), vals, color=color, alpha=.20, linewidth=1)
                means = [statistics.mean(v) for v in zip(*trajectories)]
                ax.plot(range(1, len(means) + 1), means, color=color, label=arm, linewidth=2)
            ax.set_xlabel("Generation")
        else:
            for i, (arm, color) in enumerate(colors.items()):
                vals = [statistics.mean(r["win_rate"] for r in data["runs"][f"seed{s}/{arm}"]["holdout"]["scores"])
                        for s in c["seeds"]]
                ax.scatter([i] * len(vals), vals, color=color, s=40)
                ax.plot([i - .2, i + .2], [statistics.mean(vals)] * 2, color=color, linewidth=2)
            ax.set_xticks(range(len(colors)), colors, rotation=20)
        ax.set_title(title)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.15)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Boltzmann selection temperature: matched retraining experiment\nThin lines / dots: individual seeds; thick lines: means", fontsize=13)
    fig.savefig(output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    data = json.loads(args.results.read_text())
    summary = analyze(data)
    summary["raw_result_sha256"] = hashlib.sha256(args.results.read_bytes()).hexdigest()
    summary["analysis_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.results.with_suffix(".analysis.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    args.results.with_suffix(".analysis.md").write_text(markdown(summary))
    plot(data, args.results.with_suffix(".png"))
    print(markdown(summary))
