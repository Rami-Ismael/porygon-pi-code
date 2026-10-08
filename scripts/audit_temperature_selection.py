"""Battle-free diagnostics on historical pools; not a win-rate experiment."""
import json
from pathlib import Path
import sys
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from boltzmann_selection import ARMS, selection_temperature, select_boltz
from temperature_experiment import atomic_json, digest, diversity


def main():
    anchor_file = REPO / "results/activesearch.json"
    baseline_file = REPO / "results/gradloop.json"
    anchor = json.loads(anchor_file.read_text())
    baseline = json.loads(baseline_file.read_text())["gens"]
    pastes = []
    for file in anchor["anchor"]:
        old = Path(file)
        local = REPO / "teams/reg_mb" / old.relative_to("/tmp/vgc-pilot/vgc-bench/teams/reg_mb")
        pastes.append(local.read_text())
    pastes += anchor["gens"]["gen0"]["pastes"]
    y = list(anchor["anchor"].values()) + anchor["gens"]["gen0"]["y"]
    rows = []
    for generation in range(1, 12):
        k = max(100, int(.25 * len(y)))
        for arm in ARMS:
            temperature, stats = selection_temperature(arm, generation, y)
            selected = []
            for seed in (101, 202, 303):
                indices, weights = select_boltz(y, k, temperature, np.random.default_rng(seed + generation))
                selected.append(diversity([pastes[i] for i in indices]))
            rows.append(dict(generation=generation, arm=arm, pool_size=len(y), draws=k,
                             temperature=temperature, concentration=stats,
                             weighted_measured_score=float(np.dot(weights, y)),
                             selected_diversity=selected))
        previous = baseline.get(f"combined_g{generation}")
        if previous is None:
            break
        pastes += previous["pastes"]
        y += previous["y"]
    output = REPO / "results/temperature_selection_audit.json"
    atomic_json(output, dict(kind="offline_same_pool_selection_diagnostic",
        limitation="No generator retraining or fresh battles; cannot estimate temperature's effect on win rate.",
        inputs={str(f): digest(f) for f in (anchor_file, baseline_file)}, rows=rows))
    lines = ["# Historical-pool selection diagnostics", "",
             "Every arm sees the same historical combined-loop pool. These are weight/selection calculations, "
             "not new generated-team or battle results. Higher weighted historical score follows from the "
             "selection rule and does not establish better fresh win rate.", "",
             "| Generation | Arm | T | ESS / pool | Max weight | Effective selected species sets (3-seed mean) |",
             "|---:|---|---:|---:|---:|---:|"]
    for row in rows:
        if row["generation"] not in (1, 6, 11):
            continue
        stats = row["concentration"]
        lines.append(f"| {row['generation']} | {row['arm']} | {row['temperature']:.3f} | "
                     f"{stats['ess_fraction']:.3f} | {stats['max_weight']:.4f} | "
                     f"{np.mean([v['effective_species_sets'] for v in row['selected_diversity']]):.1f} |")
    lines += ["", "Adaptive target: ESS/pool = 0.50, clamped to T ∈ [0.10, 0.30]. "
              "The detailed JSON records boundary saturation and all generations. Selection diversity is "
              "distinct from the diversity of the generator after retraining."]
    output.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
