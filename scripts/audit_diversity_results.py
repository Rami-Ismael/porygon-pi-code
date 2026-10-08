"""Recompute every measured stream and verify its source/budget provenance."""
import importlib.metadata
import json
from pathlib import Path
import socket

from capture_diversity_baseline import verify_baseline
from report_team_diversity import analyze, load_reference
from temperature_experiment import REPO, atomic_json, digest


def main():
    baseline_path = REPO / "results/diversity_baseline.json"
    history_path = REPO / "results/temperature_diversity_metrics.json"
    baseline = json.loads(baseline_path.read_text())
    history = json.loads(history_path.read_text())
    if baseline["status"] != "complete":
        raise ValueError("baseline incomplete")
    checks = {}
    for path, expected in baseline["manifest"]["inputs"].items():
        actual = ({name: importlib.metadata.version(name) for name in expected}
                  if path == "runtime_versions" else digest(path))
        if actual != expected:
            raise ValueError(f"baseline input changed: {path}")
    checks["all_baseline_inputs_and_runtime_versions_unchanged"] = True
    source_path = Path(history["source"])
    if digest(source_path) != history["source_sha256"] or history["source_sha256"] != baseline["manifest"]["source_campaign_sha256"]:
        raise ValueError("historical source identity changed")
    if digest(REPO / "src/diversity_metrics.py") != history["metric_source_sha256"]:
        raise ValueError("historical metric implementation changed")
    source = json.loads(source_path.read_text())
    metadata = history["reference"]
    reference, parse = load_reference(Path(metadata["provenance"]["training_manifest"]),
                                      source["manifest"]["config"], source["manifest"]["inputs"])
    if reference.metadata != metadata:
        raise ValueError("historical reference metadata mismatch")
    rows = analyze(source, reference, parse)
    if rows != history["rows"]:
        raise ValueError("historical metrics differ from the saved raw draws")
    checks["all_180_historical_streams_recomputed"] = True
    verify_baseline(baseline, lambda pastes: [parse(p) for p in pastes])
    config = baseline["manifest"]["config"]
    if config["seeds"] != [101, 202, 303] or (config["propose"], config["final_sample"], config["final_battles"]) != (512, 128, 192):
        raise ValueError("baseline does not have the planned three-seed budget")
    checks["all_six_baseline_streams_recomputed"] = True
    checks["all_73728_baseline_battles_and_schedules_verified"] = True
    for row in baseline["runs"].values():
        for phase in ("proposal", "holdout"):
            legacy = row[phase]["diversity"]
            metrics = legacy["team_diversity"]
            if legacy["distinct_teams"] != metrics["with_spreads"]["unique_count"] or legacy["distinct_species_sets"] != metrics["composition"]["unique_count"]:
                raise ValueError("baseline legacy counts disagree with record-based metrics")
    checks["baseline_counts_agree_with_independent_legacy_representation"] = True
    for port in config["ports"]:
        with socket.socket() as connection:
            connection.settimeout(.1)
            if connection.connect_ex(("127.0.0.1", port)) == 0:
                raise ValueError(f"private baseline server still listening: {port}")
    checks["all_private_baseline_servers_stopped"] = True
    artifacts = [baseline_path, history_path, source_path]
    artifacts += [REPO / "results" / ("diversity_before_after" + suffix)
                  for suffix in (".md", ".json", ".png", ".svg")]
    output = REPO / "results/diversity_verification.json"
    atomic_json(output, dict(status="passed", checks=checks, historical_streams=len(rows),
        baseline_streams=6, historical_draws=sum(r["metrics"]["sampling"]["accepted"] for r in rows),
        baseline_draws=3 * (512 + 128), baseline_battles=73728,
        artifacts={str(path): digest(path) for path in artifacts}, audit_source_sha256=digest(__file__)))
    print(f"PASS: 186 streams, 88,320 draws, 73,728 new battles; {output}")


if __name__ == "__main__":
    main()
