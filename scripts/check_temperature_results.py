"""Audit a completed campaign's persisted budgets, isolation, and matched cells."""
import argparse
import json
from pathlib import Path


def check(data):
    assert data["status"] == "complete", "campaign is not complete"
    manifest = data["manifest"]
    c = manifest["config"]
    expected_runs = {f"seed{s}/{a}" for s in c["seeds"] for a in manifest["arms"]}
    assert set(data["runs"]) == expected_runs
    total = 0
    for seed in c["seeds"]:
        reference = data["runs"][f"seed{seed}/{manifest['arms'][0]}"]
        for arm in manifest["arms"]:
            rec = data["runs"][f"seed{seed}/{arm}"]
            assert set(rec["generations"]) == {str(g) for g in range(1, c["generations"] + 1)}
            search_seeds = set()
            for name, cell in rec["generations"].items():
                assert len(cell["proposal"]["pastes"]) == c["propose"]
                assert len(cell["scores"]) == c["battle"]
                assert cell["training"]["beta"] == 0
                assert cell["training"]["refit_steps"] == reference["generations"][name]["training"]["refit_steps"]
                assert cell["schedule"] == reference["generations"][name]["schedule"]
                assert cell["battle_seed"] == reference["generations"][name]["battle_seed"]
                search_seeds.add(cell["battle_seed"])
                for score in cell["scores"]:
                    assert score["battles"] == c["battles"]
                    assert score["win_rate"] == score["wins"] / score["battles"]
                    total += score["battles"]
            for phase, n in (("holdout", c["final_sample"]), ("final", c["finalists"])):
                assert len(rec[phase]["scores"]) == n
                assert rec[phase]["battle_seed"] not in search_seeds
                assert rec[phase]["schedule"] == reference[phase]["schedule"]
                assert rec[phase]["battle_seed"] == reference[phase]["battle_seed"]
                for score in rec[phase]["scores"]:
                    assert score["battles"] == c["final_battles"]
                    assert score["win_rate"] == score["wins"] / score["battles"]
                    total += score["battles"]
            assert rec["holdout"]["battle_seed"] != rec["final"]["battle_seed"]
        # These arms have identical initial pool, T, selection seed, model, and proposal seed.
        left = data["runs"][f"seed{seed}/annealed"]["generations"]["1"]
        right = data["runs"][f"seed{seed}/fixed_030"]["generations"]["1"]
        assert left["training"] == {**right["training"], "concentration": left["training"]["concentration"]}
        assert left["proposal"]["pastes"] == right["proposal"]["pastes"], "paired equal-T proposal mismatch"
    expected = len(expected_runs) * (c["generations"] * c["battle"] * c["battles"]
                                    + (c["final_sample"] + c["finalists"]) * c["final_battles"])
    assert total == expected
    return dict(status="passed", completed_runs=len(expected_runs), measured_battles=total,
                paired_equal_temperature_proposals="identical")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    receipt = check(json.loads(args.results.read_text()))
    args.results.with_suffix(".verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
