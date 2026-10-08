"""Recount saved proposals and revalidate legality for the full direction study."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from corpus import parse_team, norm


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(a, b, label):
    assert (a is None and b is None) or (a is not None and b is not None and
            math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-8)), (label, a, b)


def from_paste(text):
    names = []
    for block in re.split(r"\n\s*\n", text.strip()):
        header = block.strip().splitlines()[0].split(" @ ", 1)[0].strip()
        header = re.sub(r"\s*\((M|F)\)$", "", header)
        nickname = re.match(r"^.*\((.+)\)$", header)
        names.append(norm(nickname[1] if nickname else header))
    assert len(names) == 6 and all(names), "a proposal must contain six named slots"
    return tuple(sorted(names))


def check_metrics(combinations, corpus, saved):
    counts = Counter(combinations)
    n = len(combinations)
    assert saved["n"] == n and saved["distinct_species_sets"] == len(counts)
    assert saved["novel_distinct_species_sets"] == len(set(counts) - corpus)
    if not n:
        assert saved["effective_species_sets"] == 0
        for key in ("species_set_entropy", "corpus_combination_rate", "mean_pairwise_jaccard",
                    "mean_nearest_corpus_jaccard", "rarefied_distinct_128"):
            assert saved[key] is None, key
        return
    entropy = -sum(c * math.log(c / n) for c in counts.values()) / n
    close(saved["species_set_entropy"], entropy, "entropy")
    close(saved["effective_species_sets"], math.exp(entropy), "effective sets")
    close(saved["corpus_combination_rate"], sum(counts[s] for s in corpus) / n, "corpus rate")
    unique = list(counts)
    pair_sum = 0.
    nearest_sum = 0.
    for i, a in enumerate(unique):
        sa = set(a)
        nearest_sum += counts[a] * min(1 - len(sa & set(b)) / len(sa | set(b)) for b in corpus)
        for b in unique[i + 1:]:
            distance = 1 - len(sa & set(b)) / len(sa | set(b))
            pair_sum += counts[a] * counts[b] * distance
    close(saved["mean_pairwise_jaccard"], pair_sum * 2 / (n * (n - 1)) if n > 1 else None, "pairwise distance")
    close(saved["mean_nearest_corpus_jaccard"], nearest_sum / n, "nearest corpus")
    rare = None
    if n >= 128:
        rare = 0.
        for c in counts.values():
            unseen = math.prod((n - c - i) / (n - i) for i in range(128)) if n - c >= 128 else 0.
            rare += 1 - unseen
    close(saved["rarefied_distinct_128"], rare, "rarefaction")


def audit(path, node):
    snapshot = path.read_bytes()
    data = json.loads(snapshot)
    m, runs = data["manifest"], data["runs"]
    assert data["status"] == "complete", "experiment has not completed"
    expected = dict(protocol="reverse-score-diversity-v1", attempts=512,
                    seeds=[1101, 1202, 1303], magnitude=108., chunk=48,
                    arms=["normal", "none", "reverse"], guidance="gloss",
                    decoding_temperature=1., surrogate_labels=328, corpus_teams=692,
                    training_steps=0, measured_battles=0)
    for k, v in expected.items():
        assert m[k] == v, ("protocol", k)
    c = data["controls"]
    assert c["status"] == "passed" and c["maximum_initial_logit_tilt"] > 0
    assert c["control_samples"] == 4 and c["control_seed"] not in m["seeds"]
    assert all(c[k] is True for k in ("normal_matches_original", "zero_matches_original", "reversal_is_exact"))
    assert set(runs) == {f"seed{s}/{a}" for s in m["seeds"] for a in m["arms"]}
    for p, expected_hash in m["inputs"].items():
        assert sha(p) == expected_hash, ("changed input", p)
    assert subprocess.check_output([str(node), "--version"], text=True).strip() == m["node"]
    metadata_path = ROOT / "results/temperature_p0.training.json"
    assert str(metadata_path) in m["inputs"]
    meta = json.loads(metadata_path.read_text())
    assert sha(m["checkpoint"]) == meta["checkpoint_sha256"]
    corpus = set()
    corpus_size = 0
    for p in meta["corpus"]:
        team = parse_team(p)
        if len(team) == 6:
            corpus_size += 1
            corpus.add(tuple(sorted(norm(slot["species"]) for slot in team)))
    assert corpus_size == 692
    validator_file = next(Path(p) for p in m["inputs"] if Path(p).name == "validate-teams-batch.js")
    validator = subprocess.Popen([str(node), validator_file.name], cwd=validator_file.parent,
                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    attempts = legal = 0
    try:
        for key, cell in runs.items():
            seed, arm = cell["seed"], cell["arm"]
            assert key == f"seed{seed}/{arm}"
            assert cell["strength"] == {"normal": 108., "none": 0., "reverse": -108.}[arm]
            rows = cell["records"]
            assert len(rows) == 512 and [r["index"] for r in rows] == list(range(512))
            raw_species, legal_species = [], []
            for row in rows:
                names = from_paste(row["paste"])
                assert names == tuple(row["species"])
                assert isinstance(row["valid"], bool) and row["valid"] == (row["validation_error"] is None)
                assert math.isfinite(row["surrogate_score"])
                validator.stdin.write(json.dumps({"format": "gen9championsvgc2026regmb", "team": row["paste"]}) + "\n")
                validator.stdin.flush()
                result = json.loads(validator.stdout.readline())
                assert result["valid"] == row["valid"], (key, row["index"], "legality changed")
                raw_species.append(names)
                if row["valid"]:
                    legal_species.append(names)
            check_metrics(raw_species, corpus, cell["raw"])
            check_metrics(legal_species, corpus, cell["legal"])
            close(cell["legality_rate"], len(legal_species) / 512, "legality rate")
            close(cell["mean_surrogate_score"], sum(r["surrogate_score"] for r in rows) / 512,
                  "raw surrogate mean")
            scores = [r["surrogate_score"] for r in rows if r["valid"]]
            close(cell["mean_legal_surrogate_score"], sum(scores) / len(scores) if scores else None,
                  "legal surrogate mean")
            attempts += len(rows)
            legal += len(legal_species)
            print(f"Verified {key}", flush=True)
    finally:
        validator.stdin.close()
        validator.wait()
    assert validator.returncode == 0 and attempts == 4608
    summary = json.loads(path.with_suffix(".analysis.json").read_text())
    result_hash = hashlib.sha256(snapshot).hexdigest()
    assert summary["raw_result_sha256"] == result_hash
    assert summary["analysis_source_sha256"] == sha(ROOT / "scripts/reverse_score_diversity.py")
    assert summary["measured_battles"] == 0
    for arm in m["arms"]:
        cells = [runs[f"seed{s}/{arm}"] for s in m["seeds"]]
        for metric in ("distinct_species_sets", "effective_species_sets", "novel_distinct_species_sets",
                       "mean_nearest_corpus_jaccard", "rarefied_distinct_128"):
            values = [v["legal"][metric] for v in cells]
            close(summary["means"][arm][metric], sum(values) / 3 if all(v is not None for v in values) else None,
                  "summary " + metric)
        for metric in ("legality_rate", "mean_legal_surrogate_score"):
            values = [v[metric] for v in cells]
            close(summary["means"][arm][metric], sum(values) / 3 if all(v is not None for v in values) else None,
                  "summary " + metric)
    # Invert the df=2 t CDF independently of the producer's closed-form quantile.
    lo, hi = 0., 100.
    for _ in range(80):
        mid = (lo + hi) / 2
        if .5 + mid / (2 * math.sqrt(mid * mid + 2)) < .9875:
            lo = mid
        else:
            hi = mid
    critical = (lo + hi) / 2
    for arm in ("normal", "none"):
        diffs = [runs[f"seed{s}/reverse"]["legal"]["distinct_species_sets"] - runs[f"seed{s}/{arm}"]["legal"]["distinct_species_sets"] for s in m["seeds"]]
        assert summary["reverse_minus_comparator"][arm]["differences"] == diffs
        contrast = summary["reverse_minus_comparator"][arm]
        mean = sum(diffs) / 3
        close(contrast["mean"], mean, "paired mean")
        se = math.sqrt(sum((v - mean) ** 2 for v in diffs) / 2 / 3)
        for actual, expected in zip(contrast["ci95_two_comparisons"], (mean - critical * se, mean + critical * se)):
            close(actual, expected, "adjusted paired interval")
    return dict(status="passed", cells=9, attempts=attempts, legal_attempts=legal,
                measured_battles=0, all_proposals_revalidated=True,
                input_files_checked=len(m["inputs"]), result_sha256=result_hash,
                audit_source_sha256=sha(__file__), metrics_recomputed=True,
                verified_at=datetime.now(timezone.utc).isoformat())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--node", type=Path, default=Path('/Users/ramiismael/.nvm/versions/node/v22.22.0/bin/node'))
    args = parser.parse_args()
    receipt = audit(args.results.resolve(), args.node)
    args.results.with_suffix(".verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
