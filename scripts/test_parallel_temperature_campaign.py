"""Test seed ownership, atomic snapshots, and final-source integrity in collection."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import parallel_temperature_campaign as campaign
from parallel_temperature_campaign import merge, seed_complete, PORTS

ARMS = ["annealed", "fixed_010", "fixed_020", "fixed_030", "adaptive"]


class MergeTest(unittest.TestCase):
    def test_waits_for_independent_writers_before_recording_final_hashes(self):
        base = dict(schema=1, arms=ARMS, opponents=["opponent"], inputs={"p0": "hash"},
                    config=dict(seeds=[101, 202, 303], ports=PORTS[101], epochs=40, battles=24))
        with tempfile.TemporaryDirectory() as directory:
            sources = {}
            for seed in (101, 202, 303):
                manifest = copy.deepcopy(base)
                manifest["config"].update(seeds=[seed], ports=PORTS[seed])
                raw = dict(status="running" if seed == 202 else "complete", manifest=manifest,
                           runs={f"seed{seed}/{arm}": {"final": {}} for arm in ARMS})
                path = Path(directory) / f"{seed}.json"
                path.write_text(json.dumps(raw))
                sources[seed] = (path, raw)
            self.assertEqual(merge(base, sources)["status"], "running",
                             "a writer still has its final status write pending")
            sources[202][1]["status"] = "complete"
            sources[202][0].write_text(json.dumps(sources[202][1]))
            self.assertEqual(merge(base, sources)["status"], "complete")

    def test_combines_owned_seeds_without_changing_controls(self):
        base = dict(schema=1, arms=ARMS, opponents=["opponent"], inputs={"p0": "hash"},
                    config=dict(seeds=[101, 202, 303], ports=PORTS[101], epochs=40, battles=24))
        with tempfile.TemporaryDirectory() as directory:
            sources = {}
            for seed in (101, 202, 303):
                manifest = copy.deepcopy(base)
                manifest["config"].update(seeds=[seed], ports=PORTS[seed])
                # Synthetic fixture: the collector checks ownership/control integrity;
                # the separate experiment verifier checks actual score budgets.
                raw = dict(status="complete", manifest=manifest, runs={f"seed{seed}/{arm}": {"final": {}, "marker": seed} for arm in ARMS})
                path = Path(directory) / f"{seed}.json"
                path.write_text(json.dumps(raw))
                sources[seed] = (path, raw)
            out = merge(base, sources)
            self.assertEqual(out["status"], "complete")
            self.assertEqual(len(out["runs"]), 15)
            self.assertEqual(out["manifest"]["config"]["epochs"], 40)
            self.assertEqual(out["manifest"]["execution"]["ports_by_seed"]["202"], PORTS[202])
            self.assertTrue(all(r["marker"] == int(k.split("/")[0][4:]) for k, r in out["runs"].items()))
            self.assertEqual(merge(base, {101: sources[101]})["status"], "running")
            self.assertIn("ports", base["config"], "collector mutated the original manifest")
            changed = copy.deepcopy(sources)
            changed[202][1]["manifest"]["config"]["epochs"] = 41
            with self.assertRaises(ValueError):
                merge(base, changed)
            changed = copy.deepcopy(sources)
            changed[303][1]["manifest"]["inputs"]["p0"] = "another-checkpoint"
            with self.assertRaises(ValueError):
                merge(base, changed)
            del sources[101][1]["runs"]["seed101/annealed"]["final"]
            self.assertFalse(seed_complete(sources[101][1], 101))
            self.assertEqual(merge(base, sources)["status"], "running")


class CollectorSnapshotTest(unittest.TestCase):
    def test_atomic_update_during_read_is_collected_on_next_iteration(self):
        class StopLoop(Exception):
            pass

        base = dict(schema=1, arms=ARMS, opponents=["opponent"], inputs={"p0": "hash"},
                    config=dict(seeds=[101, 202, 303], ports=PORTS[101], epochs=40, battles=24))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "results").mkdir()
            initial = root / "results/initial.json"
            output = root / "results/combined.json"
            paths = {101: initial, **{s: root / f"results/temperature_seed{s}.json" for s in (202, 303)}}
            record = dict(seed=101, arm="annealed", generations={"1": {"proposal": {"pastes": []}}})
            initial_data = dict(status="running", manifest=base, runs={"seed101/annealed": record})
            initial.write_text(json.dumps(initial_data))
            for seed in (202, 303):
                manifest = copy.deepcopy(base)
                manifest["config"].update(seeds=[seed], ports=PORTS[seed])
                paths[seed].write_text(json.dumps(dict(status="running", manifest=manifest, runs={})))
            (root / "temperature-parallel-state.json").write_text(json.dumps(dict(jobs={
                str(s): dict(pid=s, output=str(paths[s])) for s in (202, 303)})))
            commands = {s: f"python entropyloop.py compare run --output {p}" for s, p in paths.items()}
            original_open = Path.open
            reads = 0

            class ReplaceAfterRead:
                def __init__(self, stream):
                    self.stream = stream

                def __enter__(self):
                    self.stream.__enter__()
                    return self

                def __exit__(self, *args):
                    return self.stream.__exit__(*args)

                def __getattr__(self, name):
                    return getattr(self.stream, name)

                def read(self, *args):
                    nonlocal reads
                    payload = self.stream.read(*args)
                    reads += 1
                    # First read loads the base manifest. The second is the first
                    # source snapshot in the real main loop.
                    if reads == 2:
                        old_time = os.fstat(self.stream.fileno()).st_mtime_ns
                        changed = copy.deepcopy(initial_data)
                        changed["runs"]["seed101/annealed"]["generations"]["1"]["scores"] = [{"wins": 1}]
                        replacement = initial.with_suffix(".replacement")
                        replacement.write_text(json.dumps(changed))
                        os.utime(replacement, ns=(old_time + 1000000000, old_time + 1000000000))
                        replacement.replace(initial)
                    return payload

            def opened(path, *args, **kwargs):
                stream = original_open(path, *args, **kwargs)
                return ReplaceAfterRead(stream) if path == initial else stream

            argv = ["collector", "--initial", str(initial), "--adopt-pid", "101", "--output", str(output)]
            with patch.object(campaign, "REPO", root), patch.object(campaign, "RUNTIME", root), \
                    patch.object(campaign, "live_command", side_effect=lambda pid: commands.get(pid)), \
                    patch.object(campaign, "launch", side_effect=AssertionError("must adopt existing workers")), \
                    patch.object(campaign.time, "sleep", side_effect=[None, StopLoop]), \
                    patch.object(sys, "argv", argv), patch.object(Path, "open", opened):
                with self.assertRaises(StopLoop):
                    campaign.main()
            collected = json.loads(output.read_text())
            self.assertEqual(reads, 3, "test must exercise both collector iterations")
            self.assertIn("scores", collected["runs"]["seed101/annealed"]["generations"]["1"],
                          "collector skipped the new source after tagging an older snapshot with its timestamp")


if __name__ == "__main__":
    unittest.main()
