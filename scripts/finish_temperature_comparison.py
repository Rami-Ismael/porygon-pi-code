"""Finish local experiment artifacts when a running campaign exits successfully."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--pid", type=int, required=True)
    args = parser.parse_args()
    deadline = time.monotonic() + 24 * 3600
    while time.monotonic() < deadline:
        data = json.loads(args.results.read_text()) if args.results.exists() else {}
        if data.get("status") == "complete":
            for name in ("check_temperature_results.py", "analyze_temperature_comparison.py"):
                subprocess.run([sys.executable, str(Path(__file__).with_name(name)), str(args.results)], check=True)
            print("TEMPERATURE_VERIFIED_AND_ANALYZED", flush=True)
            return
        try:
            os.kill(args.pid, 0)
        except ProcessLookupError:
            raise SystemExit("Campaign process exited before completion; inspect the run log and resume.")
        time.sleep(30)
    raise SystemExit("Campaign still incomplete after 24 hours; inspect the run log.")


if __name__ == "__main__":
    main()
