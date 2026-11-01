"""Run the offline SDK recovery matrix and export its observed evidence."""

import argparse
import json
from pathlib import Path

from scripts.benchmark.checkpoint_recovery import run_suite


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=30, help="Deadline for each disposable worker")
    parser.add_argument("--output", type=Path, help="New JSON report file; existing files are preserved")
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        parser.error("output already exists; choose a new report path")
    try:
        report = run_suite(repeat=args.repeat, timeout_seconds=args.timeout_seconds)
    except ValueError as error:
        parser.error(str(error))
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8", newline="\n") as output:
            output.write(serialized)
    print(serialized, end="")
    return 0 if report["summary"]["matching_observations"] == report["summary"]["cases"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
