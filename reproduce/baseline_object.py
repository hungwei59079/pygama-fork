"""Compare two independently produced sets of baseline JSON files.

The comparison itself lives in :mod:`reproduce_utils`; this is the command
line around it.  The question it answers is whether two sets of baseline files
describe the *same* measurement -- they will never be byte-identical, since
they carry different provenance fields and were printed by different writers,
so the answer is reported as coverage, usability, and how far apart the
numbers actually are.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from reproduce_utils import (
    baseline_files,
    compare_baselines,
    format_report,
    load_baseline,
    printed_digits,
)

DEFAULT_DIR_A = Path("/pscratch/sd/h/hungwei/reproduce_temp/baseline")
DEFAULT_DIR_B = Path(
    "/pscratch/sd/h/hungwei/temp_results_p08/temp_results/parameters"
    "/baseline_individuals_20260719_234907/json"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir_a", type=Path, default=DEFAULT_DIR_A, help="first baseline directory"
    )
    parser.add_argument(
        "--dir_b", type=Path, default=DEFAULT_DIR_B, help="second baseline directory"
    )
    parser.add_argument(
        "--pattern", default="*.json", help="glob picking the JSON files out"
    )
    parser.add_argument(
        "--n_worst",
        type=int,
        default=5,
        help="how many of the largest disagreements to list",
    )
    parser.add_argument(
        "--dump",
        type=Path,
        default=None,
        help="write the per-value comparison to this JSON file",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    files_a = baseline_files(args.dir_a, args.pattern)
    files_b = baseline_files(args.dir_b, args.pattern)
    print(f"A: {len(files_a)} files in {args.dir_a}")
    print(f"B: {len(files_b)} files in {args.dir_b}")

    baseline_a = load_baseline(files_a)
    baseline_b = load_baseline(files_b)

    report = compare_baselines(baseline_a, baseline_b)
    print()
    print(
        format_report(
            report,
            label_a=str(args.dir_a),
            label_b=str(args.dir_b),
            digits_a=printed_digits(files_a),
            digits_b=printed_digits(files_b),
            n_worst=args.n_worst,
        )
    )

    if args.dump is not None:
        args.dump.parent.mkdir(parents=True, exist_ok=True)
        args.dump.write_text(
            json.dumps({**report, "digits": None}, indent=2, default=str)
        )
        print(f"per-value comparison written to {args.dump}")


if __name__ == "__main__":
    main()
