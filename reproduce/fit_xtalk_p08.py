"""Fit every p08 cross-talk column and assemble them into the matrix.

The two steps after ``xtalk_column_p08.py``, in one job:
:func:`~pygama.pargen.xtc.xtalk_histogram_fitter` runs over every column file
in ``--in_dir``, then :func:`~pygama.pargen.xtc.build_xtalk_matrix` collects
what it returned into an :class:`~pygama.pargen.xtc_utils.XTCMatrix`.

Fitting a column is fast enough -- a few thousand gaussians, no file reading
beyond the histograms themselves -- that all 101 of them fit in one serial
job, so there is no array to split here and the matrix can be built in the
same process that produced its rows.

The fitted columns are written to ``--out_dir`` rather than back over the
histograms in ``--in_dir``, so a refit with different thresholds never costs
another pass over the hit/dsp files.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from pygama.pargen.xtc import (
    FIT_STATUS,
    build_xtalk_matrix,
    read_xtalk_column,
    xtalk_histogram_fitter,
)

DEFAULT_ROOT = Path("/pscratch/sd/h/hungwei/reproduce_temp/")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--in_dir",
    type=Path,
    default=DEFAULT_ROOT / "xtalk_columns",
    help="Directory of the xtalk_column_XXXX.lh5 files to fit.",
)
parser.add_argument(
    "--out_dir",
    type=Path,
    default=DEFAULT_ROOT / "xtalk_fits",
    help="Directory the fitted columns are written to.",
)
parser.add_argument(
    "--results_dir",
    type=Path,
    default=DEFAULT_ROOT / "xtalk_matrix",
    help="Directory the matrix lh5, its heatmaps and its CSVs are written to.",
)
parser.add_argument(
    "--low_stats_threshold",
    type=float,
    default=100,
    help="Counts below which the histogram moments replace the fit.",
)
parser.add_argument(
    "--y_mask_threshold",
    type=float,
    default=0.05,
    help="Bins below this fraction of the tallest one are dropped before fitting.",
)
parser.add_argument(
    "--sharp_fit_min_points",
    type=int,
    default=5,
    help="Bins that mask must leave for the fit to use it.",
)
parser.add_argument(
    "--max_status",
    type=str,
    default="low_stats",
    choices=list(FIT_STATUS),
    help=(
        "Worst fit status still accepted into the matrix; the statuses are "
        "ordered from the most to the least trustworthy, so this is a quality cut."
    ),
)
parser.add_argument(
    "--store_in_percent",
    action="store_true",
    help="Store percent in the matrix lh5 rather than fractions.",
)
parser.add_argument(
    "--max_columns",
    type=int,
    default=None,
    help="Fit only the first N column files. For a quick smoke test.",
)
parser.add_argument(
    "--debug_mode",
    action="store_true",
    help="Re-raise instead of recording an element as fit_failed.",
)
args = parser.parse_args()

# both the fitter and the matrix builder report what they skipped through the
# logging module, so route those into the SLURM .out file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)

column_files = sorted(args.in_dir.glob("xtalk_column_*.lh5"))
if not column_files:
    msg = f"no xtalk_column_*.lh5 files in {args.in_dir}"
    raise FileNotFoundError(msg)
if args.max_columns is not None:
    column_files = column_files[: args.max_columns]

fit_config = {
    "low_stats_threshold": args.low_stats_threshold,
    "y_mask_threshold": args.y_mask_threshold,
    "sharp_fit_min_points": args.sharp_fit_min_points,
}

print(f"fitting {len(column_files)} columns from {args.in_dir}")
print(f"fit configuration: {fit_config}")

# ---------------------------------------------------------------- fit ------

fitted_columns = {}
failed_files = []

for column_file in column_files:
    try:
        histogram_data = read_xtalk_column(column_file)
    except Exception as e:
        if args.debug_mode:
            raise
        print(f"{column_file.name}: unreadable, skipping ({e})")
        failed_files.append(column_file.name)
        continue

    out_file = args.out_dir / column_file.name.replace("column", "fit")
    result = xtalk_histogram_fitter(
        histogram_data,
        config=fit_config,
        out_path=out_file,
        debug_mode=args.debug_mode,
    )

    trigger_id = int(result["trigger_id"])
    if trigger_id in fitted_columns:
        msg = (
            f"{column_file.name} triggers on ch{trigger_id}, which an earlier "
            f"file in {args.in_dir} already covers"
        )
        raise ValueError(msg)
    fitted_columns[trigger_id] = result

    n_valid = int(np.asarray(result["valid"], dtype=bool).sum())
    print(
        f"{column_file.name}: ch{trigger_id}, "
        f"{int(result['neg_success'].sum())} negative and "
        f"{int(result['pos_success'].sum())} positive fits converged "
        f"of {n_valid} filled elements -> {out_file.name}"
    )

print(f"\n{len(fitted_columns)} columns fitted, written to {args.out_dir}")
if failed_files:
    print(f"{len(failed_files)} column files could not be read: {failed_files}")

# ------------------------------------------------------------- matrix ------

args.results_dir.mkdir(parents=True, exist_ok=True)
matrix_file = args.results_dir / "par_evt_xtc.lh5"

matrix = build_xtalk_matrix(
    fitted_columns,
    out_path=matrix_file,
    config={
        "max_status": FIT_STATUS[args.max_status],
        "store_in_percent": args.store_in_percent,
    },
)

unit = "percent" if args.store_in_percent else "fractions"
print(
    f"\n{matrix.n_detectors}x{matrix.n_detectors} matrix written to "
    f"{matrix_file} ({unit})"
)

# The index of every row and column, so a matrix element can be traced back to
# the detector pair that produced it without reopening the lh5 file.
index_file = args.results_dir / "detector_index.csv"
with index_file.open("w") as f:
    f.write("index,rawid\n")
    for index, rawid in enumerate(matrix.rawids):
        f.write(f"{index},{rawid}\n")
print(f"detector index written to {index_file}")

for polarity in matrix.polarities:
    matrix.plot(polarity, args.results_dir / f"{polarity}_xtalk_matrix.png")
    np.savetxt(
        args.results_dir / f"{polarity}_xtalk_matrix.csv",
        matrix.mu[polarity],
        delimiter=",",
    )
    np.savetxt(
        args.results_dir / f"{polarity}_xtalk_matrix_sigma.csv",
        matrix.sigma[polarity],
        delimiter=",",
    )
print(f"heatmaps and CSVs written to {args.results_dir}")

# ------------------------------------------------------------ summary ------

summary = matrix.summary()
print("\nfit status of the matrix elements:")
for polarity, counts in summary.items():
    filled = matrix.n_detectors**2 - counts.get("not_filled", 0)
    finite = int(np.isfinite(matrix.mu[polarity]).sum())
    print(f"  {polarity}: {filled} elements filled, {finite} of them measured")
    for name, total in sorted(counts.items(), key=lambda item: -item[1]):
        print(f"    {name}: {total}")

summary_file = args.results_dir / "fit_summary.json"
with summary_file.open("w") as f:
    json.dump(
        {
            "n_detectors": int(matrix.n_detectors),
            "n_columns_fitted": len(fitted_columns),
            "unreadable_column_files": failed_files,
            "fit_parameters": fit_config,
            "max_status": args.max_status,
            "stored_in_percent": bool(args.store_in_percent),
            "status_counts": summary,
        },
        f,
        indent=2,
    )
print(f"summary written to {summary_file}")
