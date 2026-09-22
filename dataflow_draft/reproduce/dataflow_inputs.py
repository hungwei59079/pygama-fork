"""Write the inputs the dataflow would give the cross-talk rules of one run.

Until there are snakemake rules, this stands in for the dataflow.  It finds the
hit and dsp files, looks up the germanium channels in the channel map on the
earliest timestamp, and writes, in the output directory:

all-l200-{period}-{run}-{datatype}-{hit,dsp}.filelist
    the files of each tier, one per line
l200-{period}-{run}-{datatype}-channels.txt
    one "timestamp channel rawid" line per germanium channel, in the order the
    channel map lists them, for a detector-info array task to pick its line
    from, and for the matrix job to list the detector info files it reads and
    to pass the channel order the matrix is indexed in

Unlike the dataflow, every geds channel is kept, whether or not its detector
status is processable, so that the channels match the earlier analysis.

Several runs may be given, to measure one matrix over all of them::

    --run r008 r009 r010 r011 r012 r013 r014 r015 r016 r017 r018

Their files are then written to a single pair of filelists, ordered in time,
under the run label {first}_{last}.  The runs have to be combined here rather
than at the matrix step: prepare_detector records its triggers as entry numbers
into the whole file list it is given, so they only mean the same thing in every
channel if every channel is prepared from the same list.

Run it once before submitting detector_info_submitter.sh; it prints the --array
range.
"""

import argparse
import re
from pathlib import Path

from dbetto import TextDB

TIMESTAMP_PATTERN = re.compile(r"\d{8}T\d{6}Z")

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument(
    "--data-dir",
    type=Path,
    required=True,
    help="Production directory holding generated/ and the metadata in inputs/.",
)
parser.add_argument("--datatype", required=True, help="e.g. xtc or ssc.")
parser.add_argument("--period", required=True, help="e.g. p08.")
parser.add_argument(
    "--run",
    required=True,
    nargs="+",
    help="e.g. r015, or several runs to combine into one measurement.",
)
parser.add_argument(
    "--run-label",
    help="Run field of the written names; by default {first}_{last} of --run.",
)
parser.add_argument(
    "--output-dir",
    type=Path,
    required=True,
    help="Directory the filelists and the channel list are written to.",
)
args = parser.parse_args()

runs = sorted(args.run)
run_label = args.run_label or (runs[0] if len(runs) == 1 else f"{runs[0]}_{runs[-1]}")
keypart = f"l200-{args.period}-{run_label}-{args.datatype}"


def find_tier_files(tier: str, run: str) -> dict[str, Path]:
    """One run's files of one tier, keyed by timestamp."""
    run_keypart = f"l200-{args.period}-{run}-{args.datatype}"
    tier_dir = (
        args.data_dir / "generated" / "tier" / tier / args.datatype / args.period / run
    )
    files = sorted(tier_dir.glob(f"{run_keypart}-*-tier_{tier}.lh5"))
    if not files:
        msg = f"no {tier} files of {run_keypart} in {tier_dir}"
        raise FileNotFoundError(msg)
    return {TIMESTAMP_PATTERN.search(f.name).group(): f for f in files}


def merge(by_run: dict[str, dict[str, Path]], tier: str) -> dict[str, Path]:
    """Every run's files of one tier in one dict, so that sorting it orders
    them in time whichever run each of them came from."""
    merged: dict[str, Path] = {}
    for run in runs:
        for timestamp, file in by_run[run].items():
            if timestamp in merged:
                msg = (
                    f"two {tier} files of {args.period} are timestamped "
                    f"{timestamp}: {merged[timestamp]} and {file}"
                )
                raise ValueError(msg)
            merged[timestamp] = file
    return merged


hit_by_run = {run: find_tier_files("hit", run) for run in runs}
dsp_by_run = {run: find_tier_files("dsp", run) for run in runs}

hit_files = merge(hit_by_run, "hit")
dsp_files = merge(dsp_by_run, "dsp")
if hit_files.keys() != dsp_files.keys():
    msg = (
        f"the hit and dsp files of {keypart} do not match: "
        f"{sorted(hit_files.keys() - dsp_files.keys())} only in hit, "
        f"{sorted(dsp_files.keys() - hit_files.keys())} only in dsp"
    )
    raise ValueError(msg)

timestamp = min(hit_files)


def geds_on(timestamp: str) -> dict[str, int]:
    """The germanium channels of the channel map, name to rawid.

    In the order the channel map itself lists them, which is the order the
    matrix is indexed in and the one the earlier analysis used.  Sorting here,
    by name or by rawid, would index the matrix differently -- correct either
    way, but not comparable element by element with those results.
    """
    chmap = (
        TextDB(args.data_dir / "inputs", lazy=True)
        .hardware.configuration.channelmaps.on(timestamp, system=args.datatype)
    )
    return {
        name: channel["daq"]["rawid"]
        for name, channel in chmap.items()
        if channel["system"] == "geds"
    }


geds = geds_on(timestamp)
if not geds:
    msg = f"the channel map of {keypart} on {timestamp} has no geds channels"
    raise ValueError(msg)

# The combined measurement reads every run as one list of events, so the runs
# have to agree on which channels there are for an entry number to mean the
# same channel throughout.
for run in runs[1:]:
    run_timestamp = min(hit_by_run[run])
    run_geds = geds_on(run_timestamp)
    if run_geds != geds:
        msg = (
            f"the geds channels of {run} on {run_timestamp} are not those of "
            f"{runs[0]} on {timestamp}: "
            f"{sorted(run_geds.items() - geds.items())} only in {run}, "
            f"{sorted(geds.items() - run_geds.items())} only in {runs[0]}"
        )
        raise ValueError(msg)

args.output_dir.mkdir(parents=True, exist_ok=True)
for tier, files in (("hit", hit_files), ("dsp", dsp_files)):
    filelist = args.output_dir / f"all-{keypart}-{tier}.filelist"
    filelist.write_text("".join(f"{files[ts]}\n" for ts in sorted(files)))

channel_list = args.output_dir / f"{keypart}-channels.txt"
channel_list.write_text(
    "".join(f"{timestamp} {name} {rawid}\n" for name, rawid in geds.items())
)

print(f"{keypart}: {len(hit_files)} hit and dsp files each, from {timestamp}")
if len(runs) > 1:
    print(f"over the {len(runs)} runs {', '.join(runs)}")
print(f"{len(geds)} geds channels written to {channel_list}")
print(f"submit the array with --array=0-{len(geds) - 1}")
