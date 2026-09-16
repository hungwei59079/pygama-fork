"""Write the inputs the dataflow would give the cross-talk rules of one run.

Until there are snakemake rules, this stands in for the dataflow.  It finds the
run's hit and dsp files, looks up its germanium channels in the channel map on
the run's earliest timestamp, and writes, in the output directory:

all-l200-{period}-{run}-{datatype}-{hit,dsp}.filelist
    the files of each tier, one per line
l200-{period}-{run}-{datatype}-channels.txt
    one "timestamp channel rawid" line per germanium channel, sorted by
    channel name, for a SLURM array task to pick its line from

Unlike the dataflow, every geds channel is kept, whether or not its detector
status is processable, so that the channels match the earlier analysis.

Run it once per run before submitting the array; it prints the --array range.
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
parser.add_argument("--run", required=True, help="e.g. r015.")
parser.add_argument(
    "--output-dir",
    type=Path,
    required=True,
    help="Directory the filelists and the channel list are written to.",
)
args = parser.parse_args()

keypart = f"l200-{args.period}-{args.run}-{args.datatype}"


def find_tier_files(tier: str) -> dict[str, Path]:
    """The run's files of one tier, keyed by timestamp."""
    tier_dir = (
        args.data_dir / "generated" / "tier" / tier / args.datatype / args.period / args.run
    )
    files = sorted(tier_dir.glob(f"{keypart}-*-tier_{tier}.lh5"))
    if not files:
        msg = f"no {tier} files of {keypart} in {tier_dir}"
        raise FileNotFoundError(msg)
    return {TIMESTAMP_PATTERN.search(f.name).group(): f for f in files}


hit_files = find_tier_files("hit")
dsp_files = find_tier_files("dsp")
if hit_files.keys() != dsp_files.keys():
    msg = (
        f"the hit and dsp files of {keypart} do not match: "
        f"{sorted(hit_files.keys() - dsp_files.keys())} only in hit, "
        f"{sorted(dsp_files.keys() - hit_files.keys())} only in dsp"
    )
    raise ValueError(msg)

timestamp = min(hit_files)

chmap = (
    TextDB(args.data_dir / "inputs", lazy=True)
    .hardware.configuration.channelmaps.on(timestamp, system=args.datatype)
)
geds = sorted(name for name, channel in chmap.items() if channel["system"] == "geds")
if not geds:
    msg = f"the channel map of {keypart} on {timestamp} has no geds channels"
    raise ValueError(msg)

args.output_dir.mkdir(parents=True, exist_ok=True)
for tier, files in (("hit", hit_files), ("dsp", dsp_files)):
    filelist = args.output_dir / f"all-{keypart}-{tier}.filelist"
    filelist.write_text("".join(f"{files[ts]}\n" for ts in sorted(files)))

channel_list = args.output_dir / f"{keypart}-channels.txt"
channel_list.write_text(
    "".join(f"{timestamp} {name} {chmap[name]['daq']['rawid']}\n" for name in geds)
)

print(f"{keypart}: {len(hit_files)} hit and dsp files each, from {timestamp}")
print(f"{len(geds)} geds channels written to {channel_list}")
print(f"submit the array with --array=0-{len(geds) - 1}")
