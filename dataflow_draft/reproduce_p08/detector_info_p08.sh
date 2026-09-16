#!/bin/bash
#SBATCH --job-name=xtc_detector_info_p08
#SBATCH --output=/pscratch/sd/h/hungwei/reproduce_with_dataflow_p08/logs/detector_info_p08_%A_%a.out
#SBATCH --error=/pscratch/sd/h/hungwei/reproduce_with_dataflow_p08/logs/detector_info_p08_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=0-100%50
#SBATCH -q shared
#SBATCH -C cpu
#SBATCH -A m2676

# First stage of reproducing the p08 cross-talk matrix through the dataflow
# entry point, `xtc.py detector-info`.  One array task per detector: p08 has
# 101 channels, so 0-100, and the task id indexes chn_id in FILE_LIST and the
# channel/rawid lists in CHANNEL_LIST, which are in the same order.
#
# Submit from the repository root.  The log directory in the #SBATCH lines
# above is opened before this script runs, so it has to exist at submit time:
#   mkdir -p /pscratch/sd/h/hungwei/reproduce_with_dataflow_p08/logs

TEMP_DIR=${TEMP_DIR:-/pscratch/sd/h/hungwei/reproduce_with_dataflow_p08}
FILE_LIST=${FILE_LIST:-reproduce/test_p08.json}
CHANNEL_LIST=${CHANNEL_LIST:-dataflow_draft/reproduce_p08/channels_p08.json}
CONFIGS=${CONFIGS:-dataflow_draft/config}
DATATYPE=xtc

# Always work from the repository root
REPO_ROOT=${SLURM_SUBMIT_DIR:-$(dirname "$(dirname "$(dirname "$(readlink -f "$0")")")")}
cd "$REPO_ROOT" || exit 1

source .venv/bin/activate

set -eo pipefail

FILELIST_DIR="${TEMP_DIR}/filelists"
mkdir -p "${FILELIST_DIR}" "${TEMP_DIR}/logs" "${TEMP_DIR}/detector_info"

# Writes the hit and dsp filelists the dataflow would hand the rule, and prints
# this task's timestamp, channel and rawid.  The timestamp is the earliest of
# all the files in FILE_LIST.
TASK_INPUTS=$(python - "${FILE_LIST}" "${CHANNEL_LIST}" "${SLURM_ARRAY_TASK_ID}" "${FILELIST_DIR}" <<'EOF'
import json
import os
import re
import sys
from pathlib import Path

file_list, channel_list, index, filelist_dir = sys.argv[1:]
files = json.loads(Path(file_list).read_text())
channels = json.loads(Path(channel_list).read_text())
index = int(index)

rawid = channels["rawid"][index]
if rawid != files["chn_id"][index]:
    sys.exit(f"index {index} is rawid {rawid} in {channel_list} but {files['chn_id'][index]} in {file_list}")

# every task writes the same filelists, so writing to a temporary file and
# renaming it is enough to keep a task from reading one half written
for tier in ("hit", "dsp"):
    filelist = Path(filelist_dir) / f"all-l200-p08-r015-xtc-{tier}.filelist"
    tmp = filelist.with_name(f"{filelist.name}.{index}.tmp")
    tmp.write_text("\n".join(files[tier]) + "\n")
    os.replace(tmp, filelist)

timestamp = min(re.search(r"\d{8}T\d{6}Z", f).group() for f in files["hit"] + files["dsp"])
print(timestamp, channels["channel"][index], rawid)
EOF
)
read -r TIMESTAMP CHANNEL RAWID <<< "${TASK_INPUTS}"

# named like the dataflow's per-channel temporary pars and logs
KEY="l200-p08-r015-${DATATYPE}-${TIMESTAMP}-${CHANNEL}"
OUTPUT="${TEMP_DIR}/detector_info/${KEY}-par_xtc_detector_info.lh5"
LOG="${TEMP_DIR}/logs/${KEY}-pars_geds_xtc_detector_info.log"

date
hostname
echo "Running detector-info on ${CHANNEL} (rawid ${RAWID}, index ${SLURM_ARRAY_TASK_ID}) at ${TIMESTAMP}, results in ${OUTPUT}"

python dataflow_draft/xtc.py detector-info \
    --hit-files "${FILELIST_DIR}/all-l200-p08-r015-xtc-hit.filelist" \
    --dsp-files "${FILELIST_DIR}/all-l200-p08-r015-xtc-dsp.filelist" \
    --configs "${CONFIGS}" \
    --log "${LOG}" \
    --datatype "${DATATYPE}" \
    --timestamp "${TIMESTAMP}" \
    --channel "${CHANNEL}" \
    --rawid "${RAWID}" \
    --output "${OUTPUT}"

echo "Done."
date
