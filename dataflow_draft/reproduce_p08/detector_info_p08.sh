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
# entry point, `xtc.py detector-info`.  One array task per germanium channel:
# task n runs on line n+1 of the channel list.
#
# Submit from the repository root, after writing the filelists and the channel
# list once, which prints the --array range (0-100 for p08):
#   python dataflow_draft/dataflow_inputs.py \
#       --data-dir /global/cfs/cdirs/m2676/data/lngs/l200/scratch/crosstalk_data/xtc \
#       --datatype xtc --period p08 --run r015 \
#       --output-dir /pscratch/sd/h/hungwei/reproduce_with_dataflow_p08/filelists
#
# The log directory in the #SBATCH lines above is opened before this script
# runs, so it has to exist at submit time:
#   mkdir -p /pscratch/sd/h/hungwei/reproduce_with_dataflow_p08/logs

TEMP_DIR=${TEMP_DIR:-/pscratch/sd/h/hungwei/reproduce_with_dataflow_p08}
CONFIGS=${CONFIGS:-dataflow_draft/config}
PERIOD=p08
RUN=r015
DATATYPE=xtc

KEYPART="l200-${PERIOD}-${RUN}-${DATATYPE}"
FILELIST_DIR="${TEMP_DIR}/filelists"
CHANNEL_LIST="${FILELIST_DIR}/${KEYPART}-channels.txt"

# Always work from the repository root
REPO_ROOT=${SLURM_SUBMIT_DIR:-$(dirname "$(dirname "$(dirname "$(readlink -f "$0")")")")}
cd "$REPO_ROOT" || exit 1

source .venv/bin/activate

set -eo pipefail

mkdir -p "${TEMP_DIR}/logs" "${TEMP_DIR}/detector_info"

CHANNEL_LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${CHANNEL_LIST}")
if [[ -z "${CHANNEL_LINE}" ]]; then
    echo "task ${SLURM_ARRAY_TASK_ID} is past the last channel in ${CHANNEL_LIST}" >&2
    exit 1
fi
read -r TIMESTAMP CHANNEL RAWID <<< "${CHANNEL_LINE}"

# named like the dataflow's per-channel temporary pars and logs
KEY="${KEYPART}-${TIMESTAMP}-${CHANNEL}"
OUTPUT="${TEMP_DIR}/detector_info/${KEY}-par_xtc_detector_info.lh5"
LOG="${TEMP_DIR}/logs/${KEY}-pars_geds_xtc_detector_info.log"

date
hostname
echo "Running detector-info on ${CHANNEL} (rawid ${RAWID}, index ${SLURM_ARRAY_TASK_ID}) at ${TIMESTAMP}, results in ${OUTPUT}"

python dataflow_draft/xtc.py detector-info \
    --hit-files "${FILELIST_DIR}/all-${KEYPART}-hit.filelist" \
    --dsp-files "${FILELIST_DIR}/all-${KEYPART}-dsp.filelist" \
    --configs "${CONFIGS}" \
    --log "${LOG}" \
    --datatype "${DATATYPE}" \
    --timestamp "${TIMESTAMP}" \
    --channel "${CHANNEL}" \
    --rawid "${RAWID}" \
    --output "${OUTPUT}"

echo "Done."
date
