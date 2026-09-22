#!/bin/bash
# Change the lines marked with DATASET to run on a different dataset.

# DATASET: period and run in the job name and log paths (the logs/ of detector_info_submitter.sh)
#SBATCH --job-name=xtc_matrix_p16
#SBATCH --output=/pscratch/sd/h/hungwei/reproduce_with_dataflow_p16/logs/matrix_p16_%j.out
#SBATCH --error=/pscratch/sd/h/hungwei/reproduce_with_dataflow_p16/logs/matrix_p16_%j.err

#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

#SBATCH -q shared
#SBATCH -C cpu
#SBATCH -A m2676

# One job per measurement, over the detector info detector_info_submitter.sh
# wrote for every channel of the channel list.  PERIOD, RUN and DATATYPE have to
# be the ones that array ran with, so that KEYPART names the same files.  Submit
# from the repository root once that array has finished:
#   sbatch dataflow_draft/reproduce/matrix_submitter.sh
# or right after submitting the array, to start when all of its tasks succeed:
#   sbatch --dependency=afterok:<array job id> dataflow_draft/reproduce/matrix_submitter.sh


# DATASET: p08 is PERIOD=p08, RUN=r015, DATATYPE=xtc
PERIOD=p16
RUN=r008_r018
DATATYPE=ssc

TEMP_DIR=${TEMP_DIR:-/pscratch/sd/h/hungwei/reproduce_with_dataflow_${PERIOD}}
CONFIGS=${CONFIGS:-dataflow_draft/config}

KEYPART="l200-${PERIOD}-${RUN}-${DATATYPE}"
FILELIST_DIR="${TEMP_DIR}/filelists"
CHANNEL_LIST="${FILELIST_DIR}/${KEYPART}-channels.txt"
DETECTOR_FILELIST="${FILELIST_DIR}/all-${KEYPART}-detector_info.filelist"

# Always work from the repository root
REPO_ROOT=${SLURM_SUBMIT_DIR:-$(dirname "$(dirname "$(dirname "$(readlink -f "$0")")")")}
cd "$REPO_ROOT" || exit 1

source .venv/bin/activate

set -eo pipefail

mkdir -p "${TEMP_DIR}/logs" "${TEMP_DIR}/matrix"

# every line holds the same timestamp, the run's first
read -r TIMESTAMP _ < "${CHANNEL_LIST}"

# Listed from the channel list rather than globbed, so that a channel whose detector-info task failed
# stops the matrix instead of dropping out of it. 
: > "${DETECTOR_FILELIST}" # Empty the filelist file in case of resubmission
TASK=0
MISSING=0
RAWIDS=()
while read -r _ CHANNEL RAWID; do
    FILE="${TEMP_DIR}/detector_info/${KEYPART}-${TIMESTAMP}-${CHANNEL}-par_xtc_detector_info.lh5"
    echo "${FILE}" >> "${DETECTOR_FILELIST}"
    RAWIDS+=("${RAWID}")
    if [[ ! -f "${FILE}" ]]; then
        echo "no detector info of ${CHANNEL} (rawid ${RAWID}), rerun detector_info_submitter.sh with --array=${TASK}" >&2
        MISSING=$((MISSING + 1))
    fi
    TASK=$((TASK + 1))
done < "${CHANNEL_LIST}"

if (( MISSING > 0 )); then
    echo "${MISSING} of ${TASK} channels in ${CHANNEL_LIST} have no detector info" >&2
    exit 1
fi

# named like the dataflow's run-level pars, plots and logs
KEY="${KEYPART}-${TIMESTAMP}"
OUTPUT="${TEMP_DIR}/matrix/${KEY}-par_xtc.lh5"
PLOT_FILE="${TEMP_DIR}/matrix/${KEY}-plt_xtc.pkl"
LOG="${TEMP_DIR}/logs/${KEY}-pars_geds_xtc_matrix.log"

date
hostname
echo "Running matrix on the ${TASK} channels listed in ${DETECTOR_FILELIST} at ${TIMESTAMP}, indexed in the order of ${CHANNEL_LIST}, results in ${OUTPUT}"

python dataflow_draft/xtc.py matrix \
    --detector-files "${DETECTOR_FILELIST}" \
    --rawids "${RAWIDS[@]}" \
    --configs "${CONFIGS}" \
    --log "${LOG}" \
    --datatype "${DATATYPE}" \
    --timestamp "${TIMESTAMP}" \
    --plot-file "${PLOT_FILE}" \
    --output "${OUTPUT}"

echo "Done."
date
