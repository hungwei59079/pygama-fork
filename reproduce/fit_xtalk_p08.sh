#!/bin/bash
#SBATCH --job-name=fit_xtalk_p08
#SBATCH --output=/pscratch/sd/h/hungwei/reproduce_temp/logs/fit_xtalk_p08_%j.out
#SBATCH --error=/pscratch/sd/h/hungwei/reproduce_temp/logs/fit_xtalk_p08_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -q shared
#SBATCH -C cpu
#SBATCH -A m2676

# One serial task for all 101 columns: fitting needs no file reading beyond
# the histograms, so there is nothing to split into an array. The matrix is
# built at the end of the same run.
#
# The log directory in the #SBATCH lines above is opened before this script
# runs, so it has to exist at submit time:
#   mkdir -p /pscratch/sd/h/hungwei/reproduce_temp/logs

ROOT=${ROOT:-/pscratch/sd/h/hungwei/reproduce_temp}
IN_DIR=${IN_DIR:-$ROOT/xtalk_columns}
OUT_DIR=${OUT_DIR:-$ROOT/xtalk_fits}
RESULTS_DIR=${RESULTS_DIR:-$ROOT/xtalk_matrix}

# Always work from the repository root
REPO_ROOT=${SLURM_SUBMIT_DIR:-$(dirname "$(dirname "$(readlink -f "$0")")")}
cd "$REPO_ROOT" || exit 1

source .venv/bin/activate

date
hostname
echo "Fitting columns from ${IN_DIR}, fits in ${OUT_DIR}, matrix in ${RESULTS_DIR}"

python reproduce/fit_xtalk_p08.py \
    --in_dir "${IN_DIR}" \
    --out_dir "${OUT_DIR}" \
    --results_dir "${RESULTS_DIR}"

echo "Done."
date
