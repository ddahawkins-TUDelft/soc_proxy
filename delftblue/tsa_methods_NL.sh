#!/bin/bash -l
#
# DelftBlue SLURM array: TSA-method sensitivity, Netherlands.
#
# One array task = one experiment = one case_id.
# Each task writes only per-case result fragments. It never consolidates.
#
#SBATCH --job-name=soc-tsa-NL
#SBATCH --partition=compute
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --output=delftblue/logs/tsa_methods_NL_%A_%a.out
#SBATCH --error=delftblue/logs/tsa_methods_NL_%A_%a.err
#SBATCH --array=0-179%1
#SBATCH --account=research-tpm-ess


set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}"

module load 2026 cpu

echo "Job ID:       ${SLURM_JOB_ID}"
echo "Array task:   ${SLURM_ARRAY_TASK_ID}"
echo "Host:         $(hostname)"
echo "Working dir:  $(pwd)"
echo "Started:      $(date --iso-8601=seconds)"

export PYTHONUNBUFFERED=1

srun pixi run --as-is python -u -m scripts.experiments.run_experiment_batch \
    config/sensitivity_tsa_methods_NL.yaml \
    --index "${SLURM_ARRAY_TASK_ID}"

echo "Finished:     $(date --iso-8601=seconds)"
