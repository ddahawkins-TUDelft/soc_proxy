#!/bin/bash -l
#
# DelftBlue consolidation job.
#
# Submit this only after both country arrays have completed successfully.
# Later we can submit it automatically with an afterok dependency on both
# array-job IDs.
#
#SBATCH --job-name=soc-consolidate
#SBATCH --partition=compute
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --output=delftblue/logs/consolidate_%j.out
#SBATCH --error=delftblue/logs/consolidate_%j.err
#SBATCH --account=research-tpm-ess


set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}"

module load 2026 cpu

echo "Job ID:       ${SLURM_JOB_ID}"
echo "Host:         $(hostname)"
echo "Working dir:  $(pwd)"
echo "Started:      $(date --iso-8601=seconds)"

srun pixi run python -m scripts.experiments.consolidate_results

echo "Finished:     $(date --iso-8601=seconds)"
