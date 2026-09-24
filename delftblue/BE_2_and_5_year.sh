#!/bin/bash -l
#
# DelftBlue job: TSA-method sensitivity, Netherlands.
#
# One SLURM job runs all experiment cases sequentially.
# Each case writes its own result fragment, so an interrupted run can
# be restarted and already-completed cases can be skipped by the runner.
#
#SBATCH --job-name=BE-2-5y
#SBATCH --partition=compute
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --output=delftblue/logs/BE_2y_5y_%j.out
#SBATCH --error=delftblue/logs/BE_2y_5y_%j.err
#SBATCH --account=research-tpm-ess

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}"

module load 2026 cpu

echo "Job ID:       ${SLURM_JOB_ID}"
echo "Host:         $(hostname)"
echo "Working dir:  $(pwd)"
echo "Started:      $(date --iso-8601=seconds)"
echo "Mode:         all cases, sequential"
echo

export PYTHONUNBUFFERED=1

srun pixi run --as-is python -u -m scripts.experiments.run_experiment_batch \
    config/BE_2_and_5_year_config.yaml \
    --all

echo
echo "Finished:     $(date --iso-8601=seconds)"