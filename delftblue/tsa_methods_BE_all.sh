#!/bin/bash -l
#
# DelftBlue job: TSA-method sensitivity, Netherlands.
#
# One SLURM job runs all experiment cases sequentially.
# Each case writes its own result fragment, so an interrupted run can
# be restarted and already-completed cases can be skipped by the runner.
#
#SBATCH --job-name=soc-tsa-BE
#SBATCH --partition=compute
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=3G
#SBATCH --output=delftblue/logs/tsa_methods_BE_%j.out
#SBATCH --error=delftblue/logs/tsa_methods_BE_%j.err
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
    config/sensitivity_tsa_methods_BE.yaml \
    --all

echo
echo "Finished:     $(date --iso-8601=seconds)"