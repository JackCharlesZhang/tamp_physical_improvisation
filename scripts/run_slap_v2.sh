#!/bin/bash

#SBATCH --nodes=1                                      ## Node count
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48                              ## Give the single process lots of CPU

#SBATCH --mem=100G                                      ## RAM per node
#SBATCH --time=5:00:00                                  ## Walltime
#SBATCH --gres=gpu:1                                    ## Number of GPUs
#SBATCH --job-name=train_slap                            ## Job Name
#SBATCH --output=/scratch/gpfs/TRIDAO/jz4267/rollouts_obstacle_tower_%j.out    ## Stdout File
#SBATCH --error=/scratch/gpfs/TRIDAO/jz4267/rollouts_obstacle_tower_%j.err      ## Stderr File

set -euo pipefail

# If not running under Slurm, auto-submit this script to avoid login-node execution.
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "[INFO] Not inside a Slurm job. Submitting via sbatch to avoid running on the login node..."
  exec sbatch "$0"
fi


module load intel-mkl/2024.2  # Load Intel MKL for numpy

# Set library paths for IKFast (if using obstacle_tower)
export LAPACK_DIR="/usr/lib64"
export LIBGFORTRAN_DIR="/usr/lib64"
export BLAS_DIR="/usr/lib64"

# Set up paths
SCRATCH_DIR="/scratch/gpfs/TRIDAO/jz4267"

# Create necessary directories
mkdir -p "$SCRATCH_DIR/slurm_logs"
mkdir -p "$PIPELINE_CACHE_DIR"

# Create a timestamped run directory for this experiment
RUN_TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
RUN_DIR="$SCRATCH_DIR/outputs/last_day/rollouts_grid_world_job${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR"

# Also copy logs to the run directory (in addition to slurm_logs/)
exec > >(tee -a "$RUN_DIR/slurm_${SLURM_JOB_ID}.out")
exec 2> >(tee -a "$RUN_DIR/slurm_${SLURM_JOB_ID}.err" >&2)

cd ~/tamp_physical_improvisation && module load anaconda3/2024.10 && source .venv/bin/activate

export PYTHONPATH=/scratch/gpfs/TRIDAO/jz4267/prpl-mono/bilevel-planning/src:/scratch/gpfs/TRIDAO/jz4267/prpl-mono/prbench/src:/scratch/gpfs/TRIDAO/jz4267/prpl-mono/prbench-bilevel-planning/src:/scratch/gpfs/TRIDAO/jz4267/prpl-mono/prbench-models/src:/scratch/gpfs/TRIDAO/jz4267/pybullet-blocks/src:/scratch/gpfs/TRIDAO/jz4267/pybullet-helpers/src:/scratch/gpfs/TRIDAO/jz4267/pybullet:\$PYTHONPATH



# WandB API Key
export WANDB_API_KEY="f4337c0642e57576b4b354fb031abc4b9a367ce4"

# CUDA memory allocator: reduce fragmentation for long runs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# SLURM-specific threading
export SLURM_CPUS_PER_TASK=48
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Ensure output directories exist
mkdir -p slurm_outputs/train_slap

python -m experiments.slap_train_pipeline_v2 --config-name gridworld_fixed ${CONFIG_OVERRIDES} seed=42 | tee run_log.txt