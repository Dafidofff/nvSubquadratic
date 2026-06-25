#!/bin/bash
#SBATCH --account=all6000users
#SBATCH --partition=all6000
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=1-00:00:00
#SBATCH --mem=48G
#SBATCH --output=logs/dd_%x_%j.out

set -eo pipefail
echo "Node: $SLURM_NODELIST  Job: $SLURM_JOB_NAME ($SLURM_JOB_ID)  SEED=${SEED:-unset}"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate nvsubq

export WANDB_DIR=/ivi/zfs/s0/original_homes/dwessel/wandb
export WANDB_DATA_DIR=/ivi/zfs/s0/original_homes/dwessel/wandb
export TINYIMAGENET_PATH="${TINYIMAGENET_PATH:-/ivi/zfs/s0/original_homes/dwessel/data/tiny-imagenet}"

export PATH="/usr/local/cuda-13.0/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/tmp/triton_nocache_${SLURM_JOB_ID}
export OMP_NUM_THREADS=1

cd /home/dwessel/code/nvSubquadratic-dd
mkdir -p logs

CONFIG="$1"; shift
PYTHONPATH=. python experiments/run.py --config "$CONFIG" "$@"
