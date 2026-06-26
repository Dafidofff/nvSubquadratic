#!/bin/bash
#SBATCH --account=linuxusers
#SBATCH --partition=performance
#SBATCH --gres=gpu:rtx_6000_ada:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=192G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --signal=B:TERM@300

# ─────────────────────────────────────────────────────────────────────────────
# TinyImageNet image-generation (v2) — 2-GPU / 2-day initial run on hipster.
#
# Launches one diffusion config (Hyena / Attention / JiT) for ~48 h on the
# `performance` partition (RTX 6000 Ada, 48 GB; min 32 CPU/GPU -> 64 CPUs for 2).
# Runs are WALL-CLOCK bounded: train.iterations is a large cap and the
# WalltimeCheckpointer (train.run_time_limit_hours) saves last.ckpt + stops
# gracefully before the 48 h hard kill. Re-submitting the same line autoresumes.
#
# IMPORTANT: bare `python experiments/run.py` — NOT torchrun. construct_trainer
# uses devices=range(torch.cuda.device_count()) + strategy="ddp", so Lightning
# spawns the 2 DDP workers itself; wrapping in torchrun double-spawns and hangs.
#
# Usage (from the repo root on hipster):
#   sbatch --job-name=tin-hyena-2g examples/image_generation/v2/submit_2gpu_hipster.sh \
#       examples/image_generation/v2/vit5_hyena.py dataset.batch_size=32 train.accumulate_grad_steps=4
#   sbatch --job-name=tin-attn-2g  examples/image_generation/v2/submit_2gpu_hipster.sh \
#       examples/image_generation/v2/vit5_attention.py dataset.batch_size=16 train.accumulate_grad_steps=8
#   sbatch --job-name=tin-jit-2g   examples/image_generation/v2/submit_2gpu_hipster.sh \
#       examples/image_generation/v2/jit_baseline.py dataset.batch_size=128 train.accumulate_grad_steps=1
#
# Per-GPU batch_size + accumulate_grad_steps are passed as overrides (after the
# config path) so the same launcher serves all models; target effective batch
# ~256 = batch_size × 2 GPUs × accum. Tune via the smoke test first.
# ─────────────────────────────────────────────────────────────────────────────

set -eo pipefail

if [ -z "${1:-}" ]; then
    echo "Usage: sbatch --job-name=NAME $0 <config.py> [overrides...]"
    exit 1
fi
CONFIG="$1"; shift

# Repo root (override with REPO_DIR=... if deployed elsewhere).
REPO_DIR="${REPO_DIR:-$HOME/code/nvSubquadratic-imagegen}"
cd "${REPO_DIR}"
echo "Running on node: ${SLURM_NODELIST:-$(hostname)}   repo: ${REPO_DIR}"

# ─── Environment ──────────────────────────────────────────────────────────────
source ~/miniforge3/etc/profile.d/conda.sh
conda activate nvsubq

export PYTHONPATH="."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/tmp/triton_${SLURM_JOB_ID:-$$}
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export WANDB_DIR="${PWD}/wandb"
export WANDB_DATA_DIR="${PWD}/wandb"

# TinyImageNet HF cache (HF datasets caches itself here; no DALI / no staging).
export TINYIMAGENET_CACHE="${TINYIMAGENET_CACHE:-$HOME/.cache/tinyimagenet}"

# ─── Logging / results dirs ───────────────────────────────────────────────────
mkdir -p logs wandb runs
EXPERIMENT_NAME="$(basename "${CONFIG%.py}")_2gpu"
RESULTS_DIR="runs/${EXPERIMENT_NAME}"
mkdir -p "${RESULTS_DIR}"

# ─── Deterministic W&B run ID (stable across re-submissions / resume) ──────────
RUN_ID_FILE="${RESULTS_DIR}/run.id"
if [ -f "${RUN_ID_FILE}" ]; then
    RUN_ID=$(<"${RUN_ID_FILE}")
    echo "Resuming W&B run ID: ${RUN_ID}"
else
    array=()
    for i in {a..z} {A..Z} {0..9}; do array[$RANDOM]=$i; done
    RUN_ID=$(printf %s "${array[@]::8}")
    echo "${RUN_ID}" > "${RUN_ID_FILE}"
    echo "Fresh W&B run ID: ${RUN_ID}"
fi

# Resume from checkpoint if one exists, else start fresh attached to RUN_ID.
AUTORESUME_ARG="wandb.run_id=${RUN_ID}"
if [ -f "${RESULTS_DIR}/checkpoints/last.ckpt" ]; then
    echo "Checkpoint found — autoresume enabled"
    AUTORESUME_ARG="autoresume.enabled=True"
fi

# ─── Walltime budget (graceful checkpoint before the 48 h hard kill) ───────────
JOB_START_TIMESTAMP=$(date +%s)
TIME_LIMIT_HOURS=47.5

echo "================================================"
echo "  Config   : ${CONFIG}"
echo "  Exp dir  : ${RESULTS_DIR}  (2-GPU, ${TIME_LIMIT_HOURS}h budget)"
echo "  W&B ID   : ${RUN_ID}"
echo "  Cache    : ${TINYIMAGENET_CACHE}"
echo "  Overrides: $*"
echo "  GPUs     : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | tr '\n' ',')"
echo "================================================"

# ─── Run (2-GPU DDP owned by Lightning) ───────────────────────────────────────
python experiments/run.py \
    --config "${CONFIG}" \
    num_nodes=1 \
    experiment_dir="${RESULTS_DIR}" \
    train.run_start_time="${JOB_START_TIMESTAMP}" \
    train.run_time_limit_hours="${TIME_LIMIT_HOURS}" \
    ${AUTORESUME_ARG} \
    "$@"
