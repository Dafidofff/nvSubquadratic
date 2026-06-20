#!/usr/bin/env bash
# Submit dynamic-deformable experiment jobs to the ivi SLURM cluster.
#
# Submits (in parallel, each its own job):
#   P1-last  — spectral_film.py with FILM_LAYERS=last, seeds {42,43,44}
#   P2       — envelope_warp.py, seeds {42,43,44}
#
# Usage (from repo root on the ivi login node):
#   bash examples/dynamic_deformable/submit_ivi.sh
#
# Cluster data path: set CIFAR10_PATH in your environment or cluster.env.
# Default: /shared/data/image_datasets/cifar10
#
# Output dirs land in runs/dynamic_deformable/ (ananas-backed symlink).
# Set DD_OUT to override.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SUBMIT="$ROOT/scripts/slurm/submit_1gpu.sh"
OUT="${DD_OUT:-runs/dynamic_deformable}"
mkdir -p "$ROOT/$OUT"

export CIFAR10_PATH="${CIFAR10_PATH:-/shared/data/image_datasets/cifar10}"

SEEDS=(42 43 44)

echo "=== Submitting dynamic-deformable jobs to ivi ==="
echo "  CIFAR10_PATH=$CIFAR10_PATH"
echo "  output dir : $ROOT/$OUT"
echo ""

# ── P1 last-layer ablation (FILM_LAYERS=last, 3 seeds) ───────────────────────
for seed in "${SEEDS[@]}"; do
    exp="$OUT/spectral_film_last_seed${seed}"
    SEED=$seed FILM_LAYERS=last bash "$SUBMIT" \
        --job-name="p1last-s${seed}" \
        "examples/dynamic_deformable/spectral_film.py" \
        "experiment_dir=$exp" \
        "debug=false"
    echo "[P1-last] seed=$seed submitted -> $exp"
done

# ── P2 envelope warp (3 seeds) ───────────────────────────────────────────────
for seed in "${SEEDS[@]}"; do
    exp="$OUT/envelope_warp_seed${seed}"
    SEED=$seed bash "$SUBMIT" \
        --job-name="p2env-s${seed}" \
        "examples/dynamic_deformable/envelope_warp.py" \
        "experiment_dir=$exp" \
        "debug=false"
    echo "[P2-env] seed=$seed submitted -> $exp"
done

echo ""
echo "=== All jobs submitted. Monitor with: squeue -u \$USER ==="
