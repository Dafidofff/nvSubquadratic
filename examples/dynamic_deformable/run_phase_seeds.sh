#!/usr/bin/env bash
# Sequential multi-seed sweep for the dynamic-deformable phase.
#
# Runs each listed config across every seed in SEEDS, ONE run at a time, on the
# local RTX 3090. Resumable: a run whose checkpoints/last.ckpt already exists is
# skipped. Each run logs cleanly to its own file (env-python launch, not
# `conda run`, so stdout is not swallowed the way the first P0 baseline was).
#
# Usage:  bash examples/dynamic_deformable/run_phase_seeds.sh
set -u

ROOT=/home/davidwessels/Documents/code/nvSubquadratic-private
cd "$ROOT" || exit 1

export PYTHONUNBUFFERED=1
export PYTHONPATH=.
# Proven-local CIFAR-10 cache. ananas is mounted again, but the local cache is
# known-good and avoids any ananas hf-layout mismatch mid-sweep.
export CIFAR10_PATH="$ROOT/.data/cifar10"
PY=/home/davidwessels/miniforge3/envs/nvsubq/bin/python

SEEDS=(42 43 44)
OUT=runs/dynamic_deformable        # ananas-backed via the runs/ symlink
mkdir -p "$OUT"
SWEEPLOG="$OUT/sweep.log"

# name:config_path[:EXTRA_VAR=val,...] triples, executed in order.
# Completed runs (last.ckpt exists) are skipped automatically.
CONFIGS=(
  "baseline_hyena:examples/dynamic_deformable/baseline_hyena.py"
  "spectral_film:examples/dynamic_deformable/spectral_film.py"
  "spectral_film_last:examples/dynamic_deformable/spectral_film.py:FILM_LAYERS=last"
  "envelope_warp:examples/dynamic_deformable/envelope_warp.py"
  "baseline_hyena_native32:examples/dynamic_deformable/baseline_hyena_native32.py"
  "envelope_warp_native32:examples/dynamic_deformable/envelope_warp_native32.py"
  "sparse_mask:examples/dynamic_deformable/sparse_mask.py"
  "data_warp:examples/dynamic_deformable/data_warp.py"
  # P3 sparsity-weight ablation (SW sweep for P5 stacking decision)
  "sparse_mask_sw001:examples/dynamic_deformable/sparse_mask.py:SPARSITY_WEIGHT=0.01"
  "sparse_mask_sw005:examples/dynamic_deformable/sparse_mask.py:SPARSITY_WEIGHT=0.05"
  "sparse_mask_sw05:examples/dynamic_deformable/sparse_mask.py:SPARSITY_WEIGHT=0.5"
  # P5: P1 (spectral FiLM all) + P3 (sparse mask SW=0.5)
  "combined_best:examples/dynamic_deformable/combined_best.py"
)

echo "[$(date)] sweep start; seeds=${SEEDS[*]}; out=$OUT" | tee -a "$SWEEPLOG"
for entry in "${CONFIGS[@]}"; do
  IFS=: read -r name cfg extras <<< "$entry"
  for seed in "${SEEDS[@]}"; do
    exp="$OUT/${name}_seed${seed}"
    log="$OUT/${name}_seed${seed}.log"
    if [ -f "$exp/checkpoints/last.ckpt" ]; then
      echo "[$(date)] SKIP  $name seed=$seed (last.ckpt exists)" | tee -a "$SWEEPLOG"
      continue
    fi
    echo "[$(date)] START $name seed=$seed -> $exp" | tee -a "$SWEEPLOG"
    env SEED=$seed ${extras} "$PY" experiments/run.py --config "$cfg" experiment_dir="$exp" > "$log" 2>&1
    rc=$?
    echo "[$(date)] END   $name seed=$seed rc=$rc (log: $log)" | tee -a "$SWEEPLOG"
  done
done
echo "[$(date)] sweep complete" | tee -a "$SWEEPLOG"
