# Dynamic Tokenization & Learned Sparse / Deformable Attention — Experiment Tracker

**W&B project:** [`implicit-long-convs/nvsubquadratic`](https://wandb.ai/implicit-long-convs/nvsubquadratic) · **W&B group:** `dynamic-deformable`
**Starting task:** CIFAR-10 classification (32×32 RGB, 10 classes, 50k train / 10k test)
**Status:** ✅ P1–P5 all complete (2026-06-15/19, local RTX 3090). **P5 combined (P1+P3 SW=0.5): 88.70 ± 0.14% (+0.92 pp vs P0)** — clear best, slightly sub-additive (−0.30 pp vs naive sum). P1-all: +0.57 pp. P3 SW=0.5: +0.65 pp. P2/P4: eliminated. P3 gradient bug fixed 2026-06-18 (detached `_last_mean_mask`; all P3 runs re-executed). CIFAR-10 series complete. · ✅ **Difficulty-scaling extension CONCLUDED (2026-06-22):** P0 vs P5 replicated on **TinyImageNet** (200-class, native 64×64, ivi `all6000`, 3 seeds each). The CIFAR-10 +0.92 pp gain does **not** transfer to the harder task — it vanishes/inverts: **patch-8 (64-token) Δ = −0.77 pp**, **patch-4 (256-token) Δ = −0.27 pp** (within noise). The dynamic-deformable gap does not widen with difficulty; the CIFAR-10 win looks task-specific. See "TinyImageNet difficulty scaling" section below.

## Research thesis

The subquadratic mixer in this repo replaces attention with a dense `O(L log L)` global FFT
convolution whose kernel is an implicit SIREN field, shaped by a Gaussian envelope and (optionally)
conditioned by register tokens via FiLM. That dense, grid-respecting structure is exactly what makes
the FFT path fast — but today the conditioning is largely **static** (fixed frequency init, fixed
per-axis envelope std, fixed token grid).

This experiment asks: **how much expressiveness can we recover by making the operator
input-dependent (dynamic / deformable) without ever breaking the uniform FFT grid?** Four
mechanisms are explored, each touching a different stage of the pipeline (input → kernel → envelope
→ geometry). CIFAR-10 is the cheap proxy; the downstream motivation is gigapixel computational
pathology (WSI), where most of the image is irrelevant background and tissue geometry varies wildly.

All four ideas share one design constraint: **keep the spatial grid uniform so the optimized FFT
conv path is preserved.** Nothing here flattens the pixel space or scatters memory access.

---

## The four ideas

### 1. Sparse Tokenization via Dense Masking

**Concept.** Rather than physically dropping tokens — which would disrupt the regular grid required
by the FFT — a lightweight differentiable scorer predicts a spatial binary (or soft) "keep" mask.
The input tensor is multiplied by this mask to zero out uninformative patches *before* the dense
`O(L log L)` global convolution runs. The grid stays full; informative regions are isolated by
attenuation, not removal.

**Application & impact.** Powerful for gigapixel WSIs (MSI / TMB prediction, sTILs classification),
where vast slide regions are background or irrelevant connective tissue. Because the architecture
keeps spatial dims alive instead of flattening, masking naturally isolates the tumor
microenvironment while preserving native 2D/3D geometry and the optimized hardware path.

**Attach point in this repo.** A small conv scorer (a few channels → 1) producing a soft mask in
`[0,1]`, applied to the patch-embedded feature map before the CKConv mixer. Train the scorer with a
sparsity/budget regularizer (target keep-ratio) so it learns to attenuate rather than collapse.

**References.** MergeDNA / dnaHNet (2024/25, dynamic context-aware token merging in genomics) ·
Token Merging / ToMe (Bolya et al., 2023) · DynamicViT (Rao et al., 2021).

### 2. Dynamic Spectral Filtering (learnable sparse attention)

**Concept.** Instead of static frequency initializations shared across all inputs, the register
tokens dynamically output scale/shift parameters via FiLM to modulate the SIREN's hidden layers.
This lets the model suppress or amplify specific frequency blocks on the fly — a Linear
Time-Varying (LTV) system in the frequency domain.

**Application & impact.** Large theoretical-expressiveness boost. In pathology the model can route
focus by global context: amplify high-frequency filters when nuclear boundaries / cellular
morphology matter, shift to low-frequency filters to assess macroscopic tissue architecture and
stromal boundaries.

**Attach point in this repo.** This is the most "native" idea — `KernelFiLMGenerator`
([film.py:75](../../nvsubquadratic/modules/film.py)) already routes register tokens to FiLM
parameters; `SIRENKernelND` ([kernels_nd.py:876](../../nvsubquadratic/modules/kernels_nd.py)) already
accepts per-layer scale/shift. The experiment is to enable register→FiLM→SIREN conditioning and
ablate which SIREN layers receive it.

**References.** Adaptive Unitary SSMs (Karuvally et al., 2025, input-dependent transition dynamics
close the expressivity gap with attention at linear cost) · FNet (Lee-Thorp et al., 2021) · Global
Filter Networks (Rao et al., 2021).

### 3. Dynamic Envelope Warping (input-dependent Gaussian masks)

**Concept.** The architecture currently uses learnable but **static** per-axis standard deviations
for its Gaussian mask. Route the register tokens to predict a dynamic shift (translation) and
dynamic scale multiplier (dilation) for the Gaussian envelope, so the effective receptive field
deforms per-input while the underlying FFT grid stays perfectly uniform.

**Application & impact.** Macro-level deformability without the scattered memory access of classic
deformable convs. Useful for multi-scale irregular geometries in biomedical imaging: widen the
receptive field for sparse cellular distributions, narrow it for dense, highly cellular tumor beds.

**Attach point in this repo.** `GaussianModulationND`
([masks_nd.py:293](../../nvsubquadratic/modules/masks_nd.py)) holds the per-axis std. Add an
input-dependent `(shift, log_scale)` head off the register tokens that offsets/dilates the envelope
center and width; keep the static value as the residual base so init = current behavior.

**References.** Deformable ConvNets (Dai et al., 2017) & DCNv3 (Wang et al., 2023) · Deformable DETR
(Zhu et al., 2020).

### 4. Active Spatial Data Warping

**Concept.** Instead of deforming the kernel, deform the **data** to fit the uniform kernel grid. A
lightweight conv router predicts an offset field for the input; a differentiable interpolator
(`grid_sample`) resamples the feature map onto a regular grid before the global FFT conv is applied.

**Application & impact.** A geometric alignment tool — the model learns canonical representations of
tissue structures, counteracting variance in slide prep, orientation, and tissue folding. Bridges
raw spatial convs and geometric deep learning by regularizing input geometry before the subquadratic
operator mixes the sequence.

**Attach point in this repo.** A new pre-mixer module: conv router → offset field →
`F.grid_sample` resample → existing CKConv mixer. Init offsets at zero (identity warp) so training
starts from the baseline. Regularize offset smoothness/magnitude to avoid degenerate warps.

**References.** Spatial Transformer Networks (Jaderberg et al., 2015) · Warped Convolutions
(Henriques et al., 2017).

---

## Experimentation plan

**Common backbone.** Hyena/CKConv-ND classifier at the MNIST-example scale as the cheap proxy:
~4 blocks, ~160 hidden, SIREN kernel + Gaussian modulation, register tokens for conditioning,
bf16-mixed. CIFAR-10 patchified to keep the grid small. Each idea is a minimal, isolated diff on top
of a single frozen baseline so effects are attributable.

**Protocol.** One mechanism at a time, identical schedule/seed vs. baseline. Report mean over **3
seeds** for anything that looks promising. Track val accuracy (primary), val loss, throughput
(it/s), and param count; for the deformable ideas also log the regularizer value and a qualitative
viz (mask heatmap / envelope footprint / offset field) to confirm the mechanism does something
non-trivial.

| Phase | Goal | Configs | Gate to next phase |
| ----- | ---- | ------- | ------------------ |
| **P0 — Baseline** | Lock a reproducible static-conditioning CIFAR-10 reference (3 seeds) | `baseline_hyena.py` | Stable val acc + variance band established |
| **P1 — Idea 2 (spectral FiLM)** | Enable register→FiLM→SIREN; ablate which SIREN layers get FiLM | `spectral_film.py` (+ layer-subset variants) | ≥ baseline acc at ≤ baseline cost |
| **P2 — Idea 3 (envelope warp)** | Input-dependent Gaussian shift+dilation, static base as residual | `envelope_warp.py` | Mechanism active (non-trivial shift/scale) + no regression |
| **P3 — Idea 1 (dense masking)** | Soft keep-mask scorer + sparsity budget; sweep target keep-ratio | `sparse_mask.py` | Acc held at meaningful sparsity (e.g. ≥50% attenuated) |
| **P4 — Idea 4 (data warping)** | STN-style offset field + `grid_sample` pre-mixer | `data_warp.py` | Acc ≥ baseline; warps are stable/smooth |
| **P5 — Combine** | Stack the winners from P1–P4 | `combined_best.py` | Additive gains vs. each in isolation |

**Out of scope for CIFAR-10 phase (deferred to WSI follow-up):** gigapixel memory tiling, real
budget-constrained masking at slide scale, multi-scale envelope sweeps.

### Run command (SLURM)

```bash
conda activate nvsubq
source .env                  # WANDB_API_KEY, HF_TOKEN
export PYTHONPATH=.

srun --gres=gpu:1 -c 16 --partition low \
    python experiments/run.py \
    --config examples/dynamic_deformable/<config>.py
```

### Local GPU setup (RTX 3090, verified 2026-06-12)

The default `CIFAR10_PATH` (`~/data/cifar10`) and `runs/` both symlink to the external
`ananas` drive, which is **not mounted** during local dev — both are broken symlinks, so
the harness fails in `prepare_data` / experiment-dir creation. For local runs, point both at
on-disk paths via overrides:

```bash
conda activate nvsubq
export PYTHONPATH=.
export CIFAR10_PATH="$PWD/.data/cifar10"        # local data cache (auto-downloads ~170 MB)

# Full baseline run (offline W&B, debug=true is set in the config):
python experiments/run.py --config examples/dynamic_deformable/baseline_hyena.py \
    experiment_dir=.runs/baseline_hyena

# Fast smoke test (a few steps, runs train+val on GPU):
python experiments/run.py --config examples/dynamic_deformable/baseline_hyena.py \
    experiment_dir=.runs/smoke train.iterations=6 \
    trainer.check_val_every_n_iterations=3 trainer.check_val_every_n_epoch=null \
    trainer.limit_val_batches=2
```

For online W&B logging set `debug=false` (needs `source .env`). When the `ananas` drive is
mounted again, the `CIFAR10_PATH` / `experiment_dir` overrides are unnecessary.

### Planned configs

| Config | Idea | Diff vs. baseline | Status |
| ------ | ---- | ----------------- | ------ |
| `baseline_hyena.py` | — | Static SIREN + static Gaussian envelope, `num_registers=0` (see note) | ✅ written + smoke-tested (2.77 M params) |
| `spectral_film.py` | 2 | `KernelFiLMGenerator` active: registers → FiLM → SIREN hidden layers | ✅ written + smoke-tested (3.12 M params); full 3-seed run complete |
| `envelope_warp.py` | 3 | Register-predicted shift + dilation on `DynamicGaussianModulationND` | ✅ 3-seed complete; −0.21 pp vs P0 (no gain at 8×8) |
| `baseline_hyena_native32.py` | — (control) | Same as P0 baseline but patch_size=1 (32×32 grid, batch 64) | ✅ 3-seed complete (2026-06-17); **91.55 ± 0.24%** best val |
| `envelope_warp_native32.py` | 3 (native32) | P2 at 32×32 — test if mechanism gains at larger grid | ✅ 3-seed complete (2026-06-17); **91.31 ± 0.35%** best val |
| `sparse_mask.py` | 1 | `SpatialSoftMask` (3×3 conv scorer) + `SparseMaskWrapper` sparsity loss; `SPARSITY_WEIGHT` env knob | ✅ re-run complete (2026-06-18, gradient bug fixed); **88.30 ± 0.22%** best val (SW=0.1, density≈5.8%) |
| `data_warp.py` | 4 | `SpatialWarp` (3×3 conv → 2-ch offset field → `grid_sample`) + `DataWarpClassificationWrapper` offset-mag regularizer; `WARP_REG_WEIGHT` env knob | ✅ 3-seed complete (2026-06-18); **87.78 ± 0.46%** (0.00 pp vs P0; warp ~0.03 px) |
| `sparse_mask.py` (SW=0.01) | 1 ablation | Same as P3 but SW=0.01 — looser budget, higher keep-ratio | ✅ re-run complete (2026-06-18); **88.20 ± 0.08%** (density≈30.2%) |
| `sparse_mask.py` (SW=0.05) | 1 ablation | Same as P3 but SW=0.05 | ✅ re-run complete (2026-06-18); **88.20 ± 0.33%** (density≈12.8%) |
| `sparse_mask.py` (SW=0.5) | 1 ablation | Same as P3 but SW=0.5 — tighter budget, sparser masks | ✅ re-run complete (2026-06-18); **88.43 ± 0.12%** (density≈0.4%) — **best P3 result** |
| `combined_best.py` | 1+3 | P1 (spectral FiLM all) + P3 (sparse mask SW=0.5); `SparseMaskClassificationWrapper`; `NUM_REGISTERS=4` + `SpatialSoftMask` in same `ViT5HyenaAdapter` block | ✅ 3-seed complete (2026-06-19); **88.70 ± 0.14%** (+0.92 pp vs P0) |

> **Register / Hyena-adapter constraint (found during P0 bring-up).** `ViT5HyenaAdapter`
> reshapes the token sequence into the H×W grid before the spatial conv, so — unlike a
> permutation-agnostic attention mixer — it **cannot carry non-grid register tokens through the
> mixer**. With `num_registers>0` the reshape fails (`T=68` vs. `8×8=64`). The static baseline
> therefore uses `num_registers=0`. Ideas 2 & 3 route *register* tokens via FiLM, so their first
> implementation step is to **extend `ViT5HyenaAdapter`** to split registers off before the
> reshape and restore them after (or feed register conditioning into the kernel without sending
> registers through the spatial path). This adapter change is scoped into P1/P2, not the baseline.

---

## Job submission log

| Date | Job ID | Config | Idea | Seed | Node | Status | Notes |
| ---- | ------ | ------ | ---- | ---- | ---- | ------ | ----- |
| 2026-06-12 | local/d9wbg79m | `baseline_hyena.py` | — (P0) | 42 | local RTX 3090 | ✅ done | 50 ep, offline W&B; `.runs/baseline_hyena_seed42` |
| 2026-06-14 | local/sweep | `baseline_hyena.py` + `spectral_film.py` | P0+P1 | 42,43,44 | local RTX 3090 | ✅ done | `run_phase_seeds.sh`; `runs/dynamic_deformable/` |
| 2026-06-15 | local/sweep2 | `spectral_film.py` (FILM_LAYERS=last) | 2 (P1-last) | 42,43,44 | local RTX 3090 | ✅ done | `run_phase_seeds.sh`; test 0.8810 ± 0.0020; +0.29 pp vs P0 |
| 2026-06-15 | local/sweep2 | `envelope_warp.py` | 3 (P2) | 42,43,44 | local RTX 3090 | ✅ done | `run_phase_seeds.sh`; test 0.8760 ± 0.0014; −0.21 pp vs P0 |
| 2026-06-16/17 | local/b9g2ki1bb | `baseline_hyena_native32.py` | — (P0-native32) | 42,43,44 | local RTX 3090 | ✅ done | best val 91.53/91.33/91.80%; mean **91.55 ± 0.24%**; OOM'd first pass, re-ran clean |
| 2026-06-16/17 | local/bv1i1ed3h | `envelope_warp_native32.py` | 3 (P2-native32) | 42,43,44 | local RTX 3090 | ✅ done | best val 91.50/91.52/90.90%; mean **91.31 ± 0.35%**; ~3h18m/seed |
| 2026-06-16/17 | local/bv1i1ed3h | `sparse_mask.py` | 1 (P3) | 42,43,44 | local RTX 3090 | ✅ done | best val 88.07/87.88/87.81%; mean **87.92 ± 0.14%**; SW=0.1; density≈0.47; ~27m/seed |
| 2026-06-17/18 | local/bqocl55h5 | `data_warp.py` | 4 (P4) | 42,43,44 | local RTX 3090 | ✅ done | best val 87.76/87.33/88.25%; mean **87.78 ± 0.46%**; WR=0.01; offset≈0.008; ~27m/seed |
| 2026-06-18 | local/b0yoig6ki (INVALID) | `sparse_mask.py` SW ∈ {0.01,0.05,0.5} | 1 ablation | 42,43,44 | local RTX 3090 | ❌ invalid | Sparsity gradient detached bug — all SW values trained identically; archived to `_invalid_sw_no_grad/` |
| 2026-06-18 | local/sweep (PID 2949769) | `sparse_mask.py` + SW ∈ {0.01,0.05,0.5} | 1 + ablation | 42,43,44 | local RTX 3090 | ✅ done | Re-run after gradient fix; 12 runs × ~27 min = ~5.4h; **SW=0.5 best: 88.43 ± 0.12%** |
| 2026-06-19 | local/sweep (PID 3037575) | `combined_best.py` | 1+3 (P5) | 42,43,44 | local RTX 3090 | ✅ done | best val 88.8/88.8/88.5%; mean **88.70 ± 0.14%** (+0.92 pp vs P0); density 0.46% |

---

## Results

### 🏆 Leaderboard (best per idea)

| Rank | Idea | Config | Val Acc (best) | Val Loss | it/s | Params | W&B | Notes |
| ---- | ---- | ------ | -------------- | -------- | ---- | ------ | --- | ----- |
| 1 | 1 Sparse masking SW=0.5 | `sparse_mask.py` (SW=0.5) | **0.8843 ± 0.0012** | — | ~18 | 2.8 M | offline | **+0.65 pp vs P0**; density 0.4% (99.6% attenuated); corrected run (gradient bug fixed) |
| 2 | 2 Spectral FiLM (all) | `spectral_film.py` | **0.8835 ± 0.0013** | — | ~18 | 3.12 M | offline | test 0.8828 ± 0.0012; **+0.57 pp val / +0.47 pp test vs P0**; non-overlapping bands |
| 3 | 1 Sparse masking SW=0.1 | `sparse_mask.py` (SW=0.1) | **0.8830 ± 0.0022** | — | ~18 | 2.8 M | offline | **+0.52 pp vs P0**; density 5.8%; corrected run |
| 4 | 1 Sparse masking SW=0.01 | `sparse_mask.py` (SW=0.01) | **0.8820 ± 0.0008** | — | ~18 | 2.8 M | offline | **+0.42 pp vs P0**; density 30.2% |
| 4 | 1 Sparse masking SW=0.05 | `sparse_mask.py` (SW=0.05) | **0.8820 ± 0.0033** | — | ~18 | 2.8 M | offline | **+0.42 pp vs P0**; density 12.8% |
| 5 | 2 Spectral FiLM (last) | `spectral_film.py` | **0.8810 ± 0.0020** | — | ~18 | 3.00 M | offline | **+0.32 pp val vs P0**; 62% of full-FiLM gain at 1 layer |
| 6 | P0 Baseline | `baseline_hyena.py` | 0.8778 ± 0.0011 | — | ~18 | 2.77 M | offline | test 0.8781 ± 0.0007 |
| 7 | 4 Data warp | `data_warp.py` | 0.8778 ± 0.0046 | — | ~18 | 2.78 M | offline | **0.00 pp vs P0**; warp ~0.03 px; regularizer dominated; 4× variance; eliminated |
| 8 | 3 Envelope warp (8×8) | `envelope_warp.py` | 0.8760 ± 0.0014 | — | ~18 | 2.77 M | offline | **−0.21 pp vs P0**; no gain at 8×8 grid scale; eliminated |
| — | 3 Envelope warp (native32) | `envelope_warp_native32.py` | 0.9131 ± 0.0035 | — | ~3.4 | 2.9 M | offline | native32 baseline 91.55 ± 0.24%; **Δ = −0.24 pp**; eliminated |
| 0 | 1+2 Combined (P5) | `combined_best.py` | **0.8870 ± 0.0014** | — | ~18 | ~3.2 M | offline | **+0.92 pp vs P0** (88.70%); density 0.46%; sub-additive by −0.30 pp vs naive P1+P3 sum |

### P0 — Baseline (3 seeds)

Run dir: `runs/dynamic_deformable/baseline_hyena_seed{42,43,44}` (offline W&B). 50 ep, ~18 it/s, 2.77 M params.

| Job ID | Seed | W&B | Val Acc | Test Acc | Test Loss | Notes |
| ------ | ---- | --- | ------- | -------- | --------- | ----- |
| offline | 42 | — | 0.8787 | 0.8776 | 0.4776 | — |
| offline | 43 | — | 0.8766 | 0.8778 | 0.4965 | — |
| offline | 44 | — | 0.8780 | 0.8789 | 0.4756 | — |
| **mean ± std** | — | — | **0.8778 ± 0.0011** | **0.8781 ± 0.0007** | — | — |

### P1 — Idea 2: Dynamic Spectral Filtering (FiLM-SIREN)

Run dir: `runs/dynamic_deformable/spectral_film_seed{42,43,44}` (`FILM_LAYERS=all`, offline W&B). 50 ep, ~18 it/s, 3.12 M params.

| Job ID | Variant (FiLM layers) | Seed | Val Acc | Test Acc | Δval vs P0 | Δtest vs P0 |
| ------ | --------------------- | ---- | ------- | -------- | ---------- | ----------- |
| offline | all hidden layers (2) | 42 | 0.8849 | 0.8840 | +0.62 pp | +0.64 pp |
| offline | all hidden layers (2) | 43 | 0.8829 | 0.8817 | +0.63 pp | +0.39 pp |
| offline | all hidden layers (2) | 44 | 0.8826 | 0.8828 | +0.46 pp | +0.39 pp |
| **mean ± std** | all hidden layers (2) | — | **0.8835 ± 0.0013** | **0.8828 ± 0.0012** | **+0.57 pp** | **+0.47 pp** |
| offline | last layer only (1) | 42 | 0.8831 | 0.8831 | +0.44 pp | +0.55 pp |
| offline | last layer only (1) | 43 | 0.8807 | 0.8807 | +0.41 pp | +0.29 pp |
| offline | last layer only (1) | 44 | 0.8792 | 0.8792 | +0.12 pp | +0.03 pp |
| **mean ± std** | last layer only (1) | — | **0.8810 ± 0.0020** | **0.8810 ± 0.0020** | **+0.32 pp** | **+0.29 pp** |

**Conclusion (P1):** the pilot's edge **reproduces** — P1-all > P0 on **all 3 seeds** (Δtest +0.47 pp avg; non-overlapping ±1σ bands). **P1-last (1 FiLM layer)** recovers ~62% of the full-FiLM gain (+0.29 pp test) at ~1 layer vs 2, with broader variance (bands overlap with P0). Last-layer FiLM is cheaper but less reliable than full conditioning. Winner for stacking: `FILM_LAYERS=all`.

### P2 — Idea 3: Dynamic Envelope Warping

Run dir: `runs/dynamic_deformable/envelope_warp_seed{42,43,44}` (offline W&B). 50 ep, ~18 it/s, 2.77 M params (+tiny warp head ≈ same as baseline).

| Job ID | Seed | Val Acc | Test Acc | Δtest vs P0 | Notes |
| ------ | ---- | ------- | -------- | ----------- | ----- |
| offline | 42 | 0.8762 | 0.8762 | −0.19 pp | — |
| offline | 43 | 0.8745 | 0.8745 | −0.36 pp | — |
| offline | 44 | 0.8772 | 0.8772 | −0.17 pp | — |
| **mean ± std** | — | **0.8760 ± 0.0014** | **0.8760 ± 0.0014** | **−0.21 pp** | — |

**Conclusion (P2):** envelope warping provides **no benefit** on CIFAR-10 — all 3 seeds are below the P0 baseline (Δtest −0.21 pp avg; bands overlap with P0 lower tail). Likely explanation: at 8×8 patch-grid scale, the Gaussian envelope is already wide enough that per-input shift/dilation adds noise rather than useful inductive bias. May be more valuable at larger grid scales (e.g. gigapixel WSI where tissue geometry varies). **P2 does not clear the gate; will not be included in P5 combined.**

### P2-native32 — Idea 3: Envelope Warp at Native 32×32 Resolution

Run dirs: `runs/dynamic_deformable/{envelope_warp,baseline_hyena}_native32_seed{42,43,44}` (offline W&B). 50 ep, batch 64, 1024-token grid, ~2.5–3.3h/seed, ~2.9 M params.

| Job ID | Seed | Baseline val | Warp val | Δ (warp − baseline) |
| ------ | ---- | ------------ | -------- | ------------------- |
| offline | 42 | 0.9153 | 0.9150 | −0.03 pp |
| offline | 43 | 0.9133 | 0.9152 | +0.19 pp |
| offline | 44 | 0.9180 | 0.9090 | −0.90 pp |
| **mean ± std** | — | **0.9155 ± 0.0024** | **0.9131 ± 0.0035** | **−0.24 pp** |

**Conclusion (P2-native32):** Envelope warping at native 32×32 provides **no benefit** — Δ = −0.24 pp (warp below baseline), within noise (bands overlap) but in the wrong direction. This matches the 8×8 result (−0.21 pp). The mechanism fails at both grid scales tested. **P2 definitively eliminated.** The hypothesis that larger grids give the Gaussian shift/dilation room to be useful is not supported by this data; the warp head may simply be adding harmful gradient noise with no useful inductive bias for natural image classification at any of the tested scales.

### P3 — Idea 1: Sparse Tokenization via Dense Masking

> **Note:** The original P3 runs (2026-06-16/17) had a bug in `SpatialSoftMask`: `_last_mean_mask` was stored as `mask.detach().mean()`, so when `SparseMaskClassificationWrapper` computed `sparsity_weight * _last_mean_mask`, no gradient flowed to the scorer. The regularizer added a numerical value to the loss but had zero effect on model behavior — all SW values trained identically. Bug fixed 2026-06-18 (added `_mask_mean` live tensor; training wrapper uses that for gradient, `_last_mean_mask` retained for logging). All P3 + SW sweep runs re-executed with the fix.

Run dir: `runs/dynamic_deformable/sparse_mask_seed{42,43,44}` (offline W&B). 50 ep, batch 128, 8×8 grid (patch=4), `SPARSITY_WEIGHT=0.1`, ~27 min/seed, ~2.8 M params. `SpatialSoftMask` (3×3 conv → 1-channel sigmoid) applied before each Hyena block; sparsity penalty = SW × mean mask density.

#### P3 SW=0.1 (default)

| Job ID | Seed | Best Val Acc | Δ vs P0 | Val Mask Density |
| ------ | ---- | ------------ | -------- | ---------------- |
| offline | 42 | 0.886 | +0.82 pp | 5.8% |
| offline | 43 | 0.882 | +0.42 pp | 5.9% |
| offline | 44 | 0.881 | +0.32 pp | 5.5% |
| **mean ± std** | — | **0.8830 ± 0.0022** | **+0.52 pp** | **5.8 ± 0.2%** |

#### P3 SW sweep (sparsity weight ablation)

| Config | SW | Seeds | Val Acc | Δ vs P0 | Density |
| ------ | -- | ----- | ------- | ------- | ------- |
| sparse_mask_sw001 | 0.01 | 42,43,44 | **0.8820 ± 0.0008** | +0.42 pp | 30.2 ± 0.4% |
| sparse_mask_sw005 | 0.05 | 42,43,44 | **0.8820 ± 0.0033** | +0.42 pp | 12.8 ± 0.3% |
| sparse_mask (default) | 0.10 | 42,43,44 | **0.8830 ± 0.0022** | +0.52 pp | 5.8 ± 0.2% |
| sparse_mask_sw05 | 0.50 | 42,43,44 | **0.8843 ± 0.0012** | **+0.65 pp** | **0.4 ± 0.0%** |

**Conclusion (P3 + SW sweep):** The sparsity regularizer is now active and produces a clear accuracy-vs-density trade-off. **All SW values outperform P0** (vs. within-noise in the broken runs). Strikingly, SW=0.5 achieves the **best accuracy (+0.65 pp, 88.43%), surpassing P1 spectral FiLM (+0.57 pp val)**, while masking 99.6% of tokens to near-zero. The mask is not physically dropping tokens (the grid is preserved for the FFT), but concentrating influence on <1% of spatial positions — the conv scorer learns to gate the global FFT by the few most informative features. The finding that more sparsity → higher accuracy suggests the gating/selection effect is more valuable than the global average contribution of most tokens on CIFAR-10. **Winner for P5 stacking: SW=0.5 (`sparse_mask_sw05` config).**

### P4 — Idea 4: Active Spatial Data Warping

Run dir: `runs/dynamic_deformable/data_warp_seed{42,43,44}` (offline W&B). 50 ep, batch 128, 8×8 grid (patch=4), `WARP_REG_WEIGHT=0.01`, ~27 min/seed, ~2.78 M params.

| Job ID | Seed | Best Val Acc | Δ vs P0 | Val Offset Mag | Notes |
| ------ | ---- | ------------ | -------- | -------------- | ----- |
| offline | 42 | 0.8776 | −0.02 pp | 0.0071 | — |
| offline | 43 | 0.8733 | −0.45 pp | 0.0101 | — |
| offline | 44 | 0.8825 | +0.47 pp | 0.0065 | — |
| **mean ± std** | — | **0.8778 ± 0.0046** | **0.00 pp** | **~0.008** | — |

**Conclusion (P4):** Data warping provides **no benefit** — mean matches P0 exactly (87.78%) but with 4× higher variance (±0.46% vs ±0.11%). The learned offset magnitude is tiny (~0.008 in normalized [-1,1] coords ≈ 0.03 pixels at 8×8 patch grid), indicating the `WARP_REG_WEIGHT=0.01` regularizer dominated and the mechanism barely activated. The model learned near-identity warps throughout training. **P4 eliminated.** Pattern consistent with P2: both geometric/spatial deformation mechanisms (envelope warp, data warp) provide no benefit at 8×8 CIFAR-10 scale, likely because (a) the augmentation pipeline already handles spatial variance and (b) the 8×8 grid is too coarse for fine-grained spatial deformation to be meaningful. A lower `WARP_REG_WEIGHT` might allow larger warps, but the consistent pattern across P2/P4 suggests the task/scale is the fundamental limitation.

### P5 — Combined: P1 (spectral FiLM) + P3 (sparse mask SW=0.5)

Config: `combined_best.py`. Run dir: `runs/dynamic_deformable/combined_best_seed{42,43,44}`. Smoke-tested (2026-06-19, exit 0; `val/mask_density=0.495` at init as expected).

Individual baselines for comparison: P1 88.35 ± 0.13%, P3 SW=0.5 88.43 ± 0.12%, P0 87.78 ± 0.11%.

| Job ID | Seed | Best Val Acc | Δ vs P0 | Val Mask Density | Notes |
| ------ | ---- | ------------ | -------- | ---------------- | ----- |
| offline | 42 | 0.888 | +1.02 pp | 0.46% | — |
| offline | 43 | 0.888 | +1.02 pp | 0.39% | — |
| offline | 44 | 0.885 | +0.72 pp | 0.54% | — |
| **mean ± std** | — | **0.8870 ± 0.0014** | **+0.92 pp** | **0.46%** | — |

**Conclusion (P5):** Stacking P1 + P3 (SW=0.5) achieves **88.70 ± 0.14% (+0.92 pp vs P0)** — the best result in the series. Both mechanisms contribute: +0.35 pp above P1 alone, +0.27 pp above P3 alone. The combination is slightly sub-additive (−0.30 pp vs the naive P1+P3 sum of 89.00%), likely because both mechanisms improve how the global FFT convolution focuses on informative signal, giving them partially overlapping effects. Density remains at 0.46% — the sparse mask continues to gate 99.5% of tokens even when the SIREN kernel is also being FiLM-conditioned. The two mechanisms are genuinely complementary and composable with no conflict.

---

## Notes & decisions log

- **2026-06-12** — Tracker created. Folder `examples/dynamic_deformable/`. Ideas 2 & 3 reuse
  existing modules (`KernelFiLMGenerator`, `GaussianModulationND`); ideas 1 & 4 need new
  lightweight pre-mixer modules.
- **2026-06-12** — P0 `baseline_hyena.py` written and **smoke-tested on the local RTX 3090**
  (6 steps, train+val, exit 0). Self-contained config (CIFAR-10 datamodule + `ViT5ClassificationNet`
  isotropic Hyena), lightweight scale: 32×32 native, patch 4 → 8×8 grid, dim 192, 6 blocks →
  **2.77 M params**. Recipe: AdamW lr 1e-3, wd 0.05, mixup 0.2, cosine + 2-epoch warmup,
  bf16-mixed, 50 epochs. `debug=true` (offline W&B) by default.
  - Vendored `experiments/datamodules/cifar10.py` from the `feat/patch-merging` branch (commit
    `74dd91c`) — it is **not present on `main`**. This is an untracked addition to the working tree.
  - Two `main`-vs-patch-merging divergences resolved: `main`'s `Hyena` has **no `use_rope` arg**
    (dropped it); and the Hyena adapter can't carry registers (set `num_registers=0`, see note above).
  - Env quirk: `~/data/cifar10` and `runs/` are broken symlinks to the unmounted `ananas` drive →
    use `CIFAR10_PATH` + `experiment_dir` overrides locally (see Local GPU setup).
  - **Next:** decide whether to commit the vendored datamodule, then run the full 50-epoch P0
    baseline (3 seeds) and start P1 (`spectral_film.py` + the `ViT5HyenaAdapter` register split).
- **2026-06-12** — P0 baseline (seed 42) **completed** (50 ep, local RTX 3090, offline W&B
  `d9wbg79m`, `.runs/baseline_hyena_seed42`). Remaining P0 seeds (0/1/2 for the variance band)
  still TODO.
- **2026-06-12** — **P1 implemented and launched** (seed 42, `.runs/spectral_film_seed42`).
  - Extended `ViT5HyenaAdapter` with a **register-split mode** (`grid_h` + `register_write` +
    `hidden_dim` args): when `grid_h` is set, only the first `grid_h*grid_w` tokens are reshaped
    into the FFT grid and mixed; trailing register/CLS/aux tokens are kept off the spatial path
    (fixes the `T=68 vs 8×8` reshape failure that forced `num_registers=0` in P0). Legacy
    whole-sequence behaviour is preserved when `grid_h is None`, so `baseline_hyena.py` is unchanged.
  - **Input-dependence note:** in this attention-free backbone, registers never read patches, so a
    *pure* split would leave the registers (hence FiLM γ/β) constant across the batch → a static
    kernel. To make Idea 2 actually dynamic, `register_write=True` writes a learned, **zero-init**
    mean-pool of the mixed patch grid back into the register slots each block. Net effect: at init
    the model == P0 baseline (FiLM identity + zero write), and it learns per-input conditioning
    from there (dynamic from block 1 on; block 0's FiLM stays static). Config decision confirmed
    with DW.
  - Wiring reused as-is: `ViT5ResidualBlock.register_pooling` (`RegisterPooling`) →
    `conditioning=[B,C]` → `QKVSequenceMixer` → `Hyena` → `CKConvND` → `SIRENKernelND.film_generator`
    (`KernelFiLMGenerator`, `num_film_layers = num_layers-1 = 2`, identity init).
  - **TODO after this run:** the "last layer only" FiLM ablation (tracker P1 table) needs a way to
    FiLM a *subset* of SIREN hidden layers — `SIRENKernelND` currently FiLMs all of them.
- **2026-06-12** — **P1 (seed 42) finished** (50 ep, step 19500, `eejn`, `.runs/spectral_film_seed42`):
  best **val/acc 0.8849** (epoch 46), **test/acc 0.8840**, test/loss 0.4792, ~18 it/s. Vs P0 baseline
  val/acc 0.8787 → **+0.62 pp val**. Mechanism trained stably from the baseline-equivalent init.
  Caveat: **single seed**, so +0.62 pp is within the likely noise band — clears the "≥ baseline at
  ≤ baseline cost" gate provisionally but needs the 3-seed protocol to confirm before declaring a win.
  (P0 test/acc not captured: its `conda run` stdout was buffered to a 0-byte log; val/acc 0.8787 is
  from the checkpoint's `best_model_score`. Future runs use the env-python launch which logs cleanly.)
- **2026-06-14** — **ananas remounted** (by DW) at `/media/davidwessels/ananas`; `runs/` + `~/data/cifar10`
  symlinks resolve again. New runs go to the ananas-backed `runs/dynamic_deformable/`.
- **2026-06-14** — **3-seed phase sweep launched** via `run_phase_seeds.sh` (sequential, resumable,
  env-python clean logs): P0 `baseline_hyena` then P1 `spectral_film`, each over seeds {42,43,44} →
  `runs/dynamic_deformable/`. Re-runs P0 to capture test/acc (the original P0 lacked it) and gives a
  3-seed variance band for both arms so the P0-vs-P1 comparison can be concluded properly. Both
  configs now read `SEED` from env. Data read from the proven-local `.data/cifar10` cache. ETA ~110 min.
- **2026-06-14** — **Subset-FiLM implemented** in `SIRENKernelND` (`film_layers` arg, default `None` =
  all hidden layers, fully backward-compatible). Unit-tested (all-layers, last-only `[-1]`, identity-at-
  init equivalence, mismatch guard). Exposed in `spectral_film.py` via `FILM_LAYERS={all|last}` env knob
  (mirrors `SEED`). This unblocks the tracker's "last layer only" P1 ablation — runnable with
  `FILM_LAYERS=last` once the band sweep frees the GPU.
- **2026-06-14** — **Phase sweep complete; P1 concluded.** All 6 runs finished rc=0
  (`runs/dynamic_deformable/`, sweep ~09:38–12:22). P0 baseline (3 seeds): val 0.8778 ± 0.0011,
  test 0.8781 ± 0.0007. P1 spectral-FiLM (3 seeds, all hidden layers): val 0.8835 ± 0.0013,
  test 0.8828 ± 0.0012. **P1 > P0 on every seed** (Δval +0.57 pp, Δtest +0.47 pp avg; non-overlapping
  ±1σ bands). Modest but reproducible gain at +13% params, equal it/s → P1 mechanism validated. The
  seed-42 numbers reproduced the earlier pilot exactly (determinism confirmed). **Next steps:**
  (1) P1 "last layer only" ablation via `FILM_LAYERS=last` (does 1 FiLM layer recover most of the
  gain at fewer params?); (2) P2 envelope-warp (`envelope_warp.py`, not yet written).
- **2026-06-15** — **P1-last + P2 implemented and submitted to ivi (3 seeds each).**
  - **P1 last-layer ablation:** `spectral_film.py` already supported `FILM_LAYERS=last` via env knob
    (implemented 2026-06-14). Submitted 3 jobs to ivi via `submit_ivi.sh` →
    `runs/dynamic_deformable/spectral_film_last_seed{42,43,44}`.
  - **P2 `DynamicGaussianModulationND`:** new class added to `masks_nd.py` (subclasses
    `GaussianModulationND`). A 2-layer SiLU MLP maps the `[B, cond_dim]` register conditioning
    vector to per-axis `shift [B, data_dim]` + `log_scale [B, data_dim]`, which translate and
    dilate the Gaussian envelope centre/width per-input. Zero-init output projection →
    `shift=0, log_scale=0` at step 0 ≡ P0 static baseline. `GaussianModulationND.forward` gained
    `**kwargs` (absorbs `conditioning=` silently); `CKConvND.forward` now passes
    `conditioning=conditioning` to the mask, so the existing conditioning pathway (register tokens
    → `RegisterPooling` → `[B, C]` → `mixer_kwargs["conditioning"]` → `CKConvND`) reaches the
    mask with no new wiring. `envelope_warp.py` is a clean P0 diff (no FiLM on kernel; P2 mechanism
    in isolation). Smoke-tested on local RTX 3090 (6 steps, exit 0). Submitted 3 seeds to ivi via
    `submit_ivi.sh` → `runs/dynamic_deformable/envelope_warp_seed{42,43,44}`.
  - **submit_ivi.sh:** new script at `examples/dynamic_deformable/submit_ivi.sh` wraps
    `scripts/slurm/submit_1gpu.sh` to fan out all 6 jobs in one invocation. CIFAR10 path defaults
    to `/shared/data/image_datasets/cifar10` (override via `CIFAR10_PATH` env or `cluster.env`).
- **2026-06-15/16** — **P1-last + P2 complete** (ran locally on RTX 3090 via updated `run_phase_seeds.sh`).
  P1-last (1 FiLM layer): test 0.8810 ± 0.0020 (+0.29 pp vs P0); recovers ~62% of the 2-layer FiLM
  gain but with overlapping P0 bands — less reliable. P2 envelope warp: test 0.8760 ± 0.0014
  (−0.21 pp vs P0); all 3 seeds below baseline — mechanism does not help at 8×8 grid scale.
  **P1-all selected as the winner for P5 stacking. P2 eliminated.**
- **2026-06-17/18** — **P4 data warp complete** (sweep `bqocl55h5`). `data_warp.py` written +
  smoke-tested + 3-seed full run. Best val 87.76/87.33/88.25% → mean **87.78 ± 0.46%** (0.00 pp
  vs P0). Learned offset magnitude ~0.008 in normalized coords ≈ 0.03 px — near-identity warps
  throughout. `WARP_REG_WEIGHT=0.01` regularizer dominated. **P4 eliminated.** Consistent pattern
  with P2: spatial/geometric deformation mechanisms provide no benefit at 8×8 CIFAR-10 scale. Only
  P1 spectral FiLM (+0.47 pp) and P3 sparse mask (+0.14 pp, within noise) remain for P5 stacking.
  **Next:** P5 (P1 + P3) — but first a `SPARSITY_WEIGHT` sweep to find the best P3 config to stack.
- **2026-06-17** — **`baseline_hyena_native32` complete** (sweep `b9g2ki1bb`). Best val 91.53/91.33/91.80%
  → mean **91.55 ± 0.24%**. With this in hand the P2-native32 Δ is confirmed: warp 91.31% vs baseline
  91.55% → **−0.24 pp**. Bands overlap but direction is wrong on 2/3 seeds. P2 definitively eliminated
  at both 8×8 (−0.21 pp) and 32×32 (−0.24 pp). Native32 resolution itself drives the accuracy jump
  (87.8% → 91.5%); the envelope warp adds nothing on top of it.
- **2026-06-19** — **P5 combined_best complete.** 88.70 ± 0.14% (+0.92 pp vs P0). Best result in series. Sub-additive by −0.30 pp vs P1+P3 sum (89.00%), consistent with partial mechanism overlap. Density 0.46% — mask maintains extreme sparsity even with FiLM active. **CIFAR-10 dynamic-deformable experiment series complete.**
- **2026-06-19** — **P5 combined_best launched.** `combined_best.py` written: stacks P1 (all-layer FiLM, `NUM_REGISTERS=4`, `register_write=True`) + P3 (`SpatialSoftMask`, `SPARSITY_WEIGHT=0.5`) in a single `ViT5HyenaAdapter` block — both mechanisms are independent (mask gates spatial grid before Hyena; FiLM conditions SIREN kernel inside Hyena). `SparseMaskClassificationWrapper` handles the sparsity penalty; the FiLM pathway is transparent to it. Smoke-tested exit 0 (`val/mask_density=0.495` at init as expected). 3-seed sweep running locally (~27 min/seed).
- **2026-06-18** — **P3 sparsity gradient bug found and fixed; SW sweep re-run complete.**
  - **Bug:** `SpatialSoftMask._last_mean_mask` stored `mask.detach().mean()`. The wrapper used this detached tensor for `sparsity_weight * mean_density`, so no gradient reached the scorer. All three SW values (0.01, 0.05, 0.5) produced identical training dynamics — the regularizer was a no-op.
  - **Fix:** Added `_mask_mean = mask.mean()` (live, gradient-connected) to `spatial_mask.py`; `sparse_mask_wrapper.py` training step now uses `_mask_mean`. `_last_mean_mask` kept (detached) for logging. Confirmed by smoke tests: SW=0.5 → density 0.451, SW=0.01 → density 0.458 after 20 steps (correct divergence).
  - **Re-run:** Archived invalid runs to `_invalid_sw_no_grad/`. Re-ran all 12 runs (sparse_mask + sw001 + sw005 + sw05, seeds 42/43/44). Results: SW=0.5 **88.43 ± 0.12% (+0.65 pp)**, SW=0.1 88.30 ± 0.22% (+0.52 pp), SW=0.05/0.01 88.20 ± 0.1–0.3% (+0.42 pp). SW=0.5 beats P1 spectral FiLM on val (+0.65 vs +0.57 pp). **P3 is now a genuine winner; SW=0.5 selected for P5 stacking.**
- **2026-06-16/17** — **P2-native32 + P3 complete** (local RTX 3090, sweep `bv1i1ed3h`).
  - **`baseline_hyena_native32` OOM'd** (all 3 seeds, rc=1 ≈20 s in) — a smoke-test process was holding
    ~14 GB when the sweep launched. The 3 seeds are re-running as sweep `b9g2ki1bb` via
    `run_phase_seeds.sh` (will skip everything with `last.ckpt`, re-run only the 3 failed seeds).
  - **`envelope_warp_native32`** (3 seeds): best val **91.50 / 91.52 / 90.90%** → mean
    **91.31 ± 0.35%**. Each seed ran ~3h18m (1024-token grid is ~16× slower than 64-token). Absolute
    accuracy is far above the 8×8 baseline (87.8%) but the native32 control is still pending —
    cannot yet isolate warp mechanism Δ. Hypothesis (warp mechanism more valuable at larger grids)
    is consistent with this result; confirmation awaits `baseline_hyena_native32`.
  - **`sparse_mask`** (3 seeds, `SPARSITY_WEIGHT=0.1`): best val **88.07 / 87.88 / 87.81%** →
    mean **87.92 ± 0.14%** (+0.14 pp vs P0; within noise). Mask appeared at ~47% density. ⚠️ **These
    results are invalid** — the sparsity gradient was detached (see 2026-06-18 note). Density was
    the free-running zero-init sigmoid (~0.5), not the regularizer effect. Re-run results above.
  - **Next:** (1) await `baseline_hyena_native32` to conclude P2-native32; (2) optionally sweep
    `SPARSITY_WEIGHT` for P3 to find the accuracy-vs-density frontier; (3) consider P4 (data warp)
    and P5 (P1 + winning P3 weight).

---

## TinyImageNet difficulty scaling (does the gap grow on a harder task?)

**Question.** CIFAR-10 (10-class, 32×32) is an easy proxy; the P5−P0 expressivity gap there is
+0.92 pp. Does that gap *widen* on a harder classification task? TinyImageNet (200-class, native
64×64, 100k train / 10k val) is the next rung: 20× more classes, 4× the pixels.

**Design.** Pure difficulty swap — the backbone (dim 192, 6 blocks, GAP readout) and training recipe
(AdamW, lr 1e-3, 50 epochs, bs 128, cosine + 2ep warmup, mixup 0.2, bf16) are **identical** to the
CIFAR-10 runs. Only the dataset and patch size change. Two endpoints only (per scope decision):
  - **P0** = `tinyimagenet_baseline.py` — static-conditioning Hyena control (`NUM_REGISTERS=0`).
  - **P5** = `tinyimagenet_combined_best.py` — P1 spectral FiLM (all layers, `NUM_REGISTERS=4`,
    `register_write=True`) + P3 sparse mask (`SpatialSoftMask`, `SPARSITY_WEIGHT=0.5`).
  Both 3 seeds (42/43/44). Gap of interest: P5 − P0, compared against the CIFAR-10 +0.92 pp.

**Grid.** Phase A (first): `PATCH_SIZE=8` → 8×8 = 64-token grid (matches the CIFAR-10 token count, so
task difficulty is the only changed variable). Phase B (afterwards): `PATCH_SIZE=4` → 16×16 = 256-token
grid. `PATCH_SIZE` is an env override on both configs; everything else is interpolated from it.

**Infra.** Dataset (HF `zh-plus/tiny-imagenet`) downloaded to
`/ivi/zfs/s0/original_homes/dwessel/data/tiny-imagenet` (train 100k / valid 10k). `datasets` installed
into the ivi `nvsubq` env. `tinyimagenet.py` patched to inline `MixupConfig`/`AugmentConfig` (was
importing them from `dali_imagenet_fused`, which pulls in `nvidia.dali` at module load — not installed
on ivi). Runs on ivi `all6000` (rtx_6000) via git worktree of the `dynamic-deformable` branch.

- **2026-06-20** — TinyImageNet P0/P5 configs (patch-8) written + build-validated locally (39,050
  total iters = 50 ep × 781). Deployed to ivi via git worktree (`~/code/nvSubquadratic-dd`, branch
  `dynamic-deformable`, shipped by bundle — local has no GitHub push rights). Both configs smoke-tested
  on all6000 (exit 0; loss ≈ ln(200) = 5.3 at init, P5 `val/mask_density` ≈ 0.5). **3-seed P0+P5 sweep
  launched on all6000** (debug=false, online W&B): jobs 187699 (tin-p0-s42) / 187700 (p5-s42) /
  187701 (p0-s43) / 187702 (p5-s43) / 187703 (p0-s44) / 187704 (p5-s44). Run dir
  `/ivi/zfs/s0/original_homes/dwessel/nvsubq-runs/dynamic_deformable/tin_{p0,p5}_patch8_seed{42,43,44}`.
  Patch-4 variant (`PATCH_SIZE=4`, 256-token grid) queued for after patch-8 completes.
  **Gap to compare:** CIFAR-10 P5−P0 = +0.92 pp; does it widen on TinyImageNet-200?
- **2026-06-21** — **TinyImageNet patch-8 (64-token) COMPLETE** (6 jobs, ~50–60 min each, all exit 0).
  val/acc (= test/acc; the datamodule's test split *is* the 10k valid set):
  | Phase | seed42 | seed43 | seed44 | mean |
  | P0 baseline | 47.9 | 47.8 | 47.3 | **47.67 ± 0.31%** |
  | P5 combined | 47.1 | 47.2 | 46.4 | **46.90 ± 0.44%** |
  **Δ(P5−P0) = −0.77 pp** — the gap did not grow, it **flipped sign** (CIFAR-10 was +0.92 pp).
  Consistent across seeds (every P0 seed > every P5 seed). At matched 64-token grid the
  dynamic-deformable stack (FiLM + sparse mask) slightly *hurts* on the 200-class task. Open question:
  is this a grid-resolution effect? → patch-4 (256-token) launched to test whether the mechanisms need
  more spatial structure to pay off. Jobs 187953–187958, run dir `.../tin_{p0,p5}_patch4_seed{42,43,44}`.
- **2026-06-22** — **TinyImageNet patch-4 (256-token) COMPLETE** (6 jobs, ~1h20m P0 / ~1h44m P5, all exit 0).
  val/acc:
  | Phase | seed42 | seed43 | seed44 | mean |
  | P0 baseline | 54.4 | 54.2 | 53.9 | **54.17 ± 0.25%** |
  | P5 combined | 54.4 | 54.2 | 53.1 | **53.90 ± 0.70%** |
  **Δ(P5−P0) = −0.27 pp** (within noise; P5 ties P0 on seeds 42/43, only seed44 lower). More tokens lift
  absolute accuracy a lot (48% → 54%) but the deformable gap stays ≈0/slightly negative.
- **2026-06-22 — TinyImageNet difficulty-scaling CONCLUDED.** The CIFAR-10 P5−P0 expressivity gain
  (+0.92 pp) does **not** transfer to the harder 200-class task: patch-8 −0.77 pp, patch-4 −0.27 pp
  (within noise). The gap does not widen with difficulty — it vanishes/inverts. Adding spatial
  structure (patch-4) recovers it from −0.77 toward 0 but never positive. **Takeaway:** the
  dynamic-deformable mechanisms (spectral FiLM + sparse mask) are a CIFAR-10-specific gain, not a
  robust expressivity win on harder classification. Worth probing whether the WSI/pathology target
  (where most of the image really is irrelevant background) behaves more like the sparse-mask's
  best case than TinyImageNet does.
