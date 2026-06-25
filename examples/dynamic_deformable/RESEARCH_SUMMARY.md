# Dynamic / Deformable Conditioning for the Subquadratic FFT Mixer — Research Summary

**TL;DR.** We tested four ways to make the subquadratic FFT-convolution mixer *input-dependent*
without breaking its uniform grid. On CIFAR-10, spectral (FiLM) and sparse-masking conditioning each
help (+0.6 pp), stack to **+0.92 pp**; geometric deformation (envelope/data warp) does nothing. On
the harder TinyImageNet-200 the gains **do not transfer** — the gap vanishes/inverts. The CIFAR-10
win is task-specific, not a robust expressivity gain.

## Motivation

The mixer replaces attention with a dense `O(L log L)` global FFT convolution whose kernel is an
implicit SIREN field, shaped by a Gaussian envelope and (optionally) conditioned by register tokens
via FiLM. Today that conditioning is largely **static** (fixed frequency init, fixed envelope width,
fixed token grid). Question: *how much expressivity can we recover by making the operator
input-dependent without ever flattening/scattering the FFT grid?* Downstream motivation: gigapixel
pathology (WSI), where most of the image is irrelevant background.

## What we tried (each touches a different pipeline stage: input → kernel → envelope → geometry)

| # | Mechanism | Where it acts | How |
|---|-----------|---------------|-----|
| **P1** | **Dynamic Spectral Filtering** | kernel | Registers (pooled per-block from a patch summary) drive a FiLM generator that modulates (γ,β) the SIREN hidden layers → an input-dependent, linear time-varying frequency filter. |
| **P2** | **Dynamic Envelope Warping** | envelope | An MLP predicts per-axis shift + dilation of the Gaussian envelope from register conditioning (zero-init ⇒ static at step 0). |
| **P3** | **Sparse Tokenization via Dense Masking** | input | A lightweight 3×3 conv scorer predicts a soft spatial keep-mask applied to the patch grid *before* the FFT; a sparsity penalty drives density toward ~0.5%. Grid stays full; uninformative regions are attenuated, not removed. |
| **P4** | **Active Spatial Data Warping** | geometry | A conv router predicts a 2-ch offset field; `grid_sample` deformably warps the feature map (regularized toward identity). |

Shared constraint: **keep the spatial grid uniform** so the optimized FFT path is preserved. All
mechanisms are identity-at-init (zero-init writes / identity FiLM), so each is a clean diff over P0.

## Results — CIFAR-10 (32×32, 10-class; dim 192, 6 blocks, 64-token grid, 50 ep, 3 seeds)

| Config | Mechanism | Val acc | Δ vs P0 |
|--------|-----------|---------|---------|
| P0 | static baseline | 87.78% | — |
| P1 | spectral FiLM (all layers) | 88.35% | **+0.57** |
| P2 | envelope warp | — | eliminated (−0.21) |
| P3 | sparse mask (SW=0.5) | 88.43% | **+0.65** |
| P4 | data warp | 87.78% | eliminated (0.00) |
| **P5** | **P1 + P3** | **88.70%** | **+0.92** |

Spectral (kernel) and sparse (input) conditioning help and are partly complementary (P5 slightly
sub-additive, −0.30 pp vs the naive sum). Geometric deformation (P2/P4) provides **no** benefit at
this scale — the regularizer dominates and learned offsets stay near-identity.

## Results — TinyImageNet (64×64, 200-class; same backbone & recipe, 3 seeds) — does the gap grow?

| Grid | P0 | P5 (P1+P3) | **Δ (P5−P0)** |
|------|-----|-----------|---------------|
| patch-8 (64 tokens, matched to CIFAR) | 47.67 ± 0.31% | 46.90 ± 0.44% | **−0.77 pp** |
| patch-4 (256 tokens) | 54.17 ± 0.25% | 53.90 ± 0.70% | **−0.27 pp** (noise) |

On the harder task the +0.92 pp gain **does not transfer** — at matched grid it inverts (P5 < P0 on
every seed). Adding spatial resolution (patch-4) lifts absolute accuracy a lot (48% → 54%) and pulls
the gap back toward 0, but never positive.

## Takeaways

- **Two mechanisms matter on CIFAR-10:** spectral FiLM (kernel) and sparse masking (input). Geometric
  deformation of the envelope/grid does not — at least at 32–64 px.
- **The win is task-specific.** It does not scale with classification difficulty; on TinyImageNet-200
  it disappears. So this is *not* yet evidence of a general expressivity gain for the operator.
- **Resolution helps the deformable mechanisms more than difficulty hurts them** (patch-8 → patch-4
  recovers −0.77 → −0.27). Spatial structure, not class count, is the lever.
- **Next.** The sparse mask's intended best case is data where most of the field is irrelevant
  background — i.e. WSI/pathology, not natural images where the object fills the frame. That is the
  cleaner test of the thesis and the natural follow-up.

*Configs & full log: `examples/dynamic_deformable/` (`baseline_hyena.py`, `spectral_film.py`,
`sparse_mask.py`, `combined_best.py`, `tinyimagenet_*.py`); detailed tracker: `tracker.md`.*
