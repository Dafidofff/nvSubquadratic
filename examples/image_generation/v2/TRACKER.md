# Image Generation v2 — TinyImageNet (Hyena vs Attention)

Class-conditional **flow-matching diffusion** (JiT-style) on **TinyImageNet 64×64**
(`zh-plus/tiny-imagenet`, 200 classes, 100k/10k). Attention is the baseline;
**Hyena is the headline subquadratic model**. TinyImageNet is the cheap proxy for
initial signal — we scale to full ImageNet-64 next (see roadmap P5).

## Goal

Compare three denoisers under one diffusion recipe, then iterate on Hyena. The two
ViT5 models are **pixel-level (no patchification → 4096 tokens)**, our classification
backbones (`examples/imagenet_classification/vit_b_benchmark_tiny_imagenet/`) adapted
to generation — this is the regime where Hyena's O(N·logN) beats attention's O(N²).
`jit_baseline` is the standard patchified-DiT reference (256 tokens).

## Models

| Config | Mixer | Tokens (patch) | Params | Notes |
| ------ | ----- | -------------- | ------ | ----- |
| `jit_baseline.py`   | Attention (JiT-B/4) | 256 (patch 4)  | 130 M | Patchified DiT reference |
| `vit5_attention.py` | Attention + RoPE    | 4096 (pixel)   | 114 M | Pixel-level baseline, O(N²) |
| `vit5_hyena.py`     | Hyena (SIREN ω₀=30) | 4096 (pixel)   | 115 M | Pixel-level headline, subquadratic |

The two ViT5 models are parameter-matched (114 vs 115 M); mixer is the only
difference (both `ResidualNetwork` + `AdaLNZeroResidualBlock`, pixel-level
`Linear(3→768)` in / zero-init `Linear(768→3)` out, `gradient_checkpointing=True`).

## Shared diffusion recipe

Adam(0.9, 0.95), lr 2e-4, constant + 2–2.5 % warmup, grad-clip 1.0, EMA 0.9998,
CFG scale 2.9 / interval [0.1, 1.0], cond-dropout 0.1, flow-matching p_mean −0.8 /
p_std 0.8, 1000 train timesteps / 50 Heun inference steps, `num_classes=200`.
**Online FID is OFF** for now (no TinyImageNet reference stats / torch-fidelity on
hipster) — track flow-matching `val/loss` + W&B sample grids until P1.

## Hardware / launch (hipster)

`performance` partition, **2× RTX 6000 Ada (48 GB)**, 64 CPUs, **48 h** walltime,
bare-python DDP (Lightning owns the 2 workers — **not** torchrun). Wall-clock
bounded via `WalltimeCheckpointer` (`train.run_time_limit_hours=47.5`); re-submit to
autoresume from `runs/<cfg>_2gpu/checkpoints/last.ckpt`.

```bash
sbatch --job-name=tin-hyena-2g examples/image_generation/v2/submit_2gpu_hipster.sh \
    examples/image_generation/v2/vit5_hyena.py    dataset.batch_size=32 train.accumulate_grad_steps=4
sbatch --job-name=tin-attn-2g  examples/image_generation/v2/submit_2gpu_hipster.sh \
    examples/image_generation/v2/vit5_attention.py dataset.batch_size=16 train.accumulate_grad_steps=8
sbatch --job-name=tin-jit-2g   examples/image_generation/v2/submit_2gpu_hipster.sh \
    examples/image_generation/v2/jit_baseline.py   dataset.batch_size=128 train.accumulate_grad_steps=1
```

Target **effective batch ≈ 256** = `batch_size × 2 GPUs × accum` for all three
(confirm/adjust per the smoke test).

## Implementation notes (shared-code changes on this branch)

- `experiments/datamodules/tinyimagenet.py`: inlined `MixupConfig`/`AugmentConfig`
  to drop the module-level `nvidia.dali` import (DALI not installed locally / on
  hipster; HF loader doesn't need it). Mirrors the dynamic-deformable fix.
- `nvsubquadratic/modules/attention.py`: added `**kwargs` to `Attention.forward`
  so it accepts-and-ignores the AdaLN `conditioning` kwarg forwarded by
  `QKVSequenceMixer` (matches the documented mixer contract). Enables attention as
  a DiT/AdaLN diffusion mixer. Backward-compatible.
- `TinyImageNetDataModule` train transform applies `RandomCrop(64, pad 4)` + hflip
  even for generation — mild; revisit if it hurts sample quality.

## Status / run log

Effective batch is wall-clock bounded; log it/s + steps reached (Hyena ~slower per
step at 4096 tokens than jit_baseline; pixel-level attention slowest — equal 2-day
wall-clock ⇒ unequal step counts, expected).

| Model | Job ID | W&B run | batch/GPU × accum | eff batch | it/s | steps @ 48h | val/loss | status |
| ----- | ------ | ------- | ----------------- | --------- | ---- | ----------- | -------- | ------ |
| jit_baseline   | _tbd_ | _tbd_ | 128 × 1 | 256 | | | | 📝 not launched |
| vit5_attention | _tbd_ | _tbd_ | 16 × 8  | 256 | | | | 📝 not launched |
| vit5_hyena     | _tbd_ | _tbd_ | 32 × 4  | 256 | | | | 📝 not launched |

## Investigation roadmap

(Aligned with `examples/overview_tracker.md` → "ImageNet (Diffusion)".)

- **P0 — pipeline + initial baselines (now):** the 3 runs above. Validate the
  pipeline; first loss curves + sample grids on TinyImageNet.
- **P1 — FID infra:** `pip install torch-fidelity` in hipster `nvsubq`; generate
  TinyImageNet train reference stats (adapt `scripts/data/generate_jit_fid_stats.py`
  to `TinyImageNetDataModule`); turn on online FID → real Hyena-vs-Attention numbers.
- **P2 — Hyena masking:** ablate the `GaussianModulationND` global-conv mask
  (on/off, extent) and a spatial soft-mask.
- **P3 — hyperparameter ablations:** ω₀, lr, weight decay, dropout / drop-path.
- **P4 — patch-size analysis:** pixel-level vs patch 2 / 4 / 8 — token count vs
  quality/speed trade-off (Hyena vs attention scaling).
- **P5 — scale to full ImageNet-64:** reuse the v1 configs / recipe on more GPUs.
