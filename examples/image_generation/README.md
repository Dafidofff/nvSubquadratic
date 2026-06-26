# Image Generation (Diffusion)

Class-conditional image generation with subquadratic backbones, using the
JiT-style flow-matching diffusion stack (`experiments/lightning_wrappers/
diffusion_wrapper.py`).

## Layout

- **`v1/`** — *archive*. The original ImageNet-64/128/256 config matrix (CCNN /
  JiT / U-ViT). Designed but **never run**; kept for reference and for scaling up
  later. See `v1/README.md`. (Note: the `v1/ccnn_*` Hyena configs predate a `main`
  API change and pass a now-removed `use_rope=` arg to `Hyena` — they need a touch-up
  before running.)
- **`v2/`** — *active investigation*. Attention-vs-Hyena on **TinyImageNet 64×64**
  first (cheap proxy), scaling to full ImageNet. Start here:
  [`v2/TRACKER.md`](v2/TRACKER.md).

## v2 quickstart (hipster, 2-GPU)

```bash
sbatch --job-name=tin-hyena-2g examples/image_generation/v2/submit_2gpu_hipster.sh \
    examples/image_generation/v2/vit5_hyena.py dataset.batch_size=32 train.accumulate_grad_steps=4
```

See [`v2/TRACKER.md`](v2/TRACKER.md) for the full model set, recipe, and roadmap.
