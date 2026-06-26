# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ViT5 pixel-level Attention for class-conditional TinyImageNet 64x64 flow-matching diffusion.

The pixel-level attention baseline (O(N^2) over 4096 tokens). Architecture mirrors
our TinyImageNet classification Attention baseline (``examples/imagenet_classification/
vit_b_benchmark_tiny_imagenet/attention.py``): ViT-B scale (768 dim, 12 blocks,
12 heads, RoPE + QK-norm), **pixel-level** (no patchification, 64x64 = 4096 tokens).

Adapted to diffusion the same way as ``vit5_hyena.py``: pixel-level Linear in /
zero-init Linear out, AdaLNZeroResidualBlock for DiT-style timestep conditioning,
DiffusionWrapper + JiT flow-matching recipe. The attention mixer receives the
AdaLN ``conditioning`` kwarg via ``QKVSequenceMixer`` and simply ignores it; all
timestep conditioning flows through AdaLN-Zero. ``gradient_checkpointing=True``
keeps the 4096-token activation memory in check.

See ``examples/image_generation/v2/TRACKER.md``.
"""

import os

import torch

from experiments.datamodules.tinyimagenet import TinyImageNetDataModule
from experiments.default_cfg import (
    DiffusionConfig,
    DiffusionExperimentConfig,
    SchedulerConfig,
    TrainConfig,
    WandbConfig,
)
from experiments.lightning_wrappers.diffusion_wrapper import DiffusionWrapper
from nvsubquadratic.lazy_config import PLACEHOLDER, LazyConfig
from nvsubquadratic.modules.attention import Attention
from nvsubquadratic.modules.mlp import MLP
from nvsubquadratic.modules.patchify import Patchify, Unpatchify
from nvsubquadratic.modules.residual_block import AdaLNZeroResidualBlock
from nvsubquadratic.modules.sequence_mixer import QKVSequenceMixer
from nvsubquadratic.networks.general_purpose_resnet import ResidualNetwork
from nvsubquadratic.utils.init import partial_wang_init_fn_with_num_layers, small_init


WANDB_ENTITY = "dafidofff"

# Dataset ----------------------------------------------------------------------
BATCH_SIZE = 16  # Per GPU (override per-launch; heaviest model — 4096-token O(N^2))
NUM_WORKERS = min(12, os.cpu_count() - 2 or 4)
FINAL_IMAGE_SIZE = 64
PATCH_SIZE = 1  # pixel-level: 64x64 = 4096 tokens (no patchification)
HF_DATASET = "zh-plus/tiny-imagenet"
HF_CACHE = os.environ.get("TINYIMAGENET_CACHE", os.path.expanduser("~/.cache/tinyimagenet"))

# Network params (ViT-B scale, matching the classification Attention baseline)
INPUT_CHANNELS = 3
OUTPUT_CHANNELS = 3
DATA_DIM = 2
NUM_HIDDEN_CHANNELS = 768
NUM_BLOCKS = 12
NUM_HEADS = 12  # head_dim = 768 / 12 = 64
DROPOUT_IN_RATE = 0.0
DROPOUT_RATE = 0.1

# Optimisation -----------------------------------------------------------------
TRAINING_ITERATIONS = 300_000  # large cap; runs are wall-clock bounded (see launcher)
WARMUP_ITERATIONS_PERCENTAGE = 0.02
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.0
GRAD_CLIP = 1.0
ACCUMULATE_GRAD_STEPS = 1

# Diffusion --------------------------------------------------------------------
NUM_TRAIN_TIMESTEPS = 1000
NUM_INFERENCE_STEPS = 50  # Heun steps
NUM_SAMPLES = 16
LOG_SAMPLES = True
EMA_ENABLED = True
EMA_DECAY = 0.9998
EMA_WARMUP_STEPS = 1000
EMA_UPDATE_EVERY = 1

# CFG, per JiT
USE_CFG = True
GUIDANCE_SCALE = 2.9
CONDITION_DROPOUT_PROB = 0.1
NUM_CLASSES = 200  # TinyImageNet


def get_config() -> DiffusionExperimentConfig:
    """Build the experiment configuration."""
    config = DiffusionExperimentConfig()
    config.debug = False
    config.seed = 42

    config.dataset = LazyConfig(TinyImageNetDataModule)(
        data_dir=HF_CACHE,
        hf_dataset_name=HF_DATASET,
        hf_dataset_config=None,
        hf_auth_token=os.environ.get("HF_TOKEN"),
        image_size=FINAL_IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        seed=42,
        task="generation",  # normalizes to [-1, 1], uniform-dequantizes
        drop_labels=False,
        num_classes=NUM_CLASSES,
    )

    # Pixel-level attention denoiser (ResidualNetwork + AdaLN-Zero blocks)
    config.net = LazyConfig(ResidualNetwork)(
        in_channels=INPUT_CHANNELS,
        out_channels=OUTPUT_CHANNELS,
        num_blocks=NUM_BLOCKS,
        hidden_dim=NUM_HIDDEN_CHANNELS,
        data_dim=DATA_DIM,
        gradient_checkpointing=True,  # 4096-token activations -> recompute on backward
        # Pixel-level projections (patch_size=1): in = Linear(3->hidden),
        # out = zero-init Linear(hidden->3) (DiT final-layer zero init).
        in_proj_cfg=LazyConfig(Patchify)(
            in_features="${net.in_channels}",
            out_features="${net.hidden_dim}",
            data_dim="${net.data_dim}",
            patch_size=PATCH_SIZE,
            stride=PATCH_SIZE,
        ),
        out_proj_cfg=LazyConfig(Unpatchify)(
            in_features="${net.hidden_dim}",
            out_features="${net.out_channels}",
            data_dim="${net.data_dim}",
            patch_size="${net.in_proj_cfg.patch_size}",
            stride="${net.in_proj_cfg.stride}",
            weight_init="zeros",
        ),
        norm_cfg=LazyConfig(torch.nn.LayerNorm)(normalized_shape="${net.hidden_dim}"),
        block_cfg=LazyConfig(AdaLNZeroResidualBlock)(
            sequence_mixer_cfg=LazyConfig(QKVSequenceMixer)(
                hidden_dim="${net.hidden_dim}",
                mixer_cfg=LazyConfig(Attention)(
                    hidden_dim="${net.hidden_dim}",
                    num_heads=NUM_HEADS,
                    apply_qk_norm=True,
                    use_rope=True,
                    is_causal=False,
                    rope_base=10000.0,
                    attn_dropout=0.0,
                    # 2D RoPE over the pixel grid (64x64 = 4096 tokens at patch_size=1).
                    rope_spatial_dims=(FINAL_IMAGE_SIZE // PATCH_SIZE,) * DATA_DIM,
                ),
                init_method_in=small_init,
                init_method_out=LazyConfig(partial_wang_init_fn_with_num_layers)(num_layers="${net.num_blocks}"),
            ),
            sequence_mixer_norm_cfg="${net.norm_cfg}",
            mlp_cfg=LazyConfig(MLP)(
                dim="${net.hidden_dim}",
                activation="glu",
                expansion_factor=2.0,
                dropout_cfg=LazyConfig(torch.nn.Dropout)(p="${net.block_cfg.dropout_cfg.p}"),
                init_method_in=small_init,
                init_method_out=LazyConfig(partial_wang_init_fn_with_num_layers)(num_layers="${net.num_blocks}"),
            ),
            mlp_norm_cfg="${net.norm_cfg}",
            dropout_cfg=LazyConfig(torch.nn.Dropout)(p=DROPOUT_RATE),
            condition_norm_cfg="${net.norm_cfg}",
            hidden_dim="${net.hidden_dim}",
        ),
        dropout_in_cfg=LazyConfig(torch.nn.Dropout)(p=DROPOUT_IN_RATE),
        condition_in_proj_cfg=LazyConfig(torch.nn.Linear)(
            in_features="${net.hidden_dim}", out_features="${net.hidden_dim}"
        ),
    )

    config.lightning_wrapper_class = LazyConfig(DiffusionWrapper)()

    config.optimizer = LazyConfig(torch.optim.Adam)(
        params=PLACEHOLDER,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.95),
    )

    config.train = TrainConfig(
        batch_size="${dataset.batch_size}",
        iterations=TRAINING_ITERATIONS,
        grad_clip=GRAD_CLIP,
        accumulate_grad_steps=ACCUMULATE_GRAD_STEPS,
    )

    config.scheduler = SchedulerConfig(
        name="constant",
        warmup_iterations_percentage=WARMUP_ITERATIONS_PERCENTAGE,
        total_iterations="${train.iterations}",
        mode="min",
    )

    config.diffusion = DiffusionConfig(
        num_train_timesteps=NUM_TRAIN_TIMESTEPS,
        num_inference_steps=NUM_INFERENCE_STEPS,
        num_samples=NUM_SAMPLES,
        log_samples=LOG_SAMPLES,
        ema_enabled=EMA_ENABLED,
        ema_decay=EMA_DECAY,
        ema_update_every=EMA_UPDATE_EVERY,
        ema_warmup_steps=EMA_WARMUP_STEPS,
        # CFG
        use_classifier_free_guidance=USE_CFG,
        guidance_scale=GUIDANCE_SCALE,
        condition_dropout_prob=CONDITION_DROPOUT_PROB,
        num_classes=NUM_CLASSES,
        # JiT flow-matching params
        p_mean=-0.8,
        p_std=0.8,
        cfg_interval_start=0.1,
        cfg_interval_end=1.0,
        # Online FID OFF: no TinyImageNet reference stats / torch-fidelity yet.
        fid_online_jit=False,
        fid_stats_file="",
    )

    config.wandb = WandbConfig(
        job_group="tinyimagenet_diffusion_vit5_attention",
        entity=WANDB_ENTITY,
        tags=["attention", "diffusion", "tinyimagenet64", "pixel-level", "vit5"],
    )

    return config
