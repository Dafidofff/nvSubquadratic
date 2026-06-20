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

"""P5 — Combined: P1 (spectral FiLM) + P3 (sparse mask, SW=0.5) on CIFAR-10.

Stacks the two mechanisms that individually outperformed P0:

  * **P1 — Dynamic Spectral Filtering** (``spectral_film.py``): register tokens
    conditioned on a mean-pooled patch summary write per-block; registers feed a
    ``KernelFiLMGenerator`` that modulates every SIREN hidden layer (γ, β)
    → a Linear Time-Varying frequency-domain filter.  ``NUM_REGISTERS=4``,
    ``register_write=True``.

  * **P3 — Sparse Tokenization via Dense Masking** (``sparse_mask.py``,
    ``SPARSITY_WEIGHT=0.5``): a lightweight ``SpatialSoftMask`` (3×3 conv → sigmoid)
    applied to the ``[B, H, W, C]`` patch grid **before** each Hyena block attenuates
    uninformative positions; a sparsity penalty ``SW * mean_density`` drives density
    toward ~0.4%.  Gradient flows via ``_mask_mean`` (live tensor), not the detached
    ``_last_mean_mask`` (logging only).

Both mechanisms are independent and attach to different stages of the pipeline:
the soft mask gates the spatial feature map before the FFT; FiLM conditions the
SIREN kernel inside the FFT.  They share the same ``ViT5HyenaAdapter`` block —
``pre_mixer_mask_cfg`` applies the mask, and ``grid_h``/``register_write``/
``hidden_dim`` enable register-split mode with patch→register write-back.

Identity-at-init (both mechanisms):
  - FiLM generator initialised to identity (γ=1, β=0).
  - Register write projection zero-initialised (no register write at step 0).
  - Mask scorer output projection zero-initialised (sigmoid(0)=0.5 at step 0).

Individual results (3 seeds, same backbone, corrected SW sweep 2026-06-18):
  - P0 baseline:     87.78 ± 0.11% val
  - P1 spectral FiLM all:  88.35 ± 0.13% val  (+0.57 pp)
  - P3 SW=0.5:       88.43 ± 0.12% val  (+0.65 pp)

Run (local GPU, offline W&B)::

    conda activate nvsubq
    export PYTHONPATH=.
    export CIFAR10_PATH="$PWD/.data/cifar10"
    python experiments/run.py --config examples/dynamic_deformable/combined_best.py \\
        experiment_dir=.runs/combined_best_seed42

Smoke test::

    python experiments/run.py --config examples/dynamic_deformable/combined_best.py \\
        experiment_dir=.runs/smoke_p5 \\
        train.iterations=6 \\
        trainer.check_val_every_n_iterations=3 trainer.check_val_every_n_epoch=null \\
        trainer.limit_val_batches=2
"""

import os

import torch

from experiments.datamodules.cifar10 import (
    AugmentConfig,
    CIFAR10_NUM_CLASSES,
    CIFAR10_TRAIN_SIZE,
    CIFAR10DataModule,
    MixupConfig,
)
from experiments.default_cfg import (
    AutoResumeConfig,
    ExperimentConfig,
    SchedulerConfig,
    TrainConfig,
    TrainerConfig,
    WandbConfig,
)
from experiments.lightning_wrappers.sparse_mask_wrapper import SparseMaskClassificationWrapper
from nvsubquadratic.lazy_config import PLACEHOLDER, LazyConfig
from nvsubquadratic.modules.ckconv_nd import CKConvND
from nvsubquadratic.modules.film import KernelFiLMGenerator, RegisterPooling
from nvsubquadratic.modules.hyena_nd import Hyena
from nvsubquadratic.modules.kernels_nd import SIRENKernelND
from nvsubquadratic.modules.masks_nd import GaussianModulationND
from nvsubquadratic.modules.mlp import MLP
from nvsubquadratic.modules.rms_norm import RMSNorm
from nvsubquadratic.modules.sequence_mixer import QKVSequenceMixer
from nvsubquadratic.modules.spatial_mask import SpatialSoftMask
from nvsubquadratic.modules.vit5_hyena_adapter import ViT5HyenaAdapter
from nvsubquadratic.modules.vit5_residual_block import ViT5ResidualBlock
from nvsubquadratic.networks.vit5_classification import ViT5ClassificationNet
from nvsubquadratic.utils.init import partial_wang_init_fn_with_num_layers, small_init
from nvsubquadratic.utils.qk_norm import L2Norm


# ── Dataset ───────────────────────────────────────────────────────────────────
CIFAR10_DATA_DIR = os.environ.get("CIFAR10_PATH", "/home/davidwessels/data/cifar10")
INPUT_CHANNELS = 3
IMAGE_SIZE = 32
NUM_CLASSES = CIFAR10_NUM_CLASSES

# ── Model ─────────────────────────────────────────────────────────────────────
HIDDEN_DIM = 192
NUM_BLOCKS = 6
PATCH_SIZE = 4  # 8×8 = 64 token grid
NUM_REGISTERS = 4  # P1: register tokens for FiLM conditioning
MLP_RATIO = 4
LAYER_SCALE_INIT = 1e-4
DROP_PATH_RATE = 0.05

# SIREN kernel (FiLM-conditioned — P1)
KERNEL_MLP_HIDDEN_DIM = 32
KERNEL_NUM_LAYERS = 3
KERNEL_EMBEDDING_DIM = 32
KERNEL_OMEGA_0 = 10.0
KERNEL_HIDDEN_OMEGA_0 = 1.0

# FiLM generator (P1): all hidden layers modulated (same as spectral_film.py P1-all)
FILM_HIDDEN_DIM = 64
NUM_FILM_LAYERS = KERNEL_NUM_LAYERS - 1  # 2 hidden layers

# Sparse mask scorer (P3)
MASK_HEAD_CHANNELS = 8

# P3 sparsity weight: SW=0.5 is the winner from the ablation sweep
SPARSITY_WEIGHT = float(os.environ.get("SPARSITY_WEIGHT", "0.5"))

# ── Training recipe (identical to P0/P1/P3) ───────────────────────────────────
BATCH_SIZE = 128
EPOCHS = 50
ITERS_PER_EPOCH = CIFAR10_TRAIN_SIZE // BATCH_SIZE  # 390
WARMUP_EPOCHS = 2
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.05
GRAD_CLIP = 1.0
PRECISION = "bf16-mixed"
NUM_WORKERS = 4
SEED = int(os.environ.get("SEED", "42"))

WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "implicit-long-convs")


def get_config() -> ExperimentConfig:
    """Build the P5 combined (P1 spectral FiLM + P3 sparse mask SW=0.5) config."""
    config = ExperimentConfig()
    config.debug = True
    config.seed = SEED
    config.compile = False

    config.dataset = LazyConfig(CIFAR10DataModule)(
        data_dir=CIFAR10_DATA_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        seed=config.seed,
        image_size=32,
        final_image_size=IMAGE_SIZE,
        num_classes=NUM_CLASSES,
        drop_labels=False,
        task="classification",
        mixup_cfg=LazyConfig(MixupConfig)(
            mixup=0.2,
            cutmix=0.0,
            mixup_prob=1.0,
            mixup_switch_prob=0.5,
            mixup_mode="batch",
            smoothing=0.1,
        ),
        augment_cfg=LazyConfig(AugmentConfig)(
            use_three_augment=False,
            color_jitter=0.3,
            rand_augment=None,
            random_erasing_prob=0.0,
        ),
    )

    # P1: FiLM-conditioned SIREN kernel (all hidden layers)
    hyena_mixer_cfg = LazyConfig(QKVSequenceMixer)(
        hidden_dim=HIDDEN_DIM,
        mixer_cfg=LazyConfig(Hyena)(
            global_conv_cfg=LazyConfig(CKConvND)(
                data_dim=2,
                hidden_dim=HIDDEN_DIM,
                kernel_cfg=LazyConfig(SIRENKernelND)(
                    data_dim=2,
                    out_dim=HIDDEN_DIM,
                    mlp_hidden_dim=KERNEL_MLP_HIDDEN_DIM,
                    num_layers=KERNEL_NUM_LAYERS,
                    embedding_dim=KERNEL_EMBEDDING_DIM,
                    omega_0=KERNEL_OMEGA_0,
                    L_cache="${eval:'${net.image_size} // ${net.patch_size}'}",
                    use_bias=True,
                    hidden_omega_0=KERNEL_HIDDEN_OMEGA_0,
                    film_layers=None,  # all hidden layers (P1-all)
                    film_cfg=LazyConfig(KernelFiLMGenerator)(
                        cond_dim=HIDDEN_DIM,
                        kernel_hidden_dim=KERNEL_MLP_HIDDEN_DIM,
                        num_film_layers=NUM_FILM_LAYERS,
                        film_hidden_dim=FILM_HIDDEN_DIM,
                        init_type="identity",
                    ),
                ),
                mask_cfg=LazyConfig(GaussianModulationND)(
                    data_dim=2,
                    num_channels=HIDDEN_DIM,
                    min_attenuation_at_step=0.1,
                    max_attenuation_at_limit=0.95,
                    init_extent=1.0,
                    parametrization="direct",
                ),
                grid_type="double",
                fft_padding="zero",
                fft_backend="torch_fft",
            ),
            short_conv_cfg=LazyConfig(torch.nn.Conv2d)(
                in_channels=3 * HIDDEN_DIM,
                out_channels=3 * HIDDEN_DIM,
                kernel_size=3,
                groups=3 * HIDDEN_DIM,
                padding=1,
                bias=False,
            ),
            gate_nonlinear_cfg=LazyConfig(torch.nn.SiLU)(),
            pixelhyena_norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
            qk_norm_cfg=LazyConfig(L2Norm)(),
            output_norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
            gate_nonlinear_2_cfg=LazyConfig(torch.nn.Sigmoid)(),
        ),
        init_method_in=small_init,
        init_method_out=LazyConfig(partial_wang_init_fn_with_num_layers)(num_layers=NUM_BLOCKS),
    )

    config.net = LazyConfig(ViT5ClassificationNet)(
        in_channels=INPUT_CHANNELS,
        num_classes=NUM_CLASSES,
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        patch_size=PATCH_SIZE,
        image_size=IMAGE_SIZE,
        num_registers=NUM_REGISTERS,
        dropout_rate=0.0,
        readout="gap",
        norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
        block_cfg=LazyConfig(ViT5ResidualBlock)(
            sequence_mixer_cfg=LazyConfig(ViT5HyenaAdapter)(
                inner_mixer_cfg=hyena_mixer_cfg,
                grid_w="${eval:'${net.image_size} // ${net.patch_size}'}",
                # P1: register-split mode + patch→register write-back.
                grid_h="${eval:'${net.image_size} // ${net.patch_size}'}",
                register_write=True,
                hidden_dim=HIDDEN_DIM,
                # P3: soft spatial mask applied to the patch grid before each Hyena call.
                pre_mixer_mask_cfg=LazyConfig(SpatialSoftMask)(
                    hidden_dim=HIDDEN_DIM,
                    head_channels=MASK_HEAD_CHANNELS,
                ),
            ),
            sequence_mixer_norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
            # P1: pool register tokens → [B, C] conditioning vector for FiLM.
            register_pooling_cfg=LazyConfig(RegisterPooling)(num_registers=NUM_REGISTERS),
            num_registers=NUM_REGISTERS,
            mlp_cfg=LazyConfig(MLP)(
                dim=HIDDEN_DIM,
                activation="gelu",
                expansion_factor=float(MLP_RATIO),
                bias=False,
                dropout_cfg=LazyConfig(torch.nn.Dropout)(p=0.0),
                init_method_in=small_init,
                init_method_out=LazyConfig(partial_wang_init_fn_with_num_layers)(num_layers=NUM_BLOCKS),
            ),
            mlp_norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
            hidden_dim=HIDDEN_DIM,
            layer_scale_init=LAYER_SCALE_INIT,
            drop_path_rate=DROP_PATH_RATE,
        ),
    )

    # P3 wrapper: handles sparsity penalty; P1 FiLM is transparent to it.
    config.lightning_wrapper_class = LazyConfig(SparseMaskClassificationWrapper)(
        loss="soft_target_ce",
        sparsity_weight=SPARSITY_WEIGHT,
    )

    config.optimizer = LazyConfig(torch.optim.AdamW)(
        params=PLACEHOLDER,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    total_iters = EPOCHS * ITERS_PER_EPOCH
    config.train = TrainConfig(
        batch_size=BATCH_SIZE,
        iterations=total_iters,
        grad_clip=GRAD_CLIP,
        precision=PRECISION,
    )

    config.trainer = TrainerConfig(
        check_val_every_n_epoch=5,
        checkpoint_every_n_steps=2000,
        checkpoint_monitor="val/acc",
    )

    config.scheduler = SchedulerConfig(
        name="cosine",
        warmup_iterations_percentage=WARMUP_EPOCHS / EPOCHS,
        total_iterations="${train.iterations}",
        mode="max",
    )

    config.wandb = WandbConfig(
        job_group="dynamic-deformable",
        entity=WANDB_ENTITY,
        project="nvsubquadratic",
        tags=["dynamic-deformable", "cifar10", "p5-combined", "film-all", f"sw{SPARSITY_WEIGHT}"],
    )

    config.autoresume = AutoResumeConfig(enabled=False)
    config.callbacks = []

    return config
