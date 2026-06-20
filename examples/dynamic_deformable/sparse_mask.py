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

"""P3 — Idea 1: Sparse Tokenization via Dense Masking on CIFAR-10 (8×8 patch grid).

Minimal diff on top of ``baseline_hyena.py`` (the P0 control).  Everything is
identical except for the ``SpatialSoftMask`` pre-mixer module and the sparsity
regularizer.  Diff vs. baseline:

  1. Each ``ViT5HyenaAdapter`` block gains a ``pre_mixer_mask_cfg``
     (``SpatialSoftMask``): a lightweight 3×3 conv scorer predicts a soft
     keep-probability ``m ∈ (0, 1)`` per spatial position, which is multiplied
     into the feature map *before* the global FFT convolution.  The uniform
     spatial grid is preserved — no tokens are physically removed.
  2. ``SparseMaskClassificationWrapper`` adds a sparsity regularizer
     ``sparsity_weight * mean(mask_density_per_block)`` to the training loss,
     pushing the mask toward zero (attenuation) while the classification loss
     opens it where features are useful.  Validation/test use the unpenalised
     loss for fair metric comparison.
  3. ``SPARSITY_WEIGHT`` (env knob, default ``0.1``) controls the budget.
     Higher values → sparser masks at convergence; val/mask_density is logged
     so the effective keep-ratio can be read from the run.

No register tokens needed: the scorer is fully feed-forward (spatial context via
the 3×3 conv, no global conditioning required for this mechanism).  The static
baseline (``num_registers=0``) is therefore preserved.

Identity-at-init: the scorer's output projection is zero-initialised so
``sigmoid(0) = 0.5`` everywhere at step 0 — the initial model is equivalent to
the baseline scaled by 0.5, from which the classification signal quickly learns
to open the mask.

Run (local GPU, offline W&B)::

    conda activate nvsubq
    export PYTHONPATH=.
    export CIFAR10_PATH="$PWD/.data/cifar10"
    python experiments/run.py --config examples/dynamic_deformable/sparse_mask.py \\
        experiment_dir=.runs/sparse_mask_seed42

Tune sparsity weight::

    SPARSITY_WEIGHT=0.05  python experiments/run.py ...   # gentler budget
    SPARSITY_WEIGHT=0.5   python experiments/run.py ...   # heavier sparsity

Smoke test::

    python experiments/run.py --config examples/dynamic_deformable/sparse_mask.py \\
        experiment_dir=.runs/smoke_sparse \\
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
PATCH_SIZE = 4  # 8×8 = 64 token grid (same as P0 baseline)
NUM_REGISTERS = 0
MLP_RATIO = 4
LAYER_SCALE_INIT = 1e-4
DROP_PATH_RATE = 0.05

# SIREN kernel (static — same as P0)
KERNEL_MLP_HIDDEN_DIM = 32
KERNEL_NUM_LAYERS = 3
KERNEL_EMBEDDING_DIM = 32
KERNEL_OMEGA_0 = 10.0
KERNEL_HIDDEN_OMEGA_0 = 1.0

# Soft spatial mask scorer
MASK_HEAD_CHANNELS = 8  # lightweight: 8×C → 8×1 params per block

# Sparsity regularizer weight (env knob: higher → sparser masks at convergence)
SPARSITY_WEIGHT = float(os.environ.get("SPARSITY_WEIGHT", "0.1"))

# ── Training recipe ───────────────────────────────────────────────────────────
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
    """Build the P3 sparse-mask (soft keep-mask + sparsity budget) config for CIFAR-10."""
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
                # P3: soft spatial mask applied to the [B, H, W, C] patch grid each block.
                pre_mixer_mask_cfg=LazyConfig(SpatialSoftMask)(
                    hidden_dim=HIDDEN_DIM,
                    head_channels=MASK_HEAD_CHANNELS,
                ),
            ),
            sequence_mixer_norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
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
        tags=["dynamic-deformable", "cifar10", "p3-sparse-mask", f"sw{SPARSITY_WEIGHT}"],
    )

    config.autoresume = AutoResumeConfig(enabled=False)
    config.callbacks = []

    return config
