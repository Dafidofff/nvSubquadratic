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

"""P5 — Combined (P1 spectral FiLM + P3 sparse mask, SW=0.5) on TinyImageNet.

TinyImageNet port of ``combined_best.py`` (CIFAR-10 P5). Same backbone, same
two stacked mechanisms, same training recipe; only the dataset (TinyImageNet,
200 classes, native 64x64) and patch size change. Paired with
``tinyimagenet_baseline.py`` (P0) to measure whether the expressivity gap
(P5 − P0) grows on a harder classification task.

Mechanisms (see combined_best.py for the full rationale):
  * P1 — Dynamic Spectral Filtering: register→FiLM→SIREN, ``NUM_REGISTERS=4``,
    ``register_write=True``, all SIREN hidden layers modulated.
  * P3 — Sparse Tokenization via Dense Masking: ``SpatialSoftMask`` before each
    Hyena block, ``SPARSITY_WEIGHT=0.5``.

Grid: ``PATCH_SIZE`` defaults to 8 → 8x8 = 64-token grid (matches the CIFAR-10
runs). Override ``PATCH_SIZE=4`` for the 16x16 = 256-token variant.

Run (ivi, online W&B)::

    conda activate nvsubq
    export PYTHONPATH=.
    python experiments/run.py --config examples/dynamic_deformable/tinyimagenet_combined_best.py \\
        experiment_dir=runs/dynamic_deformable/tin_combined_best_seed42 debug=false
"""

import os

import torch

from experiments.datamodules.tinyimagenet import (
    AugmentConfig,
    MixupConfig,
    TINYIMAGENET_IMAGE_SIZE,
    TINYIMAGENET_NUM_CLASSES,
    TINYIMAGENET_TRAIN_SIZE,
    TinyImageNetDataModule,
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
TINYIMAGENET_DATA_DIR = os.environ.get(
    "TINYIMAGENET_PATH", "/ivi/zfs/s0/original_homes/dwessel/data/tiny-imagenet"
)
INPUT_CHANNELS = 3
IMAGE_SIZE = TINYIMAGENET_IMAGE_SIZE  # 64
NUM_CLASSES = TINYIMAGENET_NUM_CLASSES  # 200

# ── Model ─────────────────────────────────────────────────────────────────────
HIDDEN_DIM = 192
NUM_BLOCKS = 6
PATCH_SIZE = int(os.environ.get("PATCH_SIZE", "8"))  # 64 // 8 = 8×8 = 64 token grid
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

# FiLM generator (P1): all hidden layers modulated (P1-all)
FILM_HIDDEN_DIM = 64
NUM_FILM_LAYERS = KERNEL_NUM_LAYERS - 1  # 2 hidden layers

# Sparse mask scorer (P3)
MASK_HEAD_CHANNELS = 8
SPARSITY_WEIGHT = float(os.environ.get("SPARSITY_WEIGHT", "0.5"))

# ── Training recipe (identical to P0) ─────────────────────────────────────────
BATCH_SIZE = 128
EPOCHS = 50
ITERS_PER_EPOCH = TINYIMAGENET_TRAIN_SIZE // BATCH_SIZE  # 781
WARMUP_EPOCHS = 2
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.05
GRAD_CLIP = 1.0
PRECISION = "bf16-mixed"
NUM_WORKERS = 8
SEED = int(os.environ.get("SEED", "42"))

WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "implicit-long-convs")


def get_config() -> ExperimentConfig:
    """Build the P5 combined (P1 spectral FiLM + P3 sparse mask SW=0.5) config for TinyImageNet."""
    config = ExperimentConfig()
    config.debug = True
    config.seed = SEED
    config.compile = False

    config.dataset = LazyConfig(TinyImageNetDataModule)(
        data_dir=TINYIMAGENET_DATA_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        seed=config.seed,
        image_size=IMAGE_SIZE,
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
        tags=["dynamic-deformable", "tiny-imagenet", "p5-combined", "film-all", f"sw{SPARSITY_WEIGHT}", f"patch{PATCH_SIZE}"],
    )

    config.autoresume = AutoResumeConfig(enabled=False)
    config.callbacks = []

    return config
