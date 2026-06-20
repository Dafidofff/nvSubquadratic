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

"""P2 — Idea 3: Dynamic Envelope Warping (register -> shift + dilation -> Gaussian mask) on CIFAR-10.

Minimal diff on top of ``baseline_hyena.py`` (the P0 control).  Everything is
identical except for the dynamic Gaussian envelope mechanism.  Diff vs. baseline:

  1. ``NUM_REGISTERS = 4`` (baseline uses 0).  Register tokens exist so they
     can route per-input context into the Gaussian mask.
  2. ``GaussianModulationND`` is replaced by ``DynamicGaussianModulationND``:
     a small 2-layer MLP conditioned on the register pooling vector predicts
     per-axis shift (translation of the envelope centre) and log_scale
     (dilation of the std), making the effective receptive field input-dependent
     while keeping the FFT grid perfectly uniform.
  3. The residual block routes register tokens to the mixer via
     ``register_pooling_cfg`` (``RegisterPooling``) -> ``conditioning=[B, C]``,
     which flows QKVSequenceMixer -> Hyena -> CKConvND -> mask.forward(conditioning=...).
  4. ``ViT5HyenaAdapter`` runs in **register-split mode** (``grid_h`` set), same
     as P1, so the 4 register tokens stay off the FFT grid and a pooled patch
     summary is written back into them each block (``register_write=True``).

No FiLM on the SIREN kernel: that is P1's mechanism and is deliberately excluded
here so the P2 gain (or lack thereof) is attributable to the envelope alone.

Identity-at-init: the warp head's output projection is zero-initialised
(shift=0, log_scale=0 => exp(0)=1), so at step 0 this model is numerically
equivalent to the P0 baseline.  The mechanism is learned from there.

Run (local GPU, offline W&B, ananas unmounted)::

    conda activate nvsubq
    export PYTHONPATH=.
    export CIFAR10_PATH="$PWD/.data/cifar10"
    python experiments/run.py --config examples/dynamic_deformable/envelope_warp.py \\
        experiment_dir=.runs/envelope_warp_seed42

Smoke test::

    python experiments/run.py --config examples/dynamic_deformable/envelope_warp.py \\
        experiment_dir=.runs/smoke_envelope train.iterations=6 \\
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
from experiments.lightning_wrappers.classification_wrapper import ClassificationWrapper
from nvsubquadratic.lazy_config import PLACEHOLDER, LazyConfig
from nvsubquadratic.modules.ckconv_nd import CKConvND
from nvsubquadratic.modules.film import RegisterPooling
from nvsubquadratic.modules.hyena_nd import Hyena
from nvsubquadratic.modules.kernels_nd import SIRENKernelND
from nvsubquadratic.modules.masks_nd import DynamicGaussianModulationND
from nvsubquadratic.modules.mlp import MLP
from nvsubquadratic.modules.rms_norm import RMSNorm
from nvsubquadratic.modules.sequence_mixer import QKVSequenceMixer
from nvsubquadratic.modules.vit5_hyena_adapter import ViT5HyenaAdapter
from nvsubquadratic.modules.vit5_residual_block import ViT5ResidualBlock
from nvsubquadratic.networks.vit5_classification import ViT5ClassificationNet
from nvsubquadratic.utils.init import partial_wang_init_fn_with_num_layers, small_init
from nvsubquadratic.utils.qk_norm import L2Norm


# ── Dataset ───────────────────────────────────────────────────────────────────
CIFAR10_DATA_DIR = os.environ.get("CIFAR10_PATH", "/home/davidwessels/data/cifar10")
INPUT_CHANNELS = 3
IMAGE_SIZE = 32
NUM_CLASSES = CIFAR10_NUM_CLASSES  # 10

# ── Model ────────────────────────────────────────────────────────────────────
HIDDEN_DIM = 192
NUM_BLOCKS = 6
PATCH_SIZE = 4  # 32 // 4 = 8×8 = 64 token grid
NUM_REGISTERS = 4
MLP_RATIO = 4
LAYER_SCALE_INIT = 1e-4
DROP_PATH_RATE = 0.05

# SIREN kernel (static, same as baseline — no FiLM in P2)
KERNEL_MLP_HIDDEN_DIM = 32
KERNEL_NUM_LAYERS = 3
KERNEL_EMBEDDING_DIM = 32
KERNEL_OMEGA_0 = 10.0
KERNEL_HIDDEN_OMEGA_0 = 1.0

# Dynamic envelope warp head
WARP_HEAD_HIDDEN_DIM = 32  # MLP width inside DynamicGaussianModulationND

# ── Training recipe ───────────────────────────────────────────────────────────
BATCH_SIZE = 128
EPOCHS = 50
ITERS_PER_EPOCH = CIFAR10_TRAIN_SIZE // BATCH_SIZE  # 50_000 // 128 = 390
WARMUP_EPOCHS = 2
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.05
GRAD_CLIP = 1.0
PRECISION = "bf16-mixed"
NUM_WORKERS = 4
SEED = int(os.environ.get("SEED", "42"))

WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "implicit-long-convs")


def get_config() -> ExperimentConfig:
    """Build the P2 envelope-warp (register -> shift/dilation -> Gaussian mask) config for CIFAR-10."""
    config = ExperimentConfig()
    config.debug = True  # set debug=false for online W&B
    config.seed = SEED
    config.compile = False

    # ── Dataset ────────────────────────────────────────────────────────────────
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

    # ── Hyena mixer (static SIREN kernel + dynamic Gaussian envelope) ──────────
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
                    # No film_cfg: SIREN kernel is static in P2 (isolates envelope mechanism)
                ),
                # P2: DynamicGaussianModulationND receives the conditioning vector
                # from register tokens via CKConvND.forward(conditioning=...).
                # Identity at init (zero-init warp head): step-0 == P0 baseline.
                mask_cfg=LazyConfig(DynamicGaussianModulationND)(
                    data_dim=2,
                    num_channels=HIDDEN_DIM,
                    cond_dim=HIDDEN_DIM,
                    head_hidden_dim=WARP_HEAD_HIDDEN_DIM,
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

    # ── Network ─────────────────────────────────────────────────────────────────
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
                grid_h="${eval:'${net.image_size} // ${net.patch_size}'}",
                register_write=True,
                hidden_dim=HIDDEN_DIM,
            ),
            sequence_mixer_norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
            # Pool register tokens -> [B, C] conditioning vector for the warp head.
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

    # ── Lightning wrapper ───────────────────────────────────────────────────────
    config.lightning_wrapper_class = LazyConfig(ClassificationWrapper)(loss="soft_target_ce")

    # ── Optimizer ──────────────────────────────────────────────────────────────
    config.optimizer = LazyConfig(torch.optim.AdamW)(
        params=PLACEHOLDER,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # ── Training loop ──────────────────────────────────────────────────────────
    total_iters = EPOCHS * ITERS_PER_EPOCH
    config.train = TrainConfig(
        batch_size=BATCH_SIZE,
        iterations=total_iters,
        grad_clip=GRAD_CLIP,
        precision=PRECISION,
    )

    # ── Trainer / checkpointing ────────────────────────────────────────────────
    config.trainer = TrainerConfig(
        check_val_every_n_epoch=5,
        checkpoint_every_n_steps=2000,
        checkpoint_monitor="val/acc",
    )

    # ── LR scheduler ────────────────────────────────────────────────────────────
    config.scheduler = SchedulerConfig(
        name="cosine",
        warmup_iterations_percentage=WARMUP_EPOCHS / EPOCHS,
        total_iterations="${train.iterations}",
        mode="max",
    )

    # ── W&B ─────────────────────────────────────────────────────────────────────
    config.wandb = WandbConfig(
        job_group="dynamic-deformable",
        entity=WANDB_ENTITY,
        project="nvsubquadratic",
        tags=["dynamic-deformable", "cifar10", "p2-envelope-warp"],
    )

    config.autoresume = AutoResumeConfig(enabled=False)
    config.callbacks = []

    return config
