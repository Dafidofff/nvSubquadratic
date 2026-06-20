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

"""P1 — Idea 2: Dynamic Spectral Filtering (register -> FiLM -> SIREN) on CIFAR-10.

Minimal diff on top of ``baseline_hyena.py`` (the P0 control).  Everything is
identical except for the spectral-FiLM mechanism, so any accuracy change is
attributable to it alone.  Diff vs. baseline:

  1. ``NUM_REGISTERS = 4`` (baseline uses 0).  Register tokens now exist so they
     can carry per-input context into the SIREN kernel.
  2. ``SIRENKernelND`` gains a ``film_cfg`` (``KernelFiLMGenerator``): the
     conditioning vector is mapped to per-layer (gamma, beta) pairs that modulate
     every SIREN hidden layer, making the long-range kernel input-dependent
     (a Linear Time-Varying filter in the frequency domain).
  3. The residual block routes register tokens to the mixer via
     ``register_pooling_cfg`` (``RegisterPooling``) -> ``conditioning=[B, C]``,
     which flows QKVSequenceMixer -> Hyena -> CKConvND -> SIRENKernelND.film.
  4. ``ViT5HyenaAdapter`` runs in **register-split mode** (``grid_h`` set): the
     Hyena spatial mixer reshapes only the ``8x8`` patch grid, keeping the 4
     register tokens off the FFT grid (which is why the baseline had to use 0
     registers).  With ``register_write=True`` the adapter also writes a learned,
     mean-pooled summary of the mixed patches back into the register slots each
     block, so register->FiLM conditioning is **genuinely input-dependent**
     (registers in this attention-free backbone would otherwise stay constant
     across the batch and the FiLM would collapse to a static kernel).

Identity-at-init: the FiLM generator is identity-initialised (gamma=1, beta=0)
and the adapter's register-write projection is zero-initialised, so at step 0
this model is numerically equivalent to the P0 baseline (plus inert register
tokens).  The mechanism is learned from there.

Ablation handle: the SIREN here has ``num_layers=3`` -> 2 FiLM-modulated hidden
layers.  For the "last layer only" variant in the tracker, drop ``film_cfg``'s
``num_film_layers`` to 1 and gate it inside SIRENKernelND (follow-up).

Run (local GPU, offline W&B, ananas unmounted)::

    conda activate nvsubq
    export PYTHONPATH=.
    export CIFAR10_PATH="$PWD/.data/cifar10"
    python experiments/run.py --config examples/dynamic_deformable/spectral_film.py \\
        experiment_dir=.runs/spectral_film_seed42

Smoke test (a handful of steps)::

    python experiments/run.py --config examples/dynamic_deformable/spectral_film.py \\
        experiment_dir=.runs/smoke_spectral train.iterations=6 \\
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
from nvsubquadratic.modules.film import KernelFiLMGenerator, RegisterPooling
from nvsubquadratic.modules.hyena_nd import Hyena
from nvsubquadratic.modules.kernels_nd import SIRENKernelND
from nvsubquadratic.modules.masks_nd import GaussianModulationND
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
IMAGE_SIZE = 32  # native resolution — no 224 upsample, keeps the grid small/fast
NUM_CLASSES = CIFAR10_NUM_CLASSES  # 10

# ── Model (lightweight; bump to scale up) ─────────────────────────────────────
HIDDEN_DIM = 192
NUM_BLOCKS = 6
PATCH_SIZE = 4  # 32 // 4 = 8×8 = 64 token grid
# P1 enables register-based conditioning: 4 register tokens are pooled into the
# FiLM conditioning vector. They are kept OFF the Hyena FFT grid by the adapter's
# register-split mode (grid_h set) — the spatial mixer only sees the 8×8 patch
# grid, so the registers no longer break the reshape the way they would in the
# baseline (which therefore had to use 0 registers).
NUM_REGISTERS = 4
MLP_RATIO = 4
LAYER_SCALE_INIT = 1e-4
DROP_PATH_RATE = 0.05

# SIREN kernel (now FiLM-conditioned in P1)
KERNEL_MLP_HIDDEN_DIM = 32
KERNEL_NUM_LAYERS = 3
KERNEL_EMBEDDING_DIM = 32
KERNEL_OMEGA_0 = 10.0
KERNEL_HIDDEN_OMEGA_0 = 1.0

# FiLM generator: registers ([B, HIDDEN_DIM]) -> per-SIREN-layer (gamma, beta).
# num_film_layers must equal the number of *modulated* SIREN hidden layers,
# since film_after_pos_embed is False (the SIRENKernelND default).
FILM_HIDDEN_DIM = 64
# Which SIREN hidden layers receive FiLM — the tracker's P1 ablation:
#   FILM_LAYERS=all  -> all hidden layers (default)
#   FILM_LAYERS=last -> last hidden layer only
FILM_VARIANT = os.environ.get("FILM_LAYERS", "all").lower()
if FILM_VARIANT == "all":
    FILM_LAYER_INDICES = None  # None => modulate every hidden layer
    NUM_FILM_LAYERS = KERNEL_NUM_LAYERS - 1  # = 2
elif FILM_VARIANT == "last":
    FILM_LAYER_INDICES = [-1]  # last hidden layer only
    NUM_FILM_LAYERS = 1
else:
    raise ValueError(f"FILM_LAYERS must be 'all' or 'last', got {FILM_VARIANT!r}")

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
SEED = int(os.environ.get("SEED", "42"))  # override per-run for multi-seed sweeps

WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "implicit-long-convs")


def get_config() -> ExperimentConfig:
    """Build the P1 spectral-FiLM (register -> FiLM -> SIREN) config for CIFAR-10."""
    config = ExperimentConfig()
    config.debug = True  # offline W&B for local runs; set debug=false for online logging
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

    # ── Hyena mixer (FiLM-conditioned SIREN + static Gaussian envelope) ─────────
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
                    # P1: register -> FiLM -> SIREN. Identity-initialised so the
                    # kernel == baseline at step 0, then learns input-dependence.
                    # film_layers selects which hidden layers are modulated
                    # (all vs. last-only ablation; see FILM_LAYERS above).
                    film_layers=FILM_LAYER_INDICES,
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
                # Register-split mode: mix only the 8×8 patch grid, keep the 4
                # register tokens off the FFT grid, and route a pooled patch
                # summary back into them so register conditioning is input-dependent.
                grid_h="${eval:'${net.image_size} // ${net.patch_size}'}",
                register_write=True,
                hidden_dim=HIDDEN_DIM,
            ),
            sequence_mixer_norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
            # Pool register tokens -> [B, C] conditioning vector for FiLM.
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

    # ── Lightning wrapper (soft targets required when mixup is active) ──────────
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
        tags=["dynamic-deformable", "cifar10", "p1-spectral-film", f"film-{FILM_VARIANT}"],
    )

    config.autoresume = AutoResumeConfig(enabled=False)
    config.callbacks = []

    return config
