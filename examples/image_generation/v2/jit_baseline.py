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

"""JiT-B/4 baseline for class-conditional TinyImageNet 64x64 flow-matching diffusion.

Reference patchified DiT-style transformer (256 tokens). Derived from the
ImageNet v1 ``jit_baseline.py`` by swapping the dataset to TinyImageNet
(200 classes) and disabling online FID (no TinyImageNet reference stats yet).
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
from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.networks.jit import JiT_models


PLACEHOLDER = None
WANDB_ENTITY = "dafidofff"

# Dataset ----------------------------------------------------------------------
BATCH_SIZE = 128  # Per GPU (override per-launch; light model, 256 tokens)
NUM_WORKERS = min(12, os.cpu_count() - 2 or 4)
FINAL_IMAGE_SIZE = 64
PATCH_SIZE = FINAL_IMAGE_SIZE // 16  # 4 -> JiT-B/4 -> 16x16 = 256 tokens
HF_DATASET = "zh-plus/tiny-imagenet"
HF_CACHE = os.environ.get("TINYIMAGENET_CACHE", os.path.expanduser("~/.cache/tinyimagenet"))

# Network ----------------------------------------------------------------------
NUM_CLASSES = 200  # TinyImageNet

# Optimisation -----------------------------------------------------------------
TRAINING_ITERATIONS = 300_000  # large cap; runs are wall-clock bounded (see launcher)
WARMUP_ITERATIONS_PERCENTAGE = 0.025
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

    # JiT Model (patchified, 256 tokens)
    config.net = LazyConfig(JiT_models[f"JiT-B/{PATCH_SIZE}"])(
        input_size=FINAL_IMAGE_SIZE,
        num_classes=NUM_CLASSES,
    )

    config.lightning_wrapper_class = LazyConfig(DiffusionWrapper)()

    config.optimizer = LazyConfig(torch.optim.Adam)(
        params=PLACEHOLDER,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
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
        job_group="tinyimagenet_diffusion_jit_baseline",
        entity=WANDB_ENTITY,
        tags=["jit", "attention", "diffusion", "tinyimagenet64", "reference"],
    )

    return config
