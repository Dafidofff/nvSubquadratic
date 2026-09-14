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


"""Post-creation override: swap Hyena kernels to block-diagonal multi-ω₀ SIREN.

Operates on an already-instantiated ``ViT5HierarchicalNet`` via module surgery:
replaces ``gconv.kernel`` and ``gconv.mask`` on each stage's blocks with real
``nn.Module`` instances.

Production block-diagonal defaults (from vit5_hybrid ablation):
    num_blocks=8, omega_0_min=1.0, omega_0_max=12.0, schedule="linear",
    off_block_scale=0.1

ω₀ schedule is scaled proportionally with spatial resolution at each stage
(reference: ``examples/vit5_imagenet/vit5_hybrid/_blockdiag.py`` header note):
    ω₀_min_stage = OMEGA_0_MIN_BASE × (stage_h / 14)
    ω₀_max_stage = OMEGA_0_MAX_BASE × (stage_h / 14)

``BlockAlignedGaussianModulationND`` replaces ``GaussianModulationND`` so the
widest Gaussian channels are aligned to the lowest-ω₀ kernel blocks.

Note on grid_size:
    ``CKConvND.__init__`` normally injects ``grid_size = 2 * L_cache - 1``
    into the mask config.  Since we do post-hoc module surgery we supply it
    explicitly: ``grid_size = 2 * stage_h - 1`` (valid for grid_type="double").
"""

from examples.vit5_imagenet.v5_patchmerge._base_config import (
    KERNEL_EMBEDDING_DIM,
    KERNEL_HIDDEN_OMEGA_0,
    KERNEL_MLP_HIDDEN_DIM,
    KERNEL_NUM_LAYERS,
    STAGE_HEIGHTS,
)
from nvsubquadratic.modules.kernels_nd import BlockDiagonalMultiOmegaSIRENKernelND
from nvsubquadratic.modules.masks_nd import BlockAlignedGaussianModulationND
from nvsubquadratic.networks.vit5_hierarchical import ViT5HierarchicalNet


# ── Block-diagonal defaults ───────────────────────────────────────────────────

KERNEL_BLOCK_DIAG_NUM_BLOCKS = 8
OMEGA_0_MIN_BASE = 1.0
OMEGA_0_MAX_BASE = 12.0
KERNEL_BLOCK_DIAG_SCHEDULE = "linear"
KERNEL_BLOCK_DIAG_OFF_BLOCK_SCALE = 0.1
_REF_HEIGHT = 14


def apply_block_diag_overrides(net: ViT5HierarchicalNet) -> None:
    """Replace scalar-ω₀ SIREN kernel + mask in every stage with block-diagonal variants.

    Mutates ``net`` in place.  Call *after* ``build_hierarchical_net()``.

    Args:
        net: An already-instantiated ``ViT5HierarchicalNet``.
    """
    base_dim = net.stem.proj.out_channels  # C₁ of the first stage

    for stage_idx, (stage_blocks, stage_h) in enumerate(zip(net.stages, STAGE_HEIGHTS)):
        scale = stage_h / _REF_HEIGHT
        omega_0_min = OMEGA_0_MIN_BASE * scale
        omega_0_max = OMEGA_0_MAX_BASE * scale
        hidden_dim = base_dim * (2**stage_idx)
        # For grid_type="double", CKConvND uses grid_size = 2 * L_cache - 1.
        grid_size = 2 * stage_h - 1

        for block in stage_blocks:
            # block.sequence_mixer → QKVSequenceMixer
            # block.sequence_mixer.mixer → Hyena
            # block.sequence_mixer.mixer.global_conv → CKConvND
            gconv = block.sequence_mixer.mixer.global_conv

            gconv.kernel = BlockDiagonalMultiOmegaSIRENKernelND(
                data_dim=2,
                out_dim=hidden_dim,
                mlp_hidden_dim=KERNEL_MLP_HIDDEN_DIM,
                num_layers=KERNEL_NUM_LAYERS,
                embedding_dim=KERNEL_EMBEDDING_DIM,
                L_cache=stage_h,
                use_bias=True,
                hidden_omega_0=KERNEL_HIDDEN_OMEGA_0,
                num_blocks=KERNEL_BLOCK_DIAG_NUM_BLOCKS,
                omega_0_min=omega_0_min,
                omega_0_max=omega_0_max,
                schedule=KERNEL_BLOCK_DIAG_SCHEDULE,
                off_block_scale=KERNEL_BLOCK_DIAG_OFF_BLOCK_SCALE,
            )
            gconv.mask = BlockAlignedGaussianModulationND(
                data_dim=2,
                num_channels=hidden_dim,
                grid_size=grid_size,
                min_attenuation_at_step=0.1,
                max_attenuation_at_limit=0.95,
                init_extent=1.0,
                parametrization="direct",
            )
