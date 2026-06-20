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

"""Lightning wrapper for P3 sparse-masking experiments.

Extends :class:`~experiments.lightning_wrappers.classification_wrapper.ClassificationWrapper`
with a sparsity regularizer that penalises keeping tokens.  The penalty is
``sparsity_weight * mean(mask_density_per_block)``, harvested from
:class:`~nvsubquadratic.modules.spatial_mask.SpatialSoftMask` instances in the
network via their ``_last_mean_mask`` attribute.

Only applied during *training*; validation and test steps use the unmodified
classification loss so that metrics are comparable across configs.
"""

import torchmetrics

from experiments.default_cfg import ExperimentConfig
from experiments.lightning_wrappers.classification_wrapper import ClassificationWrapper
from nvsubquadratic.modules.spatial_mask import SpatialSoftMask


class SparseMaskClassificationWrapper(ClassificationWrapper):
    """Classification wrapper that adds a per-block sparsity penalty to the training loss.

    The penalty term ``sparsity_weight * mean_mask_density`` is added only to
    the training loss.  Each :class:`~nvsubquadratic.modules.spatial_mask.SpatialSoftMask`
    block in the network contributes its mean keep-probability (stored as
    ``_last_mean_mask`` after the forward pass); the wrapper averages across all
    blocks before scaling.

    A higher ``sparsity_weight`` pushes the masks toward zero (more attenuation);
    the classification objective then opens the mask where features are useful.
    The effective keep-ratio at convergence depends on the balance between the
    two loss terms.

    Args:
        network: The neural network module (must contain ``SpatialSoftMask`` layers).
        cfg: Experiment config.
        loss: Classification loss mode (``"cross_entropy"``, ``"soft_target_ce"``,
            ``"bce"``).
        sparsity_weight: Coefficient for the sparsity regularizer.  Default ``0.1``.
    """

    def __init__(
        self,
        network,
        cfg: ExperimentConfig,
        loss: str = "cross_entropy",
        sparsity_weight: float = 0.1,
    ):
        """Initialise the wrapper and store the sparsity weight."""
        super().__init__(network=network, cfg=cfg, loss=loss)
        self.sparsity_weight = sparsity_weight

        # Cache references to all SpatialSoftMask modules at init time.
        self._mask_modules: list[SpatialSoftMask] = [
            m for m in network.modules() if isinstance(m, SpatialSoftMask)
        ]

    def training_step(self, batch, batch_idx):
        """Training step: classification loss + sparsity penalty."""
        loss = super().training_step(batch, batch_idx)

        if self._mask_modules:
            mask_densities = [
                m._mask_mean
                for m in self._mask_modules
                if m._mask_mean is not None
            ]
            if mask_densities:
                mean_density = sum(mask_densities) / len(mask_densities)
                sparsity_loss = self.sparsity_weight * mean_density
                loss = loss + sparsity_loss
                self.log(
                    "train/mask_density",
                    mean_density.detach(),
                    on_epoch=True,
                    prog_bar=False,
                    sync_dist=self.distributed,
                )
                self.log(
                    "train/sparsity_loss",
                    sparsity_loss.detach(),
                    on_epoch=True,
                    prog_bar=False,
                    sync_dist=self.distributed,
                )

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step: log mask density alongside the classification metrics."""
        loss = super().validation_step(batch, batch_idx)

        # Log keep-ratio during validation for monitoring (no penalty applied).
        if self._mask_modules:
            mask_densities = [
                m._last_mean_mask
                for m in self._mask_modules
                if m._last_mean_mask is not None
            ]
            if mask_densities:
                mean_density = sum(mask_densities) / len(mask_densities)
                self.log(
                    "val/mask_density",
                    mean_density.detach(),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    sync_dist=self.distributed,
                )

        return loss
