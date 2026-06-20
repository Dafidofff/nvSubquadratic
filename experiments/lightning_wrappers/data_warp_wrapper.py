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

"""Lightning wrapper for P4 data-warp experiments.

Extends :class:`~experiments.lightning_wrappers.classification_wrapper.ClassificationWrapper`
with an offset-magnitude regularizer that prevents degenerate large warps.
The penalty is ``warp_reg_weight * mean(|offsets|)`` across all
:class:`~nvsubquadratic.modules.spatial_warp.SpatialWarp` blocks in the network.

Only applied during *training*; validation and test steps use the unmodified
classification loss so that metrics are comparable across configs.
"""

from experiments.default_cfg import ExperimentConfig
from experiments.lightning_wrappers.classification_wrapper import ClassificationWrapper
from nvsubquadratic.modules.spatial_warp import SpatialWarp


class DataWarpClassificationWrapper(ClassificationWrapper):
    """Classification wrapper that adds an offset regularizer to the training loss.

    The penalty ``warp_reg_weight * mean_offset_magnitude`` discourages the
    spatial warp from making large deformations that the network has not yet
    learned to use well.  At init the offsets are exactly zero (identity warp),
    so the penalty is zero and training starts clean.

    Args:
        network: The neural network module (must contain ``SpatialWarp`` layers).
        cfg: Experiment config.
        loss: Classification loss mode (``"cross_entropy"``, ``"soft_target_ce"``,
            ``"bce"``).
        warp_reg_weight: Coefficient for the offset-magnitude regularizer.
            Default ``0.01``.  Higher values keep warps closer to identity.
    """

    def __init__(
        self,
        network,
        cfg: ExperimentConfig,
        loss: str = "cross_entropy",
        warp_reg_weight: float = 0.01,
    ):
        super().__init__(network=network, cfg=cfg, loss=loss)
        self.warp_reg_weight = warp_reg_weight
        self._warp_modules: list[SpatialWarp] = [
            m for m in network.modules() if isinstance(m, SpatialWarp)
        ]

    def training_step(self, batch, batch_idx):
        loss = super().training_step(batch, batch_idx)

        if self._warp_modules:
            mags = [
                m._last_offset_magnitude
                for m in self._warp_modules
                if m._last_offset_magnitude is not None
            ]
            if mags:
                mean_mag = sum(mags) / len(mags)
                reg_loss = self.warp_reg_weight * mean_mag
                loss = loss + reg_loss
                self.log(
                    "train/warp_offset_magnitude",
                    mean_mag.detach(),
                    on_epoch=True,
                    prog_bar=False,
                    sync_dist=self.distributed,
                )
                self.log(
                    "train/warp_reg_loss",
                    reg_loss.detach(),
                    on_epoch=True,
                    prog_bar=False,
                    sync_dist=self.distributed,
                )

        return loss

    def validation_step(self, batch, batch_idx):
        loss = super().validation_step(batch, batch_idx)

        if self._warp_modules:
            mags = [
                m._last_offset_magnitude
                for m in self._warp_modules
                if m._last_offset_magnitude is not None
            ]
            if mags:
                mean_mag = sum(mags) / len(mags)
                self.log(
                    "val/warp_offset_magnitude",
                    mean_mag.detach(),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    sync_dist=self.distributed,
                )

        return loss
