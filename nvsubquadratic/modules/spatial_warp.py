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

"""Learnable spatial warp for subquadratic Hyena mixers (P4 — Idea 4).

A lightweight convolutional router predicts a per-position 2-D offset field
from the current feature map; ``F.grid_sample`` then resamples the grid onto
the predicted coordinates before the global FFT mixer runs.  The spatial grid
stays perfectly uniform after resampling — the FFT path is unaffected — but
the effective receptive field is deformed per-input.

The mean absolute offset magnitude is stored as ``_last_offset_magnitude``
after every forward so that an outer training wrapper can apply a regularizer
that prevents degenerate large warps
(see :class:`~experiments.lightning_wrappers.data_warp_wrapper.DataWarpClassificationWrapper`).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialWarp(nn.Module):
    r"""Spatial transformer that warps the feature grid before the Hyena mixer.

    Given a channels-last feature map ``x`` of shape ``[B, H, W, C]`` (the 2-D
    spatial grid inside the Hyena adapter), a lightweight conv router predicts a
    2-D offset field ``δ`` of shape ``[B, H, W, 2]`` in normalized coordinates
    ``[-1, 1]``.  The feature map is then resampled at positions
    ``base_grid + δ`` using bilinear interpolation, preserving the ``H × W``
    grid shape so the FFT path downstream sees a uniform grid.

    Architecture:

    .. code-block:: none

        x: [B, H, W, C]
            │  permute → [B, C, H, W]
            ▼
        Conv2d(C → head_channels, 3×3, padding=1)   ← spatial context
            │  GELU
            ▼
        Conv2d(head_channels → 2, 1×1)              ← (dx, dy) per position
            │
            ▼
        offsets: [B, 2, H, W]  (zero at init)
            │  permute → [B, H, W, 2]  +  base_grid [-1,1]
            ▼
        grid: [B, H, W, 2]
            │  F.grid_sample (bilinear, padding_mode='border')
            ▼
        x_warped: [B, H, W, C]

    The router output projection is zero-initialised so offsets are
    exactly zero at step 0 — training starts from the identity warp, making
    this equivalent to the P0 baseline at initialization.

    Args:
        hidden_dim: Number of input feature channels ``C``.
        head_channels: Hidden width of the 3×3 conv router.  Default ``16``.
    """

    def __init__(self, hidden_dim: int, head_channels: int = 16):
        super().__init__()
        self._router = nn.Sequential(
            nn.Conv2d(hidden_dim, head_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(head_channels, 2, kernel_size=1),
        )
        # Zero-init: offsets = 0 at step 0 → identity warp → == P0 baseline.
        nn.init.zeros_(self._router[-1].weight)
        nn.init.zeros_(self._router[-1].bias)

        self._last_offset_magnitude: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Warp the spatial feature grid with a learned per-position offset.

        Args:
            x: ``[B, H, W, C]`` feature map in channels-last layout.

        Returns:
            ``[B, H, W, C]`` resampled features (same shape and dtype as ``x``).
        """
        B, H, W, C = x.shape
        x_cf = x.permute(0, 3, 1, 2).float()   # [B, C, H, W]

        offsets = self._router(x_cf)             # [B, 2, H, W]; zero at init
        self._last_offset_magnitude = offsets.detach().abs().mean()

        # Build normalized base grid: x-coords (dim 0 of grid) go left→right,
        # y-coords (dim 1) go top→bottom, both in [-1, 1].
        # grid_sample convention: grid[..., 0] = x (width), grid[..., 1] = y (height).
        ys = torch.linspace(-1.0, 1.0, H, device=x.device, dtype=torch.float32)
        xs = torch.linspace(-1.0, 1.0, W, device=x.device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")        # [H, W]
        base = torch.stack([grid_x, grid_y], dim=-1)                   # [H, W, 2]
        base = base.unsqueeze(0).expand(B, -1, -1, -1)                 # [B, H, W, 2]

        grid = base + offsets.permute(0, 2, 3, 1)   # [B, H, W, 2]

        x_warped = F.grid_sample(
            x_cf, grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )   # [B, C, H, W]

        return x_warped.permute(0, 2, 3, 1).to(x.dtype)   # [B, H, W, C]
