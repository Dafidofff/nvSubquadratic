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

"""Learnable soft spatial token mask for subquadratic Hyena mixers (P3 — Idea 1).

Differentiable token attenuation: a lightweight conv scorer produces a per-spatial-
position keep-probability in ``[0, 1]``.  Multiplying the feature map by this mask
before the global FFT mixer concentrates the convolution on informative regions
without breaking the uniform grid (the spatial dimensions are preserved — no tokens
are physically dropped).

The mean mask density is stored on the module as ``_last_mean_mask`` after every
forward pass so that an outer training wrapper can accumulate a sparsity regularizer
(see :class:`~experiments.lightning_wrappers.sparse_mask_wrapper.SparseMaskClassificationWrapper`).
"""

import torch
import torch.nn as nn


class SpatialSoftMask(nn.Module):
    r"""Soft spatial keep-mask predicted by a lightweight convolutional scorer.

    Given a channels-last feature map ``x`` of shape ``[B, H, W, C]`` (the 2-D
    spatial grid inside the Hyena adapter), the scorer computes a per-position
    keep-probability ``m ∈ (0, 1)`` and returns ``x * m``.

    Architecture:

    .. code-block:: none

        x: [B, H, W, C]
            │  permute → [B, C, H, W]
            ▼
        Conv2d(C → head_channels, 3×3, padding=1)   ← spatial context
            │  GELU
            ▼
        Conv2d(head_channels → 1, 1×1)
            │  Sigmoid
            ▼
        mask: [B, 1, H, W]
            │  permute → [B, H, W, 1]
            ▼
        x * mask   (broadcast over C)

    The scorer is zero-initialised on its output projection so that at step 0
    the sigmoid output is 0.5 everywhere (half-keep), making the initial model
    equal to the baseline scaled by 0.5 — a safe starting point from which the
    classification signal quickly learns to open the mask where needed.

    After every forward call ``self._last_mean_mask`` holds the mean mask density
    ``mean(mask)`` for the current batch (detached from the graph for logging; the
    live tensor used for the sparsity gradient is not stored here but IS captured
    by the training wrapper through a forward hook).

    Args:
        hidden_dim: Number of input feature channels ``C``.
        head_channels: Hidden width of the 3×3 conv scorer.  Default ``8``.
    """

    def __init__(self, hidden_dim: int, head_channels: int = 8):
        """Initialise scorer with zero-init output projection."""
        super().__init__()
        self._scorer = nn.Sequential(
            nn.Conv2d(hidden_dim, head_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(head_channels, 1, kernel_size=1),
        )
        # Zero-init output projection: sigmoid(0) = 0.5 at step 0.
        nn.init.zeros_(self._scorer[-1].weight)
        nn.init.zeros_(self._scorer[-1].bias)

        self._last_mean_mask: torch.Tensor | None = None  # detached, for logging
        self._mask_mean: torch.Tensor | None = None  # live tensor, for sparsity gradient

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply soft spatial mask to channels-last feature grid.

        Args:
            x: ``[B, H, W, C]`` feature map in channels-last layout.

        Returns:
            ``[B, H, W, C]`` masked features (same shape and dtype as ``x``).
        """
        x_cf = x.permute(0, 3, 1, 2).float()          # [B, C, H, W]
        logits = self._scorer(x_cf)                     # [B, 1, H, W]
        mask = torch.sigmoid(logits).to(x.dtype)        # [B, 1, H, W]
        mask_mean = mask.mean()
        self._last_mean_mask = mask_mean.detach()       # for logging only
        self._mask_mean = mask_mean                     # live: gradient flows to scorer
        mask = mask.permute(0, 2, 3, 1)                 # [B, H, W, 1]
        return x * mask
