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

"""Adapter that plugs 2-D sequence mixers (e.g. Hyena) into the ViT-5 token-sequence architecture.

Drop-in replacement interface for :class:`~nvsubquadratic.modules.vit5_attention.ViT5Attention`.
**Important**: unlike ``ViT5Attention``, the adapter owns no QKV or output projections.
All projection dimensions (``hidden_dim``, ``num_heads``, etc.) must be configured
inside ``inner_mixer_cfg`` (e.g. as part of ``QKVSequenceMixer``).

Why an adapter is needed
------------------------
:class:`~nvsubquadratic.modules.vit5_attention.ViT5Attention` expects a flat
``[B, T, C]`` token sequence where ``T = num_patches + (1 if has_cls) + num_registers``.
It owns its own QKV and output projections and produces ``[B, T, C]`` output — the
residual block calls ``mixer(x)`` and adds the result back to ``x``.

2-D operators such as Hyena (wrapped in
:class:`~nvsubquadratic.modules.sequence_mixer.QKVSequenceMixer`) expect a spatial
grid ``[B, H, W, C]`` — they cannot consume the flat sequence directly. Moreover,
``QKVSequenceMixer`` provides its own QKV and output projections that are already
part of the inner mixer; duplicating projections in the adapter would waste memory
and parameters.

This module solves both issues with a *thin, stateless* reshape adapter:

1. Receives ``[B, T, C]`` from the residual block.
2. Reshapes to ``[B, T // grid_w, grid_w, C]`` (a 2-D spatial grid).
3. Delegates entirely to the inner mixer (any ``[B, H, W, C]``-in / ``[B, H, W, C]``-out
   module, e.g. ``QKVSequenceMixer(Hyena)``).  The inner mixer must be
   **shape-preserving** — its output must have the same ``[B, H, W, C]`` shape as its
   input; ``reshape`` will raise a cryptic error if the shape changes.
4. Reshapes back to ``[B, T, C]`` and returns.

The adapter itself adds **no parameters** — all learnable weights (input projection,
output projection, Hyena kernel generator) live inside the inner mixer.  Projection
dimensions (``hidden_dim``, ``num_heads``, etc.) must be configured inside
``inner_mixer_cfg``; the adapter accepts no ``hidden_dim`` argument itself.

Register tokens and the CLS token are **not** handled specially here: they are treated
as ordinary spatial positions in the grid.  The calling network
(:class:`~nvsubquadratic.networks.vit5_classification.ViT5ClassificationNet`) is
responsible for padding ``T`` so that it is exactly divisible by ``grid_w`` and for
arranging tokens into a layout that makes spatial sense to the mixer.  In the
hierarchical (:class:`~nvsubquadratic.networks.vit5_hierarchical_classification.ViT5HierarchicalClassificationNet`)
setting, ``grid_w`` must be consistent with the spatial width **after any patch-merging
stage** — the network supplies the correct ``grid_w`` at each stage.

Interface contract (same as ``ViT5Attention``)
----------------------------------------------
``forward(x, **mixer_kwargs) -> Tensor``

* Input:  ``x`` of shape ``[B, T, C]``.
* Output: tensor of shape ``[B, T, C]``.
* The inner mixer must return a tensor of the same shape ``[B, H, W, C]`` it received;
  downsampling or strided mixers are not supported.
* Optional kwargs (e.g. ``conditioning``) are forwarded verbatim to the inner mixer.

The module also exposes a ``flop_count(num_tokens, inference)`` method that
delegates to the inner mixer's ``flop_count``, matching the API used by the network
for FLOPs accounting.
"""

import torch
import torch.nn as nn

from nvsubquadratic.lazy_config import LazyConfig, instantiate


class ViT5HyenaAdapter(nn.Module):
    """Bridges ViT-5's ``[B, T, C]`` token sequences and Hyena's ``[B, H, W, C]`` spatial interface.

    The adapter is a **parameter-free reshape wrapper**: it does not own any QKV
    projection, output projection, or positional encoding.  All learnable components
    live inside ``inner_mixer`` (typically a
    :class:`~nvsubquadratic.modules.sequence_mixer.QKVSequenceMixer` wrapping a
    :class:`~nvsubquadratic.modules.hyena_nd.Hyena` instance).

    Data flow::

        x: [B, T, C]
            │
            ▼  reshape  (T → H × grid_w, where H = T // grid_w)
        x: [B, H, grid_w, C]
            │
            ▼  inner_mixer  (any [B, H, W, C]-preserving mixer)
        x: [B, H, grid_w, C]
            │
            ▼  reshape  back
        x: [B, T, C]

    What the adapter handles vs. what the inner mixer handles:

    * **Adapter**: shape contract (flat ↔ 2-D), ``flop_count`` delegation.
    * **Inner mixer**: input projection (C → 3C), Hyena global convolution,
      gating, output projection (C → C), any normalisation, and optional
      FiLM / AdaLN conditioning.  The mixer receives tensors in channels-last
      layout ``[B, H, W, C]``; if it uses channels-first convolution internally
      (as ``QKVSequenceMixer`` does) it handles the permutation itself.

    Register-token handling:
        Register tokens (and the CLS token, if present) are treated as ordinary
        spatial positions within the reshaped grid — no masking or special-casing
        is applied.  The upstream network is responsible for:

        1. Padding the sequence so that ``T % grid_w == 0``.
        2. Choosing a ``grid_w`` that places register/CLS tokens in a predictable
           row (e.g. a dedicated "register row" at the bottom of the grid), so
           that the spatial convolution inside Hyena sees a consistent layout.
        3. In the hierarchical case, supplying the correct ``grid_w`` at each
           stage after patch merging changes the spatial width.

    Attributes:
        inner_mixer (nn.Module): The instantiated 2-D sequence mixer.  Accepts
            and returns ``[B, H, W, C]`` tensors in channels-last layout.
            Typically a :class:`~nvsubquadratic.modules.sequence_mixer.QKVSequenceMixer`
            wrapping :class:`~nvsubquadratic.modules.hyena_nd.Hyena`.
        grid_w (int): Width of the 2-D spatial grid.  The height is inferred
            at runtime as ``T // grid_w``.
    """

    def __init__(
        self,
        inner_mixer_cfg: LazyConfig,
        grid_w: int,
        grid_h: int | None = None,
        register_write: bool = False,
        hidden_dim: int | None = None,
        pre_mixer_mask_cfg: LazyConfig | None = None,
    ):
        """Instantiate the adapter and its inner 2-D mixer.

        Args:
            inner_mixer_cfg: :class:`~nvsubquadratic.lazy_config.LazyConfig`
                describing the 2-D sequence mixer to instantiate (e.g.
                ``QKVSequenceMixer`` wrapping ``Hyena``).  The instantiated module
                must accept ``(x: Tensor[B, H, W, C], **kwargs)`` in channels-last
                layout and return a tensor of the same shape.  Any inner mixer
                that uses channels-first convolution (like ``QKVSequenceMixer``)
                handles the permutation internally.  Projection dimensions
                (``hidden_dim``, ``num_heads``, etc.) must be set inside this config;
                the adapter itself accepts no ``hidden_dim`` argument.
            grid_w: Width of the 2-D spatial grid.  In legacy (whole-sequence)
                mode every call to ``forward`` must supply a sequence length
                ``T`` that satisfies ``T % grid_w == 0``; the grid height is
                computed as ``H = T // grid_w``.  In a hierarchical network,
                pass the correct ``grid_w`` for each stage (after patch
                merging).  After a 2× patch-merging step, ``grid_w`` halves; the
                network's stage configuration (e.g.
                ``ViT5HierarchicalClassificationNet``) is the source of truth for
                each stage's ``grid_w``.
            grid_h: Height of the 2-D spatial (patch) grid.  When ``None``
                (default) the adapter runs in **legacy mode** and reshapes the
                *entire* ``T``-length sequence into ``[B, T // grid_w, grid_w, C]``
                — register/CLS/padding tokens are treated as ordinary grid
                positions, so this only works when ``T == grid_h * grid_w``
                exactly (i.e. ``num_registers == 0`` and no padding reaches the
                block).  When set, the adapter runs in **register-split mode**:
                the first ``grid_h * grid_w`` tokens are mixed as the spatial
                patch grid and any trailing tokens (register / CLS / aux tokens)
                are kept *off* the spatial path so they do not corrupt the
                ``grid_h × grid_w`` FFT structure.  This is what lets a config
                use ``num_registers > 0`` with the Hyena mixer.
            register_write: When ``True`` (requires ``grid_h`` set and
                ``hidden_dim`` given), the adapter writes a learned, mean-pooled
                summary of the *mixed* patch grid into the trailing (register)
                tokens after mixing: ``rest ← rest + W · mean(grid)``.  The write
                projection ``W`` is **zero-initialised**, so at init the registers
                pass through unchanged (the block is identical to the
                ``num_registers == 0`` baseline) and the model gradually learns to
                route per-input patch context into the registers.  Downstream
                register→FiLM conditioning (see ``ViT5ResidualBlock``) then becomes
                genuinely input-dependent.  When ``False`` the trailing tokens
                pass through untouched (pure register-split).
            hidden_dim: Channel dimension ``C`` of the tokens.  Required only
                when ``register_write=True`` (to size the write projection);
                ignored otherwise.
            pre_mixer_mask_cfg: Optional lazy config for a soft spatial mask
                applied to the ``[B, H, W, C]`` patch grid **before** the inner
                mixer runs each block.  When ``None`` (default) no masking is
                applied.  Intended for the P3 sparse-masking experiment
                (:class:`~nvsubquadratic.modules.spatial_mask.SpatialSoftMask`).
        """
        super().__init__()
        self.inner_mixer = instantiate(inner_mixer_cfg)
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.pre_mixer_mask = instantiate(pre_mixer_mask_cfg) if pre_mixer_mask_cfg is not None else None

        if register_write:
            if grid_h is None:
                raise ValueError("register_write=True requires grid_h to be set (register-split mode).")
            if hidden_dim is None:
                raise ValueError("register_write=True requires hidden_dim to size the write projection.")
            self.register_write_proj = nn.Linear(hidden_dim, hidden_dim)
            # Zero-init so registers are unchanged at init (== num_registers=0 baseline);
            # the model learns to route patch context into registers over training.
            nn.init.zeros_(self.register_write_proj.weight)
            nn.init.zeros_(self.register_write_proj.bias)
        else:
            self.register_write_proj = None

    def flop_count(self, num_tokens: int, inference: bool = False) -> int:
        """Delegate FLOPs accounting to the inner mixer.

        The adapter's reshape operations are pure metadata re-strides — zero
        arithmetic FLOPs — so the total cost is entirely determined by
        ``inner_mixer.flop_count``.

        Note:
            ``flop_count`` is a de-facto protocol, not enforced by a formal
            interface.  To guard against missing implementations use
            ``hasattr(adapter.inner_mixer, "flop_count")``.

        Args:
            num_tokens: Total flat sequence length ``T``.  Must satisfy
                ``T % grid_w == 0``.  The 2-D spatial dimensions passed to the
                inner mixer are ``(T // grid_w, grid_w)``.
            inference: Forwarded to the inner mixer.  Some mixers (e.g. those
                with cached Hyena kernels) report fewer FLOPs at inference time.

        Returns:
            Total FLOPs reported by the inner mixer for a ``(T // grid_w, grid_w)``
            spatial grid.

        Raises:
            AttributeError: If ``inner_mixer`` does not implement ``flop_count``.
        """
        if self.grid_h is None:
            spatial_dims = (num_tokens // self.grid_w, self.grid_w)
        else:
            # Register-split mode: the inner mixer only ever sees the patch grid.
            spatial_dims = (self.grid_h, self.grid_w)
        flops = self.inner_mixer.flop_count(spatial_dims, inference=inference)
        if self.register_write_proj is not None:
            # mean-pool over n_spatial tokens (~n*C adds) + Linear(C -> C) per sample.
            n_spatial = self.grid_h * self.grid_w
            C = self.register_write_proj.in_features
            flops += n_spatial * C + 2 * C * C
        return flops

    def forward(self, x: torch.Tensor, **mixer_kwargs) -> torch.Tensor:
        """Reshape to 2-D grid, apply the inner mixer, reshape back.

        Args:
            x: Input token sequence of shape ``[B, T, C]`` where

                * ``B`` — batch size.
                * ``T`` — total sequence length (must satisfy ``T % grid_w == 0``).
                  Typical layout (set by the network, not enforced here):
                  ``[patch_tokens (H_patch * W_patch), CLS (0 or 1),
                  register_tokens (R), padding (P)]``.
                * ``C`` — channel / hidden dimension.

            **mixer_kwargs: Keyword arguments forwarded verbatim to
                ``inner_mixer.forward``.  Common keys include:

                * ``conditioning`` — FiLM/AdaLN conditioning tensor used by
                  some Hyena configurations.
                * ``cp_group`` — process group for context-parallel (AllToAll)
                  sharding inside the Hyena operator.

                Any additional kwargs accepted by the concrete inner mixer are
                also forwarded; consult the inner mixer's docstring for the full
                list.

        Returns:
            Tensor of shape ``[B, T, C]`` — the token sequence after 2-D
            Hyena mixing.  The first ``reshape`` (to ``[B, H, W, C]``) is a
            zero-copy view when ``x`` is contiguous.  If ``inner_mixer`` returns
            a non-contiguous tensor, the final ``reshape`` (back to ``[B, T, C]``)
            triggers a contiguous copy; this does not affect correctness but can
            affect memory traffic in CUDA-graph or ``torch.compile`` contexts.
            In practice, ``QKVSequenceMixer`` returns a contiguous tensor (its
            output projection is a ``Linear`` on the last axis), so the final
            ``reshape`` is typically a free view.

        Raises:
            RuntimeError: Raised by ``torch.Tensor.reshape`` if
                ``T % grid_w != 0``, with a message reporting the mismatched
                total element count.
        """
        B, T, C = x.shape

        # Legacy whole-sequence mode: reshape every token into the grid.
        if self.grid_h is None:
            x = x.reshape(B, T // self.grid_w, self.grid_w, C)
            if self.pre_mixer_mask is not None:
                x = self.pre_mixer_mask(x)
            x = self.inner_mixer(x, **mixer_kwargs)
            x = x.reshape(B, T, C)
            return x

        # Register-split mode: only the first grid_h * grid_w tokens form the
        # spatial patch grid; trailing tokens (registers / CLS / aux) are kept
        # off the spatial path so they cannot corrupt the FFT grid structure.
        n_spatial = self.grid_h * self.grid_w
        if T < n_spatial:
            raise RuntimeError(
                f"ViT5HyenaAdapter register-split: sequence length T={T} is smaller than the "
                f"spatial grid grid_h*grid_w={n_spatial} (grid_h={self.grid_h}, grid_w={self.grid_w})."
            )

        grid = x[:, :n_spatial].reshape(B, self.grid_h, self.grid_w, C)
        if self.pre_mixer_mask is not None:
            grid = self.pre_mixer_mask(grid)
        grid = self.inner_mixer(grid, **mixer_kwargs)
        grid = grid.reshape(B, n_spatial, C)

        rest = x[:, n_spatial:]  # [B, T - n_spatial, C] register / aux tokens
        if self.register_write_proj is not None and rest.shape[1] > 0:
            # Route a learned, mean-pooled summary of the mixed patch grid into
            # the register tokens so register->FiLM conditioning is input-dependent.
            summary = self.register_write_proj(grid.mean(dim=1))  # [B, C]
            rest = rest + summary.unsqueeze(1)

        return torch.cat([grid, rest], dim=1)

    def extra_repr(self) -> str:
        """Return grid dims (and register-write / pre-mask flags) inserted into PyTorch's module repr."""
        s = f"grid_w={self.grid_w}"
        if self.grid_h is not None:
            s += f", grid_h={self.grid_h}, register_write={self.register_write_proj is not None}"
        if self.pre_mixer_mask is not None:
            s += f", pre_mixer_mask={type(self.pre_mixer_mask).__name__}"
        return s
