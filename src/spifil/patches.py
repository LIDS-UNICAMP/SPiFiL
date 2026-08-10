"""Seed-centered patch extraction.

For every seed, a patch is the concatenation of feature values over a
``kernel_size x kernel_size`` (dilated) neighborhood around it, with
out-of-bounds neighbors contributing zeros. Implemented as one
:func:`torch.nn.functional.unfold` over the whole feature map, then gathering
the columns at the seed coordinates --- batched, no Python-level loop over
seeds.

Patches are **channel-major** (``c * span**2 + kh * span + kw``), which is
what :func:`torch.nn.functional.unfold` produces and what lets a flattened
``Conv2d`` weight (``(out_c, in_c, kh, kw).view(out_c, -1)``) multiply them
directly. Scoring, selection and :mod:`spifil.nn.builder` all agree on this
layout, so no permutation ever happens on the hot path.

Even kernel sizes
-----------------
The neighborhood spans ``dy, dx in [-kernel_size // 2, kernel_size // 2]``, so
its side is ``2 * (kernel_size // 2) + 1``: exactly ``kernel_size`` for odd
sizes, and ``kernel_size + 1`` for even ones (``kernel_size=4`` gives offsets
``-2..2``, five values). Even kernels are therefore accepted but grow by one;
prefer odd sizes, which every shipped architecture uses.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from spifil.config import LayerSpec
from spifil.coords import to_linear
from spifil.types import PatchSet, Seeds

__all__ = ["extract_patches"]


def extract_patches(
    features: Tensor,
    seeds: Seeds,
    layer: LayerSpec,
    *,
    image_id: int = 0,
) -> PatchSet:
    """Extract one channel-major patch per seed from a feature map.

    Parameters
    ----------
    features
        ``(C, H, W)`` feature map the seeds index into (layer L-1's output).
    seeds
        Seed coordinates on a ``(H, W)`` grid matching ``features``.
    layer
        Supplies ``kernel_size`` and ``dilation`` for the adjacency — the
        *next* layer's config when scoring.
    image_id
        Value stamped into :attr:`PatchSet.image_ids`; the caller's index for
        this image (e.g. its position in a ``SpifilDataset``).

    Returns
    -------
    A :class:`~spifil.types.PatchSet` with one row per seed, in the same order
    as ``seeds``, and ``feats`` in **channel-major** layout (module docstring).
    """
    if len(seeds.grid) != 2:
        raise ValueError(f"extract_patches only supports 2D grids; got {seeds.grid}")
    if features.ndim != 3 or tuple(features.shape[1:]) != seeds.grid:
        raise ValueError(
            f"features must be (C, {seeds.grid[0]}, {seeds.grid[1]}) to match "
            f"seeds.grid; got {tuple(features.shape)}"
        )
    if layer.kernel_size < 1 or layer.dilation < 1:
        raise ValueError(
            f"kernel_size and dilation must be >= 1; got "
            f"{layer.kernel_size}, {layer.dilation}"
        )

    # The -k//2..k//2 window (see module docstring); equals kernel_size for
    # odd sizes, kernel_size + 1 for even ones.
    span = 2 * (layer.kernel_size // 2) + 1
    pad = layer.dilation * (layer.kernel_size // 2)

    # (1, C, H, W) -> (1, C*span*span, H*W); with this padding the number of
    # sliding positions is exactly H*W, one per pixel, row-major (x fastest)
    # --- the same order as to_linear.
    columns = F.unfold(
        features.unsqueeze(0), kernel_size=span, dilation=layer.dilation, padding=pad
    )
    columns = columns.squeeze(0).transpose(0, 1)  # (H*W, C*span*span)

    index = to_linear(seeds.coords, seeds.grid).to(columns.device)
    feats = columns.index_select(0, index).contiguous()

    n = len(seeds)
    return PatchSet(
        feats=feats,
        labels=seeds.labels.clone(),
        seed_rows=torch.arange(n, dtype=torch.int64, device=feats.device),
        image_ids=torch.full((n,), image_id, dtype=torch.int64, device=feats.device),
    )
