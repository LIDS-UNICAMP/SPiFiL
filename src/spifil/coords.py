"""Spatial coordinates.

SPiFiL addresses spatial positions with explicit ``(n, D)`` integer coordinate
tensors rather than flat pixel indices. The same code then serves 2D images, 3D
volumes and 4D data, and grid-dependent index arithmetic disappears (projecting
seeds onto a pooled grid is just an integer division).

Flat indices are confined to this module, where they buy something specific:
a row-major linear index is a single sortable key per position, which is how
:meth:`spifil.types.Seeds.project` groups colliding seeds and how
:func:`spifil.patches.extract_patches` gathers unfolded columns.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["from_linear", "to_linear", "project_to_grid"]


def to_linear(coords: Tensor, grid: tuple[int, ...]) -> Tensor:
    """Flatten ``(n, D)`` coordinates to linear indices, last axis fastest.

    Row-major over ``(…, y, x)`` --- ``index = y * width + x`` in 2D ---
    generalised to any number of spatial dimensions.
    """
    _check(coords, grid)
    strides = _row_major_strides(grid, coords.device)
    return (coords * strides).sum(dim=1)


def from_linear(indices: Tensor, grid: tuple[int, ...]) -> Tensor:
    """Inverse of :func:`to_linear`: linear indices to ``(n, D)`` coordinates."""
    strides = _row_major_strides(grid, indices.device)
    coords = (indices.unsqueeze(1) // strides) % torch.tensor(
        grid, device=indices.device, dtype=torch.long
    )
    return coords


def project_to_grid(
    coords: Tensor, stride: int, grid: tuple[int, ...]
) -> tuple[Tensor, tuple[int, ...]]:
    """Project coordinates onto a grid pooled by ``stride``.

    Returns the projected coordinates and the new grid shape. Coordinates may
    collide after projection; resolving collisions (which seed survives, and
    with what rank) is the caller's business, not this function's — see
    :meth:`~spifil.types.Seeds.project`.
    """
    _check(coords, grid)
    if stride <= 1:
        return coords.clone(), grid
    new_grid = tuple(-(-g // stride) for g in grid)  # ceil division
    return coords // stride, new_grid


def _row_major_strides(grid: tuple[int, ...], device: torch.device) -> Tensor:
    strides = []
    acc = 1
    for size in reversed(grid):
        strides.append(acc)
        acc *= size
    return torch.tensor(list(reversed(strides)), device=device, dtype=torch.long)


def _check(coords: Tensor, grid: tuple[int, ...]) -> None:
    if coords.ndim != 2 or coords.shape[1] != len(grid):
        raise ValueError(
            f"coords must be (n, {len(grid)}) for grid {grid}, "
            f"got {tuple(coords.shape)}"
        )
