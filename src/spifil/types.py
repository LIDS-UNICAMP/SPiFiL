"""Shared data structures.

These are the objects that flow between pipeline stages: coordinate/label
tensors plus the grid they refer to, with no behavior beyond what every stage
needs.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from spifil.coords import from_linear, project_to_grid, to_linear

__all__ = ["PatchSet", "Seeds"]


@dataclass
class Seeds:
    """Per-image labeled feature points: one per superpixel.

    Spatial-rank agnostic: ``coords`` is ``(n, D)`` for a ``D``-dimensional
    ``grid``, so the same structure serves 2D images and 3D volumes. Axes run
    slowest-first, matching torch's ``(C, [Z,] Y, X)`` feature maps.

    Attributes
    ----------
    coords
        ``(n, D)`` int64 spatial positions.
    labels
        ``(n,)`` int64 class labels. Every seed of an image carries that
        image's class, parsed from its filename.
    ranks
        ``(n,)`` int64 per-class rank, 1 = most discriminative. ``0`` means
        unranked, which excludes the seed from filter selection downstream.
    grid
        Spatial shape the coordinates index into, e.g. ``(H, W)``.
    """

    coords: Tensor
    labels: Tensor
    ranks: Tensor
    grid: tuple[int, ...]

    def __post_init__(self) -> None:
        n, d = self.coords.shape if self.coords.ndim == 2 else (-1, -1)
        if d != len(self.grid):
            raise ValueError(
                f"coords must be (n, {len(self.grid)}) for grid {self.grid}, "
                f"got {tuple(self.coords.shape)}"
            )
        if self.labels.shape != (n,) or self.ranks.shape != (n,):
            raise ValueError(
                f"labels {tuple(self.labels.shape)} and ranks "
                f"{tuple(self.ranks.shape)} must both be ({n},)"
            )

    def __len__(self) -> int:
        return int(self.coords.shape[0])

    @classmethod
    def from_coords(cls, coords: Tensor, label: int, grid: tuple[int, ...]) -> Seeds:
        """Build unranked seeds that all share one class label.

        This is the layer-0 case: superpixel centers inherit the image's class.
        """
        n = coords.shape[0]
        return cls(
            coords=coords.to(torch.int64),
            labels=torch.full((n,), label, dtype=torch.int64),
            ranks=torch.zeros(n, dtype=torch.int64),
            grid=grid,
        )

    def sorted_by_coords(self) -> Seeds:
        """Return the seeds ordered lexicographically by coordinate.

        A canonical, position-derived order, so two seed sets holding the same
        points compare equal regardless of the order they were discovered in.
        """
        # Row-major linear indices order exactly as a slowest-axis-first
        # lexicographic sort of the coordinates.
        order = torch.argsort(to_linear(self.coords, self.grid), stable=True)
        return Seeds(
            coords=self.coords[order],
            labels=self.labels[order],
            ranks=self.ranks[order],
            grid=self.grid,
        )

    def project(self, stride: int) -> Seeds:
        """Project onto a grid pooled by ``stride``, merging collisions.

        After a layer pools its features, its seeds have to follow.
        Coordinates become ``coords // stride`` on a ``ceil(size / stride)``
        grid, and seeds that land on the same pooled pixel merge into one.

        A merged seed keeps the **best (lowest positive) rank** of the group —
        the pooled pixel is as discriminative as the most discriminative seed
        inside it --- and ``0`` only if no seed in the group was ranked at all.
        Its label comes from the first seed of the group in the current order.

        Output rows are ordered lexicographically by projected coordinate, so
        the result does not depend on the order seeds arrived in.
        ``stride <= 1`` is a no-op copy, duplicates and all.
        """
        if stride <= 1:
            return Seeds(
                coords=self.coords.clone(),
                labels=self.labels.clone(),
                ranks=self.ranks.clone(),
                grid=self.grid,
            )

        coords, grid = project_to_grid(self.coords, stride, self.grid)
        index = to_linear(coords, grid)
        # Group the seeds that landed on the same pooled pixel. `pooled` is
        # ascending, which is what puts the output in coordinate order.
        pooled, group = torch.unique(index, return_inverse=True)

        n = len(self)
        # Rank 0 means "unranked". Map it to the largest int64 so it loses
        # every `amin` against a real rank, then map it back afterwards.
        unranked = torch.iinfo(torch.int64).max
        candidates = torch.where(self.ranks > 0, self.ranks, unranked)
        best = torch.full((pooled.numel(),), unranked, dtype=torch.int64)
        best.scatter_reduce_(0, group, candidates, reduce="amin")
        best = torch.where(best == unranked, torch.zeros_like(best), best)

        # Each group takes the label of its lowest-numbered member.
        rows = torch.arange(n, dtype=torch.int64)
        first = torch.full((pooled.numel(),), n, dtype=torch.int64)
        first.scatter_reduce_(0, group, rows, reduce="amin")

        return Seeds(
            coords=from_linear(pooled, grid),
            labels=self.labels[first],
            ranks=best,
            grid=grid,
        )

    def with_ranks(self, seed_rows: Tensor, ranks: Tensor) -> Seeds:
        """Return a copy with ``.ranks`` set from a scoring pass.

        ``seed_rows`` indexes into this ``Seeds`` (e.g. :attr:`PatchSet.seed_rows`
        restricted to this image); ``ranks`` holds the per-class rank for each of
        those rows. Every other seed is reset to ``0`` (unranked) first, so a
        seed that did not survive into the scored dataset drops out downstream.
        """
        new_ranks = torch.zeros(len(self), dtype=torch.int64)
        new_ranks[seed_rows] = ranks.to(torch.int64)
        return Seeds(
            coords=self.coords, labels=self.labels, ranks=new_ranks, grid=self.grid
        )


@dataclass
class PatchSet:
    """Seed-centered im2col patches: the dataset the scorers consume.

    Built by :func:`spifil.patches.extract_patches`, one per image, and merged
    across images with :meth:`cat` before scoring (every scorer runs on one
    global dataset, not per image).

    Attributes
    ----------
    feats
        ``(n, c * prod(kernel))`` float32, **channel-major** — ``F.unfold``'s
        native layout; see :mod:`spifil.patches`.
    labels
        ``(n,)`` int64 seed class, copied from the originating ``Seeds``.
    seed_rows
        ``(n,)`` int64 row index into the originating (per-image) ``Seeds``,
        used to scatter scores/ranks back with :meth:`Seeds.with_ranks`.
    image_ids
        ``(n,)`` int64 index of the image each patch came from, into whatever
        image list the caller extracted from (e.g. ``SpifilDataset`` order).
    """

    feats: Tensor
    labels: Tensor
    seed_rows: Tensor
    image_ids: Tensor

    def __post_init__(self) -> None:
        n = self.feats.shape[0] if self.feats.ndim == 2 else -1
        shapes = {
            "labels": self.labels.shape,
            "seed_rows": self.seed_rows.shape,
            "image_ids": self.image_ids.shape,
        }
        if self.feats.ndim != 2 or any(shape != (n,) for shape in shapes.values()):
            raise ValueError(
                f"feats is {tuple(self.feats.shape)}; labels, seed_rows and "
                f"image_ids must all be ({n},), got "
                f"{ {k: tuple(v) for k, v in shapes.items()} }"
            )

    def __len__(self) -> int:
        return int(self.feats.shape[0])

    def to(self, device: str | torch.device) -> PatchSet:
        """Move every field to ``device``, keeping dtypes and order.

        Scoring and selection are O(N^2) and the natural things to push to a
        GPU (device-agnostic, no ``.cuda()`` inside components), so the move
        belongs to the data, not to the components consuming it.
        """
        return PatchSet(
            feats=self.feats.to(device),
            labels=self.labels.to(device),
            seed_rows=self.seed_rows.to(device),
            image_ids=self.image_ids.to(device),
        )

    def select(self, index: Tensor) -> PatchSet:
        """Return the subset of rows named by ``index``.

        ``index`` is either a boolean mask over the rows (a
        :class:`~spifil.selection.Selector`'s output) or an integer index
        tensor, in which case the result takes *its* order.
        """
        return PatchSet(
            feats=self.feats[index],
            labels=self.labels[index],
            seed_rows=self.seed_rows[index],
            image_ids=self.image_ids[index],
        )

    @classmethod
    def cat(cls, patches: Sequence[PatchSet]) -> PatchSet:
        """Concatenate per-image patch sets into one global dataset.

        Order is preserved (first patch set first) --- this is what makes the
        global dataset, and therefore per-class ranks near score ties,
        reproducible.
        """
        if not patches:
            raise ValueError("cat() needs at least one PatchSet")
        return cls(
            feats=torch.cat([p.feats for p in patches], dim=0),
            labels=torch.cat([p.labels for p in patches], dim=0),
            seed_rows=torch.cat([p.seed_rows for p in patches], dim=0),
            image_ids=torch.cat([p.image_ids for p in patches], dim=0),
        )
