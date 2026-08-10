"""Scoring and per-class ranks.

A :class:`Scorer` turns a merged :class:`~spifil.types.PatchSet` (all images'
patches for one layer, in dataset order) into one score per sample, higher
meaning "more useful as a filter". :func:`per_class_ranks` converts scores to
the 1-based per-class rank the rest of the pipeline consumes;
:func:`scatter_ranks` writes those ranks back into each image's
:class:`~spifil.types.Seeds`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor

from spifil.metrics import DistanceMetric
from spifil.types import PatchSet, Seeds

__all__ = [
    "DistanceSumScorer",
    "FisherScorer",
    "Scorer",
    "per_class_ranks",
    "scatter_ranks",
]


@runtime_checkable
class Scorer(Protocol):
    """Turns a patch set into one discriminability score per sample."""

    def __call__(self, patches: PatchSet, metric: DistanceMetric) -> Tensor:
        """``(n,)`` scores, higher = better. Ranks are assigned separately."""
        ...


class FisherScorer:
    """``avg(dist to other classes) / (avg(dist to own class) + 1e-6)``."""

    def __call__(self, patches: PatchSet, metric: DistanceMetric) -> Tensor:
        metric.fit(patches.feats)
        dist = metric.pairwise(patches.feats, patches.feats)
        return _fisher_ratio(dist, patches.labels)


class DistanceSumScorer:
    """Unsupervised baseline: total pairwise distance to every other sample."""

    def __call__(self, patches: PatchSet, metric: DistanceMetric) -> Tensor:
        metric.fit(patches.feats)
        dist = metric.pairwise(patches.feats, patches.feats).clone()
        dist.fill_diagonal_(0.0)  # exclude the self-distance
        return dist.sum(dim=1)


def per_class_ranks(scores: Tensor, labels: Tensor) -> Tensor:
    """Convert scores to 1-based per-class ranks (1 = best in that class).

    Ties keep the order they arrive in (stable sort), so ranks are fully
    determined by the order the patches were merged in --- image order, then
    within-image seed order, which is what :meth:`~spifil.types.PatchSet.cat`
    produces.
    """
    n = scores.shape[0]
    ranks = torch.zeros(n, dtype=torch.int64, device=scores.device)
    for cls in torch.unique(labels).tolist():
        (idx,) = (labels == cls).nonzero(as_tuple=True)
        order = torch.argsort(scores[idx], descending=True, stable=True)
        ranks[idx[order]] = torch.arange(
            1, idx.numel() + 1, dtype=torch.int64, device=scores.device
        )
    return ranks


def scatter_ranks(
    patches: PatchSet, ranks: Tensor, seeds: Sequence[Seeds]
) -> list[Seeds]:
    """Write per-class ranks back into each image's ``Seeds``.

    ``seeds`` must be indexed the way :attr:`PatchSet.image_ids` is (the same
    image order the patches were extracted and merged in). A seed with no
    corresponding row in ``patches`` --- dropped before scoring --- keeps
    rank 0, meaning "unranked", and is excluded downstream.
    """
    updated = []
    for image_id, image_seeds in enumerate(seeds):
        mask = patches.image_ids == image_id
        updated.append(image_seeds.with_ranks(patches.seed_rows[mask], ranks[mask]))
    return updated


def _fisher_ratio(dist: Tensor, labels: Tensor) -> Tensor:
    n = dist.shape[0]
    eye = torch.eye(n, dtype=torch.bool, device=dist.device)
    same = (labels[:, None] == labels[None, :]) & ~eye
    diff = labels[:, None] != labels[None, :]

    cnt_intra = same.sum(dim=1)
    cnt_inter = diff.sum(dim=1)
    sum_intra = (dist * same).sum(dim=1)
    sum_inter = (dist * diff).sum(dim=1)

    zero = torch.zeros_like(sum_intra)
    avg_intra = torch.where(cnt_intra > 0, sum_intra / cnt_intra.clamp(min=1), zero)
    avg_inter = torch.where(cnt_inter > 0, sum_inter / cnt_inter.clamp(min=1), zero)
    return avg_inter / (avg_intra + 1e-6)
