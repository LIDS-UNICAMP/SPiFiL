"""Filter selection.

A :class:`Selector` turns a layer's full ranked-seed :class:`~spifil.types.PatchSet`
(every seed with ``rank > 0``) plus a per-class filter budget into a boolean
keep-mask over that same patch set.

**Candidate pooling lives inside the selector, not a separate stage.**
Patches are extracted once, for every ranked seed, and
:class:`DiversitySelector` narrows to a per-class candidate pool internally
(``per_class * pool_factor`` best-ranked seeds) before running the greedy
pick.

One property of that narrowing is not obvious from the parameters: the
pool-size cap ``total_ranked // n_classes`` is a single value shared by every
class rather than each class's own headroom, so a class with few ranked seeds
can hold back a well-stocked one from using candidates that were available.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
from torch import Tensor

from spifil.metrics import DistanceMetric, Euclidean
from spifil.types import PatchSet

__all__ = ["DiversitySelector", "Selector", "TopN", "candidate_order"]

# Below this spread the candidate pool is treated as a single point and the
# diversity term is left unnormalized, rather than dividing by ~0.
_DEGENERATE_SPREAD = 1e-6


def candidate_order(patches: PatchSet, ranks: Tensor) -> Tensor:
    """Row indices of ``patches`` in canonical candidate order.

    Image-major, rank-**descending** within each image; two stable sorts
    compose that lexicographic order.

    It is used twice: it is the order :class:`DiversitySelector`'s greedy loop
    scans, so it decides exact ties, and it is the order filters are laid out
    in a built layer. Filter order is otherwise free --- permuting a layer's
    filters permutes the next layer's input channels consistently, and
    scoring, distances and selection are all invariant under that --- so
    fixing one order costs nothing and makes runs directly comparable.
    """
    by_rank_desc = torch.argsort(-ranks, stable=True)
    return by_rank_desc[torch.argsort(patches.image_ids[by_rank_desc], stable=True)]


def _first_argmax(values: Tensor) -> int:
    """Index of the maximum, lowest index winning ties --- on CPU and CUDA
    alike.

    ``torch.argmax`` does not promise which tied index it returns, and its
    answer can differ between devices; the greedy loop needs a deterministic
    one.
    """
    (winners,) = (values == values.max()).nonzero(as_tuple=True)
    return int(winners[0].item())


@runtime_checkable
class Selector(Protocol):
    """Turns ranked candidates into a boolean keep-mask."""

    def __call__(
        self, patches: PatchSet, ranks: Tensor, n_per_class: dict[int, int]
    ) -> Tensor:
        """``(n,)`` bool — ``True`` for patches selected as filters."""
        ...


class DiversitySelector:
    """Greedy per-class selection trading rank against feature diversity.

    ``patches`` must be **every** seed with ``rank > 0`` for the layer, merged
    across images in dataset order: the pool-size cap is derived from
    ``len(patches)``, so handing this a pre-filtered subset silently shrinks
    the candidate pools. Ranks must be 1-based, as
    :func:`~spifil.scoring.per_class_ranks` produces them — a rank of 0 means
    "unranked" upstream and would be treated here as better than rank 1,
    scoring above 1.0 on the normalized scale.

    Parameters
    ----------
    alpha
        Score/diversity blend, ``1`` = pure rank, ``0`` = pure diversity
    pool_factor
        Candidate pool size per class as a multiple of the class's filter
        budget, before greedy narrowing.
    metric
        Distance used for the diversity term. Defaults to
        :class:`~spifil.metrics.Euclidean`
    """

    def __init__(
        self,
        alpha: float = 0.5,
        pool_factor: int = 3,
        metric: DistanceMetric | None = None,
    ) -> None:
        self.alpha = alpha
        self.pool_factor = pool_factor
        self.metric = metric if metric is not None else Euclidean()

    def __call__(
        self, patches: PatchSet, ranks: Tensor, n_per_class: dict[int, int]
    ) -> Tensor:
        device = patches.feats.device
        mask = torch.zeros(len(patches), dtype=torch.bool, device=device)
        n_classes = len(n_per_class)
        if n_classes == 0:
            return mask
        total_ranked = len(patches)
        ranks = ranks.to(device)

        for cls in sorted(n_per_class):
            per_class = n_per_class[cls]
            (idx,) = (patches.labels == cls).nonzero(as_tuple=True)
            if idx.numel() == 0 or per_class <= 0:
                continue

            cand_per_class = per_class * self.pool_factor
            if cand_per_class * n_classes > total_ranked:
                cand_per_class = total_ranked // n_classes
            cand_per_class = min(cand_per_class, int(idx.numel()))
            if cand_per_class <= 0:
                continue

            # Pool membership: the cand_per_class best-ranked seeds of this
            # class.
            rank_order = torch.argsort(ranks[idx], stable=True)  # ascending
            members = idx[rank_order[:cand_per_class]]

            # Scan order for the greedy tie-break: image-major (dataset
            # order), rank-descending within an image --- see
            # `candidate_order`. Two stable sorts compose a lexicographic
            # sort: sort by the secondary key first, then by the primary key.
            by_rank_desc = members[torch.argsort(-ranks[members], stable=True)]
            cand_idx = by_rank_desc[
                torch.argsort(patches.image_ids[by_rank_desc], stable=True)
            ]

            n_pick = min(per_class, cand_idx.numel())
            picked_local = self._select_class(
                patches.feats[cand_idx], ranks[cand_idx], n_pick
            )
            mask[cand_idx[picked_local]] = True

        return mask

    def _select_class(self, feats: Tensor, ranks: Tensor, n_pick: int) -> Tensor:
        """Greedy pick within one class's candidate pool.

        Returns int64 indices local to ``feats``/``ranks``.
        """
        device = feats.device
        n = feats.shape[0]
        worst_rank = max(float(ranks.max().item()), 1.0)
        norm_score = 1.0 - (ranks.to(torch.float32) - 1.0) / worst_rank

        self.metric.fit(feats)

        # The first-listed best rank wins. Ranks are unique within a class in
        # a real run, so this only bites on synthetic input.
        best_start = _first_argmax(-ranks)
        picked = [best_start]
        picked_mask = torch.zeros(n, dtype=torch.bool, device=device)
        picked_mask[best_start] = True

        min_dist = self.metric.pairwise(feats, feats[best_start : best_start + 1])
        min_dist = min_dist.squeeze(1)

        while len(picked) < n_pick:
            unpicked = ~picked_mask
            max_min_dist = min_dist[unpicked].max()
            if max_min_dist.item() < _DEGENERATE_SPREAD:
                max_min_dist = torch.ones((), dtype=min_dist.dtype, device=device)

            diversity = min_dist / max_min_dist
            combined = self.alpha * norm_score + (1.0 - self.alpha) * diversity
            combined = combined.masked_fill(picked_mask, float("-inf"))

            best_j = _first_argmax(combined)
            picked.append(best_j)
            picked_mask[best_j] = True

            new_dist = self.metric.pairwise(feats, feats[best_j : best_j + 1])
            min_dist = torch.minimum(min_dist, new_dist.squeeze(1))

        return torch.tensor(picked, dtype=torch.int64, device=device)


class TopN:
    """Per-class top-N by rank, with no diversity term.

    Equivalent to :class:`DiversitySelector` with ``alpha=1`` but without the
    candidate pooling, and useful as a baseline: it isolates how much of the
    result comes from ranking alone.
    """

    def __call__(
        self, patches: PatchSet, ranks: Tensor, n_per_class: dict[int, int]
    ) -> Tensor:
        device = patches.feats.device
        mask = torch.zeros(len(patches), dtype=torch.bool, device=device)
        ranks = ranks.to(device)
        for cls, target_n in n_per_class.items():
            if target_n <= 0:
                continue
            (idx,) = (patches.labels == cls).nonzero(as_tuple=True)
            if idx.numel() == 0:
                continue
            order = torch.argsort(ranks[idx], stable=True)
            keep = min(target_n, int(idx.numel()))
            mask[idx[order[:keep]]] = True
        return mask
