"""DiversitySelector / TopN: candidate pooling + greedy diversity pick
(the rank/diversity greedy pick).
"""

from __future__ import annotations

import pytest
import torch

from spifil.selection import DiversitySelector, Selector, TopN
from spifil.types import PatchSet


def make_patches(
    feats: list[list[float]],
    labels: list[int],
    seed_rows: list[int] | None = None,
    image_ids: list[int] | None = None,
) -> PatchSet:
    n = len(labels)
    return PatchSet(
        feats=torch.tensor(feats, dtype=torch.float32),
        labels=torch.tensor(labels, dtype=torch.int64),
        seed_rows=torch.tensor(seed_rows if seed_rows is not None else list(range(n))),
        image_ids=torch.tensor(
            image_ids if image_ids is not None else [0] * n, dtype=torch.int64
        ),
    )


def naive_diversity_select(
    feats: torch.Tensor,
    labels: torch.Tensor,
    ranks: torch.Tensor,
    image_ids: torch.Tensor,
    n_per_class: dict[int, int],
    alpha: float,
    pool_factor: int,
) -> torch.Tensor:
    """The two-stage flow written out as loops: build the candidate pool,
    then greedily select within it. An unvectorized cross-check for
    ``DiversitySelector``.

    The candidate *scan* order (as opposed to pool *membership*) is
    image-major, rank-descending within an image; see
    :func:`spifil.selection.candidate_order`.
    """
    n = feats.shape[0]
    total_ranked = n
    n_classes = len(n_per_class)
    mask = [False] * n
    labels_list = labels.tolist()
    ranks_list = ranks.tolist()
    image_ids_list = image_ids.tolist()
    feats_list = feats.tolist()

    def dist_sq(a: int, b: int) -> float:
        pairs = zip(feats_list[a], feats_list[b], strict=True)
        return sum((x - y) ** 2 for x, y in pairs)

    for cls in sorted(n_per_class):
        per_class = n_per_class[cls]
        idx = [i for i in range(n) if labels_list[i] == cls]
        if not idx or per_class <= 0:
            continue

        cand_per_class = per_class * pool_factor
        if cand_per_class * n_classes > total_ranked:
            cand_per_class = total_ranked // n_classes
        cand_per_class = min(cand_per_class, len(idx))
        if cand_per_class <= 0:
            continue

        members = sorted(idx, key=lambda i: ranks_list[i])[:cand_per_class]
        by_rank_desc = sorted(members, key=lambda i: -ranks_list[i])
        cand = sorted(by_rank_desc, key=lambda i: image_ids_list[i])
        n_pick = min(per_class, len(cand))

        max_handicap = max(1.0, max(ranks_list[i] for i in cand))
        norm_score = {i: 1.0 - (ranks_list[i] - 1) / max_handicap for i in cand}

        best_start = min(cand, key=lambda i: ranks_list[i])
        picked = [best_start]
        mask[best_start] = True
        min_dist = dict.fromkeys(cand, 1e30)
        for i in cand:
            if i != best_start:
                min_dist[i] = min(min_dist[i], dist_sq(best_start, i))

        while len(picked) < n_pick:
            unpicked = [i for i in cand if i not in picked]
            max_min_dist = max((min_dist[i] for i in unpicked), default=0.0)
            if max_min_dist < 1e-12:
                max_min_dist = 1.0

            best_j, best_val = -1, -1e30
            for i in unpicked:
                diversity = (min_dist[i] ** 0.5) / (max_min_dist**0.5)
                combined = alpha * norm_score[i] + (1 - alpha) * diversity
                if combined > best_val:
                    best_val, best_j = combined, i

            picked.append(best_j)
            mask[best_j] = True
            for i in cand:
                if i not in picked:
                    min_dist[i] = min(min_dist[i], dist_sq(best_j, i))

    return torch.tensor(mask, dtype=torch.bool)


class TestDiversitySelector:
    def test_diversity_can_beat_the_second_best_rank(self) -> None:
        """One class, points on a line: rank1=0, rank2=1, rank3=5, rank4=20.
        alpha=0.5, pool_factor large enough to admit all 4. Greedy always
        starts at rank 1; for the second pick, the far outlier (pos 20) beats
        the next-best-ranked point (pos 1) because its diversity term wins:
        combined(pos20) = .5*.25 + .5*1.0 = .625 > combined(pos1) = .5*.75 + .5*.05 = .4
        """
        patches = make_patches([[0.0], [1.0], [5.0], [20.0]], [0, 0, 0, 0])
        ranks = torch.tensor([1, 2, 3, 4])

        selector = DiversitySelector(alpha=0.5, pool_factor=4)
        mask = selector(patches, ranks, {0: 2})

        assert mask.tolist() == [True, False, False, True]

    def test_pure_score_alpha_ignores_diversity(self) -> None:
        """Same setup, alpha=1.0 (pure rank): picks the two best-ranked points."""
        patches = make_patches([[0.0], [1.0], [5.0], [20.0]], [0, 0, 0, 0])
        ranks = torch.tensor([1, 2, 3, 4])

        selector = DiversitySelector(alpha=1.0, pool_factor=4)
        mask = selector(patches, ranks, {0: 2})

        assert mask.tolist() == [True, True, False, False]

    def test_always_starts_from_the_best_rank(self) -> None:
        patches = make_patches([[0.0], [1.0], [2.0], [3.0]], [0, 0, 0, 0])
        ranks = torch.tensor([3, 1, 4, 2])  # best rank (1) is at index 1

        mask = DiversitySelector(alpha=0.9, pool_factor=4)(patches, ranks, {0: 1})

        assert mask.tolist() == [False, True, False, False]

    def test_per_class_independent_selection(self) -> None:
        patches = make_patches(
            [[0.0], [1.0], [10.0], [11.0]],
            [0, 0, 1, 1],
        )
        ranks = torch.tensor([1, 2, 1, 2])

        mask = DiversitySelector(alpha=1.0, pool_factor=1)(patches, ranks, {0: 1, 1: 1})

        assert mask.tolist() == [True, False, True, False]

    def test_zero_budget_class_selects_nothing(self) -> None:
        patches = make_patches([[0.0], [1.0]], [0, 0])
        ranks = torch.tensor([1, 2])

        mask = DiversitySelector()(patches, ranks, {0: 0})

        assert mask.tolist() == [False, False]

    def test_n_pick_clipped_by_pool_availability(self) -> None:
        patches = make_patches([[0.0], [1.0]], [0, 0])
        ranks = torch.tensor([1, 2])

        mask = DiversitySelector(pool_factor=1)(patches, ranks, {0: 5})

        assert mask.sum().item() == 2

    def test_tie_break_follows_scan_order_not_rank_order(self) -> None:
        """An exact tie in ``combined`` between two candidates from different
        images, where the winner depends on the candidate scan order.

        Image 0 has A (rank 1, pos 0) and B (rank 4, pos 4); image 1 has
        C (rank 2, pos 6) and D (rank 3, pos 8). alpha=0.5, pool_factor wide
        enough to admit all 4. Greedy always starts at A (unique rank 1);
        min_dist to A is then 4 (B), 6 (C), 8 (D), so max_min_dist=8 and:
            combined(C) = .5*.75 + .5*(6/8) = .75
            combined(D) = .5*.50 + .5*(8/8) = .75   <- exact tie with C
            combined(B) = .5*.25 + .5*(4/8) = .375  <- never wins
        The scan order after picking A is image-major then rank-descending
        within an image: [B, D, C] — D is scanned before C, so D wins. A
        rank-ascending scan ([C, D, B]) would pick C instead, so this case
        distinguishes the two.
        """
        patches = make_patches(
            [[0.0], [4.0], [6.0], [8.0]],
            [0, 0, 0, 0],
            image_ids=[0, 0, 1, 1],
        )
        ranks = torch.tensor([1, 4, 2, 3])

        mask = DiversitySelector(alpha=0.5, pool_factor=4)(patches, ranks, {0: 2})

        assert mask.tolist() == [True, False, False, True]  # A, D — not C

    def test_matches_the_naive_implementation(self) -> None:
        torch.manual_seed(0)
        feats = torch.randn(40, 3)
        labels = torch.randint(0, 3, (40,))
        ranks = torch.zeros(40, dtype=torch.int64)
        for cls in torch.unique(labels).tolist():
            (idx,) = (labels == cls).nonzero(as_tuple=True)
            perm = torch.randperm(idx.numel())
            ranks[idx[perm]] = torch.arange(1, idx.numel() + 1)

        patches = make_patches(feats.tolist(), labels.tolist())
        n_per_class = {0: 3, 1: 2, 2: 4}

        got = DiversitySelector(alpha=0.5, pool_factor=3)(patches, ranks, n_per_class)
        expected = naive_diversity_select(
            feats, labels, ranks, patches.image_ids, n_per_class, 0.5, 3
        )

        assert torch.equal(got, expected)

    def test_matches_naive_under_a_tight_global_pool_cap(self) -> None:
        """Small total_ranked relative to pool_factor forces the shared
        ``total_ranked // n_classes`` clip branch (module docstring)."""
        torch.manual_seed(1)
        feats = torch.randn(9, 2)
        labels = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
        ranks = torch.tensor([1, 2, 3, 1, 2, 3, 1, 2, 3])
        patches = make_patches(feats.tolist(), labels.tolist())
        n_per_class = {0: 2, 1: 2, 2: 2}

        got = DiversitySelector(alpha=0.5, pool_factor=5)(patches, ranks, n_per_class)
        expected = naive_diversity_select(
            feats, labels, ranks, patches.image_ids, n_per_class, 0.5, 5
        )

        assert torch.equal(got, expected)

    def test_matches_naive_across_multiple_images(self) -> None:
        """Same as the random cross-check above but with patches spread over
        several images, so the naive reference's image-major/rank-descending
        scan order is actually exercised (not degenerate to one image)."""
        torch.manual_seed(3)
        feats = torch.randn(60, 3)
        labels = torch.randint(0, 3, (60,))
        image_ids = torch.randint(0, 5, (60,))
        ranks = torch.zeros(60, dtype=torch.int64)
        for cls in torch.unique(labels).tolist():
            (idx,) = (labels == cls).nonzero(as_tuple=True)
            perm = torch.randperm(idx.numel())
            ranks[idx[perm]] = torch.arange(1, idx.numel() + 1)

        patches = make_patches(
            feats.tolist(), labels.tolist(), image_ids=image_ids.tolist()
        )
        n_per_class = {0: 3, 1: 2, 2: 4}

        got = DiversitySelector(alpha=0.5, pool_factor=3)(patches, ranks, n_per_class)
        expected = naive_diversity_select(
            feats, labels, ranks, image_ids, n_per_class, 0.5, 3
        )

        assert torch.equal(got, expected)

    def test_degenerate_pool_falls_back_to_pure_rank(self) -> None:
        """When a class's candidates are near-identical the diversity
        denominator is clamped to 1, flattening diversity to ~0 so rank alone
        decides. Without the clamp, rescaling a tiny spread back up to [0, 1]
        would let floating-point noise drive the pick.

        Points at 0, 1e-7, 2e-7, 3e-7 with ranks 1..4: the spread lands inside
        ``[1e-12, 1e-6)``, exactly where the two thresholds disagree. Clamped
        (correct), the second pick is the rank-2 point; unclamped it would be
        the far one, whose full-scale diversity term outweighs its worse rank.
        """
        patches = make_patches([[0.0], [1e-7], [2e-7], [3e-7]], [0, 0, 0, 0])
        ranks = torch.tensor([1, 2, 3, 4])

        mask = DiversitySelector(alpha=0.5, pool_factor=4)(patches, ranks, {0: 2})

        assert mask.tolist() == [True, True, False, False]
        assert torch.equal(
            mask,
            naive_diversity_select(
                patches.feats, patches.labels, ranks, patches.image_ids, {0: 2}, 0.5, 4
            ),
        )

    def test_is_deterministic(self) -> None:
        torch.manual_seed(2)
        feats = torch.randn(20, 4)
        labels = torch.randint(0, 2, (20,))
        ranks = torch.randperm(20) + 1
        patches = make_patches(feats.tolist(), labels.tolist())

        selector = DiversitySelector(alpha=0.5, pool_factor=3)
        a = selector(patches, ranks, {0: 3, 1: 3})
        b = selector(patches, ranks, {0: 3, 1: 3})

        assert torch.equal(a, b)

    def test_satisfies_the_protocol(self) -> None:
        assert isinstance(DiversitySelector(), Selector)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
    def test_runs_on_gpu_and_agrees_with_cpu(self) -> None:
        """Device-agnostic: same result, and the mask on the input's device."""
        torch.manual_seed(4)
        feats = torch.randn(30, 5)
        labels = torch.randint(0, 3, (30,))
        ranks = torch.zeros(30, dtype=torch.int64)
        for cls in torch.unique(labels).tolist():
            (idx,) = (labels == cls).nonzero(as_tuple=True)
            ranks[idx[torch.randperm(idx.numel())]] = torch.arange(1, idx.numel() + 1)

        patches = make_patches(feats.tolist(), labels.tolist())
        selector = DiversitySelector(alpha=0.5, pool_factor=3)
        n_per_class = {0: 2, 1: 2, 2: 2}

        on_cpu = selector(patches, ranks, n_per_class)
        on_gpu = selector(patches.to("cuda"), ranks.cuda(), n_per_class)

        assert on_gpu.device.type == "cuda"
        assert torch.equal(on_gpu.cpu(), on_cpu)


class TestTopN:
    def test_picks_best_ranked_per_class(self) -> None:
        patches = make_patches([[0.0], [1.0], [2.0], [3.0]], [0, 0, 1, 1])
        ranks = torch.tensor([2, 1, 1, 2])

        mask = TopN()(patches, ranks, {0: 1, 1: 1})

        assert mask.tolist() == [False, True, True, False]

    def test_clips_to_available(self) -> None:
        patches = make_patches([[0.0], [1.0]], [0, 0])
        ranks = torch.tensor([1, 2])

        mask = TopN()(patches, ranks, {0: 5})

        assert mask.sum().item() == 2

    def test_zero_budget_selects_nothing(self) -> None:
        patches = make_patches([[0.0], [1.0]], [0, 0])
        ranks = torch.tensor([1, 2])

        mask = TopN()(patches, ranks, {0: 0})

        assert mask.tolist() == [False, False]

    def test_satisfies_the_protocol(self) -> None:
        assert isinstance(TopN(), Selector)
