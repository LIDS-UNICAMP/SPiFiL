"""Scorer implementations and per-class rank assignment."""

from __future__ import annotations

import torch

from spifil.metrics import Euclidean
from spifil.scoring import (
    DistanceSumScorer,
    FisherScorer,
    Scorer,
    per_class_ranks,
    scatter_ranks,
)
from spifil.types import PatchSet, Seeds


def make_patches(feats: torch.Tensor, labels: torch.Tensor) -> PatchSet:
    n = feats.shape[0]
    return PatchSet(
        feats=feats,
        labels=labels,
        seed_rows=torch.arange(n),
        image_ids=torch.zeros(n, dtype=torch.int64),
    )


class TestFisherScorer:
    def test_matches_hand_computed_ratio(self) -> None:
        # Two tight clusters on a line, far apart: {0, 1} and {10, 11}.
        feats = torch.tensor([[0.0], [1.0], [10.0], [11.0]])
        labels = torch.tensor([0, 0, 1, 1])

        scores = FisherScorer()(make_patches(feats, labels), Euclidean())

        # sample 0: intra = dist to sample 1 = 1; inter = avg(dist to 10, 11) = 10.5
        expected_0 = 10.5 / (1.0 + 1e-6)
        assert torch.isclose(scores[0], torch.tensor(expected_0), atol=1e-4)

    def test_single_class_has_zero_inter_class_term(self) -> None:
        feats = torch.tensor([[0.0], [1.0], [3.0]])
        labels = torch.zeros(3, dtype=torch.int64)

        scores = FisherScorer()(make_patches(feats, labels), Euclidean())

        assert torch.allclose(scores, torch.zeros(3))

    def test_satisfies_the_protocol(self) -> None:
        assert isinstance(FisherScorer(), Scorer)


class TestDistanceSumScorer:
    def test_matches_hand_computed_sum(self) -> None:
        feats = torch.tensor([[0.0], [3.0], [4.0]])
        labels = torch.tensor([0, 0, 1])

        scores = DistanceSumScorer()(make_patches(feats, labels), Euclidean())

        # sample 0: dist to sample 1 (3) + dist to sample 2 (4) = 7
        assert torch.isclose(scores[0], torch.tensor(7.0))

    def test_satisfies_the_protocol(self) -> None:
        assert isinstance(DistanceSumScorer(), Scorer)


class TestPerClassRanks:
    def test_higher_score_gets_rank_one_within_class(self) -> None:
        scores = torch.tensor([5.0, 1.0, 9.0, 2.0])
        labels = torch.tensor([0, 0, 1, 1])

        ranks = per_class_ranks(scores, labels)

        assert torch.equal(ranks, torch.tensor([1, 2, 1, 2]))

    def test_ties_keep_their_original_relative_order(self) -> None:
        """A stable sort: tied scores keep the order they arrived in."""
        scores = torch.tensor([5.0, 5.0, 5.0])
        labels = torch.zeros(3, dtype=torch.int64)

        ranks = per_class_ranks(scores, labels)

        assert torch.equal(ranks, torch.tensor([1, 2, 3]))

    def test_ranks_are_dense_and_one_based_per_class(self) -> None:
        torch.manual_seed(0)
        scores = torch.randn(50)
        labels = torch.randint(0, 3, (50,))

        ranks = per_class_ranks(scores, labels)

        for cls in labels.unique().tolist():
            cls_ranks = sorted(ranks[labels == cls].tolist())
            assert cls_ranks == list(range(1, len(cls_ranks) + 1))


class TestScatterRanks:
    def test_writes_ranks_back_into_the_right_seeds(self) -> None:
        seeds0 = Seeds.from_coords(torch.tensor([[0, 0], [1, 1]]), label=0, grid=(5, 5))
        seeds1 = Seeds.from_coords(torch.tensor([[2, 2]]), label=1, grid=(5, 5))
        patches = PatchSet(
            feats=torch.zeros(3, 1),
            labels=torch.tensor([0, 0, 1]),
            seed_rows=torch.tensor([0, 1, 0]),
            image_ids=torch.tensor([0, 0, 1]),
        )
        ranks = torch.tensor([2, 1, 1])

        updated = scatter_ranks(patches, ranks, [seeds0, seeds1])

        assert torch.equal(updated[0].ranks, torch.tensor([2, 1]))
        assert torch.equal(updated[1].ranks, torch.tensor([1]))

    def test_unmatched_seeds_stay_unranked(self) -> None:
        seeds = Seeds.from_coords(
            torch.tensor([[0, 0], [1, 1], [2, 2]]), label=0, grid=(5, 5)
        )
        patches = PatchSet(
            feats=torch.zeros(1, 1),
            labels=torch.tensor([0]),
            seed_rows=torch.tensor([1]),
            image_ids=torch.tensor([0]),
        )

        updated = scatter_ranks(patches, torch.tensor([1]), [seeds])

        assert torch.equal(updated[0].ranks, torch.tensor([0, 1, 0]))
