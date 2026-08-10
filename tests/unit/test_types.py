"""The Seeds and PatchSet containers."""

from __future__ import annotations

import pytest
import torch

from spifil.types import PatchSet, Seeds


def test_from_coords_labels_everything_and_leaves_ranks_unset() -> None:
    coords = torch.tensor([[1, 2], [3, 4], [5, 6]])

    seeds = Seeds.from_coords(coords, label=7, grid=(10, 10))

    assert len(seeds) == 3
    assert torch.equal(seeds.labels, torch.full((3,), 7))
    assert torch.equal(seeds.ranks, torch.zeros(3, dtype=torch.int64))
    assert seeds.coords.dtype == torch.int64


def test_sorted_by_coords_orders_slowest_axis_first() -> None:
    seeds = Seeds.from_coords(
        torch.tensor([[2, 0], [0, 5], [0, 1], [1, 9]]), label=1, grid=(4, 10)
    )

    ordered = seeds.sorted_by_coords()

    assert torch.equal(ordered.coords, torch.tensor([[0, 1], [0, 5], [1, 9], [2, 0]]))


def test_sorted_by_coords_carries_labels_and_ranks_along() -> None:
    seeds = Seeds(
        coords=torch.tensor([[2, 0], [0, 1]]),
        labels=torch.tensor([10, 20]),
        ranks=torch.tensor([1, 2]),
        grid=(4, 4),
    )

    ordered = seeds.sorted_by_coords()

    assert torch.equal(ordered.labels, torch.tensor([20, 10]))
    assert torch.equal(ordered.ranks, torch.tensor([2, 1]))


def test_coords_must_match_grid_rank() -> None:
    with pytest.raises(ValueError, match=r"coords must be \(n, 3\)"):
        Seeds.from_coords(torch.tensor([[1, 2]]), label=1, grid=(4, 4, 4))


def test_labels_and_ranks_must_match_coords() -> None:
    with pytest.raises(ValueError, match="must both be"):
        Seeds(
            coords=torch.tensor([[1, 2], [3, 4]]),
            labels=torch.tensor([1]),
            ranks=torch.tensor([0, 0]),
            grid=(8, 8),
        )


def test_works_for_3d_grids() -> None:
    seeds = Seeds.from_coords(
        torch.tensor([[1, 2, 3], [0, 0, 0]]), label=2, grid=(4, 5, 6)
    )

    assert torch.equal(
        seeds.sorted_by_coords().coords, torch.tensor([[0, 0, 0], [1, 2, 3]])
    )


def test_with_ranks_sets_only_the_given_rows() -> None:
    seeds = Seeds.from_coords(
        torch.tensor([[0, 0], [1, 1], [2, 2]]), label=1, grid=(4, 4)
    )

    updated = seeds.with_ranks(torch.tensor([2, 0]), torch.tensor([1, 2]))

    assert torch.equal(updated.ranks, torch.tensor([2, 0, 1]))
    # coords/labels/grid are untouched
    assert torch.equal(updated.coords, seeds.coords)
    assert torch.equal(updated.labels, seeds.labels)


def test_with_ranks_resets_ranks_not_in_seed_rows() -> None:
    seeds = Seeds(
        coords=torch.tensor([[0, 0], [1, 1]]),
        labels=torch.tensor([1, 1]),
        ranks=torch.tensor([9, 9]),  # stale ranks from a previous scoring pass
        grid=(4, 4),
    )

    updated = seeds.with_ranks(torch.tensor([0]), torch.tensor([3]))

    assert torch.equal(updated.ranks, torch.tensor([3, 0]))


class TestPatchSet:
    def test_len_and_field_shapes(self) -> None:
        patches = PatchSet(
            feats=torch.randn(3, 5),
            labels=torch.tensor([0, 1, 0]),
            seed_rows=torch.tensor([0, 1, 2]),
            image_ids=torch.tensor([0, 0, 0]),
        )

        assert len(patches) == 3

    def test_rejects_mismatched_row_counts(self) -> None:
        with pytest.raises(ValueError, match="must all be"):
            PatchSet(
                feats=torch.randn(3, 5),
                labels=torch.tensor([0, 1]),
                seed_rows=torch.tensor([0, 1, 2]),
                image_ids=torch.tensor([0, 0, 0]),
            )

    def test_cat_concatenates_in_order(self) -> None:
        a = PatchSet(
            feats=torch.tensor([[1.0], [2.0]]),
            labels=torch.tensor([0, 1]),
            seed_rows=torch.tensor([0, 1]),
            image_ids=torch.tensor([0, 0]),
        )
        b = PatchSet(
            feats=torch.tensor([[3.0]]),
            labels=torch.tensor([1]),
            seed_rows=torch.tensor([0]),
            image_ids=torch.tensor([1]),
        )

        merged = PatchSet.cat([a, b])

        assert torch.equal(merged.feats, torch.tensor([[1.0], [2.0], [3.0]]))
        assert torch.equal(merged.labels, torch.tensor([0, 1, 1]))
        assert torch.equal(merged.image_ids, torch.tensor([0, 0, 1]))

    def test_cat_rejects_empty_sequence(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            PatchSet.cat([])

    def test_to_is_a_no_op_move_on_cpu(self) -> None:
        patches = PatchSet(
            feats=torch.tensor([[1.0], [2.0]]),
            labels=torch.tensor([0, 1]),
            seed_rows=torch.tensor([0, 1]),
            image_ids=torch.tensor([0, 0]),
        )

        moved = patches.to("cpu")

        assert torch.equal(moved.feats, patches.feats)
        assert moved.labels.dtype == torch.int64
        assert moved.feats.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
    def test_to_moves_every_field(self) -> None:
        patches = PatchSet(
            feats=torch.tensor([[1.0], [2.0]]),
            labels=torch.tensor([0, 1]),
            seed_rows=torch.tensor([0, 1]),
            image_ids=torch.tensor([0, 0]),
        )

        moved = patches.to("cuda")

        for field in (moved.feats, moved.labels, moved.seed_rows, moved.image_ids):
            assert field.device.type == "cuda"
        assert torch.equal(moved.feats.cpu(), patches.feats)


class TestProject:
    """``Seeds.project``: seeds follow their features onto a pooled grid."""

    def test_coordinates_are_divided_and_the_grid_is_ceiled(self) -> None:
        seeds = Seeds.from_coords(
            torch.tensor([[0, 0], [3, 8], [8, 3]]), label=1, grid=(9, 9)
        )

        projected = seeds.project(stride=2)

        assert projected.grid == (5, 5)
        assert torch.equal(projected.coords, torch.tensor([[0, 0], [1, 4], [4, 1]]))

    def test_merged_seeds_keep_the_best_positive_rank(self) -> None:
        """Two seeds in one pooled pixel: the pixel is as good as its best seed."""
        seeds = Seeds(
            coords=torch.tensor([[0, 0], [0, 1], [4, 4]]),
            labels=torch.tensor([2, 2, 2]),
            ranks=torch.tensor([7, 3, 1]),
            grid=(8, 8),
        )

        projected = seeds.project(stride=2)

        assert torch.equal(projected.coords, torch.tensor([[0, 0], [2, 2]]))
        assert torch.equal(projected.ranks, torch.tensor([3, 1]))

    def test_unranked_seeds_never_beat_a_ranked_one(self) -> None:
        """Rank 0 means "unranked", not "rank zero" — it must lose every merge."""
        seeds = Seeds(
            coords=torch.tensor([[0, 0], [0, 1], [1, 0]]),
            labels=torch.tensor([1, 1, 1]),
            ranks=torch.tensor([0, 5, 0]),
            grid=(4, 4),
        )

        projected = seeds.project(stride=2)

        assert torch.equal(projected.ranks, torch.tensor([5]))

    def test_a_group_with_no_ranked_seed_stays_unranked(self) -> None:
        seeds = Seeds.from_coords(torch.tensor([[0, 0], [1, 1]]), label=1, grid=(4, 4))

        projected = seeds.project(stride=2)

        assert torch.equal(projected.ranks, torch.tensor([0]))

    def test_labels_come_from_the_first_seed_of_the_group(self) -> None:
        seeds = Seeds(
            coords=torch.tensor([[1, 1], [0, 0]]),
            labels=torch.tensor([9, 4]),
            ranks=torch.tensor([2, 1]),
            grid=(4, 4),
        )

        projected = seeds.project(stride=2)

        assert torch.equal(projected.labels, torch.tensor([9]))

    def test_output_is_ordered_lexicographically_by_coordinate(self) -> None:
        seeds = Seeds.from_coords(
            torch.tensor([[6, 0], [0, 6], [2, 2]]), label=1, grid=(8, 8)
        )

        projected = seeds.project(stride=2)

        assert torch.equal(projected.coords, torch.tensor([[0, 3], [1, 1], [3, 0]]))

    def test_stride_one_is_a_no_op_copy(self) -> None:
        """``stride <= 1`` returns an unmerged copy, duplicates and all."""
        seeds = Seeds(
            coords=torch.tensor([[1, 1], [1, 1]]),
            labels=torch.tensor([3, 3]),
            ranks=torch.tensor([2, 1]),
            grid=(4, 4),
        )

        projected = seeds.project(stride=1)

        assert len(projected) == 2
        assert projected.grid == (4, 4)
        assert projected.coords is not seeds.coords
