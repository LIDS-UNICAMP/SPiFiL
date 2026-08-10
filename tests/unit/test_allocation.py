"""UniformAllocator: per-class budget by integer division."""

from __future__ import annotations

import torch

from spifil.allocation import FilterAllocator, UniformAllocator
from spifil.config import LayerSpec
from spifil.types import PatchSet


def make_patches(labels: list[int]) -> PatchSet:
    n = len(labels)
    return PatchSet(
        feats=torch.zeros(n, 1),
        labels=torch.tensor(labels, dtype=torch.int64),
        seed_rows=torch.arange(n),
        image_ids=torch.zeros(n, dtype=torch.int64),
    )


class TestUniformAllocator:
    def test_divides_evenly(self) -> None:
        spec = LayerSpec(out_channels=12)
        patches = make_patches([0, 0, 1, 1, 2, 2])

        result = UniformAllocator()(1, spec, patches, torch.zeros(6))

        assert result == {0: 4, 1: 4, 2: 4}

    def test_uneven_division_underproduces_by_design(self) -> None:
        """3 classes, 64 targets -> 63 filters: the remainder is dropped."""
        spec = LayerSpec(out_channels=64)
        patches = make_patches([0, 1, 2])

        result = UniformAllocator()(1, spec, patches, torch.zeros(3))

        assert result == {0: 21, 1: 21, 2: 21}
        assert sum(result.values()) == 63

    def test_classes_are_discovered_and_sorted(self) -> None:
        spec = LayerSpec(out_channels=6)
        patches = make_patches([5, 1, 1, 5, 3])

        result = UniformAllocator()(1, spec, patches, torch.zeros(5))

        assert list(result.keys()) == [1, 3, 5]

    def test_no_classes_returns_empty(self) -> None:
        spec = LayerSpec(out_channels=6)
        patches = make_patches([])

        result = UniformAllocator()(1, spec, patches, torch.zeros(0))

        assert result == {}

    def test_scores_and_layer_are_unused(self) -> None:
        """Present only for protocol symmetry with difficulty-based allocators."""
        spec = LayerSpec(out_channels=4)
        patches = make_patches([0, 0, 1, 1])

        a = UniformAllocator()(1, spec, patches, torch.zeros(4))
        b = UniformAllocator()(99, spec, patches, torch.randn(4))

        assert a == b

    def test_satisfies_the_protocol(self) -> None:
        assert isinstance(UniformAllocator(), FilterAllocator)
