"""Seed-centered patch extraction via ``F.unfold``."""

from __future__ import annotations

import pytest
import torch

from spifil.config import LayerSpec
from spifil.patches import extract_patches
from spifil.types import Seeds


def naive_patch(
    features: torch.Tensor, y: int, x: int, kernel_size: int, dilation: int
) -> torch.Tensor:
    """One patch, written out as a loop.

    Channel-major (channel outer, neighbor inner), with ``dy``/``dx`` ranging
    over the ``-k//2..k//2`` window and zero for out-of-bounds neighbors.
    Unvectorized: this is what the batched ``F.unfold`` path in
    ``extract_patches`` must agree with.
    """
    _, h, w = features.shape
    half = kernel_size // 2
    offsets = range(-half, half + 1)
    values = []
    for channel in range(features.shape[0]):
        for dy in offsets:
            for dx in offsets:
                yy, xx = y + dy * dilation, x + dx * dilation
                inside = 0 <= yy < h and 0 <= xx < w
                values.append(features[channel, yy, xx] if inside else torch.zeros(()))
    return torch.stack(values)


def test_matches_the_naive_loop_at_every_seed() -> None:
    torch.manual_seed(0)
    features = torch.randn(3, 10, 10)
    coords = torch.tensor([[0, 0], [9, 9], [4, 4], [0, 9], [9, 0]])
    seeds = Seeds.from_coords(coords, label=1, grid=(10, 10))
    layer = LayerSpec(kernel_size=3, dilation=1)

    patches = extract_patches(features, seeds, layer)

    for i, (y, x) in enumerate(coords.tolist()):
        assert torch.allclose(patches.feats[i], naive_patch(features, y, x, 3, 1))


def test_dilation_skips_pixels() -> None:
    torch.manual_seed(1)
    features = torch.randn(2, 12, 12)
    seeds = Seeds.from_coords(torch.tensor([[6, 6]]), label=0, grid=(12, 12))
    layer = LayerSpec(kernel_size=3, dilation=2)

    patches = extract_patches(features, seeds, layer)

    assert torch.allclose(patches.feats[0], naive_patch(features, 6, 6, 3, 2))


def test_zero_pads_out_of_bounds_neighbors() -> None:
    features = torch.arange(1, 10, dtype=torch.float32).reshape(1, 3, 3)
    seeds = Seeds.from_coords(torch.tensor([[0, 0]]), label=0, grid=(3, 3))
    layer = LayerSpec(kernel_size=3, dilation=1)

    patches = extract_patches(features, seeds, layer)

    # dy, dx in {-1, 0, 1} x {-1, 0, 1}, row-major; only (0,0)/(0,1)/(1,0)/(1,1)
    # land inside the 3x3 image (values 1, 2, 4, 5).
    expected = torch.tensor([0, 0, 0, 0, 1, 2, 0, 4, 5], dtype=torch.float32)
    assert torch.equal(patches.feats[0], expected)


def test_even_kernel_size_spans_kernel_size_plus_one() -> None:
    """The -k//2..k//2 window: an even kernel_size grows by one."""
    features = torch.randn(2, 8, 8)
    seeds = Seeds.from_coords(torch.tensor([[4, 4]]), label=0, grid=(8, 8))
    layer = LayerSpec(kernel_size=4, dilation=1)

    patches = extract_patches(features, seeds, layer)

    assert patches.feats.shape == (1, 2 * 5 * 5)
    assert torch.allclose(patches.feats[0], naive_patch(features, 4, 4, 4, 1))


def test_patch_set_fields_carry_seed_metadata() -> None:
    features = torch.randn(3, 5, 5)
    coords = torch.tensor([[0, 0], [2, 2], [4, 4]])
    seeds = Seeds(
        coords=coords,
        labels=torch.tensor([1, 2, 1]),
        ranks=torch.zeros(3, dtype=torch.int64),
        grid=(5, 5),
    )
    layer = LayerSpec(kernel_size=3)

    patches = extract_patches(features, seeds, layer, image_id=7)

    assert len(patches) == 3
    assert torch.equal(patches.labels, seeds.labels)
    assert torch.equal(patches.seed_rows, torch.arange(3))
    assert torch.equal(patches.image_ids, torch.full((3,), 7))


def test_rejects_grid_mismatch() -> None:
    features = torch.randn(3, 5, 5)
    seeds = Seeds.from_coords(torch.tensor([[0, 0]]), label=0, grid=(6, 6))

    with pytest.raises(ValueError, match="must be"):
        extract_patches(features, seeds, LayerSpec(kernel_size=3))


def test_rejects_non_2d_grid() -> None:
    seeds = Seeds.from_coords(torch.tensor([[0, 0, 0]]), label=0, grid=(4, 4, 4))
    features = torch.randn(3, 4, 4, 4)

    with pytest.raises(ValueError, match="2D"):
        extract_patches(features, seeds, LayerSpec(kernel_size=3))
