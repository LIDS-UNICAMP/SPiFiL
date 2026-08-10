"""The z-score fold-in: selected patches to conv weight and bias."""

from __future__ import annotations

import pytest
import torch

from spifil.config import LayerSpec
from spifil.nn.builder import build_filters
from spifil.types import PatchSet


def patch_set(feats: torch.Tensor, labels: list[int] | None = None) -> PatchSet:
    n = feats.shape[0]
    return PatchSet(
        feats=feats,
        labels=torch.tensor(labels if labels is not None else [0] * n),
        seed_rows=torch.arange(n),
        image_ids=torch.zeros(n, dtype=torch.int64),
    )


def unit_normalized_zscores(
    feats: torch.Tensor, mean: torch.Tensor, stdev: torch.Tensor
) -> torch.Tensor:
    """The filter vectors before the ``1 / stdev`` fold — spelled out."""
    zscored = (feats - mean) / stdev
    return zscored / torch.linalg.vector_norm(zscored, dim=1, keepdim=True)


# A 1-channel 1x1 "kernel" keeps the arithmetic hand-checkable: one feature per
# patch, so mean/stdev/unit-norm are all scalars.
SINGLE = LayerSpec(kernel_size=1, out_channels=2)


def test_statistics_are_population_moments_with_the_floor_added() -> None:
    """``sqrt(var)`` over N (not N-1), then ``+ stdev_factor`` — not a clamp."""
    feats = torch.tensor([[1.0], [3.0]])

    bank = build_filters(patch_set(feats), SINGLE, stdev_factor=0.5)

    assert bank.mean.item() == pytest.approx(2.0)
    assert bank.stdev.item() == pytest.approx(1.0 + 0.5)


def test_folded_filter_equals_hand_computed_values() -> None:
    feats = torch.tensor([[1.0], [3.0]])
    mean, stdev = 2.0, 1.0 + 0.5

    bank = build_filters(patch_set(feats), SINGLE, stdev_factor=0.5)

    # z = (x - mean) / stdev; unit norm of a 1-vector is its sign; K = unit / stdev.
    expected = torch.tensor([[-1.0], [1.0]]) / stdev
    assert torch.allclose(bank.weight.reshape(2, 1), expected)
    assert torch.allclose(bank.bias, -mean * expected.reshape(2))


def test_fold_in_reproduces_normalized_similarity() -> None:
    """``K @ x + bias == unit_norm(zscore(patch)) . zscore(x)`` — the point of it all.

    This is the property the z-score fold-in exists to obtain: the
    convolution measures similarity to the normalized filter in z-scored input
    space, so the encoder needs no normalization layer at inference.
    """
    torch.manual_seed(0)
    patches = patch_set(torch.randn(5, 27))
    x = torch.randn(27)

    bank = build_filters(patches, LayerSpec(kernel_size=3, out_channels=5))

    unit = unit_normalized_zscores(patches.feats, bank.mean, bank.stdev)
    expected = unit @ ((x - bank.mean) / bank.stdev)

    got = bank.weight.reshape(5, -1) @ x + bank.bias
    assert torch.allclose(got, expected, atol=1e-5)


def test_short_filter_vectors_are_left_unnormalized() -> None:
    """A patch is divided by its norm only above 1e-3; below, it is left as-is.

    Reachable only for a degenerate selection (patches that barely differ from
    the set mean), which is exactly the case the unit-norm guard exists for.
    """
    # Two nearly identical patches: after z-scoring with a large floor, both
    # z-vectors are far shorter than 1e-3.
    feats = torch.tensor([[1.0], [1.0 + 1e-6]])

    bank = build_filters(patch_set(feats), SINGLE, stdev_factor=1.0)

    zscored = (feats - bank.mean) / bank.stdev
    assert float(torch.linalg.vector_norm(zscored, dim=1).max()) < 1e-3
    assert torch.allclose(bank.weight.reshape(2, 1), zscored / bank.stdev)


def test_long_filter_vectors_are_normalized() -> None:
    feats = torch.tensor([[0.0], [1.0]])

    bank = build_filters(patch_set(feats), SINGLE, stdev_factor=0.01)

    unit = bank.weight.reshape(2, 1) * bank.stdev
    assert torch.allclose(unit.abs(), torch.ones(2, 1))


def test_weight_is_channel_major_reshape_of_the_patches() -> None:
    """No permutation: patch layout *is* the flattened Conv2d weight layout."""
    patches = patch_set(torch.randn(4, 2 * 3 * 3))

    bank = build_filters(patches, LayerSpec(kernel_size=3, out_channels=4))

    assert bank.weight.shape == (4, 2, 3, 3)
    unit = unit_normalized_zscores(patches.feats, bank.mean, bank.stdev)
    expected = unit / bank.stdev
    assert torch.allclose(bank.weight.reshape(4, -1), expected)


def test_even_kernel_size_uses_the_widened_c_window() -> None:
    """``kernel_size=4`` spans 5 pixels per axis."""
    patches = patch_set(torch.randn(3, 1 * 5 * 5))

    bank = build_filters(patches, LayerSpec(kernel_size=4, out_channels=3))

    assert bank.weight.shape == (3, 1, 5, 5)


def test_labels_are_carried_through_in_patch_order() -> None:
    patches = patch_set(torch.randn(3, 9), labels=[5, 2, 5])

    bank = build_filters(patches, LayerSpec(kernel_size=3, out_channels=3))

    assert torch.equal(bank.labels, torch.tensor([5, 2, 5]))
    assert len(bank) == 3


def test_labels_are_a_copy_not_a_view() -> None:
    patches = patch_set(torch.randn(2, 9), labels=[1, 2])

    bank = build_filters(patches, LayerSpec(kernel_size=3, out_channels=2))
    patches.labels[0] = 99

    assert int(bank.labels[0]) == 1


def test_feature_count_must_match_the_kernel_window() -> None:
    patches = patch_set(torch.randn(2, 26))  # 26 is not divisible by 3x3

    with pytest.raises(ValueError, match="do not split into a 3x3 window"):
        build_filters(patches, LayerSpec(kernel_size=3, out_channels=2))


def test_empty_selection_is_rejected() -> None:
    patches = patch_set(torch.zeros(0, 9))

    with pytest.raises(ValueError, match="at least one selected patch"):
        build_filters(patches, LayerSpec(kernel_size=3, out_channels=1))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_builds_on_the_device_the_patches_live_on() -> None:
    patches = patch_set(torch.randn(4, 9)).to("cuda")

    bank = build_filters(patches, LayerSpec(kernel_size=3, out_channels=4))

    assert bank.weight.device.type == "cuda"
    assert bank.bias.device.type == "cuda"
