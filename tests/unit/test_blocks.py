"""The conv block: padding, pooling geometry, and assignable filters."""

from __future__ import annotations

import pytest
import torch

from spifil.config import LayerSpec
from spifil.nn.blocks import SpifilConvBlock
from spifil.patches import extract_patches
from spifil.types import Seeds

LAYER = LayerSpec(kernel_size=3, out_channels=4)


def identity_bank(
    in_channels: int, spec: LayerSpec
) -> tuple[torch.Tensor, torch.Tensor]:
    """One filter per input channel, reading only the window's center pixel."""
    span = 2 * (spec.kernel_size // 2) + 1
    weight = torch.zeros(in_channels, in_channels, span, span)
    for c in range(in_channels):
        weight[c, c, span // 2, span // 2] = 1.0
    return weight, torch.zeros(in_channels)


class TestGeometry:
    @pytest.mark.parametrize(
        ("size", "stride", "expected"),
        [(64, 2, 32), (65, 2, 33), (10, 2, 5), (11, 3, 4), (7, 1, 7)],
    )
    def test_pooled_size_is_ceil_of_the_stride_division(
        self, size: int, stride: int, expected: int
    ) -> None:
        """Pooling samples window centers at 0, stride, 2*stride, ..."""
        spec = LayerSpec(kernel_size=3, out_channels=2, pool_size=3, pool_stride=stride)
        block = SpifilConvBlock(spec, in_channels=1)

        out = block(torch.randn(1, 1, size, size))

        assert out.shape == (1, 2, expected, expected)

    def test_convolution_is_shape_preserving_before_pooling(self) -> None:
        spec = LayerSpec(kernel_size=5, out_channels=2, dilation=3, pool_type="none")
        block = SpifilConvBlock(spec, in_channels=1)

        out = block(torch.randn(1, 1, 20, 24))

        assert out.shape == (1, 2, 20, 24)

    def test_pooling_ignores_out_of_bounds_neighbors_rather_than_zeroing_them(
        self,
    ) -> None:
        """Out-of-bounds neighbors are skipped, so a border max stays negative.

        Zero padding would report 0 for every border pixel of an all-negative
        feature map — the one visible difference between the two conventions.
        """
        spec = LayerSpec(
            kernel_size=1, out_channels=1, pool_size=3, pool_stride=1, relu=False
        )
        block = SpifilConvBlock(spec, in_channels=1)
        block.set_filters(*identity_bank(1, spec))

        out = block(torch.full((1, 1, 3, 3), -2.0))

        assert torch.equal(out, torch.full((1, 1, 3, 3), -2.0))

    def test_relu_can_be_switched_off(self) -> None:
        spec = LayerSpec(kernel_size=1, out_channels=1, relu=False, pool_type="none")
        block = SpifilConvBlock(spec, in_channels=1)
        block.set_filters(*identity_bank(1, spec))

        out = block(torch.full((1, 1, 2, 2), -1.0))

        assert torch.equal(out, torch.full((1, 1, 2, 2), -1.0))

    def test_pool_type_none_leaves_the_spatial_size_alone(self) -> None:
        spec = LayerSpec(kernel_size=3, out_channels=2, pool_type="none")

        out = SpifilConvBlock(spec, in_channels=1)(torch.randn(1, 1, 9, 9))

        assert out.shape[-2:] == (9, 9)

    def test_average_pooling_is_available(self) -> None:
        spec = LayerSpec(kernel_size=1, out_channels=1, pool_type="avg", pool_stride=2)

        out = SpifilConvBlock(spec, in_channels=1)(torch.randn(1, 1, 8, 8))

        assert out.shape == (1, 1, 4, 4)

    def test_unknown_pool_type_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown pool_type"):
            SpifilConvBlock(LayerSpec(pool_type="median"), in_channels=1)


def test_convolution_agrees_with_patch_extraction() -> None:
    """The block and ``extract_patches`` must read the same window.

    Both sides implement the same im2col — the block through
    ``Conv2d``, the patch extractor through ``F.unfold`` — and the filters
    built from one are applied by the other, so a disagreement about layout,
    padding or dilation would otherwise go unnoticed. Here the
    convolution's output at every pixel is checked against that pixel's patch
    dotted with the flattened weight.
    """
    torch.manual_seed(0)
    spec = LayerSpec(
        kernel_size=3, out_channels=4, dilation=2, pool_type="none", relu=False
    )
    features = torch.randn(2, 7, 9)
    weight = torch.randn(4, 2, 3, 3)
    bias = torch.randn(4)

    block = SpifilConvBlock(spec, in_channels=2)
    block.set_filters(weight, bias)
    with torch.no_grad():
        encoded = block(features.unsqueeze(0)).squeeze(0)

    every_pixel = torch.cartesian_prod(torch.arange(7), torch.arange(9))
    seeds = Seeds.from_coords(every_pixel, label=0, grid=(7, 9))
    patches = extract_patches(features, seeds, spec)

    expected = patches.feats @ weight.reshape(4, -1).T + bias
    assert torch.allclose(encoded.reshape(4, -1).T, expected, atol=1e-5)


class TestSetFilters:
    def test_assigns_weight_and_bias_by_copy(self) -> None:
        block = SpifilConvBlock(LAYER, in_channels=3)
        weight = torch.randn(4, 3, 3, 3)
        bias = torch.randn(4)

        block.set_filters(weight, bias)
        original = weight.clone()
        weight[0, 0, 0, 0] = 99.0

        assert torch.equal(block.conv.weight.detach(), original)
        assert torch.equal(block.conv.bias.detach(), bias)

    def test_accepts_fewer_filters_than_the_spec_asks_for(self) -> None:
        """``out_channels // n_classes`` can under-produce filters."""
        block = SpifilConvBlock(LAYER, in_channels=3)

        block.set_filters(torch.randn(3, 3, 3, 3), torch.randn(3))

        assert block.out_channels == 3
        assert block(torch.randn(1, 3, 8, 8)).shape[1] == 3

    def test_rejects_a_weight_with_the_wrong_input_shape(self) -> None:
        block = SpifilConvBlock(LAYER, in_channels=3)

        with pytest.raises(ValueError, match="convolves 3 channels"):
            block.set_filters(torch.randn(4, 2, 3, 3), torch.randn(4))

    def test_rejects_a_bias_that_does_not_match_the_weight(self) -> None:
        block = SpifilConvBlock(LAYER, in_channels=3)

        with pytest.raises(ValueError, match="bias must be"):
            block.set_filters(torch.randn(4, 3, 3, 3), torch.randn(5))

    def test_rejects_a_flat_weight(self) -> None:
        block = SpifilConvBlock(LAYER, in_channels=3)

        with pytest.raises(ValueError, match="weight must be"):
            block.set_filters(torch.randn(4, 27), torch.randn(4))
