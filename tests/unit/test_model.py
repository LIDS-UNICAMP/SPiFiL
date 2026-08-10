"""SpifilNet: layer numbering, filter installation, and standalone state_dict."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from spifil.config import ArchSpec, LayerSpec
from spifil.nn.blocks import SpifilConvBlock
from spifil.nn.builder import FilterBank, build_filters
from spifil.nn.model import SpifilNet
from spifil.types import PatchSet

ARCH = ArchSpec(
    layers=[
        LayerSpec(kernel_size=3, out_channels=16),
        LayerSpec(kernel_size=3, out_channels=32),
    ]
)


def bank_of(n_filters: int, in_channels: int, spec: LayerSpec) -> FilterBank:
    """A filter bank of the requested width, from random patches."""
    feats = torch.randn(n_filters, in_channels * 9)
    patches = PatchSet(
        feats=feats,
        labels=torch.arange(n_filters),
        seed_rows=torch.arange(n_filters),
        image_ids=torch.zeros(n_filters, dtype=torch.int64),
    )
    return build_filters(patches, spec)


def test_blocks_chain_channels_from_the_arch() -> None:
    net = SpifilNet(ARCH, in_channels=3)

    assert len(net) == 2
    assert net.blocks[0].in_channels == 3
    assert net.blocks[1].in_channels == 16


def test_forward_pools_once_per_layer() -> None:
    net = SpifilNet(ARCH, in_channels=3)

    out = net(torch.randn(1, 3, 40, 40))

    assert out.shape == (1, 32, 10, 10)


class TestUpto:
    def test_stops_after_the_named_layer(self) -> None:
        net = SpifilNet(ARCH, in_channels=3)

        assert net(torch.randn(1, 3, 40, 40), upto=1).shape == (1, 16, 20, 20)

    def test_zero_returns_the_input_untouched(self) -> None:
        """Layer 0 is the color transform's output, passed through."""
        net = SpifilNet(ARCH, in_channels=3)
        x = torch.randn(1, 3, 8, 8)

        assert torch.equal(net(x, upto=0), x)

    def test_out_of_range_is_rejected(self) -> None:
        net = SpifilNet(ARCH, in_channels=3)

        with pytest.raises(ValueError, match="upto must be between 0 and 2"):
            net(torch.randn(1, 3, 8, 8), upto=3)


class TestSetFilters:
    def test_installs_the_bank_on_the_named_layer(self) -> None:
        net = SpifilNet(ARCH, in_channels=3)
        bank = bank_of(16, 3, ARCH.layers[0])

        net.set_filters(1, bank)

        assert torch.equal(net.blocks[0].conv.weight.detach(), bank.weight)

    def test_layers_are_numbered_from_one(self) -> None:
        net = SpifilNet(ARCH, in_channels=3)

        with pytest.raises(ValueError, match="layer must be between 1 and 2"):
            net.set_filters(0, bank_of(16, 3, ARCH.layers[0]))

    def test_a_narrow_bank_rewires_the_next_layer(self) -> None:
        """Uneven allocation shrinks a layer; the next one has to follow.

        ``UniformAllocator`` splits by integer division, so 16 filters
        over 6 classes produce 12 and layer 2 must
        convolve 12 channels, not the 16 its spec named.
        """
        net = SpifilNet(ARCH, in_channels=3)

        net.set_filters(1, bank_of(12, 3, ARCH.layers[0]))

        assert net.blocks[0].out_channels == 12
        assert net.blocks[1].in_channels == 12
        assert net(torch.randn(1, 3, 40, 40)).shape == (1, 32, 10, 10)

    def test_a_narrow_bank_on_the_last_layer_rewires_nothing(self) -> None:
        net = SpifilNet(ARCH, in_channels=3)

        net.set_filters(2, bank_of(30, 16, ARCH.layers[1]))

        assert net.blocks[1].out_channels == 30


def test_state_dict_holds_only_conv_parameters() -> None:
    """The standalone claim: nothing in the checkpoint needs spifil to load.

    Loading into a bare ``nn.Sequential`` of ``Conv2d`` layers with matching
    shapes is the strictest cheap version of "torch alone can run this" — it
    fails if a block ever starts carrying learned state of its own.
    """
    net = SpifilNet(ARCH, in_channels=3)
    net.set_filters(1, bank_of(16, 3, ARCH.layers[0]))
    state = net.state_dict()

    assert sorted(state) == [
        "blocks.0.conv.bias",
        "blocks.0.conv.weight",
        "blocks.1.conv.bias",
        "blocks.1.conv.weight",
    ]

    plain = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1), nn.Conv2d(16, 32, 3, padding=1)
    )
    plain.load_state_dict(
        {
            "0.weight": state["blocks.0.conv.weight"],
            "0.bias": state["blocks.0.conv.bias"],
            "1.weight": state["blocks.1.conv.weight"],
            "1.bias": state["blocks.1.conv.bias"],
        }
    )
    assert torch.equal(plain[0].weight.detach(), state["blocks.0.conv.weight"])


def test_state_dict_round_trips_through_a_fresh_net() -> None:
    net = SpifilNet(ARCH, in_channels=3)
    net.set_filters(1, bank_of(16, 3, ARCH.layers[0]))
    x = torch.randn(1, 3, 24, 24)

    restored = SpifilNet(ARCH, in_channels=3)
    restored.load_state_dict(net.state_dict())

    with torch.no_grad():
        assert torch.equal(restored(x), net(x))


def test_a_custom_block_factory_is_used_for_every_layer() -> None:
    """Blocks are a swappable concern, including on rewiring."""
    built: list[tuple[int, int]] = []

    def factory(spec: LayerSpec, in_channels: int) -> nn.Module:
        built.append((in_channels, spec.out_channels))
        return SpifilConvBlock(spec, in_channels)

    net = SpifilNet(ARCH, in_channels=3, block_factory=factory)
    net.set_filters(1, bank_of(12, 3, ARCH.layers[0]))

    assert built == [(3, 16), (16, 32), (12, 32)]


def test_an_empty_arch_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one layer"):
        SpifilNet(ArchSpec(layers=[]))
