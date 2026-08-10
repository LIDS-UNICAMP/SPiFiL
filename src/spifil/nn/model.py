"""The model --- a plain ``nn.Module`` stack of :class:`SpifilConvBlock`.

This is the artifact SPiFiL exists to produce: once filters are assigned, it
is an ordinary PyTorch network. Its ``state_dict`` holds nothing but ``Conv2d``
weights and biases, so it loads and runs anywhere torch does, with no ``spifil``
import at inference time.

**Layers are numbered from 1**: layer 0 is the input, i.e. the colour
transform's output, and layer ``L >= 1`` is the output of the ``L``-th
convolutional block. :meth:`SpifilNet.forward`'s ``upto`` argument speaks the
same numbering, so ``net(x, upto=2)`` returns the second block's features.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor, nn

from spifil.config import ArchSpec, LayerSpec
from spifil.nn.blocks import ConvBlock, ConvBlockFactory, SpifilConvBlock
from spifil.nn.builder import FilterBank

__all__ = ["FORMAT_VERSION", "SpifilNet", "arch_from_json", "load_model"]

FORMAT_VERSION = 1
"""``architecture.json``'s schema version; bumped when its fields change."""


class SpifilNet(nn.Module):
    """A sequence of conv blocks built from an :class:`~spifil.config.ArchSpec`.

    Parameters
    ----------
    arch
        The layer stack to build. Every layer gets a block immediately, with
        the shape its spec asks for, so an unfitted net is still inspectable
        and exportable — the weights are just torch's default initialization
        until :meth:`set_filters` replaces them.
    in_channels
        Channels of the layer-0 features feeding block 1: 3 for the default
        Lab colour transform.
    block_factory
        How to build each block. Defaults to :class:`SpifilConvBlock`; a
        different factory is how alternative blocks (dilated pyramids,
        separable convs) enter without touching this class.
    """

    def __init__(
        self,
        arch: ArchSpec,
        in_channels: int = 3,
        block_factory: ConvBlockFactory = SpifilConvBlock,
    ) -> None:
        super().__init__()
        if not arch.layers:
            raise ValueError("SpifilNet needs an ArchSpec with at least one layer")

        self.arch = arch
        self.block_factory = block_factory
        self.blocks = nn.ModuleList()
        channels = in_channels
        for spec in arch.layers:
            self.blocks.append(block_factory(spec, channels))
            channels = spec.out_channels

    def __len__(self) -> int:
        return len(self.blocks)

    def set_filters(self, layer: int, bank: FilterBank) -> None:
        """Install layer ``layer``'s filters (1-based), rewiring the next block.

        A bank may hold fewer filters than its spec's ``out_channels``: the
        per-class integer division under-produces whenever the filter count is
        not divisible by the class count. When that happens the *next* block's
        ``in_channels`` no longer matches what this one emits, so it is rebuilt
        to fit. Rebuilding discards any filters that block already held, which
        is correct in every order the layers are actually fitted: they were
        computed for an input that no longer exists.
        """
        block = self.block(layer)
        before = block.out_channels
        block.set_filters(bank.weight, bank.bias)

        if block.out_channels != before and layer < len(self):
            following = self.arch.layers[layer]  # layer is 1-based: next spec
            stale = self.blocks[layer]
            rebuilt = self.block_factory(following, block.out_channels)
            self.blocks[layer] = rebuilt.to(_device_of(stale))

    def forward(self, x: Tensor, upto: int | None = None) -> Tensor:
        """Encode ``(B, C, H, W)`` through blocks ``1..upto`` (default: all).

        ``upto`` is a layer number, not an index: ``upto=0`` returns the
        input unchanged, which is what makes it usable as a loop bound when
        fitting layer by layer.
        """
        last = len(self) if upto is None else upto
        if not 0 <= last <= len(self):
            raise ValueError(
                f"upto must be between 0 and {len(self)} (layers are 1-based), "
                f"got {upto}"
            )
        for block in list(self.blocks)[:last]:
            x = block(x)
        return x

    def block(self, layer: int) -> ConvBlock:
        """Block ``layer`` (1-based), for encoding one layer without the stack.

        :meth:`forward` runs blocks ``1..upto`` from layer-0 input; a Learner
        fitting layer by layer already holds layer ``L-1``'s features and only
        needs the one block applied to them.
        """
        if not 1 <= layer <= len(self):
            raise ValueError(
                f"layer must be between 1 and {len(self)} (1-based), got {layer}"
            )
        return cast(ConvBlock, self.blocks[layer - 1])


def arch_from_json(described: dict[str, Any]) -> ArchSpec:
    """Rebuild the :class:`~spifil.config.ArchSpec` a bundle describes.

    Each layer's ``out_channels`` is what the fit actually **built**, which is
    not always what its spec asked for: the per-class integer division
    under-produces whenever the filter count is not divisible by the class
    count, and a narrow layer makes the next block narrower in turn.
    Rebuilding from the built counts is what makes the reconstructed stack's
    shapes match the saved weights; the request survives alongside it as
    ``target_out_channels``, for the record rather than for construction.

    Component overrides (scorer, selector, ...) are absent: they
    are fit-time concerns, and a loaded model has nothing left to fit.
    """
    version = int(described.get("format", FORMAT_VERSION))
    if version > FORMAT_VERSION:
        raise ValueError(
            f"architecture.json is format version {version}, but this spifil "
            f"understands up to {FORMAT_VERSION} — upgrade to load it"
        )
    layers = described["layers"]
    if not layers:
        raise ValueError("architecture.json describes no layers")

    return ArchSpec(
        layers=[
            LayerSpec(
                kernel_size=int(layer["kernel_size"]),
                out_channels=int(layer["out_channels"]),
                dilation=int(layer["dilation"]),
                pool_type=str(layer["pool_type"]),
                pool_size=int(layer["pool_size"]),
                pool_stride=int(layer["pool_stride"]),
                relu=bool(layer["relu"]),
            )
            for layer in layers
        ],
        stdev_factor=float(described.get("stdev_factor", ArchSpec().stdev_factor)),
    )


def load_model(
    bundle: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    block_factory: ConvBlockFactory = SpifilConvBlock,
) -> SpifilNet:
    """Load a bundle written by :meth:`Learner.export <spifil.learner.Learner.export>`.

    The inverse of ``export``: ``architecture.json`` gives the shapes and
    ``model.pt`` the weights, so a fitted encoder comes back without re-running
    anything. What comes back is an ordinary :class:`SpifilNet` — freeze it,
    fine-tune it, or compose it into a larger model. To load the colour
    transform along with it, use :func:`spifil.nn.encoder.load_encoder`, which
    reads the same bundle.

    Parameters
    ----------
    bundle
        Directory holding ``architecture.json`` and ``model.pt``.
    map_location
        Device to load the weights onto, as in :func:`torch.load`.
    block_factory
        Block implementation to rebuild with; must match what the fit used.

    Returns
    -------
    SpifilNet
        In ``eval()`` mode, with the learned filters installed.

    Raises
    ------
    FileNotFoundError
        If either required file is missing from ``bundle``.
    RuntimeError
        If the weights do not fit the described architecture. The load is
        strict, which is what makes the description a check rather than a
        label.
    """
    path = Path(bundle)
    description = path / "architecture.json"
    weights = path / "model.pt"
    for required in (description, weights):
        if not required.is_file():
            raise FileNotFoundError(
                f"{required} is missing; {path} does not look like a model "
                f"bundle (expected architecture.json and model.pt)"
            )

    described = json.loads(description.read_text())
    model = SpifilNet(
        arch_from_json(described),
        in_channels=int(described["in_channels"]),
        block_factory=block_factory,
    )
    state_dict = torch.load(weights, map_location=map_location, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    return model.eval()


def _device_of(module: nn.Module) -> torch.device:
    """The device a module's parameters live on (CPU if it has none)."""
    return next(module.parameters(), torch.empty(0)).device
