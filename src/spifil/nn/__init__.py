"""The model side of SPiFiL: patches to filters, filters to a torch network."""

from spifil.nn.blocks import ConvBlock, ConvBlockFactory, SpifilConvBlock
from spifil.nn.builder import FilterBank, build_filters
from spifil.nn.encoder import SpifilEncoder, describe_color, load_encoder, resolve_color
from spifil.nn.model import FORMAT_VERSION, SpifilNet, arch_from_json, load_model

__all__ = [
    "FORMAT_VERSION",
    "ConvBlock",
    "ConvBlockFactory",
    "FilterBank",
    "SpifilConvBlock",
    "SpifilEncoder",
    "SpifilNet",
    "arch_from_json",
    "build_filters",
    "describe_color",
    "load_encoder",
    "load_model",
    "resolve_color",
]
