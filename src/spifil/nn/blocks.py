"""The convolutional block.

A layer is im2col + matrix product + bias + ReLU + pooling, which is exactly
one ``Conv2d`` followed by ``ReLU`` and a pooling module. Three conventions
make that equivalence exact, and each is a plausible place to be off by one:

- **Zero padding.** Out-of-bounds neighbors contribute zero, and padding is
  ``dilation * (kernel_size // 2)``, so the convolution is shape-preserving
  and every pixel yields one sample.
- **Cross-correlation, not convolution.** The kernel is *not* flipped --- what
  ``Conv2d`` computes despite its name, and what makes a filter's response a
  similarity to the patch it was built from.
- **Pooling boundaries take the max over valid neighbors only**, i.e.
  ``MaxPool2d``'s ``-inf`` padding rather than zero padding. The difference is
  visible on any border pixel whose true neighborhood max is negative.

**Pooled output size** is ``floor((size - 1) / stride) + 1`` with
``padding = pool_size // 2`` and an odd window: window centers land on
``0, stride, 2*stride, ...``, so pooling and striding compose predictably.

**Filter counts can shrink.** :class:`~spifil.allocation.UniformAllocator`
splits the budget by integer division, so a layer asked for 16 filters over 6
classes produces 12. :meth:`SpifilConvBlock.set_filters` therefore accepts a
bank narrower than ``spec.out_channels`` and rebuilds its convolution around
it rather than rejecting it: the spec states the target, the selection states
the outcome.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
from torch import Tensor, nn

from spifil.config import LayerSpec

__all__ = ["ConvBlock", "ConvBlockFactory", "SpifilConvBlock"]


@runtime_checkable
class ConvBlock(Protocol):
    """What :class:`~spifil.nn.model.SpifilNet` needs from a block.

    An implementation is an ``nn.Module`` as well — this protocol only names
    the two things beyond a forward pass that the model itself calls.
    """

    @property
    def out_channels(self) -> int:
        """Filters the block currently holds; see the module docstring."""
        ...

    def set_filters(self, weight: Tensor, bias: Tensor) -> None:
        """Install a filter bank in place of the block's current weights."""
        ...

    def __call__(self, x: Tensor) -> Tensor:
        """Encode a ``(B, C, H, W)`` batch — ``nn.Module.__call__``.

        Named here so a caller holding a block *as a* ``ConvBlock`` can encode
        with it, which is how the Learner applies one layer to the features it
        already has instead of re-running the stack from layer 0.
        """
        ...


@runtime_checkable
class ConvBlockFactory(Protocol):
    """Builds the module that encodes one layer."""

    def __call__(self, spec: LayerSpec, in_channels: int) -> nn.Module:
        """An ``nn.Module`` that also satisfies :class:`ConvBlock`."""
        ...


class SpifilConvBlock(nn.Module):
    """conv (zero-padded, dilated) -> ReLU -> pool, with *assigned* weights.

    A perfectly ordinary ``nn.Module``: its parameters happen to be set by
    :meth:`set_filters` from selected patches instead of by an optimizer, but
    it can still be fine-tuned, scripted, exported, or dropped into any torch
    model. Until filters are assigned, the convolution holds torch's default
    initialization — the shape is right, the values are meaningless.

    Parameters
    ----------
    spec
        The layer's kernel/pooling configuration.
    in_channels
        Channels of the incoming feature map: 3 for layer 1 (Lab), otherwise
        the *actual* filter count of the previous layer, which the integer
        division in allocation can make smaller than its ``out_channels``.
    """

    def __init__(self, spec: LayerSpec, in_channels: int) -> None:
        super().__init__()
        if in_channels < 1:
            raise ValueError(f"in_channels must be >= 1, got {in_channels}")
        if spec.pool_type not in ("max", "avg", "none"):
            raise ValueError(
                f"unknown pool_type {spec.pool_type!r}; expected max, avg or none"
            )

        self.spec = spec
        self.conv = self._make_conv(in_channels, spec.out_channels)
        self.pool = self._make_pool(spec)

    @property
    def in_channels(self) -> int:
        """Channels the block expects — the previous layer's actual filter count."""
        return int(self.conv.in_channels)

    @property
    def out_channels(self) -> int:
        """Filters actually held — see the module docstring on shrinking."""
        return int(self.conv.out_channels)

    @torch.no_grad()
    def set_filters(self, weight: Tensor, bias: Tensor) -> None:
        """Install a filter bank, rebuilding the convolution if it is narrower.

        ``weight`` is ``(n_filters, in_channels, span, span)`` and ``bias`` is
        ``(n_filters,)`` — :func:`~spifil.nn.builder.build_filters`' output.
        Assignment is a copy, so the caller's tensors stay independent of the
        module's parameters.
        """
        if weight.ndim != 4:
            raise ValueError(
                f"weight must be (n_filters, in_channels, span, span), "
                f"got {tuple(weight.shape)}"
            )
        if bias.shape != weight.shape[:1]:
            raise ValueError(
                f"bias must be ({weight.shape[0]},) to match weight, "
                f"got {tuple(bias.shape)}"
            )
        expected = (self.in_channels, *self.conv.kernel_size)
        if tuple(weight.shape[1:]) != expected:
            raise ValueError(
                f"weight is {tuple(weight.shape)}; this block convolves "
                f"{self.in_channels} channels with a "
                f"{self.conv.kernel_size[0]}x{self.conv.kernel_size[1]} window, "
                f"so it needs (n_filters, {expected[0]}, {expected[1]}, "
                f"{expected[2]})"
            )

        n_filters = int(weight.shape[0])
        if n_filters != self.out_channels:
            self.conv = self._make_conv(self.in_channels, n_filters).to(
                self.conv.weight.device
            )
        self.conv.weight.copy_(weight)
        assert self.conv.bias is not None  # constructed with bias=True
        self.conv.bias.copy_(bias)

    def forward(self, x: Tensor) -> Tensor:
        """Encode ``(B, C, H, W)`` features into ``(B, out_channels, H', W')``."""
        x = self.conv(x)
        if self.spec.relu:
            x = torch.relu(x)
        if self.pool is not None:
            x = self.pool(x)
        return x

    def _make_conv(self, in_channels: int, out_channels: int) -> nn.Conv2d:
        spec = self.spec
        # The -k//2..k//2 window (spifil.patches): kernel_size for odd
        # sizes, kernel_size + 1 for even ones.
        span = 2 * (spec.kernel_size // 2) + 1
        return nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=span,
            padding=spec.dilation * (spec.kernel_size // 2),
            dilation=spec.dilation,
            bias=True,
        )

    @staticmethod
    def _make_pool(spec: LayerSpec) -> nn.Module | None:
        if spec.pool_type == "none":
            return None
        span = 2 * (spec.pool_size // 2) + 1
        padding = spec.pool_size // 2
        if spec.pool_type == "max":
            return nn.MaxPool2d(span, stride=spec.pool_stride, padding=padding)
        # ``count_include_pad=False`` keeps the border consistent with max
        # pooling: average over the valid neighbors, not over an imagined
        # ring of zeros.
        return nn.AvgPool2d(
            span, stride=spec.pool_stride, padding=padding, count_include_pad=False
        )
