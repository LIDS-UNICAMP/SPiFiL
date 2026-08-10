"""The composable artifact: colour transform + fitted stack, as one module.

:class:`~spifil.nn.model.SpifilNet` is an ordinary ``nn.Module``, but it does
not eat images — it eats **layer-0 features**, whatever the fit's
:class:`~spifil.color.ColorTransform` produced. Handing a downstream model the
network alone is therefore half an artifact: the filters were fitted on the
transform's output — Lab by default, which is not interchangeable with RGB
(see :mod:`spifil.color`) — and feeding them raw RGB fails silently, since the
shapes match perfectly and only the numbers are wrong.

:class:`SpifilEncoder` closes that gap by carrying both halves together, so
composing with a task head is what it should be::

    encoder = load_encoder("runs/exp/models").freeze()
    model = nn.Sequential(encoder, head)      # images in, predictions out

**It is not Lab-specific.** The encoder holds whatever satisfies
the ``ColorTransform`` protocol, so RGB, grayscale, another colour space or
already-normalized data are new transforms rather than changes here. Two shapes
of transform are supported and the difference is where the work happens:

- a **numpy** transform (the protocol's ``(H, W, 3) uint8 -> (C, H, W) f32``,
  which :class:`~spifil.color.LabNorm` is) runs per image on the CPU, at the
  boundary, exactly as the fit ran it;
- an **``nn.Module``** transform runs in the graph, batched, on whatever
  device the tensor is on, and skips the numpy branch entirely.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor, nn

from spifil.color import ColorTransform
from spifil.nn.model import SpifilNet, load_model

__all__ = ["SpifilEncoder", "describe_color", "load_encoder", "resolve_color"]

ImageBatch = Tensor | npt.NDArray[np.uint8] | list[npt.NDArray[np.uint8]]


class SpifilEncoder(nn.Module):
    """A fitted :class:`~spifil.nn.model.SpifilNet` behind its colour transform.

    Parameters
    ----------
    color
        The transform layer 0 was fitted with. Kept as a submodule when it is
        one, so a torch-native transform's own state travels with the encoder.
    net
        The fitted stack.

    Notes
    -----
    ``forward`` accepts a batch of images — a ``(B, H, W, 3)`` uint8 array, a
    list of ``(H, W, 3)`` arrays (images of differing size cannot be batched,
    so a list is the honest input for that case only when they *do* share a
    shape), or an already-transformed ``(B, C, H, W)`` float tensor, which
    passes straight to the network. Mixing the last case in is what keeps this
    usable in a pipeline whose preprocessing already ran.
    """

    def __init__(self, color: ColorTransform, net: SpifilNet) -> None:
        super().__init__()
        self.color = color
        if isinstance(color, nn.Module):
            self.add_module("color_module", color)
        self.net = net

    @property
    def in_channels(self) -> int:
        """Bands the colour transform produces — the stack's input width."""
        return int(self.color.out_channels)

    @property
    def out_channels(self) -> int:
        """Filters the last block holds."""
        return int(self.net.block(len(self.net)).out_channels)

    def freeze(self) -> SpifilEncoder:
        """Stop gradients for every parameter; returns ``self`` for chaining.

        The usual way to use a SPiFiL encoder downstream: its filters were
        *assigned*, not trained, and the head is what learns. Unfreezing is
        ``requires_grad_(True)`` — nothing here is one-way.
        """
        self.requires_grad_(False)
        return self

    def transform(self, images: ImageBatch) -> Tensor:
        """Turn a batch of images into ``(B, C, H, W)`` layer-0 features."""
        if isinstance(images, Tensor) and images.is_floating_point():
            return images

        if isinstance(self.color, nn.Module):
            batch = images if isinstance(images, Tensor) else torch.as_tensor(images)
            # A transform that is both a ColorTransform and a Module is called
            # as the Module: batched, in the graph, on the tensor's device.
            return cast(Tensor, cast(nn.Module, self.color)(batch))

        if isinstance(images, Tensor):
            frames: Any = images.detach().cpu().numpy()
        else:
            frames = images
        if isinstance(frames, np.ndarray) and frames.ndim == 3:
            frames = frames[None]
        features = [np.asarray(self.color(np.asarray(frame))) for frame in frames]
        return torch.from_numpy(np.stack(features)).to(_device_of(self.net))

    def forward(self, images: ImageBatch, upto: int | None = None) -> Tensor:
        """Encode images into features, through blocks ``1..upto`` (default all).

        ``upto`` uses the same 1-based layer numbering as
        :class:`~spifil.nn.model.SpifilNet`, so ``upto=0`` returns the colour
        transform's output and ``upto=2`` an intermediate representation.
        """
        return cast(Tensor, self.net(self.transform(images), upto=upto))


def describe_color(color: ColorTransform) -> dict[str, Any]:
    """Record which transform a bundle's features came from.

    Only the import path and the band count: enough for a loader to rebuild a
    parameterless transform and, more importantly, enough for one to *refuse*
    when the bundle and the caller disagree. Constructor arguments are not
    introspected — a parameterized transform is passed back in explicitly (see
    :func:`resolve_color`), which is also how a Hydra-configured run would do
    it.
    """
    kind = type(color)
    return {
        "class": f"{kind.__module__}.{kind.__qualname__}",
        "out_channels": int(color.out_channels),
    }


def resolve_color(
    described: dict[str, Any] | None,
    override: ColorTransform | None = None,
) -> ColorTransform:
    """The colour transform a bundle asks for, or the one the caller insists on.

    Parameters
    ----------
    described
        The bundle's ``color`` entry, or ``None`` for a bundle written before
        the field existed.
    override
        A transform to use instead — required when the recorded one takes
        constructor arguments, and the escape hatch when it is not importable
        (renamed, or defined outside spifil).

    Raises
    ------
    ValueError
        If ``override`` produces a different number of bands than the bundle
        was fitted on. Shapes alone would not catch this: two transforms can
        agree on band count and disagree on everything else, but a mismatch in
        the count is unambiguous and worth refusing outright.
    ImportError
        If the recorded transform cannot be imported and no ``override`` was
        given.
    """
    if override is not None:
        if described is not None:
            expected = int(described["out_channels"])
            if int(override.out_channels) != expected:
                raise ValueError(
                    f"this bundle was fitted on {expected}-band features from "
                    f"{described['class']}, but the given "
                    f"{type(override).__name__} produces "
                    f"{override.out_channels}"
                )
        return override

    if described is None:
        raise ValueError(
            "this bundle does not record its colour transform (written by an "
            "older spifil); pass the one the fit used as `color=`"
        )

    path = str(described["class"])
    module_name, _, attribute = path.rpartition(".")
    try:
        factory = getattr(importlib.import_module(module_name), attribute)
    except (ImportError, AttributeError) as exc:
        raise ImportError(
            f"cannot import the colour transform this bundle was fitted with "
            f"({path}); pass an instance as `color=` instead"
        ) from exc
    try:
        rebuilt: ColorTransform = factory()
    except TypeError as exc:
        raise TypeError(
            f"{path} needs constructor arguments, which the bundle does not "
            f"record; pass a configured instance as `color=`"
        ) from exc

    # A default-constructed transform is not necessarily the *configured* one:
    # a constructor argument with a default rebuilds silently and wrongly. The
    # band count is the part of that we can check, so we check it.
    expected = int(described["out_channels"])
    if int(rebuilt.out_channels) != expected:
        raise ValueError(
            f"{path}() produces {rebuilt.out_channels}-band features but this "
            f"bundle was fitted on {expected}; it was configured with "
            f"arguments the bundle does not record, so pass the configured "
            f"instance as `color=`"
        )
    return rebuilt


def load_encoder(
    bundle: str | Path,
    *,
    color: ColorTransform | None = None,
    map_location: str | torch.device = "cpu",
) -> SpifilEncoder:
    """Load a bundle as a ready-to-compose :class:`SpifilEncoder`.

    Parameters
    ----------
    bundle
        Directory written by :meth:`Learner.export <spifil.learner.Learner.export>`.
    color
        Overrides the transform recorded in the bundle — needed when it takes
        constructor arguments, refused when its band count disagrees.
    map_location
        Device to load the weights onto, as in :func:`torch.load`.
    """
    path = Path(bundle)
    net = load_model(path, map_location=map_location)
    described = json.loads((path / "architecture.json").read_text())
    return SpifilEncoder(resolve_color(described.get("color"), color), net)


def _device_of(module: nn.Module) -> torch.device:
    """The device a module's parameters live on (CPU if it has none)."""
    return next(module.parameters(), torch.empty(0)).device
