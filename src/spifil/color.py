"""Color transforms producing layer-0 multiband features.

The default is :class:`LabNorm`, the normalized-Lab conversion the SPiFiL
results were produced with. It is **not** a textbook CIELAB conversion, and the
differences are large enough to change features, so they are stated here rather
than discovered later:

1. **Asymmetric YCbCr round trip.** RGB pixels are stored as YCbCr using
   **BT.2020** and converted back to RGB using **BT.601**, so the image that
   reaches the Lab conversion is *not* the image on disk. Both directions
   quantize to integers, making the round trip lossy as well as asymmetric.
2. **No sRGB gamma decode.** The XYZ matrix is applied directly to R/G/B in
   [0, 1], without the usual sRGB linearization.
3. **Fixed-constant normalization.** The Lab channels are mapped to ~[0, 1] by
   dividing by hard-coded empirical ranges rather than by the actual extrema.

Intermediate precision is part of the definition too: the chain rounds to
float32 at specific points, and evaluating it wholly in float32 (or wholly in
float64) drifts by ~1e-7 per pixel.

For a standard colour space, :class:`Identity` passes pre-converted bands
straight through, and any callable matching :class:`ColorTransform` is
accepted.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

__all__ = ["ColorTransform", "Identity", "LabNorm"]

_F32 = np.float32


def _to_f32(values: npt.NDArray[Any]) -> npt.NDArray[np.float32]:
    """Round an intermediate to float32 (see the module docstring)."""
    return np.asarray(values, dtype=np.float32)


# Normalization value for 8-bit input.
_NORM_8BIT = 255

# D65-ish whitepoint (X, Y, Z) this conversion is defined against.
_WHITEPOINT = (0.950456, 1.0, 1.088754)

# Branch point between the cube-root and the linear segment of f(t).
_LABF_THRESHOLD = 8.85645167903563082e-3

# Hard-coded empirical Lab ranges used for the [0, 1] normalization.
_L_MAX = 99.998337
_A_MIN, _A_MAX = 86.182236, 98.258614
_B_MIN, _B_MAX = 107.867744, 94.481682


@runtime_checkable
class ColorTransform(Protocol):
    """Turns an RGB image into the multiband feature map of layer 0."""

    @property
    def out_channels(self) -> int:
        """Number of bands produced."""
        ...

    def __call__(self, image: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
        """Convert ``(H, W, 3)`` uint8 RGB to ``(C, H, W)`` float32 features."""
        ...


class LabNorm:
    """Normalized-Lab conversion — SPiFiL's layer-0 features.

    Turns an 8-bit RGB image into three ~[0, 1] bands. See the module
    docstring for how it differs from a textbook CIELAB conversion.

    Notes
    -----
    Only 8-bit **RGB** input is supported. Grayscale input raises, since
    converting it silently would produce features that are hard to interpret;
    convert it yourself, or use :class:`Identity`.
    """

    out_channels = 3

    def __call__(self, image: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
        if image.ndim != 3 or image.shape[2] not in (3, 4):
            raise ValueError(
                "LabNorm expects an (H, W, 3) RGB image; got shape "
                f"{image.shape}. Convert grayscale input to RGB first, or use "
                "Identity(out_channels=1) to feed the single band through."
            )
        rgb = image[:, :, :3]
        ycbcr = _rgb_to_ycbcr_bt2020(rgb)
        rgb_roundtrip = _ycbcr_to_rgb_bt601(ycbcr)
        lab = _rgb_to_labnorm2(rgb_roundtrip)
        return np.ascontiguousarray(np.transpose(lab, (2, 0, 1)))


class Identity:
    """Pass channels through unchanged — for data that is already features.

    The transform for input that arrives normalized: a dataset whose bands are
    already what the filters should see (RGB kept as RGB, a colour space
    converted upstream, sensor data that was never an image). It only moves
    channels to the front, since the protocol's output is ``(C, H, W)`` while
    image-shaped input is ``(H, W, C)``.

    Unlike :class:`LabNorm` it takes a band count, because nothing about the
    data announces one.

    Parameters
    ----------
    out_channels
        Bands the input carries. Checked against every image, so a mismatch
        surfaces here rather than as a shape error deep in the first
        convolution.
    """

    def __init__(self, out_channels: int = 3) -> None:
        if out_channels < 1:
            raise ValueError(f"out_channels must be >= 1, got {out_channels}")
        self.out_channels = out_channels

    def __call__(self, image: npt.NDArray[Any]) -> npt.NDArray[np.float32]:
        if image.ndim == 2:
            image = image[:, :, None]
        if image.ndim != 3 or image.shape[2] != self.out_channels:
            raise ValueError(
                f"Identity({self.out_channels}) expects (H, W, "
                f"{self.out_channels}) input; got shape {image.shape}"
            )
        moved = np.transpose(image, (2, 0, 1))
        return np.ascontiguousarray(moved, dtype=np.float32)


def _rgb_to_ycbcr_bt2020(rgb: npt.NDArray[np.uint8]) -> npt.NDArray[np.int64]:
    """BT.2020 RGB -> YCbCr, 8-bit in / 8-bit out, in double arithmetic."""
    r, g, b = (rgb[..., i].astype(np.float64) / 255.0 for i in range(3))
    y = 0.2627 * r + 0.6780 * g + 0.0593 * b
    cb = (b - y) / 1.8814
    cr = (r - y) / 1.4746
    y = np.clip(y, 0.0, 1.0)
    cb = np.clip(cb, -0.5, 0.5)
    cr = np.clip(cr, -0.5, 0.5)
    # Truncation toward zero, not rounding.
    return np.stack(
        [
            np.trunc(y * 219).astype(np.int64) + 16,
            np.trunc(cb * 224).astype(np.int64) + 128,
            np.trunc(cr * 224).astype(np.int64) + 128,
        ],
        axis=-1,
    )


def _ycbcr_to_rgb_bt601(ycbcr: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    """BT.601 YCbCr -> RGB. Truncation happens *before* clipping."""
    a, b = 16.0, 128.0  # (16/255)*255 and (128/255)*255, exact in float32
    y = ycbcr[..., 0].astype(np.float64) - a
    cb = ycbcr[..., 1].astype(np.float64) - b
    cr = ycbcr[..., 2].astype(np.float64) - b
    r = 1.164383562 * y + 1.596026786 * cr
    g = 1.164383562 * y - 0.39176229 * cb - 0.812967647 * cr
    bl = 1.164383562 * y + 2.017232143 * cb
    out = np.trunc(np.stack([r, g, bl], axis=-1)).astype(np.int64)
    return np.clip(out, 0, _NORM_8BIT)


def _labf(t: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Lab f(t): cube root above the threshold, linear below."""
    t64 = t.astype(np.float64)
    return _to_f32(
        np.where(
            t64 >= _LABF_THRESHOLD,
            np.power(np.maximum(t64, 0.0), 0.333333333333333),
            (841.0 / 108.0) * t64 + (4.0 / 29.0),
        )
    )


def _rgb_to_labnorm2(rgb: npt.NDArray[np.int64]) -> npt.NDArray[np.float32]:
    """RGB -> normalized Lab: no gamma decode, float32 at each named step."""
    chan = [
        _to_f32(rgb[..., i].astype(np.float32) / _F32(_NORM_8BIT)) for i in range(3)
    ]
    r, g, b = (c.astype(np.float64) for c in chan)

    # Double-precision matrix products, each rounded back to float32.
    x = _to_f32(
        0.4123955889674142161 * r
        + 0.3575834307637148171 * g
        + 0.1804926473817015735 * b
    )
    y = _to_f32(
        0.2125862307855955516 * r
        + 0.7151703037034108499 * g
        + 0.07220049864333622685 * b
    )
    z = _to_f32(
        0.01929721549174694484 * r
        + 0.1191838645808485318 * g
        + 0.9504971251315797660 * b
    )

    x = _labf(_to_f32(x.astype(np.float64) / _WHITEPOINT[0]))
    y = _labf(_to_f32(y.astype(np.float64) / _WHITEPOINT[1]))
    z = _labf(_to_f32(z.astype(np.float64) / _WHITEPOINT[2]))

    # From here on the arithmetic is float32.
    lightness = _to_f32(116 * y - 16)
    a_star = _to_f32(500 * (x - y))
    b_star = _to_f32(200 * (y - z))

    lightness = _to_f32(lightness / _F32(_L_MAX))
    a_star = _to_f32((a_star + _F32(_A_MIN)) / _F32(_A_MIN + _A_MAX))
    b_star = _to_f32((b_star + _F32(_B_MIN)) / _F32(_B_MIN + _B_MAX))

    return np.stack([lightness, a_star, b_star], axis=-1)
