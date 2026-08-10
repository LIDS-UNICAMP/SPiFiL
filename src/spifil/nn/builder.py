"""Selected patches to conv weights.

Turning the selected patch dataset into a kernel bank takes three steps:
z-score the dataset per feature (with ``stdev_factor`` added to every stdev),
unit-normalize each z-scored patch, then **fold the normalization into the
weights** so the encoder needs no normalization layer::

    K[k][f]  =  unit_norm(zscore(patch_k))[f] / stdev[f]
    bias[k]  = -sum_f mean[f] * K[k][f]

which makes ``K·x + bias`` equal to ``unit_norm(zscore(patch_k)) · zscore(x)``:
the convolution measures similarity to the normalized filter, in z-scored
input space, at inference time and for free.

Two details of that normalization change the numbers:

- **The stdev floor is added, not clamped.** The divisor is
  ``sqrt(population variance) + stdev_factor``, not
  ``max(stdev, stdev_factor)``, so it shifts every feature's scale slightly,
  not just the degenerate ones.
- **Unit normalization has a threshold.** A patch is divided by its norm only
  when that norm exceeds ``1e-3``; a shorter vector is left as-is rather than
  blown up. Filters that survive selection are essentially never that short,
  so the guard only fires in the degenerate case it exists for.

The returned weight is ``(out_channels, in_channels, span, span)``, built
straight from :attr:`~spifil.types.PatchSet.feats` with a ``reshape``, because
patches are kept in ``F.unfold``'s channel-major layout --- the same layout a
flattened ``Conv2d`` weight multiplies (see :mod:`spifil.patches`).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from spifil.config import LayerSpec
from spifil.types import PatchSet

__all__ = ["FilterBank", "build_filters"]

# A patch is unit-normalized only when its norm exceeds this.
_UNIT_NORM_THRESHOLD = 1e-3


@dataclass(frozen=True)
class FilterBank:
    """A layer's learned filters, plus the statistics they were folded from.

    Attributes
    ----------
    weight
        ``(out_channels, in_channels, span, span)`` float32 conv weight, ready
        for :meth:`SpifilConvBlock.set_filters
        <spifil.nn.blocks.SpifilConvBlock.set_filters>` or a plain
        ``nn.Conv2d``. Filter ``k`` comes from row ``k`` of the patch set it
        was built from, so the caller's row order is the filter order.
    bias
        ``(out_channels,)`` float32 conv bias — the folded-in mean term.
    labels
        ``(out_channels,)`` int64 class of each filter, carried over from the
        patch it was built from.
    mean, stdev
        ``(in_channels * span**2,)`` float32 per-feature z-score statistics
        over the selected patches, in **channel-major** layout. They are the
        model's provenance: ``weight`` and ``bias`` cannot be un-folded back
        into them.
    """

    weight: Tensor
    bias: Tensor
    labels: Tensor
    mean: Tensor
    stdev: Tensor

    def __len__(self) -> int:
        return int(self.weight.shape[0])


def build_filters(
    selected: PatchSet, spec: LayerSpec, *, stdev_factor: float = 0.01
) -> FilterBank:
    """Fold selected patches into conv weights and bias.

    Parameters
    ----------
    selected
        The patches chosen by a :class:`~spifil.selection.Selector`, one per
        filter. Row order becomes filter order.
    spec
        The layer being built; supplies ``kernel_size`` (and, through it, the
        spatial extent of each patch) so the flat features can be reshaped.
    stdev_factor
        Added to every per-feature stdev before dividing; comes from
        :attr:`~spifil.config.ArchSpec.stdev_factor`.

    Returns
    -------
    A :class:`FilterBank`. Everything is computed on ``selected.feats``'
    device and dtype, so a GPU selection stays on the GPU.
    """
    if len(selected) == 0:
        raise ValueError("build_filters needs at least one selected patch")

    # Same window as patch extraction (see spifil.patches): span equals
    # kernel_size only for odd sizes.
    span = 2 * (spec.kernel_size // 2) + 1
    n_feats = int(selected.feats.shape[1])
    if n_feats % (span * span) != 0:
        raise ValueError(
            f"{n_feats} features do not split into a {span}x{span} window; "
            f"kernel_size={spec.kernel_size} cannot be the one these patches "
            "were extracted with"
        )
    in_channels = n_feats // (span * span)

    feats = selected.feats
    mean = feats.mean(dim=0)
    # Population variance (divide by N), then the floor — added, not clamped.
    stdev = torch.sqrt(((feats - mean) ** 2).mean(dim=0)) + stdev_factor

    zscored = (feats - mean) / stdev
    norms = torch.linalg.vector_norm(zscored, dim=1, keepdim=True)
    zscored = torch.where(norms > _UNIT_NORM_THRESHOLD, zscored / norms, zscored)

    kernels = zscored / stdev
    bias = -(kernels * mean).sum(dim=1)

    return FilterBank(
        weight=kernels.reshape(len(selected), in_channels, span, span).contiguous(),
        bias=bias.contiguous(),
        labels=selected.labels.clone(),
        mean=mean,
        stdev=stdev,
    )
