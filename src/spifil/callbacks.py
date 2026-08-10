"""Cross-cutting concerns, hooked to the Learner's events.

Everything here is I/O or reporting — the things that would otherwise turn the
layer loop into a log of itself. A callback is any object with methods named
after the events in :data:`~spifil.learner.EVENTS`; :class:`Callback` is a
convenience base that defines them all as no-ops, not a requirement.

Callbacks receive the Learner and may read anything on it:
``learn.layer`` is where the fit currently is, ``learn.state`` is what every
layer has produced so far, ``learn.selected`` is the current layer's picks.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO

import torch

from spifil.learner import CancelLayerException, LayerState
from spifil.types import Seeds

if TYPE_CHECKING:  # pragma: no cover
    from spifil.learner import Learner

__all__ = ["CSVLogger", "Callback", "Checkpoint", "ProgressBar"]


class Callback:
    """No-op implementations of every event; override the ones you need.

    The docstrings below are the event contract: when each one fires, and what
    is on the Learner by the time it does. ``learn.layer`` is the layer being
    fitted throughout — **including ``0``** during layer 0's scoring pass, so
    ``after_scoring`` never has to guess which pass it is looking at.
    """

    def before_fit(self, learn: Learner) -> None:
        """Once, at the top of :meth:`~spifil.learner.Learner.fit`.

        Before layer 0 is prepared, so a callback that restores state (as
        :class:`Checkpoint` does) can populate ``learn.state[0]`` here and have
        the idempotent ``prepare()`` accept it instead of recomputing.
        """

    def before_layer(self, learn: Learner) -> None:
        """At the start of each layer, before any of its work.

        Raising :class:`~spifil.learner.CancelLayerException` skips the layer
        entirely, which is how :class:`Checkpoint` resumes a finished one.
        """

    def after_scoring(self, learn: Learner) -> None:
        """After a scoring pass, with ``patches``, ``scores`` and ``ranks`` set.

        Fires once per scored layer, layer 0 included. The last layer is never
        scored — it has no successor to rank seeds for — so this does not fire
        for it.
        """

    def after_selection(self, learn: Learner) -> None:
        """After the selector picks; ``learn.selected`` holds the patches."""

    def after_model_build(self, learn: Learner) -> None:
        """After the filters are folded and installed on the layer's block.

        ``learn.state[layer].bank`` is the new :class:`~spifil.nn.builder.FilterBank`,
        and its size is what the layer actually produced — which per-class
        integer division can make smaller than the spec's target.
        """

    def after_encode(self, learn: Learner) -> None:
        """After every image is re-encoded and its seeds projected.

        ``learn.state[layer]`` is complete at this point except for the ranks,
        which the layer's own scoring pass overwrites immediately afterwards.
        """

    def after_layer(self, learn: Learner) -> None:
        """At the end of a layer, after its scoring pass has run."""

    def after_fit(self, learn: Learner) -> None:
        """Once, at the end — including when
        :class:`~spifil.learner.CancelFitException` ended the fit early."""


class Checkpoint(Callback):
    """Persist each layer's state and skip layers already done on re-run.

    - A layer is complete when its ``layer{L}.done`` marker exists. The marker
      is written after the payload and removed before rewriting it, so an
      interrupted save leaves an incomplete layer looking incomplete rather
      than looking finished and corrupt.
    - Completed layers are restored, not recomputed: ``before_layer`` loads
      the state and raises :class:`~spifil.learner.CancelLayerException`.
    - ``resume_from=L`` deletes markers from ``L`` on before the fit starts.

    **Ranks are restored; scores are not.** The payload carries features, seeds
    (ranks included) and the filter bank, which is everything the next layer
    consumes. Raw scores are dropped, so a resumed run hands a score-aware
    allocator an empty score vector rather than pretending to remember; the
    O(N²) scoring is not repeated either way.
    """

    def __init__(self, out_dir: str | Path, *, resume_from: int | None = None) -> None:
        self.dir = Path(out_dir)
        self.resume_from = resume_from

    # -- paths ---------------------------------------------------------

    def payload(self, layer: int) -> Path:
        """Where layer ``layer``'s features, seeds and filters are stored."""
        return self.dir / f"layer{layer}.pt"

    def marker(self, layer: int) -> Path:
        """The ``layer{L}.done`` file whose existence means "finished"."""
        return self.dir / f"layer{layer}.done"

    def is_complete(self, layer: int) -> bool:
        """Whether ``layer`` can be restored: both marker and payload present.

        Requiring both is what makes an interrupted save look incomplete —
        :meth:`_save` removes the marker first and writes it last.
        """
        return self.marker(layer).exists() and self.payload(layer).exists()

    # -- events --------------------------------------------------------

    def before_fit(self, learn: Learner) -> None:
        """Honour ``resume_from``, then restore layer 0 if it is complete."""
        self.dir.mkdir(parents=True, exist_ok=True)
        if self.resume_from is not None:
            self.clear_from(self.resume_from)
        if self.is_complete(0):
            learn.state[0] = self._load(0)

    def before_layer(self, learn: Learner) -> None:
        """Restore a completed layer and skip it, or do nothing and let it run."""
        assert learn.layer is not None
        if not self.is_complete(learn.layer):
            return
        learn.state[learn.layer] = self._load(learn.layer)
        bank = learn.state[learn.layer].bank
        if bank is not None:
            # The model has to carry the restored filters too: the next layer
            # encodes with it, and export() reads its weights.
            learn.model.set_filters(learn.layer, bank)
        raise CancelLayerException

    def after_scoring(self, learn: Learner) -> None:
        """Save layer 0, which is only finished once its ranks exist."""
        # The next layer selects from those ranks, so layer 0 is not
        # restorable without them. Later layers save at ``after_layer``,
        # downstream of their own re-scoring pass.
        if learn.layer == 0:
            self._save(learn, 0)

    def after_layer(self, learn: Learner) -> None:
        """Save the layer that just finished, marker last."""
        assert learn.layer is not None
        self._save(learn, learn.layer)

    # -- persistence ---------------------------------------------------

    def clear_from(self, layer: int) -> None:
        """Invalidate every layer from ``layer`` up, so the fit redoes them."""
        for stale in range(layer, 1024):
            marker, payload = self.marker(stale), self.payload(stale)
            if not marker.exists() and not payload.exists():
                if stale > layer:
                    break
                continue
            marker.unlink(missing_ok=True)
            payload.unlink(missing_ok=True)

    def _save(self, learn: Learner, layer: int) -> None:
        state = learn.state[layer]
        self.dir.mkdir(parents=True, exist_ok=True)
        self.marker(layer).unlink(missing_ok=True)
        torch.save(
            {
                "features": [f.cpu() for f in state.features],
                "seeds": [
                    {
                        "coords": s.coords,
                        "labels": s.labels,
                        "ranks": s.ranks,
                        "grid": s.grid,
                    }
                    for s in state.seeds
                ],
                "bank": state.bank,
            },
            self.payload(layer),
        )
        self.marker(layer).write_text("")

    def _load(self, layer: int) -> LayerState:
        blob: dict[str, Any] = torch.load(
            self.payload(layer), map_location="cpu", weights_only=False
        )
        return LayerState(
            features=list(blob["features"]),
            seeds=[Seeds(**{**s, "grid": tuple(s["grid"])}) for s in blob["seeds"]],
            bank=blob["bank"],
        )


class CSVLogger(Callback):
    """Two CSVs per fit: one row per layer, and one row per scored seed.

    ``layers.csv`` is the summary a sweep is read from: how many candidates
    each layer had, how many filters it actually produced (integer division
    can make that smaller than the target), and the class each filter came
    from. ``scores.csv`` holds the per-seed score and rank behind every
    selection, which is the only way to see *why* a filter was picked after
    the fact.
    """

    def __init__(self, out_dir: str | Path) -> None:
        self.dir = Path(out_dir)

    def before_fit(self, learn: Learner) -> None:
        """Write both headers, truncating whatever a previous run left."""
        self.dir.mkdir(parents=True, exist_ok=True)
        with (self.dir / "layers.csv").open("w", newline="") as handle:
            csv.writer(handle).writerow(
                ["layer", "n_candidates", "n_selected", "target_out_channels", "labels"]
            )
        with (self.dir / "scores.csv").open("w", newline="") as handle:
            csv.writer(handle).writerow(
                ["layer", "image_id", "seed_row", "label", "score", "rank"]
            )

    def after_scoring(self, learn: Learner) -> None:
        """Append one ``scores.csv`` row per scored seed.

        Silently does nothing for a layer restored from a checkpoint: those
        carry ranks but no patches or scores.
        """
        assert learn.layer is not None
        layer = learn.layer
        state = learn.state[layer]
        patches, scores, ranks = state.patches, state.scores, state.ranks
        if patches is None or scores is None or ranks is None:
            return
        rows = zip(
            patches.image_ids.tolist(),
            patches.seed_rows.tolist(),
            patches.labels.tolist(),
            scores.tolist(),
            ranks.tolist(),
            strict=True,
        )
        with (self.dir / "scores.csv").open("a", newline="") as handle:
            writer = csv.writer(handle)
            for image_id, seed_row, label, score, rank in rows:
                writer.writerow(
                    [layer, image_id, seed_row, label, f"{score:.9g}", rank]
                )

    def after_model_build(self, learn: Learner) -> None:
        """Append the layer's ``layers.csv`` summary row."""
        assert learn.layer is not None and learn.selected is not None
        previous = learn.state[learn.layer - 1]
        n_candidates = (
            int((previous.ranks > 0).sum()) if previous.ranks is not None else 0
        )
        with (self.dir / "layers.csv").open("a", newline="") as handle:
            csv.writer(handle).writerow(
                [
                    learn.layer,
                    n_candidates,
                    len(learn.selected),
                    learn.spec(learn.layer).out_channels,
                    " ".join(str(label) for label in learn.selected.labels.tolist()),
                ]
            )


class ProgressBar(Callback):
    """Plain-text progress on the layer loop.

    One line per step rather than a redrawn bar: the loop has a handful of
    long steps, not thousands of fast ones, and plain text needs no extra
    dependency.
    """

    def __init__(self, stream: TextIO | None = None) -> None:
        self.stream = stream if stream is not None else sys.stderr

    def _say(self, message: str) -> None:
        print(message, file=self.stream, flush=True)

    def before_fit(self, learn: Learner) -> None:
        """Announce the shape of the run: layers, images, device."""
        self._say(
            f"[spifil] fitting {learn.n_layers} layers over "
            f"{len(learn.data)} images on {learn.device}"
        )

    def after_scoring(self, learn: Learner) -> None:
        """Report how many seeds the pass ranked."""
        assert learn.layer is not None
        state = learn.state[learn.layer]
        n = 0 if state.patches is None else len(state.patches)
        self._say(f"[spifil] layer {learn.layer}: scored {n} seeds")

    def after_model_build(self, learn: Learner) -> None:
        """Report the filter count, naming the target when it fell short."""
        assert learn.layer is not None and learn.selected is not None
        target = learn.spec(learn.layer).out_channels
        built = len(learn.selected)
        shortfall = "" if built == target else f" (target {target})"
        self._say(f"[spifil] layer {learn.layer}: built {built} filters{shortfall}")

    def after_encode(self, learn: Learner) -> None:
        """Report the pooled grid the next layer will work on."""
        assert learn.layer is not None
        state = learn.state[learn.layer]
        grid = tuple(state.features[0].shape[1:]) if state.features else ()
        self._say(f"[spifil] layer {learn.layer}: encoded to {grid}")

    def after_fit(self, learn: Learner) -> None:
        """Say so."""
        self._say("[spifil] done")
