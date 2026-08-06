"""Zero-copy evaluation views over validated immutable prediction windows.

A :class:`~prob4d.data.PredictionWindow` already enforces shapes, masks,
finite active values, immutable ownership, and canonical dense-storage dtypes.
When such a window is deliberately used as an internal engineering reference,
copying it through :class:`~prob4d.metrics.TruthSequence` duplicates every dense
point and flow field. This module provides an explicit subtype that reuses those
validated read-only arrays without weakening the ordinary public truth contract.
"""

from __future__ import annotations

import numpy as np

from .data import PredictionWindow
from .metrics import TruthSequence


class PredictionWindowTruthView(TruthSequence):
    """A read-only ``TruthSequence`` view backed by one ``PredictionWindow``.

    Construction accepts only a fully validated ``PredictionWindow`` (including
    memory-mapped execution-store windows). The retained arrays must already be
    read-only and are assigned by identity: no dtype conversion, defensive copy,
    or dense materialization is performed.

    This adapter is intended for internal parity, ablation, and engineering
    diagnostics where a prediction artifact is intentionally the common
    reference. It does not turn that artifact into external ground truth and
    must not be used to support an accuracy claim.
    """

    _source_window_id: str

    def __init__(self, window: PredictionWindow) -> None:
        if not isinstance(window, PredictionWindow):
            raise TypeError("window must be a validated PredictionWindow")

        arrays = {
            "frame_indices": window.frame_indices,
            "point_map": window.point_map,
            "valid_mask": window.valid_mask,
            "scene_flow": window.scene_flow,
            "deform_mask": window.deform_mask,
        }
        for name, value in arrays.items():
            if value is not None and np.asarray(value).flags.writeable:
                raise ValueError(
                    f"validated prediction field {name!r} unexpectedly remains writeable"
                )

        object.__setattr__(self, "frame_indices", window.frame_indices)
        object.__setattr__(self, "point_map", window.point_map)
        object.__setattr__(self, "valid_mask", window.valid_mask)
        object.__setattr__(self, "scene_flow", window.scene_flow)
        object.__setattr__(self, "deform_mask", window.deform_mask)
        object.__setattr__(self, "_source_window_id", window.window_id)

    @property
    def source_window_id(self) -> str:
        """Return the exact prediction-window identity backing this view."""

        return self._source_window_id

    @property
    def retained_array_bytes(self) -> int:
        """Return dense bytes referenced by the view, without implying ownership."""

        values = (
            self.frame_indices,
            self.point_map,
            self.valid_mask,
            self.scene_flow,
            self.deform_mask,
        )
        return sum(np.asarray(value).nbytes for value in values if value is not None)


def prediction_window_truth_view(window: PredictionWindow) -> PredictionWindowTruthView:
    """Return a zero-copy truth-compatible view of a validated prediction window."""

    return PredictionWindowTruthView(window)


__all__ = ["PredictionWindowTruthView", "prediction_window_truth_view"]
