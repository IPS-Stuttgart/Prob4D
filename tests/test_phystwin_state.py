import numpy as np
import pytest

from prob4d.data import PredictionWindow
from prob4d.lineage import motioncrafter_temporal_lineage_manifest
from prob4d.phystwin_state import (
    anchored_physics_rollout,
    paired_frame_block_bootstrap,
    validate_causal_source_lineage,
)


def test_anchored_rollout_preserves_observed_endpoint_offset() -> None:
    trajectory = np.zeros((4, 2, 3))
    trajectory[:, 0, 0] = np.arange(4)
    trajectory[:, 1, 1] = np.arange(4)
    initial = np.array([[1.25, 0.0, 0.0], [np.nan, np.nan, np.nan]])

    result = anchored_physics_rollout(
        initial,
        trajectory,
        endpoint_frame=1,
        output_frame_count=4,
    )

    np.testing.assert_allclose(result[:, 0, 0], [0.25, 1.25, 2.25, 3.25])
    assert np.all(np.isnan(result[:, 1]))

    association_only = anchored_physics_rollout(
        initial,
        trajectory,
        endpoint_frame=1,
        output_frame_count=4,
        preserve_endpoint_offset=False,
    )
    np.testing.assert_allclose(association_only[:, 0, 0], [0.0, 1.0, 2.0, 3.0])


def test_paired_block_bootstrap_detects_uniform_improvement() -> None:
    baseline = np.full(12, 0.02)
    method = np.full(12, 0.01)
    frames = np.repeat(np.arange(6), 2)

    result = paired_frame_block_bootstrap(
        method,
        baseline,
        frames,
        repetitions=100,
        seed=3,
    )

    assert np.isclose(result["method_minus_baseline_mean_m"], -0.01)
    np.testing.assert_allclose(result["interval_95_m"], [-0.01, -0.01])
    assert result["probability_method_better"] == 1.0
    assert len(result["paired_frame_rows"]) == 6
    assert result["paired_frame_rows"][0]["count"] == 2


def _prediction(frames: np.ndarray) -> PredictionWindow:
    return PredictionWindow(
        "state",
        frames,
        np.zeros((len(frames), 1, 1, 3)),
        np.ones((len(frames), 1, 1), dtype=bool),
    )


def _manifest() -> dict[str, object]:
    return {
        "format_version": 1,
        "config": {"window_size": 25, "overlap": 8},
        "temporal_lineage": motioncrafter_temporal_lineage_manifest(
            window_size=25,
            overlap=8,
        ),
    }


def test_causal_source_lineage_accepts_prefix_aligned_disjoint_endpoint() -> None:
    audit = validate_causal_source_lineage(
        _prediction(np.arange(109, 159)),
        _manifest(),
        product="disjoint",
        fit_end_frame=134,
    )

    assert audit["admissible"] is True
    assert audit["source_frame_max"] == 133


def test_causal_source_lineage_rejects_future_dependent_endpoint() -> None:
    with pytest.raises(ValueError, match="depends on source frame 134"):
        validate_causal_source_lineage(
            _prediction(np.arange(110, 160)),
            _manifest(),
            product="disjoint",
            fit_end_frame=134,
        )


def test_causal_source_lineage_rejects_latent_overlap() -> None:
    with pytest.raises(ValueError, match="depends on source frame 150"):
        validate_causal_source_lineage(
            _prediction(np.arange(109, 159)),
            _manifest(),
            product="latent_linear",
            fit_end_frame=134,
        )
