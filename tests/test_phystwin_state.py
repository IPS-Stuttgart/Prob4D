import numpy as np

from prob4d.phystwin_state import (
    anchored_physics_rollout,
    paired_frame_block_bootstrap,
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
