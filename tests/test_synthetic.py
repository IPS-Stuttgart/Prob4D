import numpy as np

from prob4d.synthetic import make_synthetic_problem


def test_synthetic_problem_has_correlated_overlaps_and_metric_anchors() -> None:
    problem = make_synthetic_problem(seed=11, num_frames=55, height=4, width=6)

    assert len(problem.overlap_windows) >= 3
    assert problem.overlap_windows[0].common_frames(problem.overlap_windows[1]).size > 0
    assert problem.disjoint_windows[0].common_frames(problem.disjoint_windows[1]).size == 0
    assert problem.scale_anchors
    for anchor in problem.scale_anchors:
        truth_scale = problem.true_overlap_gauges[anchor.window_id].scale
        np.testing.assert_allclose(anchor.scale, truth_scale, rtol=0.08)
