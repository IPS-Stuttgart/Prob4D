import numpy as np

from prob4d.data import PredictionWindow
from prob4d.uncertainty import DepthDisagreementModel, StructuredCovariance


def make_window() -> PredictionWindow:
    point_map = np.zeros((2, 2, 3, 3))
    point_map[..., 2] = np.array([[[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]] * 2)
    return PredictionWindow(
        "window",
        np.array([0, 1]),
        point_map,
        np.ones((2, 2, 3), dtype=bool),
    )


def _imbalanced_group_calibration() -> tuple[np.ndarray, StructuredCovariance, np.ndarray]:
    small_count = 4
    large_count = 400
    count = small_count + large_count
    rays = np.zeros((count, 3))
    rays[:, 2] = 1.0
    covariance = StructuredCovariance(
        ray_directions=rays,
        parallel_variance=np.ones(count),
        lateral_variance=np.ones(count),
    )
    errors = np.zeros((count, 3))
    errors[:small_count] = np.array([1.0, 1.0, 1.0])
    errors[small_count:] = np.array([3.0, 3.0, 3.0])
    groups = np.asarray(
        ["small"] * small_count + ["large"] * large_count,
        dtype=object,
    )
    return errors, covariance, groups


def test_depth_model_grows_along_ray_uncertainty() -> None:
    covariance = DepthDisagreementModel().predict(make_window())

    assert covariance.parallel_variance[0, 0, 2] > covariance.parallel_variance[0, 0, 0]
    assert covariance.parallel_variance[0, 0, 2] > covariance.lateral_variance[0, 0, 2]
    np.testing.assert_allclose(
        covariance.matrices()[0, 0, 0, :2, :2], np.eye(2) * covariance.lateral_variance[0, 0, 0]
    )


def test_calibration_recovers_known_variance_scale() -> None:
    generator = np.random.default_rng(2)
    window = make_window()
    model = DepthDisagreementModel()
    covariance = model.predict(window)
    errors = np.zeros_like(window.point_map)
    errors[..., 2] = generator.normal(
        scale=np.sqrt(4.0 * covariance.parallel_variance), size=window.shape
    )
    errors[..., 0] = generator.normal(
        scale=np.sqrt(2.0 * covariance.lateral_variance), size=window.shape
    )
    errors[..., 1] = generator.normal(
        scale=np.sqrt(2.0 * covariance.lateral_variance), size=window.shape
    )

    calibrated, report = model.calibrate(errors, covariance, trim_quantile=1.0)

    assert calibrated.parallel_scale > 1.0
    assert calibrated.lateral_scale > 1.0
    assert report.count == np.prod(window.shape)


def test_calibration_scales_disagreement_variance_too() -> None:
    window = make_window()
    evidence = np.ones(window.shape)
    from prob4d.uncertainty import DisagreementEvidence

    disagreement = DisagreementEvidence(evidence.copy(), evidence.copy(), evidence.copy())
    model = DepthDisagreementModel(parallel_scale=3.0, lateral_scale=5.0)
    unscaled = DepthDisagreementModel().predict(window, disagreement)
    scaled = model.predict(window, disagreement)

    np.testing.assert_allclose(scaled.parallel_variance, 3.0 * unscaled.parallel_variance)
    np.testing.assert_allclose(scaled.lateral_variance, 5.0 * unscaled.lateral_variance)


def test_group_balanced_calibration_assigns_equal_mass_to_sequences() -> None:
    errors, covariance, groups = _imbalanced_group_calibration()
    model = DepthDisagreementModel()

    balanced, report = model.calibrate_group_balanced(
        errors,
        covariance,
        groups,
        trim_quantile=1.0,
    )
    pooled, _ = model.calibrate(errors, covariance, trim_quantile=1.0)

    np.testing.assert_allclose(balanced.parallel_scale, 5.0)
    np.testing.assert_allclose(balanced.lateral_scale, 5.0)
    assert pooled.parallel_scale > 8.9
    assert pooled.lateral_scale > 8.9
    assert report.group_ids == ("large", "small")
    assert report.group_counts == (400, 4)
    assert report.group_count == 2
    assert report.to_dict()["aggregation"] == (
        "equal-group-mean-of-within-group-upper-winsorized-ratios-v2"
    )


def test_group_balanced_calibration_is_row_order_invariant() -> None:
    errors, covariance, groups = _imbalanced_group_calibration()
    model = DepthDisagreementModel()
    forward, forward_report = model.calibrate_group_balanced(
        errors,
        covariance,
        groups,
        trim_quantile=1.0,
    )
    permutation = np.arange(len(errors))[::-1]
    reversed_covariance = StructuredCovariance(
        ray_directions=covariance.ray_directions[permutation],
        parallel_variance=covariance.parallel_variance[permutation],
        lateral_variance=covariance.lateral_variance[permutation],
    )
    reverse, reverse_report = model.calibrate_group_balanced(
        errors[permutation],
        reversed_covariance,
        groups[permutation],
        trim_quantile=1.0,
    )

    assert forward == reverse
    assert forward_report == reverse_report


def test_group_balanced_calibration_rejects_empty_active_group() -> None:
    errors, covariance, groups = _imbalanced_group_calibration()
    groups = groups.copy()
    groups[0] = ""

    with np.testing.assert_raises_regex(ValueError, "group IDs must be non-empty"):
        DepthDisagreementModel().calibrate_group_balanced(
            errors,
            covariance,
            groups,
        )
