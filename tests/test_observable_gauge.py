import numpy as np

from prob4d.observable_gauge import (
    CLUSTER_OBSERVABLE_INFORMATION,
    IID_OBSERVABLE_INFORMATION,
    estimate_observable_sim3_factor,
)
from prob4d.observable_gauge_study import (
    ObservableGaugeStudyConfig,
    run_observable_gauge_study,
)
from prob4d.sim3 import Sim3, so3_exp


def _line_case(seed: int = 4):
    generator = np.random.default_rng(seed)
    coordinate = np.linspace(-1.0, 1.0, 48)
    source = np.column_stack((coordinate, np.zeros_like(coordinate), np.zeros_like(coordinate)))
    base_rotation = so3_exp(np.array([0.0, 0.25, -0.15]))
    truth = Sim3(
        scale=1.1,
        rotation=base_rotation @ so3_exp(np.array([0.6, 0.0, 0.0])),
        translation=np.array([0.2, -0.1, 0.05]),
    )
    target = truth.transform_points(source) + generator.normal(scale=0.01, size=source.shape)
    return source, target, truth


def test_centroid_chart_round_trip() -> None:
    source, target, _ = _line_case()
    factor = estimate_observable_sim3_factor(source, target)
    local = np.array([0.02, 0.03, -0.01, 0.04, 0.1, -0.2, 0.05])

    recovered = factor.chart.to_local(factor.chart.from_local(local))

    np.testing.assert_allclose(recovered, local, atol=1e-10, rtol=1e-10)


def test_collinear_overlap_retains_six_dimensional_information() -> None:
    source, target, truth = _line_case()

    factor = estimate_observable_sim3_factor(source, target)

    assert factor.rank == 6
    assert factor.nullspace_basis.shape == (7, 1)
    assert factor.covariance_method == IID_OBSERVABLE_INFORMATION
    np.testing.assert_allclose(factor.normalized_geometry_spectrum[:6], 1.0, atol=1e-12)
    assert factor.normalized_geometry_spectrum[6] < 1e-12
    null_direction = factor.nullspace_basis[:, 0]
    assert factor.quadratic_cost_local(0.5 * null_direction) < 1e-18
    truth_local = factor.chart.to_local(truth)
    assert abs(float(null_direction @ truth_local)) > 0.5


def test_standard_sim3_covariance_transports_into_intrinsic_chart() -> None:
    source, target, truth = _line_case()
    factor = estimate_observable_sim3_factor(source, target)
    covariance_vector = np.diag(
        np.array([0.03, 0.04, 0.05, 0.06, 0.02, 0.03, 0.04]) ** 2
    )

    transported = factor.chart.transport_vector_gaussian(truth, covariance_vector)
    fused = factor.fuse_vector_gaussian(truth, covariance_vector)

    np.testing.assert_allclose(
        transported.mean_transform.rotation,
        truth.rotation,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        transported.mean_transform.translation, truth.translation, atol=1e-10
    )
    assert np.all(np.linalg.eigvalsh(transported.covariance_local) > 0.0)
    assert np.trace(fused.covariance_local) < np.trace(transported.covariance_local)


def test_gaussian_prior_supplies_only_the_missing_direction() -> None:
    source, target, _ = _line_case()
    factor = estimate_observable_sim3_factor(source, target)
    null_direction = factor.nullspace_basis[:, 0]
    prior_covariance = 0.04 * np.eye(7)
    prior_mean = 0.3 * null_direction

    posterior = factor.fuse_local_gaussian(prior_mean, prior_covariance)

    np.testing.assert_allclose(
        null_direction @ posterior.mean_local,
        0.3,
        atol=1e-10,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        null_direction @ posterior.covariance_local @ null_direction,
        0.04,
        atol=1e-10,
        rtol=1e-10,
    )
    observable_variance = np.diag(
        factor.observable_basis.T
        @ posterior.covariance_local
        @ factor.observable_basis
    )
    assert np.all(observable_variance < 0.04)


def test_generic_three_dimensional_overlap_is_full_rank() -> None:
    generator = np.random.default_rng(10)
    source = generator.normal(size=(200, 3))
    truth = Sim3(
        scale=0.9,
        rotation=so3_exp(np.array([0.1, -0.2, 0.05])),
        translation=np.array([0.4, -0.2, 0.1]),
    )
    target = truth.transform_points(source) + generator.normal(scale=0.005, size=source.shape)

    factor = estimate_observable_sim3_factor(source, target)

    assert factor.rank == 7
    assert factor.nullspace_basis.shape == (7, 0)
    np.testing.assert_allclose(
        factor.chart.linearization.transform_points(source),
        truth.transform_points(source),
        atol=3e-3,
    )


def test_cluster_robust_factor_stays_within_observable_subspace() -> None:
    generator = np.random.default_rng(22)
    points_per_cluster = 10
    clusters = np.repeat(np.arange(20), points_per_cluster)
    source = generator.normal(size=(clusters.size, 3))
    truth = Sim3(
        scale=1.05,
        rotation=so3_exp(np.array([0.08, -0.04, 0.03])),
        translation=np.array([0.2, -0.1, 0.05]),
    )
    shared_error = generator.normal(scale=0.01, size=(20, 3))
    target = truth.transform_points(source) + shared_error[clusters]
    target += generator.normal(scale=0.001, size=source.shape)

    factor = estimate_observable_sim3_factor(
        source,
        target,
        covariance_cluster_ids=clusters,
    )

    assert factor.rank == 7
    assert factor.covariance_method == CLUSTER_OBSERVABLE_INFORMATION
    assert factor.num_covariance_clusters == 20
    assert np.all(np.linalg.eigvalsh(factor.observable_information) > 0.0)


def test_controlled_dlo_mechanism_is_decisive() -> None:
    result = run_observable_gauge_study(ObservableGaugeStudyConfig(trials=100))
    observable = result["methods"]["observable_subspace_factor"]
    completion = result["methods"]["isotropic_nullspace_completion_control"]

    assert result["geometry"]["rank_counts"] == {"6": 100}
    assert observable["support_rmse_improvement_fraction"] > 0.95
    assert observable["probe_rmse_improvement_fraction"] > 0.70
    assert observable["harmful_probe_fraction"] == 0.0
    assert completion["empirical_90pct_coverage"] < 0.10
