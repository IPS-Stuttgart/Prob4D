"""Reproducible, constructed mechanism study; not a real-provider experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .conditional_gauge_design import (
    ConditionalGaugeSession,
    CorrelatedGaugeDesign,
    GaussianGaugeBelief,
    select_query_window,
)

PROTOCOL: dict[str, Any] = {
    "schema": "prob4d.conditional-query-design-study-v1",
    "classification": "constructed-linear-Gaussian-mechanism-study",
    "seed": 20260830,
    "episodes": 10000,
    "bootstrap_seed": 8302026,
    "bootstrap_replicates": 2000,
    "prior_standard_deviation": 0.05,
    "history_noise_standard_deviation": 0.01,
    "repeat_noise_correlation": 0.999,
    "complement_noise_variance": 0.00025,
    "global_only_noise_variance": 0.000001,
    "cloud_scale_m": 0.1,
    "query_point_in_cloud_radii": [0.0, 0.02, 0.01],
    "new_window_budget": 1,
    "candidate_cost": 1.0,
    "ellipsoid_90_chi_square_df3": 6.251388631170325,
    "independent_unit": "synthetic episode, not coordinate or point",
    "targets_opened": False,
    "provider_executed": False,
}


def make_design(correlation: float = 0.999) -> CorrelatedGaugeDesign:
    """Line-like rank-six history, repeat, complementary support, and twist-only."""
    history = np.eye(7)[[0, 2, 3, 4, 5, 6]]
    complement = np.eye(7)[[0, 1, 3, 4, 5, 6]]
    global_only = np.eye(7)[[1]]
    design = np.vstack((history, history, complement, global_only))
    noise = np.diag([0.0001] * 12 + [0.00025] * 6 + [0.000001])
    noise[:6, 6:12] = correlation * 0.0001 * np.eye(6)
    noise[6:12, :6] = correlation * 0.0001 * np.eye(6)
    return CorrelatedGaugeDesign(
        "centroid-normalized-local-sim3-study-v1",
        "known-generative-noise-v1",
        ("history", "near_repeat", "complement", "global_only"),
        (6, 6, 6, 1),
        design,
        noise,
    )


def query_jacobian() -> np.ndarray:
    point = np.array(PROTOCOL["query_point_in_cloud_radii"], dtype=float)
    x, y, z = point
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return 0.1 * np.column_stack((point, -skew, np.eye(3)))


def prior_belief(model: CorrelatedGaugeDesign) -> GaussianGaugeBelief:
    return GaussianGaugeBelief(model.chart_id, np.zeros(7), 0.0025 * np.eye(7))


def _history_session(model: CorrelatedGaugeDesign) -> ConditionalGaugeSession:
    session = ConditionalGaugeSession(model, prior_belief(model))
    session.assimilate("history", np.zeros(6))
    return session


def _vectorized_prediction(
    model: CorrelatedGaugeDesign, observations: np.ndarray, selected: str | None
) -> tuple[np.ndarray, np.ndarray]:
    """Apply exactly the session's conditional factor to all independent episodes."""
    session = _history_session(model)
    old = session.belief
    h = model.design_matrix[:6]
    prior = prior_belief(model)
    k0 = np.linalg.solve(
        model.noise_covariance[:6, :6] + h @ prior.covariance @ h.T,
        h @ prior.covariance,
    ).T
    means = observations[:, :6] @ k0.T
    if selected is not None:
        factor = model.conditional_factor(("history",), selected)
        rows = list(model.rows((selected,)))
        values = observations[:, rows] - observations[:, :6] @ factor.regression.T
        a = factor.whitened_design
        k = np.linalg.solve(np.eye(a.shape[0]) + a @ old.covariance @ a.T, a @ old.covariance).T
        means += (values @ factor.whitener.T - means @ a.T) @ k.T
        session.assimilate(selected, np.zeros(len(rows)))
    return means, session.belief.covariance


def _paired_bootstrap_interval(
    differences: np.ndarray, replicates: int, seed: int
) -> list[float]:
    rng = np.random.default_rng(seed)
    means = []
    for start in range(0, replicates, 100):
        size = min(100, replicates - start)
        indices = rng.integers(0, differences.size, size=(size, differences.size))
        means.extend(np.mean(differences[indices], axis=1).tolist())
    return np.quantile(means, [0.025, 0.975]).tolist()


def _reference_channel_control() -> dict[str, float]:
    design = np.vstack((np.eye(7)[0], np.zeros(7)))
    model = CorrelatedGaugeDesign(
        "scalar-reference-embedded-in-gauge", "known-reference-noise",
        ("signal", "noise_reference"), (1, 1), design,
        np.array([[1.0, 1.0], [1.0, 1.1]]),
    )
    session = ConditionalGaugeSession(
        model, GaussianGaugeBelief(model.chart_id, np.zeros(7), np.eye(7))
    )
    query = np.eye(7)[[0]]
    standalone = session.preview_query("noise_reference", query)
    session.assimilate("signal", np.zeros(1))
    conditional = session.preview_query("noise_reference", query)
    return {
        "standalone_reference_gain": standalone.variance_reduction,
        "conditional_reference_gain": conditional.variance_reduction,
        "variance_before_reference": conditional.prior_metric_variance,
        "variance_after_reference": conditional.posterior_metric_variance,
    }


def run_study(*, episodes: int = 10000, bootstrap_replicates: int = 2000) -> dict[str, Any]:
    if episodes < 100 or bootstrap_replicates < 20:
        raise ValueError("at least 100 episodes and 20 bootstrap replicates are required")
    protocol = dict(PROTOCOL, episodes=episodes, bootstrap_replicates=bootstrap_replicates)
    model = make_design()
    independent_noise = np.diag(np.diag(model.noise_covariance))
    independence_model = CorrelatedGaugeDesign(
        model.chart_id, "deliberately-invalid-independent-noise-control",
        model.window_ids, model.window_sizes, model.design_matrix, independent_noise,
    )
    query = query_jacobian()
    candidates = ("near_repeat", "complement", "global_only")
    correct = _history_session(model)
    independent = _history_session(independence_model)
    correct_utilities = tuple(correct.preview_query(w, query) for w in candidates)
    marginal_utilities = tuple(independent.preview_query(w, query) for w in candidates)
    global_utilities = tuple(correct.preview_query(w, np.eye(7)) for w in candidates)
    query_choice = select_query_window(correct_utilities)
    marginal_choice = select_query_window(marginal_utilities)
    global_choice = select_query_window(global_utilities)
    # Selection is complete before the first observation or latent episode is drawn.
    rng = np.random.default_rng(PROTOCOL["seed"])
    truth = 0.05 * rng.normal(size=(episodes, 7))
    noise = rng.normal(size=(episodes, 19)) @ np.linalg.cholesky(model.noise_covariance).T
    observations = truth @ model.design_matrix.T + noise
    arms: dict[str, tuple[CorrelatedGaugeDesign, str | None]] = {
        "history_only": (model, None),
        "marginal_query_selection_independent_update": (independence_model, marginal_choice),
        "marginal_query_selection_correct_update": (model, marginal_choice),
        "global_variance_selection_correct_update": (model, global_choice),
        "conditional_query_selection_correct_update": (model, query_choice),
    }
    errors: dict[str, np.ndarray] = {}
    covariances: dict[str, np.ndarray] = {}
    parity_maximum = 0.0
    for name, (assumed_model, selected) in arms.items():
        means, covariance = _vectorized_prediction(assumed_model, observations, selected)
        selected_ids = ("history",) if selected is None else ("history", selected)
        rows = list(assumed_model.rows(selected_ids))
        h = assumed_model.design_matrix[rows]
        r = assumed_model.noise_covariance[np.ix_(rows, rows)]
        p = prior_belief(assumed_model).covariance
        batch_gain = np.linalg.solve(r + h @ p @ h.T, h @ p).T
        batch_means = observations[:, rows] @ batch_gain.T
        batch_covariance = np.linalg.inv(np.linalg.inv(p) + h.T @ np.linalg.solve(r, h))
        parity_maximum = max(parity_maximum, float(np.max(np.abs(means - batch_means))),
                             float(np.max(np.abs(covariance - batch_covariance))))
        errors[name] = (means - truth) @ query.T
        covariances[name] = query @ covariance @ query.T
    baseline_loss = np.sum(errors["history_only"] ** 2, axis=1)
    metrics: dict[str, Any] = {}
    for name, error in errors.items():
        covariance = covariances[name]
        loss = np.sum(error**2, axis=1)
        nees = np.sum(error * np.linalg.solve(covariance, error.T).T, axis=1)
        sign, logdet = np.linalg.slogdet(covariance)
        if sign <= 0:
            raise ValueError("query covariance must be positive definite for scoring")
        improvement_mm2 = 1e6 * (baseline_loss - loss)
        interval = _paired_bootstrap_interval(improvement_mm2, bootstrap_replicates,
                                              PROTOCOL["bootstrap_seed"])
        metrics[name] = {
            "selected_window": arms[name][1],
            "query_euclidean_rmse_mm": float(1000 * np.sqrt(np.mean(loss))),
            "query_expected_rmse_mm": float(1000 * np.sqrt(np.trace(covariance))),
            "normalized_query_nees": float(np.mean(nees) / 3),
            "query_ellipsoid_90_coverage": float(np.mean(nees <= PROTOCOL["ellipsoid_90_chi_square_df3"])),
            "gaussian_query_nll_nats": float(0.5 * (3 * np.log(2 * np.pi) + logdet + np.mean(nees))),
            "harmful_episode_fraction_vs_history": float(np.mean(loss > baseline_loss + 1e-18)),
            "paired_mean_squared_query_error_improvement_mm2": float(np.mean(improvement_mm2)),
            "paired_95_percentile_bootstrap_interval_mm2": interval,
        }
    correlation_sweep = []
    for correlation in (0.0, 0.5, 0.9, 0.999, 1.0):
        sweep_session = _history_session(make_design(correlation))
        utilities = tuple(sweep_session.preview_query(w, query) for w in candidates)
        correlation_sweep.append({
            "correlation": correlation,
            "selected_window": select_query_window(utilities),
            "query_variance_reduction_mm2": {u.candidate_id: 1e6 * u.variance_reduction for u in utilities},
        })
    return {
        "schema": "prob4d.conditional-query-design-result-v1",
        "protocol": protocol,
        "protocol_sha256": hashlib.sha256(json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "query_variance_reduction_mm2": {
            "correct_conditional": {u.candidate_id: 1e6 * u.variance_reduction for u in correct_utilities},
            "incorrect_marginal": {u.candidate_id: 1e6 * u.variance_reduction for u in marginal_utilities},
        },
        "arms": metrics,
        "correlation_sweep": correlation_sweep,
        "noise_reference_non_submodularity_control": _reference_channel_control(),
        "maximum_kernel_vs_independent_dense_reference_error": parity_maximum,
        "boundaries": [
            "Known, correctly specified Gaussian noise except the explicitly invalid control.",
            "Constructed local-linear Sim(3) chart, not nonlinear perception or physical simulation.",
            "Independent synthetic episodes, not real objects, frames, or target evaluations.",
            "Utility guarantees expected squared-query-loss reduction only under the assumed model.",
            "Nonzero harmful-episode fractions are retained; no per-update safety claim.",
            "No PointWorld, BayesianPhysTwin, or Causal4D runtime or empirical benefit is established.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args()
    result = run_study(episodes=args.episodes, bootstrap_replicates=args.bootstrap_replicates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
