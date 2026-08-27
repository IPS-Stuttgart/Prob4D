"""Controlled DLO-like study for observability-aware Sim(3) gauge factors.

The synthetic overlap is a one-dimensional centerline.  It constrains six of the
seven similarity-gauge directions and leaves twist around the transformed line
unobservable.  Complete object/session trials are the Monte Carlo units.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .observable_gauge import GaugeGaussianPosterior, estimate_observable_sim3_factor
from .sim3 import Sim3, so3_exp

_SCHEMA = "prob4d.observable-gauge-study"
_SCHEMA_VERSION = 1
_CHI2_7_90 = 12.017036623780532


@dataclass(frozen=True)
class ObservableGaugeStudyConfig:
    seed: int = 20_260_827
    trials: int = 1_000
    points_per_line: int = 48
    observation_sigma_m: float = 0.01
    line_half_extent_m: float = 1.0
    probe_radius_m: float = 0.25
    truth_scale: float = 1.1
    truth_twist_rad: float = 0.6
    rank_threshold: float = 1e-8
    nullspace_precision_ratio: float = 1.0
    prior_log_scale_std: float = 0.06
    prior_rotation_std_rad: float = 0.14
    prior_centroid_translation_std: float = 0.10

    def validate(self) -> None:
        for name in ("seed", "trials", "points_per_line"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")
        if self.trials < 20:
            raise ValueError("trials must be at least 20")
        if self.points_per_line < 8:
            raise ValueError("points_per_line must be at least eight")
        positive = (
            "observation_sigma_m",
            "line_half_extent_m",
            "probe_radius_m",
            "truth_scale",
            "rank_threshold",
            "nullspace_precision_ratio",
            "prior_log_scale_std",
            "prior_rotation_std_rad",
            "prior_centroid_translation_std",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.rank_threshold >= 1.0:
            raise ValueError("rank_threshold must be smaller than one")
        if not np.isfinite(self.truth_twist_rad):
            raise ValueError("truth_twist_rad must be finite")


@dataclass(frozen=True)
class _MethodTrial:
    support_rmse_m: float
    probe_rmse_m: float
    nll: float
    nees: float
    mean_std_local: float


def _source_line(config: ObservableGaugeStudyConfig) -> np.ndarray:
    coordinate = np.linspace(
        -config.line_half_extent_m,
        config.line_half_extent_m,
        config.points_per_line,
    )
    return np.column_stack((coordinate, np.zeros_like(coordinate), np.zeros_like(coordinate)))


def _probe_tube(config: ObservableGaugeStudyConfig) -> np.ndarray:
    coordinate = np.linspace(
        -config.line_half_extent_m,
        config.line_half_extent_m,
        max(8, config.points_per_line // 2),
    )
    angles = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    points = []
    for position in coordinate:
        for angle in angles:
            points.append(
                [
                    position,
                    config.probe_radius_m * np.cos(angle),
                    config.probe_radius_m * np.sin(angle),
                ]
            )
    return np.asarray(points, dtype=np.float64)


def _truth_transform(config: ObservableGaugeStudyConfig) -> Sim3:
    base_rotation = so3_exp(np.array([0.0, 0.25, -0.15]))
    unobservable_twist = so3_exp(np.array([config.truth_twist_rad, 0.0, 0.0]))
    return Sim3(
        scale=config.truth_scale,
        rotation=base_rotation @ unobservable_twist,
        translation=np.array([0.20, -0.10, 0.05]),
    )


def _prior_covariance(config: ObservableGaugeStudyConfig) -> np.ndarray:
    standard_deviations = np.array(
        [
            config.prior_log_scale_std,
            config.prior_rotation_std_rad,
            config.prior_rotation_std_rad,
            config.prior_rotation_std_rad,
            config.prior_centroid_translation_std,
            config.prior_centroid_translation_std,
            config.prior_centroid_translation_std,
        ]
    )
    return np.diag(standard_deviations**2)


def _fuse_information(
    chart,
    prior_mean: np.ndarray,
    prior_covariance: np.ndarray,
    factor_information: np.ndarray,
) -> GaugeGaussianPosterior:
    prior_information = np.linalg.solve(prior_covariance, np.eye(7))
    posterior_information = prior_information + factor_information
    posterior_covariance = np.linalg.solve(posterior_information, np.eye(7))
    posterior_covariance = 0.5 * (posterior_covariance + posterior_covariance.T)
    posterior_mean = posterior_covariance @ (prior_information @ prior_mean)
    return GaugeGaussianPosterior(
        chart=chart,
        mean_local=posterior_mean,
        covariance_local=posterior_covariance,
    )


def _trial_metrics(
    posterior: GaugeGaussianPosterior,
    truth_local: np.ndarray,
    truth: Sim3,
    source: np.ndarray,
    probes: np.ndarray,
) -> _MethodTrial:
    mean_transform = posterior.mean_transform
    support_error = mean_transform.transform_points(source) - truth.transform_points(source)
    probe_error = mean_transform.transform_points(probes) - truth.transform_points(probes)
    delta = truth_local - posterior.mean_local
    precision_delta = np.linalg.solve(posterior.covariance_local, delta)
    nees = float(delta @ precision_delta)
    sign, logdet = np.linalg.slogdet(posterior.covariance_local)
    if sign <= 0.0:
        raise ValueError("posterior covariance must have positive determinant")
    nll = 0.5 * (7.0 * np.log(2.0 * np.pi) + logdet + nees)
    return _MethodTrial(
        support_rmse_m=float(np.sqrt(np.mean(support_error**2))),
        probe_rmse_m=float(np.sqrt(np.mean(probe_error**2))),
        nll=float(nll),
        nees=nees,
        mean_std_local=float(np.sqrt(np.trace(posterior.covariance_local) / 7.0)),
    )


def _aggregate(
    trials: list[_MethodTrial],
    fallback: list[_MethodTrial] | None = None,
) -> dict[str, Any]:
    support = np.asarray([trial.support_rmse_m for trial in trials])
    probe = np.asarray([trial.probe_rmse_m for trial in trials])
    nll = np.asarray([trial.nll for trial in trials])
    nees = np.asarray([trial.nees for trial in trials])
    width = np.asarray([trial.mean_std_local for trial in trials])
    result: dict[str, Any] = {
        "mean_support_rmse_mm": float(1_000.0 * np.mean(support)),
        "mean_probe_rmse_mm": float(1_000.0 * np.mean(probe)),
        "mean_gaussian_nll": float(np.mean(nll)),
        "normalized_nees": float(np.mean(nees) / 7.0),
        "empirical_90pct_coverage": float(np.mean(nees <= _CHI2_7_90)),
        "mean_local_standard_deviation": float(np.mean(width)),
    }
    if fallback is not None:
        fallback_support = np.asarray([trial.support_rmse_m for trial in fallback])
        fallback_probe = np.asarray([trial.probe_rmse_m for trial in fallback])
        result.update(
            {
                "support_rmse_improvement_fraction": float(
                    1.0 - np.mean(support) / np.mean(fallback_support)
                ),
                "probe_rmse_improvement_fraction": float(
                    1.0 - np.mean(probe) / np.mean(fallback_probe)
                ),
                "harmful_probe_fraction": float(np.mean(probe > fallback_probe)),
            }
        )
    return result


def run_observable_gauge_study(
    config: ObservableGaugeStudyConfig | None = None,
) -> dict[str, Any]:
    """Run the frozen controlled mechanism study."""

    if config is None:
        config = ObservableGaugeStudyConfig()
    config.validate()
    generator = np.random.default_rng(config.seed)
    source = _source_line(config)
    probes = _probe_tube(config)
    truth = _truth_transform(config)
    prior_covariance = _prior_covariance(config)
    fallback_trials: list[_MethodTrial] = []
    observable_trials: list[_MethodTrial] = []
    completion_trials: list[_MethodTrial] = []
    ranks: list[int] = []
    nullspace_truth_magnitudes: list[float] = []

    for _ in range(config.trials):
        target = truth.transform_points(source) + generator.normal(
            scale=config.observation_sigma_m,
            size=source.shape,
        )
        factor = estimate_observable_sim3_factor(
            source,
            target,
            rank_threshold=config.rank_threshold,
        )
        ranks.append(factor.rank)
        truth_local = factor.chart.to_local(truth)
        prior_mean = truth_local + generator.multivariate_normal(
            np.zeros(7),
            prior_covariance,
        )
        nullspace_truth_magnitudes.append(
            float(np.linalg.norm(factor.nullspace_basis.T @ truth_local))
        )
        fallback = GaugeGaussianPosterior(
            chart=factor.chart,
            mean_local=prior_mean,
            covariance_local=prior_covariance,
        )
        observable = factor.fuse_local_gaussian(prior_mean, prior_covariance)
        weakest_observable_precision = float(
            np.min(np.linalg.eigvalsh(factor.observable_information))
        )
        completed_information = factor.information_matrix.copy()
        if factor.nullspace_basis.shape[1]:
            completed_information += (
                config.nullspace_precision_ratio
                * weakest_observable_precision
                * factor.nullspace_basis
                @ factor.nullspace_basis.T
            )
        completion = _fuse_information(
            factor.chart,
            prior_mean,
            prior_covariance,
            completed_information,
        )
        fallback_trials.append(
            _trial_metrics(fallback, truth_local, truth, source, probes)
        )
        observable_trials.append(
            _trial_metrics(observable, truth_local, truth, source, probes)
        )
        completion_trials.append(
            _trial_metrics(completion, truth_local, truth, source, probes)
        )

    fallback_result = _aggregate(fallback_trials)
    observable_result = _aggregate(observable_trials, fallback_trials)
    completion_result = _aggregate(completion_trials, fallback_trials)
    rank_counts = {
        str(rank): int(np.count_nonzero(np.asarray(ranks) == rank))
        for rank in sorted(set(ranks))
    }
    criteria = {
        "all_trials_rank_six": rank_counts == {"6": config.trials},
        "observable_support_rmse_improves_at_least_50pct": (
            observable_result["support_rmse_improvement_fraction"] >= 0.50
        ),
        "observable_probe_rmse_improves": (
            observable_result["probe_rmse_improvement_fraction"] > 0.0
        ),
        "observable_nll_improves": (
            observable_result["mean_gaussian_nll"] < fallback_result["mean_gaussian_nll"]
        ),
        "observable_coverage_is_calibrated": (
            0.86 <= observable_result["empirical_90pct_coverage"] <= 0.94
        ),
        "completion_is_more_harmful_than_subspace": (
            completion_result["harmful_probe_fraction"]
            >= observable_result["harmful_probe_fraction"] + 0.25
        ),
        "completion_is_undercovered": (
            completion_result["empirical_90pct_coverage"] <= 0.50
        ),
    }
    return {
        "schema": _SCHEMA,
        "schema_version": _SCHEMA_VERSION,
        "claim_boundary": (
            "controlled DLO-like gauge-observability mechanism only; not real-provider "
            "competence, downstream physical benefit, or deployment calibration"
        ),
        "config": asdict(config),
        "geometry": {
            "expected_observable_rank": 6,
            "rank_counts": rank_counts,
            "mean_unobservable_truth_magnitude_rad": float(
                np.mean(nullspace_truth_magnitudes)
            ),
            "current_full_covariance_alignment_accepts_rank_deficient_geometry": False,
        },
        "methods": {
            "exact_physical_prior_fallback": fallback_result,
            "observable_subspace_factor": observable_result,
            "isotropic_nullspace_completion_control": completion_result,
        },
        "criteria": criteria,
        "decision": "pass" if all(criteria.values()) else "fail",
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m prob4d.observable_gauge_study",
        description="run the controlled DLO-like observable-gauge study",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--trials", type=int, default=ObservableGaugeStudyConfig.trials)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    config = ObservableGaugeStudyConfig(trials=arguments.trials)
    result = run_observable_gauge_study(config)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(rendered, end="")
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered, encoding="utf-8")
    return 0 if result["decision"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
