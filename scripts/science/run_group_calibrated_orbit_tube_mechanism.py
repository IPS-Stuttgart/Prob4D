#!/usr/bin/env python3
"""Controlled study for complete-group versus pooled-frame orbit calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "prob4d.group-calibrated-orbit-tube-mechanism.v1"


def _finite_sample_rank(count: int, miscoverage: float) -> int:
    return math.ceil((count + 1) * (1.0 - miscoverage))


def _summarize_arm(
    *,
    radii: np.ndarray,
    test_scores: np.ndarray,
    exact_advantages: np.ndarray,
) -> dict[str, float | int]:
    covered = test_scores <= radii
    accepted = radii < exact_advantages
    harmful = test_scores > exact_advantages
    harmful_accepted = accepted & harmful
    accepted_count = int(np.count_nonzero(accepted))
    harmful_count = int(np.count_nonzero(harmful_accepted))
    return {
        "trials": int(radii.size),
        "mean_radius": float(np.mean(radii)),
        "group_coverage": float(np.mean(covered)),
        "acceptance_fraction": float(np.mean(accepted)),
        "harmful_accepted_fraction": float(np.mean(harmful_accepted)),
        "harmful_fraction_among_accepted": (
            float(harmful_count / accepted_count) if accepted_count else 0.0
        ),
        "accepted": accepted_count,
        "harmful_accepted": harmful_count,
    }


def run_study(protocol: dict[str, Any]) -> dict[str, Any]:
    settings = protocol["settings"]
    seed = int(settings["seed"])
    trials = int(settings["trials"])
    calibration_groups = int(settings["calibration_groups"])
    frames_per_group = int(settings["frames_per_group"])
    miscoverage = float(settings["miscoverage"])
    chunk_size = int(settings["chunk_size"])

    if trials <= 0 or calibration_groups <= 0 or frames_per_group <= 1:
        raise ValueError("trial, group, and frame counts must be positive")
    if not 0.0 < miscoverage < 1.0:
        raise ValueError("miscoverage must lie strictly between zero and one")

    group_rank = _finite_sample_rank(calibration_groups, miscoverage)
    if group_rank > calibration_groups:
        raise ValueError("protocol requests an unsupported finite group radius")
    frame_count = calibration_groups * frames_per_group
    frame_rank = _finite_sample_rank(frame_count, miscoverage)
    if frame_rank > frame_count:
        raise ValueError("protocol requests an unsupported finite frame radius")

    rng = np.random.default_rng(seed)
    group_radii: list[np.ndarray] = []
    pooled_radii: list[np.ndarray] = []
    maximum_radii: list[np.ndarray] = []
    zero_radii: list[np.ndarray] = []
    test_scores_parts: list[np.ndarray] = []
    advantage_parts: list[np.ndarray] = []

    completed = 0
    while completed < trials:
        batch = min(chunk_size, trials - completed)
        total_groups = calibration_groups + 1

        # Each exchangeable trajectory has one short-lived maximum departure
        # from its estimated orbit. The remaining frames are correlated, small
        # departures. This makes a frame-wise quantile a deliberately invalid
        # substitute for complete-trajectory coverage.
        group_peaks = (
            float(settings["peak_scale"])
            * rng.beta(
                float(settings["peak_beta_a"]),
                float(settings["peak_beta_b"]),
                size=(batch, total_groups),
            )
        )
        residuals = group_peaks[:, :, None] * rng.uniform(
            float(settings["background_fraction_min"]),
            float(settings["background_fraction_max"]),
            size=(batch, total_groups, frames_per_group),
        )
        peak_indices = rng.integers(
            0,
            frames_per_group,
            size=(batch, total_groups),
        )
        batch_indices = np.arange(batch)[:, None]
        group_indices = np.arange(total_groups)[None, :]
        residuals[batch_indices, group_indices, peak_indices] = group_peaks

        calibration_residuals = residuals[:, :calibration_groups, :]
        calibration_group_scores = np.max(calibration_residuals, axis=2)
        test_group_scores = np.max(residuals[:, -1, :], axis=1)

        group_radius = np.partition(
            calibration_group_scores,
            group_rank - 1,
            axis=1,
        )[:, group_rank - 1]
        pooled = calibration_residuals.reshape(batch, frame_count)
        pooled_radius = np.partition(
            pooled,
            frame_rank - 1,
            axis=1,
        )[:, frame_rank - 1]
        maximum_radius = np.max(calibration_group_scores, axis=1)

        exact_advantage = rng.uniform(
            float(settings["advantage_min"]),
            float(settings["advantage_max"]),
            size=batch,
        )

        group_radii.append(group_radius)
        pooled_radii.append(pooled_radius)
        maximum_radii.append(maximum_radius)
        zero_radii.append(np.zeros(batch, dtype=float))
        test_scores_parts.append(test_group_scores)
        advantage_parts.append(exact_advantage)
        completed += batch

    test_scores = np.concatenate(test_scores_parts)
    advantages = np.concatenate(advantage_parts)
    arms = {
        "complete_group_split_conformal": _summarize_arm(
            radii=np.concatenate(group_radii),
            test_scores=test_scores,
            exact_advantages=advantages,
        ),
        "pooled_frames_as_independent": _summarize_arm(
            radii=np.concatenate(pooled_radii),
            test_scores=test_scores,
            exact_advantages=advantages,
        ),
        "largest_calibration_group": _summarize_arm(
            radii=np.concatenate(maximum_radii),
            test_scores=test_scores,
            exact_advantages=advantages,
        ),
        "exact_orbit_without_tube": _summarize_arm(
            radii=np.concatenate(zero_radii),
            test_scores=test_scores,
            exact_advantages=advantages,
        ),
    }

    group_arm = arms["complete_group_split_conformal"]
    coverage_failure = 1.0 - float(group_arm["group_coverage"])
    harmful_acceptance = float(group_arm["harmful_accepted_fraction"])
    if harmful_acceptance > coverage_failure + 1e-12:
        raise AssertionError("harmful group acceptance must imply tube noncoverage")

    result: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": 1,
        "evidence_class": "constructed-exchangeable-group-mechanism",
        "protocol_id": protocol["protocol_id"],
        "settings": settings,
        "finite_sample": {
            "group_quantile_rank": group_rank,
            "group_coverage_lower_bound": group_rank
            / (calibration_groups + 1.0),
            "minimum_miscoverage_for_finite_radius": 1.0
            / (calibration_groups + 1.0),
            "pooled_frame_quantile_rank": frame_rank,
            "pooled_frame_claim": (
                "random-frame coverage only; not simultaneous future-group coverage"
            ),
        },
        "arms": arms,
        "algebraic_checks": {
            "harmful_group_accept_implies_noncoverage": True,
            "group_harmful_accepted_fraction_no_larger_than_noncoverage": True,
        },
        "claim_boundary": [
            "This is a constructed mechanism study, not public real-data evidence.",
            "The exchangeable unit is one complete simulated trajectory.",
            "The exact orbit estimator, metric, and Lipschitz constants are assumed.",
            "The study does not establish conditional subgroup coverage.",
            "The study does not establish provider competence or deployment safety.",
        ],
    }
    canonical = json.dumps(
        result,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    result["result_id"] = hashlib.sha256(canonical).hexdigest()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("protocols/group-calibrated-orbit-tube-mechanism-v1.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    result = run_study(protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
