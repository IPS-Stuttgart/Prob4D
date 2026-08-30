#!/usr/bin/env python3
"""Verify evidence hashes and analytic controls without importing Prob4D."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    evidence, source = args.evidence_dir.resolve(), args.source_root.resolve()
    manifest = json.loads((evidence / "manifest.json").read_text())
    result = json.loads((evidence / "result.json").read_text())
    verified_files = 0
    for name, expected in manifest["source_files"].items():
        path = (source / name).resolve()
        if not path.is_relative_to(source):
            raise ValueError("nonlocal source path")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"source digest mismatch: {name}")
        verified_files += 1
    for name, expected in manifest["output_files"].items():
        path = (evidence / name).resolve()
        if not path.is_relative_to(evidence):
            raise ValueError("nonlocal output path")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"evidence digest mismatch: {name}")
        verified_files += 1
    control = result["analytic_control"]
    radius = control["radius_mm"] / 1000
    sigma = control["phase_stddev_rad"]
    threshold = control["violation_below_mm"] / 1000
    expected_mean_mm = radius * math.exp(-sigma**2 / 2) * 1000
    expected_std_mm = radius * (-math.expm1(-sigma**2)) / math.sqrt(2) * 1000
    boundary = math.acos(threshold / radius)
    # Independent lifted-interval formula, evaluated with the standard library.
    normal_cdf = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
    expected_probability = math.fsum(
        normal_cdf((2 * math.pi - boundary + 2 * k * math.pi) / sigma)
        - normal_cdf((boundary + 2 * k * math.pi) / sigma)
        for k in range(-8, 9)
    )
    checks = {
        "radial_mean_mm_error": abs(control["exact_mean_mm"] - expected_mean_mm),
        "radial_stddev_mm_error": abs(control["exact_stddev_mm"] - expected_std_mm),
        "event_probability_error": abs(control["exact_violation_probability"] - expected_probability),
        "repeated_constraint_probability_error": abs(result["shared_phase_controls"][0]["exact_joint_probability"] - 0.1),
        "disjoint_constraint_probability_error": abs(result["shared_phase_controls"][1]["exact_joint_probability"] - 0.5),
    }
    if max(checks.values()) > 1e-12:
        raise ValueError(f"analytic verification failed: {checks}")
    if any(result["information_boundary"].values()):
        raise ValueError("unexpected real-data or pipeline access in controlled evidence")
    lo, hi = control["monte_carlo"]["wilson_95_interval"]
    report = {
        "status": "passed",
        "prob4d_imported": False,
        "files_hash_verified": verified_files,
        "analytic_checks": checks,
        "analytic_risk_inside_observed_monte_carlo_95_interval": lo <= expected_probability <= hi,
        "statistical_note": "Monte Carlo agreement is a check on this controlled generator, not independent real-world calibration.",
        "upstream_repository_ci_run": False,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
