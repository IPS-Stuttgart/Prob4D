"""Controlled Sim(3)-linearized study; no provider, data or target access.

Run from the repository root with PYTHONPATH=src. The dense solver is an
independent reference for a small controlled study, not a production backend.
The protocol is hashed before drawing any outcomes. Output is a new directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path

import numpy as np

from prob4d.posterior_preserving_compression import (
    compress_shared_factor_for_posterior,
)


class DenseReference:
    def __init__(self, covariance: np.ndarray) -> None:
        self.covariance = covariance
        self.dimension = covariance.shape[0]
        self.observation_count = self.dimension // 3

    def solve(self, value: object) -> np.ndarray:
        raw = np.asarray(value)
        return np.linalg.solve(
            self.covariance, raw.reshape(self.dimension, -1)
        ).reshape(raw.shape)


def skew(point: np.ndarray) -> np.ndarray:
    x, y, z = point
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def physical_map(point: np.ndarray) -> np.ndarray:
    # Three translation, three rotation, and three bending modes, in metres.
    return np.column_stack([
        0.008 * np.eye(3), -0.025 * skew(point),
        0.008 * (1.0 + (point[0] / 0.25) ** 2) * np.eye(3),
    ])


def make_design(protocol: dict, windows: int, seed: int):
    rng = np.random.default_rng(seed)
    n = protocol["points_per_window"]
    points = rng.normal(size=(n, 3)) * np.array([0.16, 0.06, 0.04])
    # Every window observes the same material points. Shared anchor and
    # cumulative relative-gauge errors produce a correlated gauge chain.
    local_j = np.concatenate([
        np.column_stack([point, -skew(point), np.eye(3)]) for point in points
    ], axis=0)
    increments = np.diag(protocol["gauge_increment_standard_deviations"])
    gauge_root = np.kron(np.tril(np.ones((windows, windows))), increments)
    u = np.kron(np.eye(windows), local_j) @ gauge_root
    f = np.tile(np.concatenate([physical_map(point) for point in points]), (windows, 1))
    query = physical_map(np.array([0.25, 0.04, -0.02]))
    noise_std = protocol["conditional_standard_deviation_m"]
    d = noise_std**2 * np.eye(3 * n * windows)
    return u, f, query, d


def posterior(prior, cross, covariance):
    gain = np.linalg.solve(covariance, cross.T).T
    result = prior - gain @ cross.T
    return gain, 0.5 * (result + result.T)


def counterexample():
    d = np.diag([1.0, 1e-4, 1.0])
    f = np.array([[1.0], [1.0], [0.0]])
    u = np.array([[1000.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    a, prior, cross = d + f @ f.T, np.ones((1, 1)), f.T
    s = a + u @ u.T
    result = compress_shared_factor_for_posterior(
        u.reshape(1, 3, 2), prior_query_covariance=prior,
        query_observation_cross_covariance=cross,
        innovation_operator=DenseReference(s), maximum_rank=1,
    )
    _, full = posterior(prior, cross, s)
    _, naive = posterior(prior, cross, a + u[:, :1] @ u[:, :1].T)
    reduced = result.compressed_factor_m.reshape(3, -1)
    _, exact = posterior(prior, cross, a + reduced @ reduced.T)
    return {
        "shared_trace_fraction_retained_by_naive": float(1e6 / (1e6 + 1.0)),
        "registered_marginal_projection": [1.0, 0.0, 0.0],
        "marginal_projection_variance_full_and_naive": 1e6,
        "full_posterior_variance": float(full[0, 0]),
        "naive_posterior_variance": float(naive[0, 0]),
        "posterior_preserving_variance": float(exact[0, 0]),
        "naive_variance_understatement_factor": float(full[0, 0] / naive[0, 0]),
        "compression": result.summary(),
    }


def evaluate_design(protocol, windows, seed):
    u, f, query, d = make_design(protocol, windows, seed)
    dimension, rank = u.shape
    qdim = query.shape[0]
    prior, cross = query @ query.T, query @ f.T
    a = d + f @ f.T
    s = a + u @ u.T
    reference_gain, reference_covariance = posterior(prior, cross, s)
    compression = compress_shared_factor_for_posterior(
        u.reshape(-1, 3, rank), prior_query_covariance=prior,
        query_observation_cross_covariance=cross,
        innovation_operator=DenseReference(s), maximum_rank=qdim,
        rank_relative_tolerance=protocol["rank_relative_tolerance"],
        parity_relative_tolerance=protocol["parity_relative_tolerance"],
    )
    v = np.linalg.svd(u, full_matrices=False)[2].T[:, :qdim]
    arms = {
        "full": u,
        "posterior-preserving": compression.compressed_factor_m.reshape(dimension, -1),
        "equal-rank-covariance-pca": u @ v,
        "conditional-only": np.empty((dimension, 0)),
        "cached-full-query-message": None,
    }
    rng = np.random.default_rng(protocol["draw_seed_offset"] + 1000 * windows + seed)
    samples = protocol["draws_per_design"]
    state = rng.normal(size=(f.shape[1], samples))
    innovation = (
        f @ state + u @ rng.normal(size=(rank, samples))
        + protocol["conditional_standard_deviation_m"]
        * rng.normal(size=(dimension, samples))
    )
    truth = query @ state
    results = []
    for name, factor in arms.items():
        if factor is None:
            gain, covariance = reference_gain, reference_covariance
            stored_rank, stored_bytes = None, 0
        else:
            gain, covariance = posterior(prior, cross, a + factor @ factor.T)
            stored_rank, stored_bytes = factor.shape[1], factor.nbytes
        root = np.linalg.cholesky(covariance)
        error = truth - gain @ innovation
        white_error = np.linalg.solve(root, error)
        nees = np.sum(white_error**2, axis=0)
        logdet = 2 * np.log(np.diag(root)).sum()
        nll = 0.5 * (nees + logdet + qdim * math.log(2 * math.pi))
        gain_error = np.linalg.norm(gain - reference_gain) / np.linalg.norm(reference_gain)
        covariance_error = (
            np.linalg.norm(covariance - reference_covariance)
            / np.linalg.norm(reference_covariance)
        )
        results.append({
            "method": name, "retained_rank": stored_rank,
            "shared_factor_payload_bytes": stored_bytes,
            "query_message_payload_bytes": gain.nbytes + covariance.nbytes,
            "relative_gain_error": float(gain_error),
            "relative_covariance_error": float(covariance_error),
            "mean_query_nll_nats": float(np.mean(nll)),
            "mean_normalized_nees": float(np.mean(nees) / qdim),
            "coverage_90": float(np.mean(nees <= protocol["chi_square_3_90_percent"])),
            "query_rmse_m": float(np.sqrt(np.mean(np.sum(error**2, axis=0)))),
        })
    return {
        "windows": windows, "geometry_seed": seed, "observation_dimension": dimension,
        "original_shared_rank": rank, "draw_count": samples,
        "compression": compression.summary(), "methods": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path(
        "protocols/posterior-compression-controlled-v1.json"
    ))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    args = parser.parse_args()
    protocol_bytes = args.protocol.read_bytes()
    protocol = json.loads(protocol_bytes)
    if protocol["schema"] != "prob4d.posterior-compression-controlled.v1":
        raise ValueError("unsupported protocol")
    if args.output_dir.exists():
        raise FileExistsError("output directory already exists; never overwrite a run")
    args.output_dir.mkdir(parents=True)
    # Retain exactly the protocol bytes BEFORE generating any study outcomes.
    (args.output_dir / "protocol.json").write_bytes(protocol_bytes)
    root = Path(__file__).resolve().parents[2]
    source_paths = [
        Path("src/prob4d/posterior_preserving_compression.py"),
        Path("scripts/science/run_posterior_compression_study.py"),
    ]
    manifest = {
        "source_revision": args.source_revision,
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "source_sha256": {
            str(path): hashlib.sha256((root / path).read_bytes()).hexdigest()
            for path in source_paths
        },
        "python": platform.python_version(), "numpy": np.__version__,
        "real_provider_evidence": False, "sealed_data_accessed": False,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = {
        "evidence_class": protocol["evidence_class"],
        "counterexample": counterexample(),
        "designs": [
            evaluate_design(protocol, windows, seed)
            for windows in protocol["windows"] for seed in protocol["geometry_seeds"]
        ],
    }
    report_bytes = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    (args.output_dir / "result.json").write_bytes(report_bytes)
    manifest["result_sha256"] = hashlib.sha256(report_bytes).hexdigest()
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(args.output_dir), **manifest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
