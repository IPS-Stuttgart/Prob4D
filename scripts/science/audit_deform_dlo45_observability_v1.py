"""Source-only observability audit for official DEFORM DLO4/DLO5 trajectories.

Only the public training split is opened. The script reports the intrinsic
centroid-normalized Sim(3) geometry spectrum for several fixed spatial and
temporal supports. It does not fit a provider, inspect evaluation trajectories,
or make a target-performance claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "prob4d.deform-dlo45-observability-source-audit"
SCHEMA_VERSION = 1
REQUEST_SCHEMA = "prob4d.deform-dlo45-observability-source-request"
EXPECTED_ROOT = Path("/mnt/seagate10tb/florianpfaff/datasets/deform/data_set")


class _RestrictedNumpyUnpickler(pickle.Unpickler):
    """Load only the NumPy reconstruction globals used by official trajectories."""

    _NUMPY_MULTIARRAY_MODULES = {
        "numpy.core.multiarray",
        "numpy._core.multiarray",
    }

    def find_class(self, module: str, name: str) -> Any:
        if module == "numpy" and name == "ndarray":
            return np.ndarray
        if module == "numpy" and name == "dtype":
            return np.dtype
        if module in self._NUMPY_MULTIARRAY_MODULES and name in {
            "_reconstruct",
            "scalar",
        }:
            return getattr(np.core.multiarray, name)
        raise pickle.UnpicklingError(f"forbidden pickle global: {module}.{name}")

    def persistent_load(self, pid: object) -> Any:
        raise pickle.UnpicklingError(f"persistent pickle ID is forbidden: {pid!r}")


def canonical_sha256(value: dict[str, Any]) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_request(path: Path) -> dict[str, Any]:
    request = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(request, dict):
        raise TypeError("request must be a JSON object")
    if request.get("schema") != REQUEST_SCHEMA or request.get("schema_version") != 1:
        raise ValueError("unsupported request schema")
    supplied = request.get("request_id")
    unhashed = dict(request)
    unhashed.pop("request_id", None)
    if supplied != canonical_sha256(unhashed):
        raise ValueError("request_id does not match canonical request contents")
    if request.get("stage") != "source-audit":
        raise ValueError("this script accepts only stage=source-audit")
    if Path(str(request.get("dataset_root"))) != EXPECTED_ROOT:
        raise ValueError(f"dataset_root must be exactly {EXPECTED_ROOT}")
    if request.get("dlo_types") != ["DLO4", "DLO5"]:
        raise ValueError("dlo_types must be exactly DLO4 and DLO5")
    boundary = request.get("information_boundary")
    expected_boundary = {
        "opened_split": "train",
        "evaluation_file_contents_opened": False,
        "provider_predictions_opened": False,
        "bayesian_phystwin_outcomes_opened": False,
        "causal4d_outcomes_opened": False,
    }
    if boundary != expected_boundary:
        raise ValueError("information boundary changed")
    return request


def load_trajectory(path: Path) -> np.ndarray:
    with path.open("rb") as stream:
        loaded = _RestrictedNumpyUnpickler(stream).load()
    array = np.asarray(loaded)
    if array.dtype.hasobject:
        raise ValueError(f"{path} contains an object-valued trajectory")
    array = np.asarray(array, dtype=np.float64).squeeze()
    if array.ndim != 3:
        raise ValueError(f"{path} has unsupported shape {array.shape}")
    if array.shape[-1] == 3:
        frames = array
    elif array.shape[1] == 3:
        frames = np.transpose(array, (0, 2, 1))
    else:
        raise ValueError(f"{path} has no identifiable xyz axis: {array.shape}")
    if frames.shape[0] < 2 or frames.shape[1] < 5:
        raise ValueError(f"{path} contains too few frames or vertices: {frames.shape}")
    if not np.all(np.isfinite(frames)):
        raise ValueError(f"{path} contains non-finite coordinates")
    return np.asarray(frames, dtype=np.float64)


def vertex_indices(count: int, mode: str) -> np.ndarray:
    if mode == "full":
        return np.arange(count, dtype=np.int64)
    if mode == "interior":
        if count < 7:
            raise ValueError("interior support needs at least seven vertices")
        return np.arange(1, count - 1, dtype=np.int64)
    if mode == "local5":
        start = (count - 5) // 2
        return np.arange(start, start + 5, dtype=np.int64)
    raise ValueError(f"unknown vertex mode {mode!r}")


def geometry_spectrum(points: np.ndarray) -> tuple[np.ndarray, float, float]:
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    rho = float(np.sqrt(np.mean(np.sum(centered**2, axis=1))))
    if not math.isfinite(rho) or rho <= np.finfo(np.float64).eps:
        raise ValueError("support has no spatial extent")
    information = np.zeros((7, 7), dtype=np.float64)
    identity = np.eye(3)
    for point in centered:
        skew = np.array(
            [
                [0.0, -point[2], point[1]],
                [point[2], 0.0, -point[0]],
                [-point[1], point[0], 0.0],
            ],
            dtype=np.float64,
        )
        jacobian = np.empty((3, 7), dtype=np.float64)
        jacobian[:, 0] = point
        jacobian[:, 1:4] = -skew
        jacobian[:, 4:7] = rho * identity
        information += jacobian.T @ jacobian
    eigenvalues = np.linalg.eigvalsh(0.5 * (information + information.T))[::-1]
    maximum = float(eigenvalues[0])
    if maximum <= 0.0:
        raise ValueError("support carries no gauge information")
    normalized = np.maximum(eigenvalues / maximum, 0.0)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    line_ratio = float(singular_values[-1] / max(singular_values[0], np.finfo(float).eps))
    return normalized, rho, line_ratio


def quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(np.min(array)),
        "q05": float(np.quantile(array, 0.05)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "q75": float(np.quantile(array, 0.75)),
        "q95": float(np.quantile(array, 0.95)),
        "maximum": float(np.max(array)),
    }


def run(request: dict[str, Any]) -> dict[str, Any]:
    root = EXPECTED_ROOT
    stride = int(request["frame_stride"])
    if stride < 1:
        raise ValueError("frame_stride must be positive")
    thresholds = [float(value) for value in request["rank_threshold_candidates"]]
    if not thresholds or any(not 0.0 < value < 1.0 for value in thresholds):
        raise ValueError("rank thresholds must lie in (0,1)")
    configs = request["support_configs"]
    if not isinstance(configs, list) or not configs:
        raise ValueError("support_configs must be a nonempty list")

    manifest: list[dict[str, Any]] = []
    spectra: dict[str, list[np.ndarray]] = defaultdict(list)
    radii: dict[str, list[float]] = defaultdict(list)
    line_ratios: dict[str, list[float]] = defaultdict(list)
    cases_by_object: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    trajectory_shapes: dict[str, list[list[int]]] = defaultdict(list)

    for dlo_type in request["dlo_types"]:
        directory = root / dlo_type / "train"
        files = sorted(directory.glob("*.pkl"), key=lambda path: int(path.stem))
        if len(files) != 56:
            raise ValueError(
                f"expected 56 official training files for {dlo_type}, found {len(files)}"
            )
        for path in files:
            manifest.append(
                {
                    "path": str(path.relative_to(root)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
            frames = load_trajectory(path)
            trajectory_shapes[dlo_type].append([int(value) for value in frames.shape])
            for config in configs:
                name = str(config["name"])
                mode = str(config["vertex_mode"])
                span = int(config["frame_span"])
                if span < 1 or span > 9:
                    raise ValueError("frame_span must lie in [1,9]")
                indices = vertex_indices(frames.shape[1], mode)
                for start in range(0, frames.shape[0] - span + 1, stride):
                    support = frames[start : start + span, indices].reshape(-1, 3)
                    normalized, rho, line_ratio = geometry_spectrum(support)
                    key = f"{dlo_type}/{name}"
                    spectra[key].append(normalized)
                    radii[key].append(rho)
                    line_ratios[key].append(line_ratio)
                    cases_by_object[dlo_type][name] += 1

    manifest_record: dict[str, Any] = {
        "files": sorted(manifest, key=lambda row: row["path"]),
        "file_count": len(manifest),
        "total_bytes": int(sum(row["bytes"] for row in manifest)),
    }
    manifest_record["manifest_sha256"] = canonical_sha256(manifest_record)

    groups: dict[str, Any] = {}
    for key, group_spectra in sorted(spectra.items()):
        matrix = np.asarray(group_spectra, dtype=np.float64)
        rank_rows: dict[str, Any] = {}
        for threshold in thresholds:
            ranks = np.sum(matrix >= threshold, axis=1)
            rank_rows[f"{threshold:.12g}"] = {
                "rank_deficient_fraction": float(np.mean(ranks < 7)),
                "rank_six_fraction": float(np.mean(ranks == 6)),
                "full_rank_fraction": float(np.mean(ranks == 7)),
                "rank_counts": {
                    str(rank): int(np.count_nonzero(ranks == rank))
                    for rank in sorted(set(int(value) for value in ranks))
                },
            }
        groups[key] = {
            "cases": int(matrix.shape[0]),
            "lambda7_over_lambda1": quantiles(matrix[:, -1].tolist()),
            "lambda6_over_lambda1": quantiles(matrix[:, -2].tolist()),
            "cloud_radius_m": quantiles(radii[key]),
            "smallest_to_largest_centered_singular_value": quantiles(line_ratios[key]),
            "rank_by_threshold": rank_rows,
        }

    result: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "stage": "source-audit",
        "request_id": request["request_id"],
        "evidence_class": "public-real-trajectory-source-only-geometry-audit",
        "dataset": {
            "name": "DEFORM",
            "objects": request["dlo_types"],
            "opened_split": "train",
            "manifest": manifest_record,
            "trajectory_shapes": trajectory_shapes,
        },
        "design": {
            "frame_stride": stride,
            "rank_threshold_candidates": thresholds,
            "support_configs": configs,
            "cases_by_object_and_support": cases_by_object,
        },
        "groups": groups,
        "information_boundary": request["information_boundary"],
        "claim_boundary": [
            "Only official DLO4/DLO5 training trajectories were opened.",
            "The audit measures real trajectory geometry and motion support, not learned-provider accuracy.",
            "No evaluation file content, BayesianPhysTwin outcome, or Causal4D outcome was opened.",
            "Any evaluation threshold must be frozen from this source-only artifact before evaluation access.",
        ],
    }
    result["result_id"] = canonical_sha256(result)
    return result


def write_summary(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# DEFORM DLO4/DLO5 source-only observability audit",
        "",
        f"- Result ID: `{result['result_id']}`",
        f"- Training files opened: `{result['dataset']['manifest']['file_count']}`",
        f"- Manifest SHA-256: `{result['dataset']['manifest']['manifest_sha256']}`",
        "",
        "## Geometry groups",
        "",
    ]
    for name, group in result["groups"].items():
        lines.append(
            f"- **{name}:** cases={group['cases']}, median lambda7/lambda1="
            f"{group['lambda7_over_lambda1']['median']:.6g}"
        )
        for threshold, row in group["rank_by_threshold"].items():
            lines.append(
                f"  - threshold `{threshold}`: rank<7={row['rank_deficient_fraction']:.3f}, "
                f"rank6={row['rank_six_fraction']:.3f}, rank7={row['full_rank_fraction']:.3f}"
            )
    lines.extend(["", "## Claim boundary", ""])
    lines.extend(f"- {entry}" for entry in result["claim_boundary"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    request = load_request(args.request)
    result = run(request)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_summary(result, args.output_dir / "summary.md")
    print(json.dumps({"result_id": result["result_id"], "output": str(args.output_dir)}))


if __name__ == "__main__":
    main()
