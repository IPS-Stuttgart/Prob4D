#!/usr/bin/env python3
"""Held-out Deform360 rope pilot for finite-orbit physical-query admission."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import traceback
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

QUERIES = ("span_change", "centroid_progress", "named_endpoint_progress")
ARMS = (
    "fallback",
    "ungated_candidate",
    "local_canonical",
    "reject_noninvariant",
    "independent_finite_orbit",
    "shared_finite_orbit",
)


class PilotError(RuntimeError):
    """Bounded protocol or data failure."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("schema") != "prob4d.deform360-rope-query-pilot-v1":
        raise PilotError("unexpected protocol schema")
    split = protocol["episode_split"]
    source = tuple(split["source"])
    calibration = tuple(split["calibration"])
    target = tuple(split["target"])
    if sorted(source + calibration + target) != list(range(10)):
        raise PilotError("episodes 0..9 must be assigned exactly once")
    if set(source) & set(calibration) or set(source) & set(target):
        raise PilotError("episode cohorts overlap")
    if set(calibration) & set(target):
        raise PilotError("episode cohorts overlap")
    if protocol["evaluation"]["queries"] != list(QUERIES):
        raise PilotError("query roster changed")
    if protocol["evaluation"]["arms"] != list(ARMS):
        raise PilotError("arm roster changed")
    return protocol


def episode_number(path: Path) -> int | None:
    for part in path.parts:
        if part.startswith("episode_") and part[8:].isdigit():
            return int(part[8:])
    return None


def local_clouds(root: Path) -> dict[int, list[Path]]:
    candidates = (
        root / "processed" / "001-rope",
        root / "processed-repository" / "processed" / "001-rope",
        root / "deform360_processed" / "processed" / "001-rope",
    )
    result: dict[int, list[Path]] = defaultdict(list)
    for candidate in candidates:
        for path in candidate.glob("episode_*/pcd_clean/0000*.npz"):
            episode = episode_number(path)
            if episode is not None:
                result[episode].append(path)
    return {key: sorted(value) for key, value in result.items()}


def resolve_clouds(
    dataset_root: Path, protocol: Mapping[str, Any], download_root: Path
) -> tuple[dict[int, list[Path]], dict[str, Any]]:
    local = local_clouds(dataset_root)
    if all(len(local.get(episode, ())) >= 20 for episode in range(10)):
        return local, {"mode": "local", "root": str(dataset_root)}
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise PilotError("huggingface_hub is unavailable") from exc
    source = protocol["processed_source"]
    try:
        snapshot = Path(
            snapshot_download(
                repo_id=source["repository"],
                repo_type="dataset",
                revision=source["revision"],
                allow_patterns=source["allow_patterns"],
                local_dir=download_root,
            )
        )
    except Exception as exc:
        raise PilotError(f"processed-data download failed: {exc}") from exc
    result: dict[int, list[Path]] = defaultdict(list)
    for path in snapshot.glob("processed/001-rope/episode_*/pcd_clean/0000*.npz"):
        episode = episode_number(path)
        if episode is not None:
            result[episode].append(path)
    return (
        {key: sorted(value) for key, value in result.items()},
        {
            "mode": "pinned-public-download",
            "repository": source["repository"],
            "revision": source["revision"],
            "root": str(snapshot),
        },
    )


def load_points(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        if "pts" not in archive.files:
            raise PilotError(f"{path} has no pts array")
        points = np.asarray(archive["pts"], dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 100:
        raise PilotError(f"invalid point-cloud shape {points.shape} in {path}")
    if not np.isfinite(points).all():
        raise PilotError(f"nonfinite point in {path}")
    return points


def canonical_axis(vector: np.ndarray) -> np.ndarray:
    axis = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-12:
        raise PilotError("degenerate endpoint axis")
    axis /= norm
    pivot = int(np.argmax(np.abs(axis)))
    return -axis if axis[pivot] < 0.0 else axis


def endpoint_trajectory(
    paths: Sequence[Path], protocol: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray, str]:
    geometry = protocol["geometry"]
    stride = int(geometry["frame_stride"])
    limit = int(geometry["maximum_selected_frames"])
    selected = tuple(paths[::stride][:limit])
    if len(selected) < int(geometry["minimum_selected_frames"]):
        raise PilotError(f"only {len(selected)} selected point-cloud frames")
    first = load_points(selected[0])
    centered = first - np.mean(first, axis=0)
    values, vectors = np.linalg.eigh(centered.T @ centered)
    order = np.argsort(values)[::-1]
    ratio = float(values[order[0]] / max(values[order[1]], 1e-15))
    if ratio < float(geometry["minimum_axis_ratio"]):
        raise PilotError(f"principal-axis ratio {ratio:.3f} below gate")
    axis = canonical_axis(vectors[:, order[0]])
    projection = centered @ axis
    fraction = float(geometry["tail_fraction"])
    low, high = np.quantile(projection, [fraction, 1.0 - fraction])
    low_ids = np.flatnonzero(projection <= low)
    high_ids = np.flatnonzero(projection >= high)
    if min(len(low_ids), len(high_ids)) < 5:
        raise PilotError("too few fixed endpoint-tail points")
    endpoints: list[np.ndarray] = []
    spans: list[float] = []
    digest = hashlib.sha256()
    for path in selected:
        points = load_points(path)
        if points.shape != first.shape:
            raise PilotError("persistent point identity changed within episode")
        pair = np.stack((np.mean(points[low_ids], axis=0), np.mean(points[high_ids], axis=0)))
        span = float(np.linalg.norm(pair[1] - pair[0]))
        if span < float(geometry["minimum_span_m"]):
            raise PilotError(f"endpoint span {span:.6f} m below gate")
        endpoints.append(pair)
        spans.append(span)
        digest.update(path.name.encode())
        digest.update(bytes.fromhex(sha256_file(path)))
    return np.asarray(endpoints), np.asarray(spans), digest.hexdigest()


def cohort_map(protocol: Mapping[str, Any]) -> dict[int, str]:
    return {
        int(episode): cohort
        for cohort in ("source", "calibration", "target")
        for episode in protocol["episode_split"][cohort]
    }


def load_episodes(
    clouds: Mapping[int, Sequence[Path]], protocol: Mapping[str, Any]
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    episodes: dict[int, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    cohorts = cohort_map(protocol)
    for episode in range(10):
        try:
            endpoints, spans, digest = endpoint_trajectory(clouds.get(episode, ()), protocol)
            episodes[episode] = {
                "cohort": cohorts[episode],
                "endpoints": endpoints,
                "spans": spans,
                "digest": digest,
                "frames": len(endpoints),
            }
        except Exception as exc:
            failures.append({"episode": episode, "cohort": cohorts[episode], "reason": str(exc)})
    return episodes, failures


def fit_damping(
    episodes: Mapping[int, Mapping[str, Any]], horizons: Sequence[int]
) -> dict[int, float]:
    result: dict[int, float] = {}
    for horizon in horizons:
        numerator = 0.0
        denominator = 0.0
        for episode in episodes.values():
            if episode["cohort"] != "source":
                continue
            points = episode["endpoints"]
            for frame in range(1, len(points) - horizon):
                design = horizon * (points[frame] - points[frame - 1])
                response = points[frame + horizon] - points[frame]
                numerator += float(np.sum(design * response))
                denominator += float(np.sum(design * design))
        result[horizon] = float(np.clip(numerator / denominator if denominator else 0.0, 0.0, 1.5))
    return result


def query_value(
    current: np.ndarray, future: np.ndarray, axis: np.ndarray, query: str, gauge: int
) -> float:
    if query == "span_change":
        return float(
            np.linalg.norm(future[1] - future[0]) - np.linalg.norm(current[1] - current[0])
        )
    if query == "centroid_progress":
        return float((np.mean(future, axis=0) - np.mean(current, axis=0)) @ axis)
    if query == "named_endpoint_progress":
        return float((future[gauge] - current[gauge]) @ axis)
    raise PilotError(f"unknown query {query}")


def episode_metrics(
    episodes: Mapping[int, Mapping[str, Any]], damping: Mapping[int, float], horizons: Sequence[int]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for episode_id, episode in sorted(episodes.items()):
        points = episode["endpoints"]
        spans = episode["spans"]
        for horizon in horizons:
            grouped: dict[tuple[str, int], list[tuple[float, float, float]]] = defaultdict(list)
            for frame in range(1, len(points) - horizon):
                current = points[frame]
                truth = points[frame + horizon]
                candidate = current + damping[horizon] * horizon * (current - points[frame - 1])
                axis = canonical_axis(current[1] - current[0])
                span = float(spans[frame])
                predictions = {
                    query: [
                        query_value(current, candidate, axis, query, gauge) / span
                        for gauge in (0, 1)
                    ]
                    for query in QUERIES
                }
                for query in QUERIES:
                    diameter = abs(predictions[query][0] - predictions[query][1])
                    for gauge in (0, 1):
                        actual = query_value(current, truth, axis, query, gauge) / span
                        candidate_error = abs(predictions[query][gauge] - actual)
                        fallback_error = abs(actual)
                        grouped[(query, gauge)].append((candidate_error, fallback_error, diameter))
            for (query, gauge), values in sorted(grouped.items()):
                rows.append(
                    {
                        "episode": episode_id,
                        "cohort": episode["cohort"],
                        "query": query,
                        "horizon": horizon,
                        "gauge": gauge,
                        "candidate_error": float(np.mean([value[0] for value in values])),
                        "fallback_error": float(np.mean([value[1] for value in values])),
                        "advantage": float(np.mean([value[1] - value[0] for value in values])),
                        "orbit_diameter": float(np.mean([value[2] for value in values])),
                    }
                )
    return rows


def conservative_quantile(values: Iterable[float], probability: float) -> float:
    array = np.sort(np.asarray(list(values), dtype=np.float64))
    if not len(array):
        return float("nan")
    index = int(math.ceil((len(array) + 1) * probability) - 1)
    return float(array[min(max(index, 0), len(array) - 1)])


def calibration_model(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[tuple[str, int, int], dict[str, float]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["cohort"] == "calibration":
            grouped[(str(row["query"]), int(row["horizon"]), int(row["gauge"]))].append(row)
    model: dict[tuple[str, int, int], dict[str, float]] = {}
    output: list[dict[str, Any]] = []
    for key, values in sorted(grouped.items()):
        shared = conservative_quantile((float(value["advantage"]) for value in values), 0.10)
        independent = conservative_quantile(
            (float(value["fallback_error"]) for value in values), 0.10
        ) - conservative_quantile((float(value["candidate_error"]) for value in values), 0.90)
        model[key] = {
            "shared_lcb": shared,
            "independent_lcb": independent,
            "orbit_diameter": max(float(value["orbit_diameter"]) for value in values),
        }
        output.append({"query": key[0], "horizon": key[1], "gauge": key[2], **model[key]})
    return model, output


def accept(
    arm: str,
    query: str,
    horizon: int,
    model: Mapping[tuple[str, int, int], Mapping[str, float]],
    protocol: Mapping[str, Any],
) -> bool:
    if arm == "fallback":
        return False
    if arm == "ungated_candidate":
        return True
    records = [model.get((query, horizon, gauge), {}) for gauge in (0, 1)]
    margin = float(protocol["evaluation"]["advantage_margin_normalized"])
    if arm == "local_canonical":
        return float(records[0].get("shared_lcb", float("nan"))) > margin
    if arm == "reject_noninvariant":
        diameter = max(float(record.get("orbit_diameter", float("inf"))) for record in records)
        return diameter <= float(protocol["evaluation"]["orbit_tolerance_normalized"])
    field = "independent_lcb" if arm == "independent_finite_orbit" else "shared_lcb"
    return all(float(record.get(field, float("nan"))) > margin for record in records)


def bootstrap_ci(values: Sequence[float], seed: int, repetitions: int) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    means = np.asarray(
        [
            float(np.mean(generator.choice(array, size=len(array), replace=True)))
            for _ in range(repetitions)
        ]
    )
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def evaluate(
    rows: Sequence[Mapping[str, Any]],
    model: Mapping[tuple[str, int, int], Mapping[str, float]],
    protocol: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    target: dict[tuple[int, str, int], dict[int, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row["cohort"] == "target":
            target[(int(row["episode"]), str(row["query"]), int(row["horizon"]))][
                int(row["gauge"])
            ] = row
    episode_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    harm_margin = float(protocol["evaluation"]["harm_margin_normalized"])
    repetitions = int(protocol["evaluation"]["bootstrap_repetitions"])
    seed = int(protocol["evaluation"]["bootstrap_seed"])
    for query in QUERIES:
        for horizon in protocol["prediction"]["horizons_frames"]:
            keys = [key for key in target if key[1:] == (query, horizon)]
            for arm in ARMS:
                admitted = accept(arm, query, horizon, model, protocol)
                arm_rows: list[dict[str, Any]] = []
                for episode, _, _ in sorted(keys):
                    gauges = target[(episode, query, horizon)]
                    if set(gauges) != {0, 1}:
                        continue
                    candidate = max(float(gauges[g]["candidate_error"]) for g in (0, 1))
                    fallback = max(float(gauges[g]["fallback_error"]) for g in (0, 1))
                    deployed = candidate if admitted else fallback
                    regret = deployed - fallback
                    record = {
                        "episode": episode,
                        "query": query,
                        "horizon": horizon,
                        "arm": arm,
                        "accepted": int(admitted),
                        "candidate_error_worst_gauge": candidate,
                        "fallback_error_worst_gauge": fallback,
                        "deployed_error_worst_gauge": deployed,
                        "regret_vs_fallback": regret,
                        "harmful_accepted": int(admitted and regret > harm_margin),
                    }
                    arm_rows.append(record)
                    episode_rows.append(record)
                regrets = [float(row["regret_vs_fallback"]) for row in arm_rows]
                errors = [float(row["deployed_error_worst_gauge"]) for row in arm_rows]
                lower, upper = bootstrap_ci(
                    regrets,
                    seed + horizon + sum(ord(char) for char in query + arm),
                    repetitions,
                )
                summaries.append(
                    {
                        "query": query,
                        "horizon": horizon,
                        "arm": arm,
                        "target_episodes": len(arm_rows),
                        "accepted_episodes": sum(int(row["accepted"]) for row in arm_rows),
                        "acceptance_rate": float(np.mean([row["accepted"] for row in arm_rows])),
                        "mean_error_worst_gauge": float(np.mean(errors)),
                        "mean_regret_vs_fallback": float(np.mean(regrets)),
                        "regret_ci95_low": lower,
                        "regret_ci95_high": upper,
                        "harmful_accepted_rate": float(
                            np.mean(
                                [row["harmful_accepted"] for row in arm_rows if row["accepted"]]
                            )
                        )
                        if admitted
                        else 0.0,
                        "worst_episode_regret": max(regrets),
                    }
                )
    return episode_rows, summaries


def report(result: Mapping[str, Any]) -> str:
    lines = [
        "# Deform360 rope finite-orbit query pilot",
        "",
        f"Decision: **{result['decision']}**",
        "",
        "This is a within-object held-out-episode pilot; frames are repeated observations.",
        "",
        "| Query | H | Arm | Acceptance | Error | Regret | 95% CI | Harm |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in result.get("target_summary", []):
        lines.append(
            f"| {row['query']} | {row['horizon']} | {row['arm']} | "
            f"{row['acceptance_rate']:.3f} | {row['mean_error_worst_gauge']:.6f} | "
            f"{row['mean_regret_vs_fallback']:.6f} | "
            f"[{row['regret_ci95_low']:.6f}, {row['regret_ci95_high']:.6f}] | "
            f"{row['harmful_accepted_rate']:.3f} |"
        )
    lines.extend(("", "## Claim boundary", "", str(result["claim_boundary"]), ""))
    return "\n".join(lines)


def execute(args: argparse.Namespace) -> dict[str, Any]:
    protocol = read_protocol(args.protocol)
    dataset_root = args.dataset_root.resolve(strict=True)
    metadata_candidates = [dataset_root / path for path in protocol["dataset"]["metadata_paths"]]
    metadata = next((path for path in metadata_candidates if path.is_file()), None)
    if metadata is None:
        raise PilotError("001-rope metadata is absent from the mounted raw dataset")
    clouds, processed_source = resolve_clouds(dataset_root, protocol, args.download_root)
    episodes, failures = load_episodes(clouds, protocol)
    counts = {
        cohort: sum(episode["cohort"] == cohort for episode in episodes.values())
        for cohort in ("source", "calibration", "target")
    }
    required = protocol["evaluation"]["minimum_usable_episodes"]
    reasons = [
        f"{cohort}: {counts[cohort]} usable, require {required[cohort]}"
        for cohort in counts
        if counts[cohort] < int(required[cohort])
    ]
    decision = "support-negative" if reasons else "completed-held-out-real-data-pilot"
    manifest = [
        {
            "episode": episode,
            "cohort": value["cohort"],
            "frames": value["frames"],
            "median_span_m": float(np.median(value["spans"])),
            "source_digest": value["digest"],
        }
        for episode, value in sorted(episodes.items())
    ]
    write_csv(args.output_dir / "episode_manifest.csv", manifest)
    write_json(args.output_dir / "support_failures.json", {"failures": failures})
    result: dict[str, Any] = {
        "schema": "prob4d.deform360-rope-query-pilot-result-v1",
        "decision": decision,
        "promotion_authorized": False,
        "dataset_root": str(dataset_root),
        "metadata_sha256": sha256_file(metadata),
        "processed_source": processed_source,
        "usable_episodes": len(episodes),
        "cohort_counts": counts,
        "support_reasons": reasons,
        "support_failures": failures,
        "claim_boundary": protocol["claim_boundary"],
        "target_summary": [],
    }
    if not reasons:
        horizons = tuple(int(value) for value in protocol["prediction"]["horizons_frames"])
        damping = fit_damping(episodes, horizons)
        metric_rows = episode_metrics(episodes, damping, horizons)
        model, calibration = calibration_model(metric_rows)
        episode_rows, summary = evaluate(metric_rows, model, protocol)
        write_csv(args.output_dir / "all_episode_metrics.csv", metric_rows)
        write_csv(args.output_dir / "calibration_bounds.csv", calibration)
        write_csv(args.output_dir / "target_episode_scores.csv", episode_rows)
        write_csv(args.output_dir / "target_summary.csv", summary)
        result.update({"damping": damping, "calibration": calibration, "target_summary": summary})
    result["result_id"] = hashlib.sha256(
        json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    write_json(args.output_dir / "result.json", result)
    (args.output_dir / "REPORT.md").write_text(report(result), encoding="utf-8")
    return result


def self_test() -> None:
    values = [
        {"candidate_error": 0.19, "fallback_error": 0.20, "advantage": 0.01},
        {"candidate_error": 0.79, "fallback_error": 0.80, "advantage": 0.01},
    ]
    shared = conservative_quantile((value["advantage"] for value in values), 0.10)
    independent = conservative_quantile(
        (value["fallback_error"] for value in values), 0.10
    ) - conservative_quantile((value["candidate_error"] for value in values), 0.90)
    assert shared > 0.0 and independent < 0.0
    current = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    future = np.asarray([[0.1, 0.0, 0.0], [1.3, 0.0, 0.0]])
    axis = np.asarray([1.0, 0.0, 0.0])
    assert query_value(current, future, axis, "span_change", 0) == query_value(
        current, future, axis, "span_change", 1
    )
    assert query_value(current, future, axis, "named_endpoint_progress", 0) != query_value(
        current, future, axis, "named_endpoint_progress", 1
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--download-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("self-test passed")
        return 0
    if any(
        value is None
        for value in (args.protocol, args.dataset_root, args.download_root, args.output_dir)
    ):
        parser.error("all path arguments are required")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        result = execute(args)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        failure = {
            "schema": "prob4d.deform360-rope-query-pilot-result-v1",
            "decision": "technical-negative",
            "promotion_authorized": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        write_json(args.output_dir / "result.json", failure)
        (args.output_dir / "REPORT.md").write_text(
            f"# Deform360 rope pilot\n\nDecision: **technical-negative**\n\n`{failure['error']}`\n",
            encoding="utf-8",
        )
        print(json.dumps(failure, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    import sys

    raise SystemExit(main())
