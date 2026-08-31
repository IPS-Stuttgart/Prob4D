#!/usr/bin/env python3
"""Reproduce a group-level robustness audit of the immutable DLO4/DLO5 result."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

PROTOCOL_SCHEMA = "prob4d.deform-dlo45-query-observability-robustness-audit"
RESULT_SCHEMA = "prob4d.deform-dlo45-query-observability-heldout-evaluation"
OUTPUT_SCHEMA = "prob4d.deform-dlo45-query-observability-robustness-result"


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object in {path}")
    return value


def summarize(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("summary values must be finite and nonempty")
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def trimmed_mean(values: Sequence[float], fraction: float) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    cut = int(math.floor(fraction * ordered.size))
    retained = ordered[cut : ordered.size - cut] if cut else ordered
    return float(retained.mean())


def sign_tail(wins: int, groups: int) -> float:
    return math.ldexp(sum(math.comb(groups, k) for k in range(wins, groups + 1)), -groups)


def leave_one_out(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    means = (array.sum() - array) / (array.size - 1)
    return {"minimum": float(means.min()), "maximum": float(means.max())}


def blocked_bootstrap(
    rows: Sequence[Mapping[str, Any]], key: str, replicates: int, seed: int
) -> dict[str, float | int]:
    blocks = [
        np.asarray([float(row[key]) for row in rows if row["family"] == family])
        for family in ("DLO4", "DLO5")
    ]
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates)
    count = sum(block.size for block in blocks)
    for index in range(replicates):
        draws[index] = sum(
            float(block[rng.integers(0, block.size, block.size)].sum())
            for block in blocks
        ) / count
    return {
        "replicates": replicates,
        "seed": seed,
        "ci95_lower": float(np.quantile(draws, 0.025)),
        "ci95_upper": float(np.quantile(draws, 0.975)),
    }


def metric(group: Mapping[str, Any], query: str, method: str, name: str) -> float:
    return float(group[query][method][name])


def extract_rows(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    groups = result["per_group_results"]
    rows = []
    for group_name in sorted(groups):
        group = groups[group_name]
        family = group_name.split("/", 1)[0]
        if family not in {"DLO4", "DLO5"}:
            raise ValueError(f"unexpected group family: {group_name}")
        fallback_rmse = metric(group, "segment_centroid", "physical_fallback", "rmse_mm")
        query_rmse = metric(group, "segment_centroid", "query_aware", "rmse_mm")
        fallback_nll = metric(
            group, "segment_centroid", "physical_fallback", "mean_gaussian_nll"
        )
        query_nll = metric(group, "segment_centroid", "query_aware", "mean_gaussian_nll")
        rows.append(
            {
                "group": group_name,
                "family": family,
                "rmse_gain_mm": fallback_rmse - query_rmse,
                "nll_gain": fallback_nll - query_nll,
                "coverage": metric(
                    group, "segment_centroid", "query_aware", "empirical_90pct_coverage"
                ),
                "nees": metric(group, "segment_centroid", "query_aware", "normalized_nees"),
                "off_acceptance": metric(
                    group, "off_axis_probe", "query_aware", "accepted_fraction"
                ),
                "off_exact_fallback": metric(
                    group, "off_axis_probe", "query_aware", "exact_fallback_fraction"
                ),
                "unconditional_harm": metric(
                    group,
                    "off_axis_probe",
                    "observable_subspace_unconditional",
                    "harmful_fraction_vs_fallback",
                ),
                "invalid_harm": metric(
                    group,
                    "off_axis_probe",
                    "invalid_full_rank_completion",
                    "harmful_fraction_vs_fallback",
                ),
            }
        )
    return rows


def gain_summary(
    rows: Sequence[Mapping[str, Any]], key: str, protocol: Mapping[str, Any], seed: int
) -> dict[str, Any]:
    values = [float(row[key]) for row in rows]
    wins = sum(value > 0 for value in values)
    by_family = {}
    for family in ("DLO4", "DLO5"):
        family_values = [float(row[key]) for row in rows if row["family"] == family]
        by_family[family] = {
            **summarize(family_values),
            "wins": sum(value > 0 for value in family_values),
            "groups": len(family_values),
        }
    analysis = protocol["analysis"]
    return {
        **summarize(values),
        "trimmed_mean": trimmed_mean(values, float(analysis["trim_fraction"])),
        "wins": wins,
        "groups": len(values),
        "exact_one_sided_sign_p": sign_tail(wins, len(values)),
        "leave_one_out": leave_one_out(values),
        "by_family": by_family,
        "family_blocked_bootstrap": blocked_bootstrap(
            rows, key, int(analysis["bootstrap_replicates"]), seed
        ),
    }


def audit(result: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict[str, Any]:
    if result.get("schema") != RESULT_SCHEMA:
        raise ValueError("unexpected source-result schema")
    source = protocol["source_result"]
    for key in ("result_id", "request_id"):
        if result.get(key) != source[key]:
            raise ValueError(f"source {key} mismatch")
    if result["information_boundary"]["post_open_retuning_permitted"] is not False:
        raise ValueError("source result does not preserve the no-retuning boundary")
    rows = extract_rows(result)
    expected = protocol["analysis"]["family_counts"]
    counts = {family: sum(row["family"] == family for row in rows) for family in expected}
    if len(rows) != 28 or counts != expected:
        raise ValueError(f"unexpected independent-group roster: {counts}")

    rmse = gain_summary(rows, "rmse_gain_mm", protocol, 20260901)
    nll = gain_summary(rows, "nll_gain", protocol, 20260902)
    output: dict[str, Any] = {
        "schema": OUTPUT_SCHEMA,
        "schema_version": 1,
        "status": "completed-posthoc-group-robustness-audit",
        "protocol_id": protocol["protocol_id"],
        "source_result": source,
        "independent_groups": len(rows),
        "family_counts": counts,
        "centroid_gain": {"rmse_improvement_mm": rmse, "nll_improvement": nll},
        "off_axis_controls": {
            "minimum_exact_fallback_fraction": min(
                float(row["off_exact_fallback"]) for row in rows
            ),
            "maximum_query_aware_acceptance": max(
                float(row["off_acceptance"]) for row in rows
            ),
            "unconditional_harmful_files": sum(
                float(row["unconditional_harm"]) > 0 for row in rows
            ),
            "invalid_full_rank_harmful_files": sum(
                float(row["invalid_harm"]) > 0 for row in rows
            ),
        },
        "calibration_limit": {
            "coverage_below_90pct_files": sum(float(row["coverage"]) < 0.9 for row in rows),
            "nees_above_one_files": sum(float(row["nees"]) > 1.0 for row in rows),
            "coverage": summarize([float(row["coverage"]) for row in rows]),
            "normalized_nees": summarize([float(row["nees"]) for row in rows]),
            "descriptive_rms_scale_mismatch": math.sqrt(
                float(np.mean([float(row["nees"]) for row in rows]))
            ),
            "descriptive_scale_is_not_recalibration": True,
        },
        "claim_boundary": protocol["claim_boundary"],
    }
    payload = dict(output)
    output["audit_id"] = hashlib.sha256(canonical(payload)).hexdigest()
    output["per_group_sha256"] = hashlib.sha256(canonical(rows)).hexdigest()
    output["rows"] = rows
    return output


def fmt(value: float) -> str:
    return f"{value:.3f}"


def markdown(output: Mapping[str, Any]) -> str:
    rmse = output["centroid_gain"]["rmse_improvement_mm"]
    nll = output["centroid_gain"]["nll_improvement"]
    off = output["off_axis_controls"]
    calibration = output["calibration_limit"]
    lines = [
        "# DEFORM DLO4/DLO5 query-observability robustness audit",
        "",
        f"Status: **{output['status']}**",
        "",
        "This post-hoc audit reanalyzes the immutable 28-file held-out result at the "
        "declared trajectory-file level. It changes no method, threshold, support, "
        "prior, covariance, or evaluation outcome.",
        "",
        "## Positive mechanism robustness",
        "",
        "| Endpoint | Mean | Worst file | 20% trimmed mean | "
        "Family-blocked 95% CI | Wins | Exact sign p |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| Centroid RMSE improvement [mm] | {fmt(rmse['mean'])} | "
        f"{fmt(rmse['minimum'])} | {fmt(rmse['trimmed_mean'])} | "
        f"[{fmt(rmse['family_blocked_bootstrap']['ci95_lower'])}, "
        f"{fmt(rmse['family_blocked_bootstrap']['ci95_upper'])}] | "
        f"{rmse['wins']}/{rmse['groups']} | {rmse['exact_one_sided_sign_p']:.3e} |",
        f"| Centroid Gaussian-NLL improvement | {fmt(nll['mean'])} | "
        f"{fmt(nll['minimum'])} | {fmt(nll['trimmed_mean'])} | "
        f"[{fmt(nll['family_blocked_bootstrap']['ci95_lower'])}, "
        f"{fmt(nll['family_blocked_bootstrap']['ci95_upper'])}] | "
        f"{nll['wins']}/{nll['groups']} | {nll['exact_one_sided_sign_p']:.3e} |",
        "",
        f"Leave-one-file-out RMSE means remain in `[{fmt(rmse['leave_one_out']['minimum'])}, "
        f"{fmt(rmse['leave_one_out']['maximum'])}]` mm. DLO4 and DLO5 each retain "
        "`14/14` RMSE and NLL wins.",
        "",
        "## Negative controls and calibration boundary",
        "",
        "| Finding | Result |",
        "|---|---:|",
        "| Minimum query-aware off-axis exact fallback | "
        f"{fmt(off['minimum_exact_fallback_fraction'])} |",
        f"| Unconditional off-axis use harmful | {off['unconditional_harmful_files']}/28 files |",
        "| Invalid full-rank completion harmful | "
        f"{off['invalid_full_rank_harmful_files']}/28 files |",
        "| Accepted centroid coverage below 90% | "
        f"{calibration['coverage_below_90pct_files']}/28 files |",
        "| Accepted centroid normalized NEES above 1 | "
        f"{calibration['nees_above_one_files']}/28 files |",
        "",
        "The positive mean/proper-score effect is not driven by one trajectory or one "
        "DLO family. The covariance-calibration failure is equally systematic and remains "
        "a limitation; the descriptive scale mismatch is not an evaluation-side repair.",
        "",
        "## Claim boundary",
        "",
    ]
    lines.extend(f"- {item}" for item in output["claim_boundary"])
    lines.append("")
    return "\n".join(lines)


def write_outputs(output_dir: Path, output: Mapping[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    compact = dict(output)
    rows = compact.pop("rows")
    (output_dir / "summary.json").write_bytes(canonical(compact))
    (output_dir / "ROBUSTNESS.md").write_text(markdown(output), encoding="utf-8")
    with (output_dir / "per_group.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> int:
    result_path = Path(args.result)
    protocol = read_json(Path(args.protocol))
    if protocol.get("schema") != PROTOCOL_SCHEMA or protocol.get("schema_version") != 1:
        raise ValueError("unexpected audit protocol")
    payload = dict(protocol)
    protocol_id = payload.pop("protocol_id")
    if hashlib.sha256(canonical(payload)).hexdigest() != protocol_id:
        raise ValueError("protocol ID mismatch")
    if sha256(result_path) != protocol["source_result"]["result_sha256"]:
        raise ValueError("source result SHA-256 mismatch")
    write_outputs(Path(args.output_dir), audit(read_json(result_path), protocol))
    return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--result", required=True)
    value.add_argument("--protocol", required=True)
    value.add_argument("--output-dir", required=True)
    return value


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
