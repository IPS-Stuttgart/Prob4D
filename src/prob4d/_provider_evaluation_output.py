"""JSON, CSV, and Markdown output for provider competence evaluation."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ._provider_evaluation_manifest import EvaluationModeName


def _csv_rows(records: list[dict[str, Any]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in records:
        artifact = record["artifact"]
        assert isinstance(artifact, dict)
        row: dict[str, object] = {
            "case_id": record["case_id"],
            "group_id": record["group_id"],
            "method_id": record["method_id"],
            "prediction_path": record["prediction_path"],
            "truth_path": record["truth_path"],
            "fusion_method": artifact["fusion_method"],
            "covariance_semantics": artifact["covariance_semantics"],
            "correlation_assumption": artifact["correlation_assumption"],
            "legacy_unspecified": artifact["legacy_unspecified"],
        }
        numeric = record["_numeric"]
        assert isinstance(numeric, dict)
        row.update(numeric)
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = sorted({field for row in rows for field in row})
    preferred = [
        "case_id",
        "group_id",
        "method_id",
        "fusion_method",
        "covariance_semantics",
        "correlation_assumption",
        "legacy_unspecified",
        "prediction_path",
        "truth_path",
    ]
    fieldnames = preferred + [field for field in fields if field not in preferred]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _format_summary(value: Mapping[str, Any] | None) -> str:
    if value is None:
        return "—"
    return (
        f"{float(value['mean']):.6g} "
        f"[{float(value['ci95_lower']):.6g}, {float(value['ci95_upper']):.6g}]"
    )


def _table_rows(
    summaries: Mapping[str, Any],
    *,
    primary_mode: EvaluationModeName,
    selected: tuple[str, ...],
) -> list[str]:
    rows: list[str] = []
    for method_id, summary in sorted(summaries.items()):
        metrics = summary["metrics"]
        values = [
            _format_summary(metrics.get(f"{primary_mode}.metrics.{metric}"))
            for metric in selected
        ]
        safe_method = str(method_id).replace("|", "\\|")
        rows.append(
            "| "
            + " | ".join(
                [
                    safe_method,
                    str(summary["group_count"]),
                    str(summary["case_count"]),
                    *values,
                ]
            )
            + " |"
        )
    return rows


def _write_markdown(
    path: Path,
    *,
    primary_mode: EvaluationModeName,
    reference_method: str,
    aggregate: Mapping[str, Any],
    comparisons: Mapping[str, Any],
) -> None:
    selected = (
        "metric_point_rmse",
        "point_rmse",
        "endpoint_point_rmse",
        "drift_slope",
        "seam_rmse",
        "coverage_95",
        "gaussian_nll",
    )
    headers = [
        "Method",
        "Groups",
        "Cases",
        "Metric point RMSE",
        "Point RMSE",
        "Endpoint RMSE",
        "Drift slope",
        "Seam RMSE",
        "Coverage 95%",
        "Gaussian NLL",
    ]
    separator = "| " + " | ".join(["---"] + ["---:"] * (len(headers) - 1)) + " |"
    lines = [
        "# Prob4D provider evaluation",
        "",
        f"Primary evaluation mode: `{primary_mode}`.",
        f"Registered reference method: `{reference_method}`.",
        "",
        "Intervals are deterministic group-bootstrap 95% intervals. Each physical "
        "object or independent acquisition session contributes equal aggregate mass.",
        "",
        "| " + " | ".join(headers) + " |",
        separator,
        *_table_rows(aggregate, primary_mode=primary_mode, selected=selected),
        "",
        "## Paired differences from the reference",
        "",
        "Every value below is `candidate - reference`, calculated within each case, "
        "averaged within group, and bootstrapped over groups. Negative values favor "
        "the candidate only for lower-is-better metrics.",
        "",
        "| " + " | ".join(headers) + " |",
        separator,
        *_table_rows(comparisons, primary_mode=primary_mode, selected=selected),
        "",
        "Covariance semantics are read from each prediction archive and must remain "
        "consistent within a method. Legacy archives are rejected by default.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_provider_evaluation_outputs(
    output_directory: str | Path,
    *,
    report: Mapping[str, Any],
    records: list[dict[str, Any]],
    primary_mode: EvaluationModeName,
    reference_method: str,
    aggregate: Mapping[str, Any],
    comparisons: Mapping[str, Any],
) -> None:
    """Write the complete report and compact human- and table-readable views."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    (output / "provider_evaluation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(output / "provider_evaluation.csv", _csv_rows(records))
    _write_markdown(
        output / "provider_evaluation.md",
        primary_mode=primary_mode,
        reference_method=reference_method,
        aggregate=aggregate,
        comparisons=comparisons,
    )


__all__ = ["write_provider_evaluation_outputs"]
