"""No-clobber JSON and Markdown output for finite-sample capability reports."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from ._finite_sample_capability_common import FINITE_SAMPLE_CAPABILITY_CLAIM_BOUNDARY
from ._finite_sample_capability_model import FiniteSampleCapabilityV1
from ._finite_sample_capability_records import target_design_record


def _atomic_create(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"finite-sample capability output already exists: {path}")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def render_finite_sample_capability_markdown(
    report: FiniteSampleCapabilityV1,
) -> str:
    """Render a compact deterministic evidence summary."""

    lines = [
        "# Finite-sample capability preflight",
        "",
        f"- capability ID: `{report.capability_id}`",
        f"- promotion lock: `{report.promotion_lock_id}`",
        f"- calibration groups: **{len(report.calibration_group_ids)}**",
        f"- target groups: **{len(report.target_group_ids)}**",
        "",
        "| Population | Groups | Coverage | Rank | Finite | Lower bound | Minimum groups |",
        "| --- | ---: | ---: | ---: | :---: | ---: | ---: |",
    ]
    for population in report.populations:
        population_id = cast(str, population["population_id"])
        count = cast(int, population["group_count"])
        levels = population["levels"]
        assert isinstance(levels, list)
        for level in levels:
            assert isinstance(level, Mapping)
            bound = cast(float | None, level["guaranteed_coverage_lower_bound"])
            lines.append(
                f"| {population_id} | {count} | "
                f"{cast(float, level['nominal_coverage']):.3f} | "
                f"{cast(int, level['order_statistic_rank'])} | "
                f"{'yes' if level['finite_threshold'] else 'no'} | "
                f"{'—' if bound is None else f'{bound:.3f}'} | "
                f"{cast(int, level['minimum_group_count_for_finite_threshold'])} |"
            )
    target = target_design_record(
        len(report.target_group_ids),
        report.bootstrap_resamples,
    )
    lines.extend(
        [
            "",
            "## Target-group diagnostic resolution",
            "",
            "- empirical bootstrap mass resolution: "
            f"`{cast(float, target['bootstrap_empirical_mass_resolution']):.6g}`",
            "- leave-one-group-out replications: "
            f"`{cast(int, target['leave_one_group_out_replications'])}`",
            "- all-favorable one-sided sign probability: "
            f"`{cast(float, target['all_favorable_one_sided_sign_probability']):.6g}`",
            "",
            f"- all primary levels finite: **{report.primary_levels_finite}**",
            f"- all stratum levels finite: **{report.stratum_levels_finite}**",
            "",
            FINITE_SAMPLE_CAPABILITY_CLAIM_BOUNDARY,
            "",
        ]
    )
    return "\n".join(lines)


def write_finite_sample_capability(
    report: FiniteSampleCapabilityV1,
    output: Path,
    *,
    markdown: Path | None = None,
) -> None:
    """Publish JSON and optional Markdown without replacing retained evidence."""

    destinations = [output] if markdown is None else [output, markdown]
    existing = [path for path in destinations if path.exists()]
    if existing:
        raise FileExistsError(
            "finite-sample capability output already exists: "
            + ", ".join(str(path) for path in existing)
        )
    encoded = json.dumps(report.to_dict(), sort_keys=True, indent=2).encode() + b"\n"
    _atomic_create(output, encoded)
    if markdown is not None:
        _atomic_create(
            markdown,
            render_finite_sample_capability_markdown(report).encode("utf-8"),
        )


__all__ = [
    "render_finite_sample_capability_markdown",
    "write_finite_sample_capability",
]
