from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from prob4d.finite_sample_capability import (
    build_finite_sample_capability,
    finite_sample_capability_from_dict,
    load_finite_sample_capability,
    preflight_cli,
    write_finite_sample_capability,
)


def _lock() -> SimpleNamespace:
    return SimpleNamespace(
        promotion_lock_id="a" * 64,
        calibration_group_ids=tuple(f"calibration-{index:02d}" for index in range(10)),
        target_group_ids=tuple(f"target-{index:02d}" for index in range(12)),
        bootstrap_resamples=5000,
        minimum_mean_accepted_coverage=0.90,
    )


def _strata() -> dict[str, tuple[str, ...]]:
    lock = _lock()
    return {
        "sheet": lock.calibration_group_ids[:5],
        "volumetric": lock.calibration_group_ids[5:],
    }


def test_round_trip_recomputes_derived_fields_and_rejects_aliases(tmp_path: Path) -> None:
    report = build_finite_sample_capability(
        _lock(),
        coverage_levels=(0.90, 0.95),
        calibration_strata=_strata(),
    )
    output = tmp_path / "capability.json"
    markdown = tmp_path / "capability.md"

    write_finite_sample_capability(report, output, markdown=markdown)

    assert load_finite_sample_capability(output) == report
    assert report.capability_id in markdown.read_text(encoding="utf-8")
    assert "calibration-sheet" in markdown.read_text(encoding="utf-8")
    with pytest.raises(FileExistsError, match="output already exists"):
        write_finite_sample_capability(report, output)

    tampered = report.to_dict()
    populations = tampered["populations"]
    assert isinstance(populations, list)
    levels = populations[0]["levels"]
    assert isinstance(levels, list)
    levels[0]["order_statistic_rank"] = 99
    with pytest.raises(ValueError, match="do not match recomputation"):
        finite_sample_capability_from_dict(tampered)

    coercive = report.to_dict()
    target = coercive["target_design"]
    assert isinstance(target, dict)
    target["target_group_count"] = 12.0
    with pytest.raises(ValueError, match="do not match recomputation"):
        finite_sample_capability_from_dict(coercive)


def test_duplicate_keys_and_nonfinite_json_fail_closed(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_name":"x","schema_name":"y"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        load_finite_sample_capability(duplicate)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"value":NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite JSON constant"):
        load_finite_sample_capability(nonfinite)


def test_serialized_report_is_canonical_finite_json() -> None:
    report = build_finite_sample_capability(
        _lock(),
        coverage_levels=(0.90,),
    )

    encoded = json.dumps(report.to_dict(), sort_keys=True, allow_nan=False)

    assert report.capability_id in encoded


def test_cli_retains_valid_negative_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "prob4d.finite_sample_capability.load_promotion_lock",
        lambda _path: _lock(),
    )
    output = tmp_path / "capability.json"
    markdown = tmp_path / "capability.md"

    result = preflight_cli(
        [
            str(tmp_path / "lock.json"),
            "--coverage",
            "0.95",
            "--output",
            str(output),
            "--markdown",
            str(markdown),
            "--require-primary-finite",
        ]
    )

    assert result == 3
    assert output.is_file()
    assert markdown.is_file()
    assert load_finite_sample_capability(output).primary_levels_finite is False
