from __future__ import annotations

import json
import math
from statistics import NormalDist
from types import SimpleNamespace

import pytest

from prob4d.study_sensitivity import (
    STUDY_SENSITIVITY_SCHEMA,
    PairedDifferenceScenarioV1,
    build_study_sensitivity,
    load_study_sensitivity,
    one_sided_binomial_upper_bound,
    study_sensitivity_from_dict,
    write_study_sensitivity,
)


def _lock() -> SimpleNamespace:
    return SimpleNamespace(
        promotion_lock_id="a" * 64,
        target_group_ids=tuple(f"target-{index:02d}" for index in range(12)),
        minimum_target_group_count=12,
        query_superiority_margin_mm=1.0,
        harmful_update_margin_mm=0.25,
        maximum_harmful_accepted_updates=0,
        maximum_worst_group_regression_mm=0.5,
    )


def _report():
    return build_study_sensitivity(
        _lock(),
        source_summary_id="b" * 64,
        source_metric="deployed_minus_physical_rmse_mm",
        paired_difference_scenarios=(
            PairedDifferenceScenarioV1("conservative", 1.0),
            PairedDifferenceScenarioV1("source-estimate", 0.75),
        ),
        power_levels=(0.80, 0.90),
        confidence_level=0.95,
        alternative="two-sided",
        accepted_group_counts=(6, 12),
    )


def test_paired_effect_resolution_uses_declared_normal_approximation() -> None:
    report = _report()
    records = report.paired_effect_sensitivity
    conservative_eighty = next(
        record
        for record in records
        if record["scenario_id"] == "conservative" and record["power"] == 0.80
    )
    critical = NormalDist().inv_cdf(0.975)
    expected = (critical + NormalDist().inv_cdf(0.80)) / math.sqrt(12)

    assert conservative_eighty["minimum_detectable_effect_mm"] == pytest.approx(expected)
    assert conservative_eighty["confidence_interval_half_width_mm"] == pytest.approx(
        critical / math.sqrt(12)
    )
    assert conservative_eighty["query_margin_detectable"] is True
    assert report.query_margin_detectable is True


def test_harmful_update_resolution_reports_exact_upper_rate_bounds() -> None:
    report = _report()
    all_groups = next(
        record
        for record in report.harmful_update_resolution
        if record["accepted_group_count_scenario"] == 12
    )

    expected_zero_upper = 1.0 - 0.05 ** (1.0 / 12)
    assert all_groups["zero_harm_one_sided_upper_rate_bound"] == pytest.approx(
        expected_zero_upper
    )
    assert all_groups["allowed_count_one_sided_upper_rate_bound"] == pytest.approx(
        expected_zero_upper
    )
    assert all_groups["one_event_rate_resolution"] == pytest.approx(1 / 12)
    assert one_sided_binomial_upper_bound(12, 12, 0.95) == 1.0


def test_report_is_content_addressed_and_replays_derived_values() -> None:
    report = _report()
    payload = report.to_dict()

    assert payload["schema_name"] == STUDY_SENSITIVITY_SCHEMA
    assert payload["target_outcomes_opened"] is False
    assert len(report.sensitivity_id) == 64
    assert study_sensitivity_from_dict(payload) == report

    changed = json.loads(json.dumps(payload))
    changed["paired_effect_sensitivity"][0]["minimum_detectable_effect_mm"] += 0.1
    with pytest.raises(ValueError, match="deterministic replay"):
        study_sensitivity_from_dict(changed)


def test_strict_loader_and_no_clobber_publication(tmp_path) -> None:
    report = _report()
    output = tmp_path / "sensitivity.json"
    markdown = tmp_path / "sensitivity.md"

    write_study_sensitivity(report, output, markdown=markdown)
    loaded = load_study_sensitivity(output)
    assert loaded == report
    assert report.sensitivity_id in markdown.read_text(encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        write_study_sensitivity(report, output, markdown=markdown)

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_name":1,"schema_name":2}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        load_study_sensitivity(duplicate)


def test_invalid_design_inputs_fail_closed() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        PairedDifferenceScenarioV1("invalid", 0.0)
    with pytest.raises(ValueError, match="sorted by unique scenario_id"):
        build_study_sensitivity(
            _lock(),
            source_summary_id="b" * 64,
            source_metric="metric",
            paired_difference_scenarios=(
                PairedDifferenceScenarioV1("duplicate", 1.0),
                PairedDifferenceScenarioV1("duplicate", 1.5),
            ),
        )
    with pytest.raises(ValueError, match="cannot exceed"):
        build_study_sensitivity(
            _lock(),
            source_summary_id="b" * 64,
            source_metric="metric",
            paired_difference_scenarios=(
                PairedDifferenceScenarioV1("source", 1.0),
            ),
            accepted_group_counts=(13,),
        )


def test_generated_effect_resolution_is_monotone_in_dispersion_and_group_count() -> None:
    source_id = "b" * 64
    for target_count in range(4, 25):
        lock = SimpleNamespace(
            **{
                **_lock().__dict__,
                "target_group_ids": tuple(
                    f"target-{index:02d}" for index in range(target_count)
                ),
                "minimum_target_group_count": target_count,
            }
        )
        previous = None
        for standard_deviation in (0.25, 0.5, 1.0, 2.0):
            report = build_study_sensitivity(
                lock,
                source_summary_id=source_id,
                source_metric="metric",
                paired_difference_scenarios=(
                    PairedDifferenceScenarioV1("source", standard_deviation),
                ),
                power_levels=(0.80,),
            )
            effect = report.paired_effect_sensitivity[0][
                "minimum_detectable_effect_mm"
            ]
            assert isinstance(effect, float)
            if previous is not None:
                assert effect > previous
            previous = effect

    small = build_study_sensitivity(
        _lock(),
        source_summary_id=source_id,
        source_metric="metric",
        paired_difference_scenarios=(PairedDifferenceScenarioV1("source", 1.0),),
        power_levels=(0.80,),
    ).paired_effect_sensitivity[0]["minimum_detectable_effect_mm"]
    larger_lock = SimpleNamespace(
        **{
            **_lock().__dict__,
            "target_group_ids": tuple(f"target-{index:02d}" for index in range(48)),
            "minimum_target_group_count": 48,
        }
    )
    large = build_study_sensitivity(
        larger_lock,
        source_summary_id=source_id,
        source_metric="metric",
        paired_difference_scenarios=(PairedDifferenceScenarioV1("source", 1.0),),
        power_levels=(0.80,),
    ).paired_effect_sensitivity[0]["minimum_detectable_effect_mm"]
    assert isinstance(small, float)
    assert isinstance(large, float)
    assert large < small


def test_generated_binomial_upper_bounds_are_monotone() -> None:
    for trials in range(2, 40):
        bounds = [
            one_sided_binomial_upper_bound(events, trials, 0.95)
            for events in range(trials + 1)
        ]
        assert bounds == sorted(bounds)

    zero_event_bounds = [
        one_sided_binomial_upper_bound(0, trials, 0.95)
        for trials in range(2, 40)
    ]
    assert all(
        later < earlier
        for earlier, later in zip(
            zero_event_bounds,
            zero_event_bounds[1:],
            strict=False,
        )
    )
