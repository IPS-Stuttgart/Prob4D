from __future__ import annotations

import hashlib
import json
from pathlib import Path

from prob4d.causal_gauge_graph_monte_carlo import GaugeGraphStudyScenario
from prob4d.cycle_guard_monte_carlo import (
    CYCLE_GUARD_MONTE_CARLO_SCHEMA,
    run_cycle_guard_monte_carlo,
)


def _tiny_scenarios() -> tuple[GaugeGraphStudyScenario, ...]:
    return (
        GaugeGraphStudyScenario("independent_clean", correlation=0.0),
        GaugeGraphStudyScenario("correlated_clean", correlation=0.75),
        GaugeGraphStudyScenario("highly_correlated_clean", correlation=0.95),
        GaugeGraphStudyScenario(
            "correlated_mild_outliers",
            correlation=0.75,
            outlier_probability=1.0,
            outlier_translation=0.1,
        ),
        GaugeGraphStudyScenario(
            "correlated_strong_outliers",
            correlation=0.75,
            outlier_probability=1.0,
            outlier_translation=0.3,
        ),
        GaugeGraphStudyScenario(
            "highly_correlated_strong_outliers",
            correlation=0.95,
            outlier_probability=1.0,
            outlier_translation=0.3,
        ),
    )


def test_tiny_cycle_guard_study_writes_auditable_decision(tmp_path: Path) -> None:
    report = run_cycle_guard_monte_carlo(
        tmp_path,
        scenarios=_tiny_scenarios(),
        calibration_trials=3,
        target_trials_per_scenario=1,
        calibration_seed=43,
        target_seed=71,
        threshold_quantile=1.0,
        representative_radius=1.0,
        num_frames=12,
        height=3,
        width=4,
        window_size=7,
        overlap=5,
        bootstrap_resamples=10,
        source_revision="a" * 40,
    )

    assert report["schema_name"] == CYCLE_GUARD_MONTE_CARLO_SCHEMA
    assert report["configuration"]["target_trials_per_scenario"] == 1
    assert report["calibration"]["raw_cycle_threshold"] > 0.0
    assert report["calibration"]["uncertainty_normalized_cycle_threshold"] > 0.0
    assert len(report["trials"]) == 6 * 4
    assert len(report["aggregate"]) == 6 * 4
    decision = report["decision"]
    assert set(decision["criteria"]) == {
        "strong_detection_at_least_0_95",
        "strong_detection_noninferior_to_raw_by_0_05",
        "mild_detection_at_least_0_90",
        "mild_detection_noninferior_to_raw_by_0_05",
        "worst_clean_false_fallback_at_most_0_10",
        "worst_clean_false_fallback_halved",
    }
    assert isinstance(decision["overall_passed"], bool)

    json_path = tmp_path / "cycle_guard_monte_carlo.json"
    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    body = {key: value for key, value in loaded.items() if key != "report_id"}
    expected = hashlib.sha256(
        json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    assert loaded["report_id"] == expected
    assert (tmp_path / "cycle_guard_monte_carlo.csv").exists()
    assert (tmp_path / "cycle_guard_monte_carlo_trials.csv").exists()
    assert (tmp_path / "cycle_guard_monte_carlo.md").exists()
    assert (tmp_path / "SHA256SUMS").exists()
