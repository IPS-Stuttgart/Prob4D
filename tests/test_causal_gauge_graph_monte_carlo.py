from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from prob4d.alignment import AlignmentResult, WindowAlignment
from prob4d.causal_gauge_graph_monte_carlo import (
    GAUGE_GRAPH_MONTE_CARLO_SCHEMA,
    GaugeGraphStudyScenario,
    _inject_inconsistent_skip_edge,
    _representative_displacement,
    run_gauge_graph_monte_carlo,
)
from prob4d.sim3 import Sim3


def _alignment(reference_id: str, moving_id: str, translation: float) -> WindowAlignment:
    return WindowAlignment(
        reference_id=reference_id,
        moving_id=moving_id,
        common_frames=np.array([1, 2], dtype=np.int64),
        result=AlignmentResult(
            transform=Sim3(translation=np.array([translation, 0.0, 0.0])),
            covariance=np.eye(7) * 1e-4,
            residual_rms=0.01,
            inlier_fraction=1.0,
            num_correspondences=20,
        ),
    )


def test_scenario_validation_rejects_unscaled_outlier_probability() -> None:
    with pytest.raises(ValueError, match="positive outlier probability"):
        GaugeGraphStudyScenario(
            "invalid",
            correlation=0.5,
            outlier_probability=0.2,
        )


def test_representative_displacement_is_zero_for_equal_transforms() -> None:
    transform = Sim3.from_vector(
        np.array([0.02, 0.01, -0.02, 0.03, 0.4, -0.2, 0.1])
    )

    assert _representative_displacement(transform, transform, radius=1.0) == 0.0


def test_outlier_injection_changes_one_cycle_supported_skip_edge() -> None:
    alignments = (
        _alignment("w0", "w1", 1.0),
        _alignment("w1", "w2", 1.0),
        _alignment("w0", "w2", 2.0),
    )

    changed, injected, edge_id = _inject_inconsistent_skip_edge(
        alignments,
        ("w0", "w1", "w2"),
        generator=np.random.default_rng(7),
        probability=1.0,
        translation_magnitude=0.3,
    )

    assert injected is True
    assert edge_id == "w0<-w2"
    np.testing.assert_array_equal(
        changed[0].result.transform.as_vector(),
        alignments[0].result.transform.as_vector(),
    )
    np.testing.assert_array_equal(
        changed[1].result.transform.as_vector(),
        alignments[1].result.transform.as_vector(),
    )
    assert not np.array_equal(
        changed[2].result.transform.as_vector(),
        alignments[2].result.transform.as_vector(),
    )


def test_tiny_study_writes_content_addressed_outputs(tmp_path: Path) -> None:
    scenarios = (
        GaugeGraphStudyScenario("clean", correlation=0.5),
        GaugeGraphStudyScenario(
            "outlier",
            correlation=0.5,
            outlier_probability=1.0,
            outlier_translation=0.3,
        ),
    )

    report = run_gauge_graph_monte_carlo(
        tmp_path,
        scenarios=scenarios,
        calibration_trials=1,
        target_trials_per_scenario=1,
        calibration_seed=17,
        target_seed=29,
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

    assert report["schema_name"] == GAUGE_GRAPH_MONTE_CARLO_SCHEMA
    assert report["configuration"]["target_trials_per_scenario"] == 1
    assert len(report["aggregate"]) == 2 * 4
    assert len(report["trials"]) == 2 * 4
    assert sum(
        bool(record["outlier_injected"])
        for record in report["trials"]
        if record["scenario_id"] == "outlier"
    ) == 4
    json_path = tmp_path / "gauge_graph_monte_carlo.json"
    markdown_path = tmp_path / "gauge_graph_monte_carlo.md"
    assert json_path.exists()
    assert markdown_path.exists()
    assert (tmp_path / "gauge_graph_monte_carlo.csv").exists()
    assert (tmp_path / "gauge_graph_monte_carlo_trials.csv").exists()
    assert (tmp_path / "SHA256SUMS").exists()
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
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "candidate - production tree" in markdown
    assert "Controlled synthetic estimator mechanics only" in loaded["claim_boundary"]
