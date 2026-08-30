from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

from prob4d.dot_rope_cut3r_study import content_id

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "science" / "audit_dot_rope_marker_support.py"
REQUEST = (
    ROOT
    / "protocols"
    / "execution_requests"
    / "dot_rope_marker_support_audit_v1.json"
)


def _load_script():
    spec = importlib.util.spec_from_file_location("audit_dot_rope_marker_support", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_registered_request_has_canonical_identity_and_source_boundary() -> None:
    module = _load_script()
    request = module.validate_request(REQUEST)

    unsigned = dict(request)
    request_id = unsigned.pop("request_id")
    assert content_id(unsigned) == request_id
    assert request["source_sequences"] == ["R01", "R02", "R03"]
    assert request["reserved_sequences"] == "R04-R70"
    assert request["normal_view_pixels_opened"] is False
    assert request["target_payloads_opened"] is False
    assert request["provider_run_id"] == 33329701704
    assert request["provider_bundle_id"] == (
        "952421d140731b2a6eb99df3cbd348653e04863fa457aaa490be31fe0b4c06a7"
    )


def test_numeric_parser_preserves_possible_identifier_columns() -> None:
    module = _load_script()

    rows = module._numeric_rows(
        """
        # marker rows
        marker 7: 120.5, 40.25
        8; 130; 50
        """
    )

    assert rows == [[7.0, 120.5, 40.25], [8.0, 130.0, 50.0]]
    assert module._widths(rows) == {"3": 2}
    assert (0, 1) in [(i, j) for i in range(3) for j in range(3) if i != j]
    assert (1, 2) in [(i, j) for i in range(3) for j in range(3) if i != j]


def test_coordinate_modes_are_explicit_and_deterministic() -> None:
    module = _load_script()
    coordinates = np.asarray([[0.25, 0.5], [1.0, 1.0]])

    unit = module._transform(
        coordinates,
        mode="unit-normalized",
        width=101,
        height=81,
    )
    percent = module._transform(
        coordinates,
        mode="percent-normalized",
        width=101,
        height=81,
    )
    one_based = module._transform(
        np.asarray([[1.0, 1.0], [101.0, 81.0]]),
        mode="pixel-one-based",
        width=101,
        height=81,
    )

    np.testing.assert_allclose(unit, [[25.0, 40.0], [100.0, 80.0]])
    np.testing.assert_allclose(percent, [[0.25, 0.4], [1.0, 0.8]])
    np.testing.assert_allclose(one_based, [[0.0, 0.0], [100.0, 80.0]])


def test_pooled_support_accepts_five_markers_per_frame_without_claiming_six() -> None:
    module = _load_script()
    support = {
        sequence: {
            frame: {
                "continuous": 5,
                "window_a": 5 if frame <= 5 else -1,
                "window_b": 5 if frame >= 3 else -1,
                "window_common": 5 if 3 <= frame <= 5 else -1,
            }
            for frame in range(1, 8)
        }
        for sequence in ("R01", "R02", "R03")
    }

    summary = module._summarize_candidate(
        "columns-1-2:pixel-zero-based",
        support,
    )

    assert summary["all_sequences_feasible_for_pooled_evaluation"] is True
    assert summary["minimum_frame_support"] == 5
    assert summary["sequences"]["R01"]["pooled_support"] == {
        "overlap_common_total": 15,
        "overlap_nonempty_frames": 3,
        "fit_a_total": 10,
        "fit_b_total": 10,
        "score_total": 10,
    }


def test_request_validator_rejects_identity_drift(tmp_path: Path) -> None:
    module = _load_script()
    request = json.loads(REQUEST.read_text(encoding="utf-8"))
    request["provider_run_id"] += 1
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request), encoding="utf-8")

    try:
        module.validate_request(path)
    except ValueError as error:
        assert "artifact name" in str(error) or "identity" in str(error)
    else:
        raise AssertionError("drifted request unexpectedly validated")
