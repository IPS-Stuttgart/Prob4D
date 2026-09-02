from __future__ import annotations

import pytest

from prob4d.symmetry_complete_belief_study import run_study


def test_controlled_study_passes_and_keeps_scope_bounded() -> None:
    result = run_study(seed=20260902, cases=32)
    assert result["decision"] == "controlled-contract-passed"
    assert all(result["criteria"].values())
    assert result["invariant_update"]["maximum_conditional_l1_change"] == 0.0
    assert result["invariant_update"]["maximum_gauge_information_nats"] == pytest.approx(
        0.0,
        abs=1e-14,
    )
    assert "does not infer a symmetry" in result["protocol"]["claim_boundary"]


def test_study_rejects_invalid_case_count() -> None:
    with pytest.raises(ValueError, match="positive"):
        run_study(seed=1, cases=0)
