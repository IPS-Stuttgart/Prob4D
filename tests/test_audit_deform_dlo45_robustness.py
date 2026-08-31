from __future__ import annotations

import importlib.util
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT.parents[0] / "scripts/science/audit_deform_dlo45_robustness.py"


def module():
    spec = importlib.util.spec_from_file_location("dlo45_robustness", SCRIPT)
    assert spec is not None and spec.loader is not None
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_exact_sign_tail_for_28_wins() -> None:
    assert module().sign_tail(28, 28) == math.ldexp(1, -28)


def test_leave_one_out_is_stable() -> None:
    result = module().leave_one_out([1.0, 2.0, 3.0])
    assert result == {"minimum": 1.5, "maximum": 2.5}


def _method(rmse: float, nll: float, coverage: float, nees: float) -> dict[str, float]:
    return {
        "rmse_mm": rmse,
        "mean_gaussian_nll": nll,
        "empirical_90pct_coverage": coverage,
        "normalized_nees": nees,
    }


def test_audit_preserves_group_level_positive_and_negative_findings() -> None:
    value = module()
    groups = {}
    for family in ("DLO4", "DLO5"):
        for index in range(14):
            groups[f"{family}/{index}.pkl"] = {
                "segment_centroid": {
                    "physical_fallback": _method(15.0, -8.0, 0.9, 1.0),
                    "query_aware": _method(1.0, -15.0, 0.75, 2.0),
                },
                "off_axis_probe": {
                    "query_aware": {
                        **_method(20.0, -7.0, 0.9, 1.0),
                        "accepted_fraction": 0.0,
                        "exact_fallback_fraction": 1.0,
                    },
                    "observable_subspace_unconditional": {
                        "harmful_fraction_vs_fallback": 0.1
                    },
                    "invalid_full_rank_completion": {
                        "harmful_fraction_vs_fallback": 0.2
                    },
                },
            }
    result = {
        "schema": value.RESULT_SCHEMA,
        "result_id": "result",
        "request_id": "request",
        "information_boundary": {"post_open_retuning_permitted": False},
        "per_group_results": groups,
    }
    protocol = {
        "protocol_id": "audit",
        "analysis": {
            "bootstrap_replicates": 20,
            "family_counts": {"DLO4": 14, "DLO5": 14},
            "trim_fraction": 0.2,
        },
        "source_result": {"result_id": "result", "request_id": "request"},
        "claim_boundary": [],
    }
    output = value.audit(result, protocol)
    assert output["centroid_gain"]["rmse_improvement_mm"]["wins"] == 28
    assert output["centroid_gain"]["nll_improvement"]["wins"] == 28
    assert output["off_axis_controls"]["minimum_exact_fallback_fraction"] == 1.0
    assert output["calibration_limit"]["coverage_below_90pct_files"] == 28
