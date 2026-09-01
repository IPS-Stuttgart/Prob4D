from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts/science/run_dot_rope_query_selective_source_support_v2_recovery.py"
WORKFLOW = (
    ROOT / ".github/workflows/dot-rope-query-selective-source-support-v2-gpuserver6000-recovery.yml"
)


def test_wrapper_is_syntax_valid_and_hash_pins_frozen_v2() -> None:
    text = WRAPPER.read_text(encoding="utf-8")
    ast.parse(text)
    assert "7d63f9d3b718f53b15036e55d538add2560e855a" in text
    assert 'destination / "frames"' in text
    assert 'command != "runtime-smoke"' in text
    assert "original_run_base_provider(args, command)" in text


def test_recovery_is_request_only_and_retires_unopened_predecessor() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "pull_request:" in text
    assert "push:" in text
    assert "pull_request_target:" not in text
    assert "dot_rope_query_selective_source_support_v2_gpuserver6000.json" in text
    assert 'ORIGINAL_RUN_ID: "33522102387"' in text
    assert "Seal marker-free R11-R20 CUT3R predictions" in text
    assert "provider may have started; recovery is forbidden" in text
    assert "actions: write" in text
    assert "/cancel" in text


def test_recovery_preserves_source_only_data_boundary() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "R11-20.zip" in text
    assert "23ce3e7067465d3edabe20b4c7cfa388" in text
    assert "R21-30.zip" not in text
    assert "R31-R70 remain unopened" in text
    assert "source RMSE/NLL/proper score" in text
    assert "BayesianPhysTwin" in text
    assert "Causal4D" in text


def test_recovery_uses_reviewed_runtime_repairs_on_gpuserver6000() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "runs-on: [self-hosted, gpuserver6000]" in text
    assert 'test "$RUNNER_NAME" = "workstation2"' in text
    assert "run_dot_rope_query_selective_source_support_v2_recovery.py" in text
    assert "9778ec434a6e2d9ae8be162295d60e06d86bf0fbadbb66d516eb46c666ab547d" in text
    assert "tokens.scalar_type()" in text
    assert "compute_89,code=sm_89" in text
    assert "weights_only=False" in text
    assert "environment: trusted-self-hosted-validation" in text
    assert "secrets." not in text
