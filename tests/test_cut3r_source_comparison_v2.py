from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/ci/cut3r_source_comparison_v2.py"


def _module():
    spec = importlib.util.spec_from_file_location("cut3r_source_comparison_v2", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_current_repository_satisfies_v2_zero_information_contract() -> None:
    module = _module()

    details = module.validate_repository(ROOT)

    assert details["predecessor_smoke_id"] == module.PREDECESSOR_SMOKE_ID
    assert details["predecessor_plan_id"] == module.PREDECESSOR_PLAN_ID
    assert details["source_freeze_id"] == module.SOURCE_FREEZE_ID
    assert details["zero_information_predecessor"] is True
    assert details["localized_repair_present"] is True


def test_predecessor_smoke_rejects_any_opened_information() -> None:
    module = _module()
    smoke = module._load_json(
        ROOT / module.PREDECESSOR_SMOKE_PATH,
        label="predecessor smoke",
    )
    drift = copy.deepcopy(smoke)
    drift["information_boundary"]["source_rgb_frames_decoded"] = True

    with pytest.raises(module.ContractError, match="zero-information"):
        module.validate_predecessor_smoke(drift)


def test_predecessor_smoke_rejects_changed_failure_localization() -> None:
    module = _module()
    smoke = module._load_json(
        ROOT / module.PREDECESSOR_SMOKE_PATH,
        label="predecessor smoke",
    )
    drift = copy.deepcopy(smoke)
    drift["failure"]["terminal_stage"] = "provider-inference"

    with pytest.raises(module.ContractError, match="terminal_stage mismatch"):
        module.validate_predecessor_smoke(drift)


def test_v2_runner_contract_requires_internal_dust3r_package_path() -> None:
    module = _module()
    runner = (ROOT / "scripts/science/run_cut3r_source_comparison.py").read_text(
        encoding="utf-8"
    )

    module.validate_runner_repair(runner)

    with pytest.raises(module.ContractError, match="repair is incomplete"):
        module.validate_runner_repair(
            runner.replace(
                'for candidate in (checkout, checkout / "src"):',
                "for candidate in (checkout,):",
            )
        )


def test_dispatch_command_binds_exact_predecessor_and_source_freeze() -> None:
    module = _module()

    assert module.DISPATCH_COMMAND == (
        "/prob4d-run-cut3r-source-comparison-v2 "
        f"{module.PREDECESSOR_SMOKE_ID} "
        f"{module.PREDECESSOR_PLAN_ID} "
        f"{module.SOURCE_FREEZE_ID}"
    )
