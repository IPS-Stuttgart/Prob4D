from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "ci" / "build_ecosystem_release_capsule.py"
SPEC = importlib.util.spec_from_file_location("build_ecosystem_release_capsule", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _log() -> str:
    return "\n".join(
        [
            "ordinary build output",
            f"{'a' * 64}  /tmp/wheelhouse/prob4d-0.3.1-py3-none-any.whl",
            f"{'b' * 64}  /tmp/wheelhouse/bayesian_phystwin-0.4.0-py3-none-any.whl",
            f"{'c' * 64}  /tmp/wheelhouse/causal4d-0.4.1-py3-none-any.whl",
            "3 passed",
        ]
    )


def _capsule() -> dict[str, object]:
    return MODULE.build_capsule(
        golden_path_log=_log(),
        prob4d_revision="1" * 40,
        bayesian_phystwin_revision="2" * 40,
        causal4d_revision="3" * 40,
        python_version="3.12.11",
        runner_os="Linux",
        run_id=123,
        run_attempt=2,
        run_url="https://github.com/IPS-Stuttgart/Prob4D/actions/runs/123",
    )


def test_parse_wheel_hashes_is_ordered_and_exact() -> None:
    hashes = MODULE.parse_wheel_hashes(_log())
    assert tuple(hashes) == ("prob4d", "bayesian_phystwin", "causal4d")
    assert hashes["prob4d"]["sha256"] == "a" * 64
    assert hashes["bayesian_phystwin"]["filename"].startswith("bayesian_phystwin-")


def test_parse_wheel_hashes_rejects_missing_or_conflicting_wheels() -> None:
    with pytest.raises(ValueError, match="omitted wheel hashes"):
        MODULE.parse_wheel_hashes(f"{'a' * 64}  prob4d-0.3.1-py3-none-any.whl")

    conflicting = _log() + f"\n{'d' * 64}  /tmp/other/prob4d-0.3.2-py3-none-any.whl\n"
    with pytest.raises(ValueError, match="conflicting prob4d wheels"):
        MODULE.parse_wheel_hashes(conflicting)


def test_capsule_round_trip_and_identity_are_deterministic(tmp_path: Path) -> None:
    first = _capsule()
    second = _capsule()
    assert first == second
    assert first["capsule_id"] == second["capsule_id"]
    assert MODULE.validate_capsule(first) == first

    path = tmp_path / "capsule.json"
    MODULE._atomic_write_json(path, first)
    restored = json.loads(path.read_text(encoding="utf-8"))
    assert MODULE.validate_capsule(restored) == first


def test_capsule_rejects_tampering_even_when_structure_remains_valid() -> None:
    capsule = _capsule()
    tampered = copy.deepcopy(capsule)
    tampered["wheels"]["prob4d"]["sha256"] = "d" * 64
    with pytest.raises(ValueError, match="capsule_id"):
        MODULE.validate_capsule(tampered)


def test_capsule_rejects_coercion_aliases_and_unknown_fields() -> None:
    capsule = _capsule()
    aliased = copy.deepcopy(capsule)
    aliased["schema_version"] = True
    with pytest.raises(ValueError, match="schema_version"):
        MODULE.validate_capsule(aliased)

    extended = copy.deepcopy(capsule)
    extended["extra"] = "not allowed"
    with pytest.raises(ValueError, match="noncanonical keys"):
        MODULE.validate_capsule(extended)
