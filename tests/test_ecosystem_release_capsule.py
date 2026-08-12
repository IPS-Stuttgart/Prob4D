from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "ci" / "build_ecosystem_release_capsule.py"
SPEC = importlib.util.spec_from_file_location("build_ecosystem_release_capsule", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

_REVISIONS = {
    "prob4d": "1" * 40,
    "bayesian_phystwin": "2" * 40,
    "causal4d": "3" * 40,
}
_WHEEL_DIGESTS = {
    "prob4d": "a" * 64,
    "bayesian_phystwin": "b" * 64,
    "causal4d": "c" * 64,
}
_PACKAGE_VERSIONS = {
    "prob4d": "0.4.1",
    "bayesian_phystwin": "0.4.0",
    "causal4d": "0.4.1",
}


def _canonical(value: dict[str, Any], *, newline: bool = False) -> bytes:
    text = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    if newline:
        text += "\n"
    return text.encode("utf-8")


def _content_addressed(
    descriptor: dict[str, Any],
    *,
    id_field: str,
    newline: bool = False,
) -> dict[str, Any]:
    payload = copy.deepcopy(descriptor)
    payload[id_field] = hashlib.sha256(_canonical(payload, newline=newline)).hexdigest()
    return payload


def _log() -> str:
    return "\n".join(
        [
            "ordinary build output",
            f"{'a' * 64}  /tmp/wheelhouse/prob4d-0.4.1-py3-none-any.whl",
            f"{'b' * 64}  /tmp/wheelhouse/bayesian_phystwin-0.4.0-py3-none-any.whl",
            f"{'c' * 64}  /tmp/wheelhouse/causal4d-0.4.1-py3-none-any.whl",
            "3 passed",
        ]
    )


def _array_identity(seed: bytes) -> dict[str, Any]:
    descriptor = {
        "dtype": "<f4",
        "shape": [1],
        "nbytes": 4,
        "payload_sha256": hashlib.sha256(seed).hexdigest(),
    }
    return _content_addressed(descriptor, id_field="array_id")


def _run_manifest() -> dict[str, Any]:
    scientific = {
        "schema_name": "bayesian_phystwin.run_manifest",
        "schema_version": 2,
        "run_id": "three-repository-installed-wheel-golden-path",
        "repository": "FlorianPfaff/Bayesian-PhysTwin",
        "revision": _REVISIONS["bayesian_phystwin"],
        "dirty": False,
        "related_repositories": [],
        "command": ["python", "-I", "-m", "pytest"],
        "classification": "infrastructure",
        "statistical_unit": "deterministic three-repository fixture",
        "information_boundary": {"future_prediction_payloads_opened": 0},
        "configuration": {"provider_api_version": 2},
        "seeds": [],
        "inputs": [],
        "outputs": [],
        "package_versions": _PACKAGE_VERSIONS,
        "runtime_environment": {"wheel_sha256": _WHEEL_DIGESTS},
        "claim_ids": ["bpt.infrastructure.three_repository_golden_path"],
        "method_freeze_id": "three-repository-installed-wheel-v1",
        "protocol_id": "three-repository-installed-wheel-v1",
        "split_id": "deterministic-fixture-v1",
        "baseline_id": "exact-zero-update-fallback-v1",
    }
    evidence_fingerprint = hashlib.sha256(_canonical(scientific)).hexdigest()
    descriptor = {
        **scientific,
        "evidence_fingerprint": evidence_fingerprint,
        "created_utc": "2026-08-12T00:00:00+00:00",
        "notes": "",
    }
    return _content_addressed(descriptor, id_field="manifest_id")


def _selection(
    *,
    decision: str,
    run_manifest: dict[str, Any],
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    accepted = decision == "accepted"
    descriptor = {
        "schema_name": "bayesian_phystwin.three_repository_golden_path_selection",
        "schema_version": 1,
        "case_id": "three-repository-golden-path",
        "protocol_id": "three-repository-installed-wheel-v1",
        "decision": decision,
        "reason": (
            "candidate-accepted"
            if accepted
            else "registered-guard-rejected-exact-baseline-fallback"
        ),
        "inference_admissible": True,
        "regret_guard_present": True,
        "regret_guard_accepted": accepted,
        "candidate_accepted": accepted,
        "observation_artifact_id": "4" * 64,
        "twin_belief_id": "5" * 64,
        "physical_posterior_id": "6" * 64,
        "provider_manifest_id": "7" * 64,
        "run_manifest_id": run_manifest["manifest_id"],
        "evidence_fingerprint": run_manifest["evidence_fingerprint"],
        "repository_revisions": _REVISIONS,
        "wheel_sha256": _WHEEL_DIGESTS,
        "package_versions": _PACKAGE_VERSIONS,
        "baseline_identity": baseline,
        "candidate_identity": candidate,
        "selected_identity": candidate if accepted else baseline,
        "exact_fallback_identity": None if accepted else baseline["array_id"],
        "metadata": {"claim_authorized": False},
    }
    return _content_addressed(descriptor, id_field="artifact_id")


def _public_api_manifest() -> dict[str, Any]:
    descriptor = {
        "schema": "prob4d.public-api-manifest",
        "schema_version": 1,
        "package": {
            "name": "prob4d",
            "version": _PACKAGE_VERSIONS["prob4d"],
            "project_id": "github-repository-id:1295794737",
        },
        "surfaces": {
            "compatibility_root": {
                "module": "prob4d",
                "surface_version": 1,
                "loading": "lazy-compatibility-shim-v1",
                "exports": ["Sim3", "__version__"],
            },
            "api_v1": {
                "module": "prob4d.api.v1",
                "api_version": 1,
                "provider_api_version": 1,
                "exports": ["API_VERSION"],
            },
            "api_v2": {
                "module": "prob4d.api.v2",
                "api_version": 2,
                "provider_api_version": 2,
                "provider_factor_api_version": 2,
                "exports": ["API_VERSION"],
            },
        },
        "claim_boundary": MODULE._PUBLIC_API_CLAIM_BOUNDARY,
    }
    return _content_addressed(descriptor, id_field="manifest_id", newline=True)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _evidence_root(tmp_path: Path) -> Path:
    root = tmp_path / "evidence"
    root.mkdir(parents=True)
    run_manifest = _run_manifest()
    baseline = _array_identity(b"baseline")
    candidate = _array_identity(b"candidate")
    accepted = _selection(
        decision="accepted",
        run_manifest=run_manifest,
        baseline=baseline,
        candidate=candidate,
    )
    rejected = _selection(
        decision="rejected",
        run_manifest=run_manifest,
        baseline=baseline,
        candidate=candidate,
    )
    bundle = _content_addressed(
        {
            "schema_name": "bayesian_phystwin.three_repository_golden_path_bundle",
            "schema_version": 1,
            "accepted": accepted,
            "rejected": rejected,
        },
        id_field="bundle_id",
    )
    _write_json(root / "accepted-selection.json", accepted)
    _write_json(root / "rejected-selection.json", rejected)
    _write_json(root / "golden-path-bundle.json", bundle)
    _write_json(root / "run-manifest-v2.json", run_manifest)
    _write_json(root / "public-api-manifest.json", _public_api_manifest())
    for name in (
        "exact-prob4d-observation.npz",
        "lineage-bound-physical-posterior.npz",
        "lineage-bound-twin-belief.npz",
        "profile-bound.npz",
    ):
        (root / name).write_bytes(f"fixture:{name}\n".encode())
    return root


def _capsule(tmp_path: Path) -> dict[str, object]:
    return MODULE.build_capsule(
        golden_path_log=_log(),
        evidence_root=_evidence_root(tmp_path),
        prob4d_revision=_REVISIONS["prob4d"],
        bayesian_phystwin_revision=_REVISIONS["bayesian_phystwin"],
        causal4d_revision=_REVISIONS["causal4d"],
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
        MODULE.parse_wheel_hashes(f"{'a' * 64}  prob4d-0.4.1-py3-none-any.whl")

    conflicting = _log() + f"\n{'d' * 64}  /tmp/other/prob4d-0.4.2-py3-none-any.whl\n"
    with pytest.raises(ValueError, match="conflicting prob4d wheels"):
        MODULE.parse_wheel_hashes(conflicting)


def test_capsule_round_trip_and_identity_are_deterministic(tmp_path: Path) -> None:
    first = _capsule(tmp_path / "first")
    second = _capsule(tmp_path / "second")
    assert first == second
    assert first["capsule_id"] == second["capsule_id"]
    assert MODULE.validate_capsule(first) == first
    assert first["schema_version"] == 2
    assert first["evidence"]["public_api_manifest_id"] == (
        _public_api_manifest()["manifest_id"]
    )

    path = tmp_path / "capsule.json"
    MODULE._atomic_write_json(path, first)
    restored = json.loads(path.read_text(encoding="utf-8"))
    assert MODULE.validate_capsule(restored) == first


def test_capsule_rejects_tampering_even_when_structure_remains_valid(
    tmp_path: Path,
) -> None:
    capsule = _capsule(tmp_path)
    tampered = copy.deepcopy(capsule)
    tampered["wheels"]["prob4d"]["sha256"] = "d" * 64
    with pytest.raises(ValueError, match="capsule_id"):
        MODULE.validate_capsule(tampered)


def test_capsule_rejects_coercion_aliases_and_unknown_fields(tmp_path: Path) -> None:
    capsule = _capsule(tmp_path)
    aliased = copy.deepcopy(capsule)
    aliased["schema_version"] = True
    with pytest.raises(ValueError, match="schema_version"):
        MODULE.validate_capsule(aliased)

    extended = copy.deepcopy(capsule)
    extended["extra"] = "not allowed"
    with pytest.raises(ValueError, match="noncanonical keys"):
        MODULE.validate_capsule(extended)


def test_builder_rejects_bundle_selection_disagreement(tmp_path: Path) -> None:
    root = _evidence_root(tmp_path)
    accepted_path = root / "accepted-selection.json"
    accepted = json.loads(accepted_path.read_text(encoding="utf-8"))
    accepted["metadata"]["changed"] = True
    accepted = _content_addressed(
        {key: value for key, value in accepted.items() if key != "artifact_id"},
        id_field="artifact_id",
    )
    _write_json(accepted_path, accepted)

    with pytest.raises(ValueError, match="bundle and selection files disagree"):
        MODULE.build_capsule(
            golden_path_log=_log(),
            evidence_root=root,
            prob4d_revision=_REVISIONS["prob4d"],
            bayesian_phystwin_revision=_REVISIONS["bayesian_phystwin"],
            causal4d_revision=_REVISIONS["causal4d"],
            python_version="3.12.11",
            runner_os="Linux",
            run_id=123,
            run_attempt=1,
            run_url="https://example.invalid/run/123",
        )


def test_builder_rejects_public_manifest_tampering(tmp_path: Path) -> None:
    root = _evidence_root(tmp_path)
    path = root / "public-api-manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["surfaces"]["api_v2"]["provider_api_version"] = 3
    manifest = _content_addressed(
        {key: value for key, value in manifest.items() if key != "manifest_id"},
        id_field="manifest_id",
        newline=True,
    )
    _write_json(path, manifest)

    with pytest.raises(ValueError, match="provider versions changed"):
        MODULE.build_capsule(
            golden_path_log=_log(),
            evidence_root=root,
            prob4d_revision=_REVISIONS["prob4d"],
            bayesian_phystwin_revision=_REVISIONS["bayesian_phystwin"],
            causal4d_revision=_REVISIONS["causal4d"],
            python_version="3.12.11",
            runner_os="Linux",
            run_id=123,
            run_attempt=1,
            run_url="https://example.invalid/run/123",
        )


def test_verify_capsule_evidence_detects_changed_artifact_bytes(tmp_path: Path) -> None:
    root = _evidence_root(tmp_path)
    capsule = MODULE.build_capsule(
        golden_path_log=_log(),
        evidence_root=root,
        prob4d_revision=_REVISIONS["prob4d"],
        bayesian_phystwin_revision=_REVISIONS["bayesian_phystwin"],
        causal4d_revision=_REVISIONS["causal4d"],
        python_version="3.12.11",
        runner_os="Linux",
        run_id=123,
        run_attempt=1,
        run_url="https://example.invalid/run/123",
    )
    (root / "profile-bound.npz").write_bytes(b"tampered\n")

    with pytest.raises(ValueError, match="does not match the supplied execution bytes"):
        MODULE.verify_capsule_evidence(
            capsule,
            golden_path_log=_log(),
            evidence_root=root,
        )


def test_builder_rejects_inexact_fallback_identity(tmp_path: Path) -> None:
    root = _evidence_root(tmp_path)
    rejected_path = root / "rejected-selection.json"
    rejected = json.loads(rejected_path.read_text(encoding="utf-8"))
    rejected["exact_fallback_identity"] = "f" * 64
    rejected = _content_addressed(
        {key: value for key, value in rejected.items() if key != "artifact_id"},
        id_field="artifact_id",
    )
    _write_json(rejected_path, rejected)

    with pytest.raises(ValueError, match="exact baseline fallback"):
        MODULE.build_capsule(
            golden_path_log=_log(),
            evidence_root=root,
            prob4d_revision=_REVISIONS["prob4d"],
            bayesian_phystwin_revision=_REVISIONS["bayesian_phystwin"],
            causal4d_revision=_REVISIONS["causal4d"],
            python_version="3.12.11",
            runner_os="Linux",
            run_id=123,
            run_attempt=1,
            run_url="https://example.invalid/run/123",
        )


def test_builder_rejects_forged_run_evidence_fingerprint(tmp_path: Path) -> None:
    root = _evidence_root(tmp_path)
    path = root / "run-manifest-v2.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["evidence_fingerprint"] = "f" * 64
    manifest = _content_addressed(
        {key: value for key, value in manifest.items() if key != "manifest_id"},
        id_field="manifest_id",
    )
    _write_json(path, manifest)

    with pytest.raises(ValueError, match="evidence fingerprint"):
        MODULE.build_capsule(
            golden_path_log=_log(),
            evidence_root=root,
            prob4d_revision=_REVISIONS["prob4d"],
            bayesian_phystwin_revision=_REVISIONS["bayesian_phystwin"],
            causal4d_revision=_REVISIONS["causal4d"],
            python_version="3.12.11",
            runner_os="Linux",
            run_id=123,
            run_attempt=1,
            run_url="https://example.invalid/run/123",
        )
