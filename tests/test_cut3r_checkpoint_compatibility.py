from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from prob4d import cut3r_runtime_contract as contract


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_pinned_runtime_and_checkpoint_enable_legacy_torch_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "cut3r.pth"
    checkpoint.write_bytes(b"frozen trusted CUT3R checkpoint")
    runtime_id = "a" * 64
    monkeypatch.setattr(contract, "_PINNED_NATIVE_ROPE_ARTIFACT_ID", runtime_id)
    monkeypatch.setattr(contract, "_PINNED_CUT3R_CHECKPOINT_SHA256", _sha256(checkpoint))
    monkeypatch.setenv("CUT3R_RUNTIME_CHECKPOINT", str(checkpoint))
    monkeypatch.delenv("TORCH_FORCE_WEIGHTS_ONLY_LOAD", raising=False)
    monkeypatch.delenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", raising=False)

    contract._activate_pinned_checkpoint_compatibility(runtime_id)

    assert os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] == "1"


def test_unregistered_runtime_does_not_change_torch_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "cut3r.pth"
    checkpoint.write_bytes(b"frozen trusted CUT3R checkpoint")
    monkeypatch.setenv("CUT3R_RUNTIME_CHECKPOINT", str(checkpoint))
    monkeypatch.delenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", raising=False)

    contract._activate_pinned_checkpoint_compatibility("b" * 64)

    assert "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD" not in os.environ


def test_pinned_runtime_rejects_checkpoint_hash_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "cut3r.pth"
    checkpoint.write_bytes(b"different checkpoint")
    runtime_id = "c" * 64
    monkeypatch.setattr(contract, "_PINNED_NATIVE_ROPE_ARTIFACT_ID", runtime_id)
    monkeypatch.setattr(contract, "_PINNED_CUT3R_CHECKPOINT_SHA256", "0" * 64)
    monkeypatch.setenv("CUT3R_RUNTIME_CHECKPOINT", str(checkpoint))

    with pytest.raises(
        contract.Cut3RRuntimeContractError,
        match="checkpoint differs from the frozen SHA-256 identity",
    ):
        contract._activate_pinned_checkpoint_compatibility(runtime_id)


def test_conflicting_weights_only_policy_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "cut3r.pth"
    checkpoint.write_bytes(b"frozen trusted CUT3R checkpoint")
    runtime_id = "d" * 64
    monkeypatch.setattr(contract, "_PINNED_NATIVE_ROPE_ARTIFACT_ID", runtime_id)
    monkeypatch.setattr(contract, "_PINNED_CUT3R_CHECKPOINT_SHA256", _sha256(checkpoint))
    monkeypatch.setenv("CUT3R_RUNTIME_CHECKPOINT", str(checkpoint))
    monkeypatch.setenv("TORCH_FORCE_WEIGHTS_ONLY_LOAD", "true")

    with pytest.raises(
        contract.Cut3RRuntimeContractError,
        match="conflicting PyTorch weights-only checkpoint policy",
    ):
        contract._activate_pinned_checkpoint_compatibility(runtime_id)
