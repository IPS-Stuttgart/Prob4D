from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

import prob4d.provider_v2 as provider
from prob4d.gauge import GaugeEstimate
from prob4d.observation_factors import ObservationFactor, ObservationFactorBundle
from prob4d.provider_attestation import build_provider_attestation
from prob4d.provider_v2_factor_bundle import (
    CLAIM_BEARING_FACTOR_BUNDLE_SCHEMA,
    load_claim_bearing_observation_factor_bundle,
    seal_claim_bearing_observation_factor_bundle,
)
from prob4d.sim3 import Sim3

_REVISION = "a" * 40


def _runtime() -> dict[str, object]:
    return {
        "expected_revision": _REVISION,
        "observed_revision": _REVISION,
        "source": "source_checkout",
        "clean_checkout": True,
        "matched": True,
        "independently_verified": True,
    }


def _attestation() -> dict[str, object]:
    return build_provider_attestation(
        provider_manifest=provider.prob4d_provider_manifest(
            provider_revision=_REVISION
        ),
        provider_revision=_REVISION,
        export_mode="calibrated",
        calibration_compatibility_validated=True,
        calibration_artifact_ids={
            "gauge_artifact_id": "1" * 64,
            "point_artifact_id": "2" * 64,
        },
        covariance_root_mode="canonical_eigenspaces",
        composition_jacobian_mode="analytic",
        runtime_revision=_runtime(),
    )


def _lineage() -> dict[str, object]:
    return {
        "schema_version": 1,
        "producer": "Prob4D",
        "motioncrafter_lineage_schema_version": 1,
        "motioncrafter_windowing_model": "motioncrafter-sliding-window-v1",
        "source_product": "independently_decoded_overlap_windows",
        "causal_frame_stop_exclusive": 5,
        "admissibility_rule": "source_frame_max < causal_frame_stop_exclusive",
        "future_prediction_payloads_opened": 0,
        "selected_windows": [
            {
                "window_id": "window-0",
                "source_frame_start": 0,
                "source_frame_stop_exclusive": 3,
                "source_frame_max": 2,
                "frame_indices_sha256": "3" * 64,
                "payload_sha256": "4" * 64,
            },
            {
                "window_id": "window-1",
                "source_frame_start": 2,
                "source_frame_stop_exclusive": 5,
                "source_frame_max": 4,
                "frame_indices_sha256": "5" * 64,
                "payload_sha256": "6" * 64,
            },
        ],
        "source_artifact_sha256": "7" * 64,
        "source_digest_scope": "prefix-only fixture",
    }


def _factor(window_id: str, frame_index: int, point_id: int) -> ObservationFactor:
    return ObservationFactor(
        factor_id=f"{window_id}:frame-{frame_index}",
        frame_index=frame_index,
        view_id="camera-0",
        window_id=window_id,
        gauge_id=window_id,
        point_ids=np.asarray([point_id], dtype=np.int64),
        points_local_m=np.asarray([[0.0, 0.0, 1.0]], dtype=np.float64),
        valid_mask=np.asarray([True]),
        local_covariance_m2=np.asarray([np.eye(3) * 1e-3]),
        association_probability=np.asarray([0.9]),
        prior_reliability=np.asarray([0.8]),
        prior_nominal_probability=0.95,
        composite_weight=0.5,
        correlation_group_id=f"camera-0:frame-{frame_index}",
        causal_frame_stop=5,
    )


def _bundle(
    *,
    covariance_semantics: str = "joint-cross-window",
    second_frame: int = 3,
) -> ObservationFactorBundle:
    marginal = np.eye(7, dtype=np.float64) * 1e-4
    gauges = (
        GaugeEstimate("window-0", Sim3.identity(), marginal),
        GaugeEstimate("window-1", Sim3.identity(), marginal),
    )
    joint = np.zeros((14, 14), dtype=np.float64)
    joint[:7, :7] = marginal
    joint[7:, 7:] = marginal
    if covariance_semantics == "joint-cross-window":
        joint[:7, 7:] = np.eye(7) * 2e-5
        joint[7:, :7] = np.eye(7) * 2e-5
    return ObservationFactorBundle(
        sequence_id="sequence-a",
        case_id="case-a",
        stream_id="prob4d:factors:camera-0",
        factors=(
            _factor("window-0", 1, 7),
            _factor("window-1", second_frame, 8),
        ),
        gauges=gauges,
        source_repository="FlorianPfaff/Prob4D",
        source_revision=_REVISION,
        causal_frame_stop=5,
        joint_gauge_covariance=joint,
        gauge_covariance_semantics=covariance_semantics,
    )


def test_claim_bearing_factor_bundle_roundtrip_and_relocation(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    envelope_path = source / "claim.json"
    validated = seal_claim_bearing_observation_factor_bundle(
        _bundle(),
        envelope_path,
        causal_source_lineage=_lineage(),
        provider_attestation=_attestation(),
        metadata={"protocol": "prob4d-to-bpt-factor-v1"},
    )
    record = json.loads(envelope_path.read_text(encoding="utf-8"))

    assert record["schema"] == CLAIM_BEARING_FACTOR_BUNDLE_SCHEMA
    assert validated.bundle.gauge_covariance_semantics == "joint-cross-window"
    assert validated.envelope.factor_count == 2
    assert validated.envelope.observation_count == 2
    assert validated.gauge_calibration_id == "1" * 64
    assert validated.point_calibration_id == "2" * 64
    assert validated.envelope.runtime_revision_independently_verified is True
    with pytest.raises(TypeError, match="immutable"):
        validated.envelope.calibration_artifact_ids["gauge_artifact_id"] = "0" * 64

    moved = tmp_path / "moved"
    shutil.copytree(source, moved)
    relocated = load_claim_bearing_observation_factor_bundle(moved / "claim.json")
    assert relocated.artifact_id == validated.artifact_id


def test_claim_bearing_factor_bundle_rejects_manifest_tampering(
    tmp_path: Path,
) -> None:
    envelope_path = tmp_path / "claim.json"
    seal_claim_bearing_observation_factor_bundle(
        _bundle(),
        envelope_path,
        causal_source_lineage=_lineage(),
        provider_attestation=_attestation(),
    )
    manifest = tmp_path / "claim.bundle.json"
    manifest.write_text(manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="no longer matches its bundle manifest"):
        load_claim_bearing_observation_factor_bundle(envelope_path)


def test_claim_bearing_factor_bundle_rejects_payload_tampering(
    tmp_path: Path,
) -> None:
    envelope_path = tmp_path / "claim.json"
    seal_claim_bearing_observation_factor_bundle(
        _bundle(),
        envelope_path,
        causal_source_lineage=_lineage(),
        provider_attestation=_attestation(),
    )
    with (tmp_path / "claim.bundle.npz").open("ab") as stream:
        stream.write(b"tampered")

    with pytest.raises(ValueError, match="payload checksum mismatch"):
        load_claim_bearing_observation_factor_bundle(envelope_path)


def test_claim_bearing_factor_bundle_rejects_contradictory_provenance(
    tmp_path: Path,
) -> None:
    envelope_path = tmp_path / "claim.json"
    seal_claim_bearing_observation_factor_bundle(
        _bundle(),
        envelope_path,
        causal_source_lineage=_lineage(),
        provider_attestation=_attestation(),
    )
    record = json.loads(envelope_path.read_text(encoding="utf-8"))
    record["calibration_artifact_ids"]["gauge_artifact_id"] = "8" * 64
    envelope_path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="differ from provider attestation"):
        load_claim_bearing_observation_factor_bundle(envelope_path)


def test_claim_bearing_factor_bundle_rejects_path_traversal(
    tmp_path: Path,
) -> None:
    envelope_path = tmp_path / "claim.json"
    seal_claim_bearing_observation_factor_bundle(
        _bundle(),
        envelope_path,
        causal_source_lineage=_lineage(),
        provider_attestation=_attestation(),
    )
    record = json.loads(envelope_path.read_text(encoding="utf-8"))
    record["bundle_manifest_path"] = "../claim.bundle.json"
    envelope_path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="safe POSIX relative path"):
        load_claim_bearing_observation_factor_bundle(envelope_path)


def test_claim_bearing_factor_bundle_rejects_marginal_only_covariance_before_write(
    tmp_path: Path,
) -> None:
    envelope_path = tmp_path / "claim.json"
    with pytest.raises(ValueError, match="requires joint gauge covariance"):
        seal_claim_bearing_observation_factor_bundle(
            _bundle(covariance_semantics="marginal-blocks-only"),
            envelope_path,
            causal_source_lineage=_lineage(),
            provider_attestation=_attestation(),
        )
    assert not envelope_path.exists()
    assert not (tmp_path / "claim.bundle.json").exists()


def test_claim_bearing_factor_bundle_rejects_factor_outside_source_window(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="outside its causal source window"):
        seal_claim_bearing_observation_factor_bundle(
            _bundle(second_frame=1),
            tmp_path / "claim.json",
            causal_source_lineage=_lineage(),
            provider_attestation=_attestation(),
        )
