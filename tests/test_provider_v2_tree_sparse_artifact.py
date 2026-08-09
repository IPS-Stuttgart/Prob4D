from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from prob4d.gauge_tree_prior import GaugeTreeSquareRootPriorV1
from prob4d.provider_attestation import (
    build_provider_attestation,
    compute_provider_manifest_id,
    validate_provider_manifest,
)
from prob4d.provider_v2_tree_sparse_artifact import (
    CLAIM_BEARING_TREE_SPARSE_OBSERVATION_SCHEMA,
    load_claim_bearing_tree_sparse_observation,
    seal_claim_bearing_tree_sparse_observation,
)
from prob4d.provider_v2_tree_sparse_manifest import (
    TREE_SPARSE_PROVIDER_CAPABILITIES,
    prob4d_tree_sparse_provider_manifest,
)
from prob4d.tree_sparse_observation_factors import (
    TreeSparseStackedObservationFactors,
    build_tree_sparse_observation_factors,
)

_REVISION = "a" * 40
_GAUGE_CALIBRATION_ID = "b" * 64
_POINT_CALIBRATION_ID = "c" * 64


def _runtime() -> dict[str, object]:
    return {
        "expected_revision": _REVISION,
        "observed_revision": _REVISION,
        "source": "source_checkout",
        "clean_checkout": True,
        "matched": True,
        "independently_verified": True,
    }


def _attestation(
    *,
    manifest: dict[str, object] | None = None,
) -> dict[str, object]:
    selected_manifest = (
        prob4d_tree_sparse_provider_manifest(provider_revision=_REVISION)
        if manifest is None
        else manifest
    )
    return build_provider_attestation(
        provider_manifest=selected_manifest,
        provider_revision=_REVISION,
        export_mode="calibrated",
        calibration_compatibility_validated=True,
        calibration_artifact_ids={
            "gauge_artifact_id": _GAUGE_CALIBRATION_ID,
            "point_artifact_id": _POINT_CALIBRATION_ID,
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
                "source_frame_stop_exclusive": 2,
                "source_frame_max": 1,
                "frame_indices_sha256": "d" * 64,
                "payload_sha256": "e" * 64,
            },
            {
                "window_id": "window-1",
                "source_frame_start": 2,
                "source_frame_stop_exclusive": 5,
                "source_frame_max": 4,
                "frame_indices_sha256": "f" * 64,
                "payload_sha256": "1" * 64,
            },
        ],
        "source_artifact_sha256": "2" * 64,
        "source_digest_scope": "prefix-only fixture",
    }


def _prior() -> GaugeTreeSquareRootPriorV1:
    transitions = np.zeros((2, 7, 7), dtype=np.float64)
    transitions[1] = np.eye(7, dtype=np.float64) * 0.2
    innovations = np.stack(
        (
            np.eye(7, dtype=np.float64) * 2.0e-4,
            np.eye(7, dtype=np.float64) * 3.0e-4,
        )
    )
    return GaugeTreeSquareRootPriorV1.from_transition_covariances(
        gauge_ids=("window-0", "window-1"),
        parent_indices=np.asarray([-1, 0], dtype=np.int64),
        transition_matrices=transitions,
        innovation_covariances=innovations,
    )


def _factors(
    *,
    second_frame: int = 3,
    mean_offset: float = 0.0,
) -> TreeSparseStackedObservationFactors:
    local_jacobian = np.zeros((4, 3, 7), dtype=np.float64)
    local_jacobian[:, :, 4:7] = np.eye(3, dtype=np.float64)[None]
    return build_tree_sparse_observation_factors(
        _prior(),
        world_mean_m=np.asarray(
            [
                [0.0, 0.0, 1.0],
                [0.2, 0.0, 1.1],
                [0.1, 0.2, 1.2],
                [0.3, 0.1, 1.3],
            ],
            dtype=np.float64,
        )
        + mean_offset,
        conditional_world_covariance_m2=np.repeat(
            np.eye(3, dtype=np.float64)[None] * 1.0e-3,
            4,
            axis=0,
        ),
        local_gauge_jacobian=local_jacobian,
        gauge_indices=np.asarray([0, 0, 1, 1], dtype=np.int64),
        association_probability=np.asarray([0.9, 0.8, 0.85, 0.75]),
        prior_reliability=np.asarray([0.95, 0.9, 0.88, 0.82]),
        prior_nominal_probability=np.asarray([0.94, 0.94, 0.91, 0.91]),
        composite_weight=np.asarray([0.5, 0.5, 0.4, 0.4]),
        point_ids=np.asarray([10, 11, 20, 21], dtype=np.int64),
        frame_indices=np.asarray([1, 1, second_frame, second_frame], dtype=np.int64),
        view_ids=("camera-0", "camera-0", "camera-0", "camera-0"),
        factor_ids=("factor-0", "factor-0", "factor-1", "factor-1"),
        correlation_group_ids=(
            "factor-0:camera-0:frame-1",
            "factor-0:camera-0:frame-1",
            f"factor-1:camera-0:frame-{second_frame}",
            f"factor-1:camera-0:frame-{second_frame}",
        ),
        causal_frame_stop=5,
    )


def _seal(
    factors: TreeSparseStackedObservationFactors,
    path: Path,
    *,
    attestation: dict[str, object] | None = None,
    metadata: dict[str, object] | None = None,
):
    return seal_claim_bearing_tree_sparse_observation(
        factors,
        path,
        sequence_id="sequence-a",
        case_id="case-a",
        stream_id="prob4d:tree-sparse:camera-0",
        source_revision=_REVISION,
        causal_source_lineage=_lineage(),
        provider_attestation=_attestation() if attestation is None else attestation,
        artifact_metadata={"split": "calibration"},
        metadata={"protocol": "tree-sparse-to-bpt-v1"} if metadata is None else metadata,
    )


def test_tree_sparse_provider_manifest_extends_valid_provider_v2_contract() -> None:
    manifest = prob4d_tree_sparse_provider_manifest(provider_revision=_REVISION)
    validated = validate_provider_manifest(manifest, expected_revision=_REVISION)

    assert validated["manifest_id"] == manifest["manifest_id"]
    capabilities = validated["capabilities"]
    assert set(TREE_SPARSE_PROVIDER_CAPABILITIES).issubset(capabilities)
    schemas = validated["artifact_schema_versions"]
    assert schemas["TreeSparseObservationArtifactV1"] == 1
    assert schemas["ClaimBearingTreeSparseObservationEnvelopeV1"] == 1


def test_claim_bearing_tree_sparse_roundtrip_and_relocation(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    envelope_path = source / "claim.json"
    validated = _seal(_factors(), envelope_path)
    record = json.loads(envelope_path.read_text(encoding="utf-8"))

    assert record["schema"] == CLAIM_BEARING_TREE_SPARSE_OBSERVATION_SCHEMA
    assert validated.envelope.observation_count == 4
    assert validated.envelope.gauge_ids == ("window-0", "window-1")
    assert validated.provider_manifest_id == validated.envelope.provider_manifest_id
    assert validated.gauge_calibration_id == _GAUGE_CALIBRATION_ID
    assert validated.point_calibration_id == _POINT_CALIBRATION_ID
    assert validated.envelope.runtime_revision_independently_verified is True
    assert not hasattr(validated.observation.factors, "gauge_prior_covariance")
    assert "gauge_prior_covariance" not in validated.observation.manifest.array_members

    moved = tmp_path / "moved"
    shutil.copytree(source, moved)
    relocated = load_claim_bearing_tree_sparse_observation(moved / "claim.json")
    assert relocated.artifact_id == validated.artifact_id
    np.testing.assert_array_equal(
        relocated.observation.factors.world_mean_m,
        validated.observation.factors.world_mean_m,
    )


def test_claim_bearing_tree_sparse_rejects_row_outside_source_window_before_write(
    tmp_path: Path,
) -> None:
    path = tmp_path / "claim.json"
    with pytest.raises(ValueError, match="outside its causal source window"):
        _seal(_factors(second_frame=1), path)
    assert list(tmp_path.iterdir()) == []


def test_claim_bearing_tree_sparse_rejects_missing_provider_capability_before_write(
    tmp_path: Path,
) -> None:
    manifest = prob4d_tree_sparse_provider_manifest(provider_revision=_REVISION)
    manifest.pop("manifest_id")
    capabilities = list(manifest["capabilities"])
    capabilities.remove("strict_claim_bearing_tree_sparse_observation_loading")
    manifest["capabilities"] = capabilities
    manifest["manifest_id"] = compute_provider_manifest_id(manifest)
    attestation = _attestation(manifest=manifest)

    with pytest.raises(ValueError, match="lacks tree-sparse claim capabilities"):
        _seal(_factors(), tmp_path / "claim.json", attestation=attestation)
    assert list(tmp_path.iterdir()) == []


def test_claim_bearing_tree_sparse_rejects_observation_manifest_tampering(
    tmp_path: Path,
) -> None:
    envelope_path = tmp_path / "claim.json"
    validated = _seal(_factors(), envelope_path)
    observation_path = tmp_path / validated.envelope.observation_manifest_path
    observation_path.write_text(
        observation_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="no longer matches its observation manifest"):
        load_claim_bearing_tree_sparse_observation(envelope_path)


def test_claim_bearing_tree_sparse_rejects_envelope_path_traversal(
    tmp_path: Path,
) -> None:
    envelope_path = tmp_path / "claim.json"
    _seal(_factors(), envelope_path)
    record = json.loads(envelope_path.read_text(encoding="utf-8"))
    record["observation_manifest_path"] = "../claim.tree-sparse.json"
    envelope_path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="safe POSIX relative path"):
        load_claim_bearing_tree_sparse_observation(envelope_path)


def test_claim_bearing_tree_sparse_rejects_provider_and_calibration_tampering(
    tmp_path: Path,
) -> None:
    envelope_path = tmp_path / "claim.json"
    _seal(_factors(), envelope_path)
    original = envelope_path.read_text(encoding="utf-8")
    record = json.loads(original)
    record["calibration_artifact_ids"]["gauge_artifact_id"] = "9" * 64
    envelope_path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="differ from provider attestation"):
        load_claim_bearing_tree_sparse_observation(envelope_path)

    envelope_path.write_text(original, encoding="utf-8")
    record = json.loads(original)
    record["provider_manifest_id"] = "8" * 64
    envelope_path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="differs from provider attestation"):
        load_claim_bearing_tree_sparse_observation(envelope_path)


def test_claim_bearing_tree_sparse_rejects_duplicate_and_nonfinite_json(
    tmp_path: Path,
) -> None:
    envelope_path = tmp_path / "claim.json"
    _seal(_factors(), envelope_path)
    payload = envelope_path.read_text(encoding="utf-8")
    duplicate = payload.replace(
        '"schema":',
        '"schema":"duplicate","schema":',
        1,
    )
    envelope_path.write_text(duplicate, encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_claim_bearing_tree_sparse_observation(envelope_path)

    fresh = tmp_path / "fresh.json"
    _seal(_factors(), fresh)
    nonfinite = fresh.read_text(encoding="utf-8").replace(
        '"metadata":{"protocol":"tree-sparse-to-bpt-v1"}',
        '"metadata":{"value":NaN}',
    )
    fresh.write_text(nonfinite, encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_claim_bearing_tree_sparse_observation(fresh)


def test_claim_bearing_tree_sparse_rejects_invalid_metadata_before_write(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="finite"):
        _seal(
            _factors(),
            tmp_path / "claim.json",
            metadata={"invalid": float("nan")},
        )
    assert list(tmp_path.iterdir()) == []


def test_claim_bearing_tree_sparse_publication_is_idempotent_no_clobber(
    tmp_path: Path,
) -> None:
    path = tmp_path / "claim.json"
    first = _seal(_factors(), path)
    retained = path.read_bytes()
    repeated = _seal(_factors(), path)

    assert repeated.artifact_id == first.artifact_id
    assert path.read_bytes() == retained
    with pytest.raises((FileExistsError, ValueError)):
        _seal(_factors(mean_offset=0.01), path)
    assert path.read_bytes() == retained
    assert load_claim_bearing_tree_sparse_observation(path).artifact_id == (first.artifact_id)


def test_claim_bearing_tree_sparse_rejects_boolean_schema_version(
    tmp_path: Path,
) -> None:
    path = tmp_path / "claim.json"
    _seal(_factors(), path)
    record = json.loads(path.read_text(encoding="utf-8"))
    record["schema_version"] = True
    path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="version"):
        load_claim_bearing_tree_sparse_observation(path)
