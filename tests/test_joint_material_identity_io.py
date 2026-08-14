import json
from pathlib import Path

import numpy as np
import pytest

from prob4d.joint_material_identity import (
    build_joint_material_identity_posterior,
    load_joint_material_identity_posterior,
    write_joint_material_identity_posterior,
)
from prob4d.material_identity_mixture import (
    LocalTrackEndpoint,
    MaterialIdentityCandidateV1,
    MaterialIdentityMixtureV1,
)

RULE_ID = "a" * 64
CALIBRATION_ID = "b" * 64
TRACKLET_REVISION = "c" * 40
ASSOCIATION_REVISION = "d" * 40
RESULT_ID = "e" * 64


def mixture(
    *,
    target_window: str,
    target_track: int,
    window_order: tuple[str, ...],
    source_endpoint: LocalTrackEndpoint | None,
    linked_weight: float = 9.0,
    null_weight: float = 1.0,
    result_id: str = RESULT_ID,
) -> MaterialIdentityMixtureV1:
    candidates = [
        MaterialIdentityCandidateV1(
            source_endpoint=None,
            association_result_id=None,
            source_score=None,
            calibrated_log_weight=np.log(null_weight),
            metadata={"role": "exact-local-fallback"},
        )
    ]
    if source_endpoint is not None:
        candidates.append(
            MaterialIdentityCandidateV1(
                source_endpoint=source_endpoint,
                association_result_id=result_id,
                source_score=0.9,
                calibrated_log_weight=np.log(linked_weight),
                metadata={"source": "calibration-only"},
            )
        )
    return MaterialIdentityMixtureV1(
        target_endpoint=LocalTrackEndpoint(target_window, target_track),
        window_order=window_order,
        causal_frame_stop=20 + window_order.index(target_window),
        association_rule_id=RULE_ID,
        calibration_id=CALIBRATION_ID,
        tracklet_producer_revision=TRACKLET_REVISION,
        association_revision=ASSOCIATION_REVISION,
        candidates=tuple(candidates),
        metadata={"stage": "source-only"},
    )


def conflicting_pair() -> tuple[MaterialIdentityMixtureV1, ...]:
    source = LocalTrackEndpoint("window-0", 0)
    return (
        mixture(
            target_window="window-1",
            target_track=0,
            window_order=("window-0", "window-1"),
            source_endpoint=source,
        ),
        mixture(
            target_window="window-1",
            target_track=1,
            window_order=("window-0", "window-1"),
            source_endpoint=source,
        ),
    )


def test_self_contained_round_trip_and_derived_tamper_rejection(tmp_path: Path) -> None:
    posterior = build_joint_material_identity_posterior(
        conflicting_pair(),
        window_order=("window-0", "window-1"),
        metadata={"nested": {"source_only": True}},
    )
    path = tmp_path / "joint.json"
    write_joint_material_identity_posterior(path, posterior)
    loaded = load_joint_material_identity_posterior(path)

    assert loaded == posterior
    assert path.read_bytes().endswith(b"\n")
    with pytest.raises(FileExistsError):
        write_joint_material_identity_posterior(path, posterior)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["assignments"][0]["probability"] += 0.01
    payload["posterior_id"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="exact joint-posterior replay"):
        load_joint_material_identity_posterior(path)


def test_duplicate_json_keys_and_embedded_mixture_tampering_fail_closed(
    tmp_path: Path,
) -> None:
    posterior = build_joint_material_identity_posterior(
        conflicting_pair(),
        window_order=("window-0", "window-1"),
    )
    path = tmp_path / "joint.json"
    write_joint_material_identity_posterior(path, posterior)
    duplicate = path.read_text(encoding="utf-8").replace(
        '"schema_version":1',
        '"schema_version":1,"schema_version":1',
        1,
    )
    path.write_text(duplicate, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        load_joint_material_identity_posterior(path)

    write_joint_material_identity_posterior(path, posterior, overwrite=True)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["mixtures"][0]["candidates"][0]["candidate_id"] = "1" * 64
    payload["posterior_id"] = "2" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="candidate ID mismatch"):
        load_joint_material_identity_posterior(path)


def test_window_prefix_and_calibration_mismatches_are_rejected() -> None:
    first, second = conflicting_pair()
    with pytest.raises(ValueError, match="global prefix"):
        build_joint_material_identity_posterior(
            (first,),
            window_order=("window-extra", "window-0", "window-1"),
        )

    changed = MaterialIdentityMixtureV1(
        target_endpoint=second.target_endpoint,
        window_order=second.window_order,
        causal_frame_stop=second.causal_frame_stop,
        association_rule_id=second.association_rule_id,
        calibration_id="9" * 64,
        tracklet_producer_revision=second.tracklet_producer_revision,
        association_revision=second.association_revision,
        candidates=second.candidates,
        metadata=second.metadata,
    )
    with pytest.raises(ValueError, match="calibration_id"):
        build_joint_material_identity_posterior(
            (first, changed),
            window_order=("window-0", "window-1"),
        )


def test_metadata_and_probability_arrays_are_immutable() -> None:
    posterior = build_joint_material_identity_posterior(
        conflicting_pair(),
        window_order=("window-0", "window-1"),
        metadata={"nested": {"source_only": True}},
    )
    with pytest.raises(TypeError, match="immutable"):
        posterior.metadata["changed"] = True  # type: ignore[index]
    with pytest.raises(ValueError, match="read-only"):
        posterior.marginals[0].probabilities[0] = 0.0
