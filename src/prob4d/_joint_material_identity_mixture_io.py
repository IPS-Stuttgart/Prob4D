"""Independent reconstruction of embedded local identity mixtures."""

from __future__ import annotations

from ._joint_material_identity_common import _sha256
from ._joint_material_identity_json import (
    _CANDIDATE_FIELDS,
    _MIXTURE_FIELDS,
    _fields,
    _list,
    _mapping,
)
from .material_identity_mixture import (
    CLAIM_BOUNDARY as LOCAL_CLAIM_BOUNDARY,
    MATERIAL_IDENTITY_MIXTURE_SCHEMA,
    MATERIAL_IDENTITY_MIXTURE_VERSION,
    LocalTrackEndpoint,
    MaterialIdentityCandidateV1,
    MaterialIdentityMixtureV1,
)


def _mixture(value: object, *, index: int) -> MaterialIdentityMixtureV1:
    record = _mapping(value, name=f"mixtures[{index}]")
    _fields(record, _MIXTURE_FIELDS, name=f"mixtures[{index}]")
    if record["schema"] != MATERIAL_IDENTITY_MIXTURE_SCHEMA:
        raise ValueError("embedded mixture schema changed")
    if record["schema_version"] != MATERIAL_IDENTITY_MIXTURE_VERSION:
        raise ValueError("embedded mixture version changed")
    if record["claim_boundary"] != LOCAL_CLAIM_BOUNDARY:
        raise ValueError("embedded mixture claim boundary changed")
    target = LocalTrackEndpoint.from_mapping(
        record["target_endpoint"],
        name=f"mixtures[{index}].target_endpoint",
    )
    candidates: list[MaterialIdentityCandidateV1] = []
    supplied_ids: list[str] = []
    for candidate_index, raw in enumerate(
        _list(record["candidates"], name=f"mixtures[{index}].candidates")
    ):
        name = f"mixtures[{index}].candidates[{candidate_index}]"
        candidate_record = _mapping(raw, name=name)
        _fields(candidate_record, _CANDIDATE_FIELDS, name=name)
        source_raw = candidate_record["source_endpoint"]
        source = (
            None
            if source_raw is None
            else LocalTrackEndpoint.from_mapping(
                source_raw,
                name=f"{name}.source_endpoint",
            )
        )
        kind = candidate_record["kind"]
        if kind not in {"null", "linked"} or (kind == "null") != (source is None):
            raise ValueError(f"{name}.kind does not match source_endpoint")
        candidates.append(
            MaterialIdentityCandidateV1(
                source_endpoint=source,
                association_result_id=candidate_record["association_result_id"],
                source_score=candidate_record["source_score"],
                calibrated_log_weight=candidate_record["calibrated_log_weight"],
                metadata=_mapping(candidate_record["metadata"], name=f"{name}.metadata"),
            )
        )
        supplied_ids.append(
            _sha256(candidate_record["candidate_id"], name=f"{name}.candidate_id")
        )
    mixture = MaterialIdentityMixtureV1(
        target_endpoint=target,
        window_order=tuple(_list(record["window_order"], name="window_order")),
        causal_frame_stop=record["causal_frame_stop"],
        association_rule_id=record["association_rule_id"],
        calibration_id=record["calibration_id"],
        tracklet_producer_revision=record["tracklet_producer_revision"],
        association_revision=record["association_revision"],
        candidates=tuple(candidates),
        metadata=_mapping(record["metadata"], name=f"mixtures[{index}].metadata"),
        weight_semantics=record["weight_semantics"],
        null_hypothesis_semantics=record["null_hypothesis_semantics"],
        mixture_id=record["mixture_id"],
    )
    if tuple(supplied_ids) != mixture.candidate_ids:
        raise ValueError("embedded material-identity candidate ID mismatch")
    return mixture
