from __future__ import annotations

import pytest

from prob4d._heldout_promotion_lock import ProviderPromotionIdentityV1


def _provider_identity() -> dict[str, object]:
    return {
        "schema_name": "prob4d.heldout-provider-promotion-identity",
        "schema_version": 1,
        "provider_family": "cut3r",
        "provider_repository": "naver/CUT3R",
        "provider_revision": "c" * 40,
        "model_set_id": "d" * 64,
        "loader_id": "7" * 64,
        "coordinate_semantics": "sequence-local-sim3",
        "point_semantics": "dense-point-map",
        "flow_semantics": "absent",
        "ray_semantics": "absent",
        "source_dependency_semantics": (
            "per-output-exclusive-source-frame-interval-v1"
        ),
    }


@pytest.mark.parametrize("coercive_version", [True, 1.0])
def test_provider_promotion_identity_rejects_coercive_schema_versions(
    coercive_version: object,
) -> None:
    identity = _provider_identity()
    identity["schema_version"] = coercive_version

    with pytest.raises(ValueError, match="schema_version"):
        ProviderPromotionIdentityV1.from_dict(identity)
