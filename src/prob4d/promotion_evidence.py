"""Version-dispatched evidence cards for held-out provider promotion."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import _promotion_evidence_v1 as _v1
from ._heldout_promotion_common import _load_json
from ._heldout_promotion_lock import (
    HELDOUT_PROMOTION_LOCK_PROVIDER_NEUTRAL_VERSION,
    ProviderPromotionIdentityV1,
    promotion_lock_from_dict,
)
from ._immutable_json import plain_json
from ._selection_evidence_common import _sha256_json

PROMOTION_EVIDENCE_CARD_SCHEMA = _v1.PROMOTION_EVIDENCE_CARD_SCHEMA
PROMOTION_EVIDENCE_CARD_VERSION = _v1.PROMOTION_EVIDENCE_CARD_VERSION
PROMOTION_EVIDENCE_CARD_PROVIDER_NEUTRAL_VERSION = 2

_REPOSITORY_FIELDS_V2 = {
    "prob4d",
    "bayesian_phystwin",
    "provider",
}


def _descriptor(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: plain_json(item)
        for key, item in value.items()
        if key != "evidence_card_id"
    }


def _provider_identity(value: Mapping[str, Any]) -> ProviderPromotionIdentityV1:
    repositories = _v1._strict_mapping(value["repositories"], name="repositories")
    _v1._exact_keys(
        repositories,
        _REPOSITORY_FIELDS_V2,
        name="repositories",
    )
    return ProviderPromotionIdentityV1.from_dict(repositories["provider"])


def _as_legacy_card(value: Mapping[str, Any]) -> dict[str, Any]:
    """Create a validation-only V1 view without rewriting retained V2 bytes."""

    identity = _provider_identity(value)
    repositories = _v1._strict_mapping(value["repositories"], name="repositories")
    legacy = plain_json(value)
    if not isinstance(legacy, dict):
        raise AssertionError("promotion evidence card is not a mapping")
    legacy["schema_version"] = PROMOTION_EVIDENCE_CARD_VERSION
    legacy["repositories"] = {
        "prob4d": plain_json(repositories["prob4d"]),
        "bayesian_phystwin": plain_json(repositories["bayesian_phystwin"]),
        "motioncrafter": {
            "revision": identity.provider_revision,
            "model_set_id": identity.model_set_id,
        },
    }
    legacy["evidence_card_id"] = _sha256_json(_descriptor(legacy))
    return legacy


def build_promotion_evidence_card(
    lock_value: object,
    report_value: object,
) -> dict[str, Any]:
    """Build a V1 replay card or a provider-neutral V2 evidence card."""

    lock = promotion_lock_from_dict(lock_value)
    card = _v1.build_promotion_evidence_card(lock_value, report_value)
    if lock.schema_version != HELDOUT_PROMOTION_LOCK_PROVIDER_NEUTRAL_VERSION:
        return card
    identity = lock.provider_identity
    if identity is None:
        raise AssertionError("provider-neutral lock has no provider identity")
    repositories = _v1._strict_mapping(card["repositories"], name="repositories")
    descriptor = _descriptor(card)
    descriptor["schema_version"] = PROMOTION_EVIDENCE_CARD_PROVIDER_NEUTRAL_VERSION
    descriptor["repositories"] = {
        "prob4d": plain_json(repositories["prob4d"]),
        "bayesian_phystwin": plain_json(repositories["bayesian_phystwin"]),
        "provider": identity.to_dict(),
    }
    return promotion_evidence_card_from_dict(
        {
            **descriptor,
            "evidence_card_id": _sha256_json(descriptor),
        }
    )


def promotion_evidence_card_from_dict(value: object) -> dict[str, Any]:
    """Validate historical V1 and provider-neutral V2 evidence cards."""

    card = _v1._strict_mapping(value, name="promotion evidence card")
    version = _v1._strict_integer(
        card.get("schema_version"),
        name="schema_version",
        minimum=1,
    )
    if version == PROMOTION_EVIDENCE_CARD_VERSION:
        return _v1.promotion_evidence_card_from_dict(card)
    if version != PROMOTION_EVIDENCE_CARD_PROVIDER_NEUTRAL_VERSION:
        raise ValueError("unsupported promotion evidence card version")
    if card.get("schema_name") != PROMOTION_EVIDENCE_CARD_SCHEMA:
        raise ValueError("unsupported promotion evidence card schema")

    _v1.promotion_evidence_card_from_dict(_as_legacy_card(card))
    _provider_identity(card)
    supplied = _v1._digest(card["evidence_card_id"], name="evidence_card_id")
    if supplied != _sha256_json(_descriptor(card)):
        raise ValueError("promotion evidence card ID mismatch")
    result = plain_json(card)
    if not isinstance(result, dict):
        raise AssertionError("validated promotion evidence card is not a mapping")
    return result


def load_promotion_evidence_card(path: str | Path) -> dict[str, Any]:
    mapping, _ = _load_json(Path(path), name="promotion evidence card")
    return promotion_evidence_card_from_dict(mapping)


def render_promotion_evidence_markdown(card: Mapping[str, Any]) -> str:
    value = promotion_evidence_card_from_dict(card)
    if value["schema_version"] == PROMOTION_EVIDENCE_CARD_VERSION:
        return _v1.render_promotion_evidence_markdown(value)

    identity = _provider_identity(value)
    legacy = _as_legacy_card(value)
    rendered = _v1.render_promotion_evidence_markdown(legacy)
    anchor = next(
        line
        for line in rendered.splitlines()
        if line.startswith("- BayesianPhysTwin:")
    )
    provider_line = (
        f"- Provider: `{identity.provider_family}` from "
        f"`{identity.provider_repository}@{identity.provider_revision}`"
    )
    return rendered.replace(anchor, f"{anchor}\n{provider_line}", 1)


def write_promotion_evidence_card(
    card: Mapping[str, Any],
    json_path: str | Path,
    markdown_path: str | Path,
) -> None:
    value = promotion_evidence_card_from_dict(card)
    json_destination = Path(json_path)
    markdown_destination = Path(markdown_path)
    if json_destination == markdown_destination:
        raise ValueError("JSON and Markdown evidence-card paths must differ")
    if json_destination.exists():
        raise FileExistsError(json_destination)
    if markdown_destination.exists():
        raise FileExistsError(markdown_destination)
    _v1._atomic_write_json(json_destination, value)
    try:
        _v1._atomic_write_text(
            markdown_destination,
            render_promotion_evidence_markdown(value),
        )
    except Exception:
        json_destination.unlink(missing_ok=True)
        raise


__all__ = [
    "PROMOTION_EVIDENCE_CARD_PROVIDER_NEUTRAL_VERSION",
    "PROMOTION_EVIDENCE_CARD_SCHEMA",
    "PROMOTION_EVIDENCE_CARD_VERSION",
    "build_promotion_evidence_card",
    "load_promotion_evidence_card",
    "promotion_evidence_card_from_dict",
    "render_promotion_evidence_markdown",
    "write_promotion_evidence_card",
]
