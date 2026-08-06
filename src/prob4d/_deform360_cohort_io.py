"""I/O, admission, and CLI helpers for Deform360 cohort bindings."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ._deform360_cohort_binding import (
    Deform360OfficialHubCohortBindingV1,
    build_deform360_official_hub_cohort_binding,
    deform360_cohort_binding_from_dict,
)
from ._deform360_cohort_schema import (
    BAYESIAN_PHYSTWIN_REPOSITORY,
    DEFORM360_SELECTION_PATH,
    _strict_string_list,
)
from ._heldout_promotion_common import _atomic_write_json, _load_json
from ._selection_evidence_common import (
    _strict_integer,
    _strict_list,
    _strict_mapping,
)


def validate_deform360_cohort_binding_against_selection(
    binding: Deform360OfficialHubCohortBindingV1,
    selection_value: Any,
) -> None:
    """Rebind a portable cohort artifact to the exact source selection bytes."""

    if not isinstance(binding, Deform360OfficialHubCohortBindingV1):
        raise TypeError("binding must be Deform360OfficialHubCohortBindingV1")
    rebuilt = build_deform360_official_hub_cohort_binding(
        selection_value,
        source_repository=binding.source_repository,
        source_revision=binding.source_revision,
        source_path=binding.source_path,
    )
    if rebuilt.to_dict() != binding.to_dict():
        raise ValueError("Deform360 cohort binding disagrees with the exact source selection")


def validate_promotion_config_against_deform360_binding(
    config_value: Any,
    binding: Deform360OfficialHubCohortBindingV1,
) -> None:
    """Require a promotion configuration to use exactly the sealed 10/12 split."""

    if not isinstance(binding, Deform360OfficialHubCohortBindingV1):
        raise TypeError("binding must be Deform360OfficialHubCohortBindingV1")
    config = _strict_mapping(config_value, name="promotion lock configuration")
    if config.get("bayesian_phystwin_repository") != binding.source_repository:
        raise ValueError(
            "promotion configuration bayesian_phystwin_repository disagrees with cohort binding"
        )
    if config.get("bayesian_phystwin_revision") != binding.source_revision:
        raise ValueError(
            "promotion configuration bayesian_phystwin_revision disagrees with cohort binding"
        )
    for field_name, expected in (
        ("calibration_group_ids", list(binding.calibration_group_ids)),
        ("target_group_ids", list(binding.target_group_ids)),
    ):
        observed = _strict_list(config.get(field_name), name=field_name)
        if observed != expected:
            raise ValueError(f"promotion configuration {field_name} disagrees with cohort binding")
    development = _strict_string_list(
        config.get("development_group_ids"),
        name="development_group_ids",
    )
    if set(development) & (set(binding.calibration_group_ids) | set(binding.target_group_ids)):
        raise ValueError("promotion development groups overlap the bound BPT cohort")
    minimum_target = _strict_integer(
        config.get("minimum_target_group_count"),
        name="minimum_target_group_count",
        minimum=1,
    )
    if minimum_target != len(binding.target_group_ids):
        raise ValueError(
            "minimum_target_group_count must equal the complete bound confirmation cohort"
        )
    frozen = _strict_mapping(config.get("frozen_artifact_ids"), name="frozen_artifact_ids")
    if frozen.get("cohort_binding") != binding.cohort_binding_id:
        raise ValueError(
            "frozen_artifact_ids.cohort_binding must equal the Deform360 cohort_binding_id"
        )


def write_deform360_cohort_binding(
    binding: Deform360OfficialHubCohortBindingV1,
    path: str | Path,
) -> None:
    """Publish one cohort binding atomically without rewriting existing evidence."""

    if not isinstance(binding, Deform360OfficialHubCohortBindingV1):
        raise TypeError("binding must be Deform360OfficialHubCohortBindingV1")
    _atomic_write_json(Path(path), binding.to_dict())


def load_deform360_cohort_binding(
    path: str | Path,
) -> Deform360OfficialHubCohortBindingV1:
    """Load and fully replay one portable cohort binding."""

    value, _ = _load_json(Path(path), name="Deform360 cohort binding")
    return deform360_cohort_binding_from_dict(value)


def bind_cli(arguments: Sequence[str]) -> int:
    """Implement the grouped ``cohort-bind`` command."""

    parser = argparse.ArgumentParser(
        prog="prob4d experiment heldout-provider cohort-bind",
        description=(
            "Bind promotion to BayesianPhysTwin's committed official-Hub "
            "Deform360 Stage-0 selection."
        ),
    )
    parser.add_argument("selection", type=Path)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument(
        "--source-repository",
        default=BAYESIAN_PHYSTWIN_REPOSITORY,
    )
    parser.add_argument("--source-path", default=DEFORM360_SELECTION_PATH)
    parser.add_argument("--output", type=Path, required=True)
    parsed = parser.parse_args(arguments)
    selection, _ = _load_json(parsed.selection, name="Deform360 official-Hub selection")
    binding = build_deform360_official_hub_cohort_binding(
        selection,
        source_repository=parsed.source_repository,
        source_revision=parsed.source_revision,
        source_path=parsed.source_path,
    )
    write_deform360_cohort_binding(binding, parsed.output)
    print(binding.cohort_binding_id)
    return 0


def verify_cli(arguments: Sequence[str]) -> int:
    """Implement the grouped ``cohort-verify`` command."""

    parser = argparse.ArgumentParser(
        prog="prob4d experiment heldout-provider cohort-verify",
        description="Replay a Deform360 cohort binding and optionally rebind its selection.",
    )
    parser.add_argument("binding", type=Path)
    parser.add_argument("--selection", type=Path)
    parsed = parser.parse_args(arguments)
    binding = load_deform360_cohort_binding(parsed.binding)
    selection_verified = False
    if parsed.selection is not None:
        selection, _ = _load_json(
            parsed.selection,
            name="Deform360 official-Hub selection",
        )
        validate_deform360_cohort_binding_against_selection(binding, selection)
        selection_verified = True
    print(
        json.dumps(
            {
                "cohort_binding_id": binding.cohort_binding_id,
                "selection_artifact_sha256": binding.selection_artifact_sha256,
                "selection_verified": selection_verified,
                "calibration_object_count": len(binding.calibration_group_ids),
                "target_object_count": len(binding.target_group_ids),
                "dataset_resolved_revision": binding.dataset_resolved_revision,
                "processing_revision": binding.processing_revision,
            },
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
    )
    return 0
