"""Audit CUT3R source metrics from canonical rows and exact score references.

The merged common-support v2 layer compares candidate and baseline support
identities. This additive audit independently reconstructs those identities from
retained canonical row arrays and binds the exact arm-neutral proper-score
reference bytes before source scores are opened. Existing v2 artifacts remain
strictly replay-compatible; claim-bearing readiness gates use the audit receipt.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ._cut3r_source_competence_audit_common import (
    LOCK_SCHEMA,
    MANIFEST_SCHEMA,
    PROPER_SCORE_REFERENCE_FIT_SCOPE,
    REPORT_SCHEMA,
    VERSION,
)
from ._cut3r_source_competence_audit_lock import (
    build_cut3r_source_competence_audit_lock,
    validate_cut3r_source_competence_audit_lock,
    verify_cut3r_source_competence_audit_reference,
)
from ._cut3r_source_competence_audit_manifest import (
    build_cut3r_metric_support_manifest,
    metric_support_from_manifest_entry,
    validate_cut3r_metric_support_manifest,
)
from ._cut3r_source_competence_audit_report import (
    build_cut3r_source_competence_support_audit_report,
    source_competence_gates_audited,
    validate_cut3r_source_competence_support_audit_report,
)
from .cut3r_comparison import load_cut3r_comparison_lock
from .cut3r_source_competence import (
    _publish_json,
    _strict_json,
    load_cut3r_source_competence_lock,
)
from .cut3r_source_competence_v2 import (
    load_cut3r_source_competence_v2_lock,
    load_cut3r_source_competence_v2_report,
)


def _reference_bytes(path: str | Path) -> bytes:
    source = Path(path)
    if source.is_symlink():
        raise ValueError("proper-score reference path must not be a symbolic link")
    payload = source.read_bytes()
    if not payload:
        raise ValueError("proper-score reference must contain exact nonempty bytes")
    return payload


def write_cut3r_source_competence_audit_lock(
    comparison_lock: Any,
    source_competence_lock: Any,
    common_support_lock: Any,
    path: str | Path,
    lock: Mapping[str, Any],
    proper_score_reference_bytes: bytes,
) -> dict[str, Any]:
    payload = validate_cut3r_source_competence_audit_lock(
        comparison_lock,
        source_competence_lock,
        common_support_lock,
        lock,
    )
    verify_cut3r_source_competence_audit_reference(
        payload,
        proper_score_reference_bytes,
    )
    return _publish_json(
        path,
        payload,
        load_existing=lambda existing: load_cut3r_source_competence_audit_lock(
            comparison_lock,
            source_competence_lock,
            common_support_lock,
            existing,
        ),
    )


def load_cut3r_source_competence_audit_lock(
    comparison_lock: Any,
    source_competence_lock: Any,
    common_support_lock: Any,
    path: str | Path,
) -> dict[str, Any]:
    return validate_cut3r_source_competence_audit_lock(
        comparison_lock,
        source_competence_lock,
        common_support_lock,
        _strict_json(path),
    )


def write_cut3r_metric_support_manifest(
    comparison_lock: Any,
    source_competence_lock: Any,
    common_support_lock: Any,
    audit_lock: Any,
    path: str | Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    payload = validate_cut3r_metric_support_manifest(
        comparison_lock,
        source_competence_lock,
        common_support_lock,
        audit_lock,
        manifest,
    )
    return _publish_json(
        path,
        payload,
        load_existing=lambda existing: load_cut3r_metric_support_manifest(
            comparison_lock,
            source_competence_lock,
            common_support_lock,
            audit_lock,
            existing,
        ),
    )


def load_cut3r_metric_support_manifest(
    comparison_lock: Any,
    source_competence_lock: Any,
    common_support_lock: Any,
    audit_lock: Any,
    path: str | Path,
) -> dict[str, Any]:
    return validate_cut3r_metric_support_manifest(
        comparison_lock,
        source_competence_lock,
        common_support_lock,
        audit_lock,
        _strict_json(path),
    )


def write_cut3r_source_competence_support_audit_report(
    comparison_lock: Any,
    source_competence_lock: Any,
    common_support_lock: Any,
    audit_lock: Any,
    records: Any,
    metric_support_manifest: Any,
    source_competence_report_v2: Any,
    path: str | Path,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    payload = validate_cut3r_source_competence_support_audit_report(
        comparison_lock,
        source_competence_lock,
        common_support_lock,
        audit_lock,
        records,
        metric_support_manifest,
        source_competence_report_v2,
        report,
    )
    return _publish_json(
        path,
        payload,
        load_existing=lambda existing: (
            load_cut3r_source_competence_support_audit_report(
                comparison_lock,
                source_competence_lock,
                common_support_lock,
                audit_lock,
                records,
                metric_support_manifest,
                source_competence_report_v2,
                existing,
            )
        ),
    )


def load_cut3r_source_competence_support_audit_report(
    comparison_lock: Any,
    source_competence_lock: Any,
    common_support_lock: Any,
    audit_lock: Any,
    records: Any,
    metric_support_manifest: Any,
    source_competence_report_v2: Any,
    path: str | Path,
) -> dict[str, Any]:
    return validate_cut3r_source_competence_support_audit_report(
        comparison_lock,
        source_competence_lock,
        common_support_lock,
        audit_lock,
        records,
        metric_support_manifest,
        source_competence_report_v2,
        _strict_json(path),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    freeze = commands.add_parser("freeze")
    freeze.add_argument("comparison_lock")
    freeze.add_argument("source_competence_lock")
    freeze.add_argument("common_support_lock")
    freeze.add_argument("specification")
    freeze.add_argument("proper_score_reference")
    freeze.add_argument("--output", required=True)

    verify_lock = commands.add_parser("verify-lock")
    verify_lock.add_argument("comparison_lock")
    verify_lock.add_argument("source_competence_lock")
    verify_lock.add_argument("common_support_lock")
    verify_lock.add_argument("audit_lock")
    verify_lock.add_argument("proper_score_reference")

    manifest = commands.add_parser("manifest")
    manifest.add_argument("comparison_lock")
    manifest.add_argument("source_competence_lock")
    manifest.add_argument("common_support_lock")
    manifest.add_argument("audit_lock")
    manifest.add_argument("manifest_input")
    manifest.add_argument("--output", required=True)

    verify_manifest = commands.add_parser("verify-manifest")
    verify_manifest.add_argument("comparison_lock")
    verify_manifest.add_argument("source_competence_lock")
    verify_manifest.add_argument("common_support_lock")
    verify_manifest.add_argument("audit_lock")
    verify_manifest.add_argument("metric_support_manifest")

    report = commands.add_parser("report")
    report.add_argument("comparison_lock")
    report.add_argument("source_competence_lock")
    report.add_argument("common_support_lock")
    report.add_argument("audit_lock")
    report.add_argument("records")
    report.add_argument("metric_support_manifest")
    report.add_argument("source_competence_report_v2")
    report.add_argument("--output", required=True)
    report.add_argument("--require-pass", action="store_true")

    verify_report = commands.add_parser("verify-report")
    verify_report.add_argument("comparison_lock")
    verify_report.add_argument("source_competence_lock")
    verify_report.add_argument("common_support_lock")
    verify_report.add_argument("audit_lock")
    verify_report.add_argument("records")
    verify_report.add_argument("metric_support_manifest")
    verify_report.add_argument("source_competence_report_v2")
    verify_report.add_argument("report")
    verify_report.add_argument("--require-pass", action="store_true")

    gates = commands.add_parser("gates")
    gates.add_argument("comparison_lock")
    gates.add_argument("source_competence_lock")
    gates.add_argument("common_support_lock")
    gates.add_argument("audit_lock")
    gates.add_argument("records")
    gates.add_argument("metric_support_manifest")
    gates.add_argument("source_competence_report_v2")
    gates.add_argument("report")
    return parser


def _load_base(arguments: argparse.Namespace) -> tuple[dict[str, Any], ...]:
    comparison = load_cut3r_comparison_lock(arguments.comparison_lock)
    source_lock = load_cut3r_source_competence_lock(
        comparison,
        arguments.source_competence_lock,
    )
    v2_lock = load_cut3r_source_competence_v2_lock(
        comparison,
        source_lock,
        arguments.common_support_lock,
    )
    return comparison, source_lock, v2_lock


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    comparison, source_lock, v2_lock = _load_base(arguments)
    if arguments.command == "freeze":
        reference = _reference_bytes(arguments.proper_score_reference)
        lock = build_cut3r_source_competence_audit_lock(
            comparison,
            source_lock,
            v2_lock,
            _strict_json(arguments.specification),
            reference,
        )
        write_cut3r_source_competence_audit_lock(
            comparison,
            source_lock,
            v2_lock,
            arguments.output,
            lock,
            reference,
        )
        print(lock["support_audit_lock_id"])
        return 0
    audit = load_cut3r_source_competence_audit_lock(
        comparison,
        source_lock,
        v2_lock,
        arguments.audit_lock,
    )
    if arguments.command == "verify-lock":
        verify_cut3r_source_competence_audit_reference(
            audit,
            _reference_bytes(arguments.proper_score_reference),
        )
        print(audit["support_audit_lock_id"])
        return 0
    if arguments.command == "manifest":
        artifact = build_cut3r_metric_support_manifest(
            comparison,
            source_lock,
            v2_lock,
            audit,
            _strict_json(arguments.manifest_input),
        )
        write_cut3r_metric_support_manifest(
            comparison,
            source_lock,
            v2_lock,
            audit,
            arguments.output,
            artifact,
        )
        print(artifact["metric_support_manifest_id"])
        return 0
    manifest = load_cut3r_metric_support_manifest(
        comparison,
        source_lock,
        v2_lock,
        audit,
        arguments.metric_support_manifest,
    )
    if arguments.command == "verify-manifest":
        print(manifest["metric_support_manifest_id"])
        return 0
    records = _strict_json(arguments.records)
    v2_report = load_cut3r_source_competence_v2_report(
        comparison,
        source_lock,
        v2_lock,
        records,
        arguments.source_competence_report_v2,
    )
    if arguments.command == "report":
        artifact = build_cut3r_source_competence_support_audit_report(
            comparison,
            source_lock,
            v2_lock,
            audit,
            records,
            manifest,
            v2_report,
        )
        write_cut3r_source_competence_support_audit_report(
            comparison,
            source_lock,
            v2_lock,
            audit,
            records,
            manifest,
            v2_report,
            arguments.output,
            artifact,
        )
        print(artifact["source_competence_support_audit_report_id"])
        return 0 if artifact["audited_source_competence_pass"] else (
            3 if arguments.require_pass else 0
        )
    artifact = load_cut3r_source_competence_support_audit_report(
        comparison,
        source_lock,
        v2_lock,
        audit,
        records,
        manifest,
        v2_report,
        arguments.report,
    )
    if arguments.command == "verify-report":
        print(artifact["source_competence_support_audit_report_id"])
        return 0 if artifact["audited_source_competence_pass"] else (
            3 if arguments.require_pass else 0
        )
    mean_gate, identity_gate = source_competence_gates_audited(artifact)
    print(
        json.dumps(
            {
                "source_mean": mean_gate.to_dict(),
                "identity_reliability": identity_gate.to_dict(),
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


CUT3R_SOURCE_COMPETENCE_AUDIT_LOCK_SCHEMA = LOCK_SCHEMA
CUT3R_METRIC_SUPPORT_MANIFEST_SCHEMA = MANIFEST_SCHEMA
CUT3R_SOURCE_COMPETENCE_AUDIT_REPORT_SCHEMA = REPORT_SCHEMA
CUT3R_SOURCE_COMPETENCE_AUDIT_VERSION = VERSION
CUT3R_PROPER_SCORE_REFERENCE_FIT_SCOPE = PROPER_SCORE_REFERENCE_FIT_SCOPE

__all__ = [
    "CUT3R_METRIC_SUPPORT_MANIFEST_SCHEMA",
    "CUT3R_PROPER_SCORE_REFERENCE_FIT_SCOPE",
    "CUT3R_SOURCE_COMPETENCE_AUDIT_LOCK_SCHEMA",
    "CUT3R_SOURCE_COMPETENCE_AUDIT_REPORT_SCHEMA",
    "CUT3R_SOURCE_COMPETENCE_AUDIT_VERSION",
    "build_cut3r_metric_support_manifest",
    "build_cut3r_source_competence_audit_lock",
    "build_cut3r_source_competence_support_audit_report",
    "load_cut3r_metric_support_manifest",
    "load_cut3r_source_competence_audit_lock",
    "load_cut3r_source_competence_support_audit_report",
    "metric_support_from_manifest_entry",
    "source_competence_gates_audited",
    "validate_cut3r_metric_support_manifest",
    "validate_cut3r_source_competence_audit_lock",
    "validate_cut3r_source_competence_support_audit_report",
    "verify_cut3r_source_competence_audit_reference",
    "write_cut3r_metric_support_manifest",
    "write_cut3r_source_competence_audit_lock",
    "write_cut3r_source_competence_support_audit_report",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
