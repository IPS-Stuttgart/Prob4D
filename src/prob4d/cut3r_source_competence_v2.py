"""Exact common-support and paired-endpoint CUT3R source competence.

This additive version-2 layer keeps the existing CUT3R comparison lock and stable
``SourceProviderCompetenceReportV1``. It proves that candidate and baseline
metrics use identical frozen support, restricts the source-mean proper score to
one arm-neutral fixed scale, and adds paired seam, drift, association, identity,
and support decisions. Target access remains forbidden.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ._cut3r_source_competence_v2_common import (
    PROPER_SCORE_SEMANTICS,
    RECORDS_SCHEMA,
)
from ._cut3r_source_competence_v2_lock import (
    build_cut3r_source_competence_v2_lock,
    validate_cut3r_source_competence_v2_lock,
)
from ._cut3r_source_competence_v2_report import (
    build_cut3r_source_competence_v2_report,
    source_competence_gates_v2,
    validate_cut3r_source_competence_v2_report,
)
from ._cut3r_source_competence_v2_records import _normalize_v2_records
from .cut3r_comparison import load_cut3r_comparison_lock
from .cut3r_source_competence import (
    _publish_json,
    _strict_json,
    build_cut3r_source_competence_report,
    load_cut3r_source_competence_lock,
    write_cut3r_source_competence_report,
)


def write_cut3r_source_competence_v2_lock(
    comparison_lock: Any,
    source_competence_lock: Any,
    path: str | Path,
    lock: Mapping[str, Any],
) -> dict[str, Any]:
    payload = validate_cut3r_source_competence_v2_lock(
        comparison_lock,
        source_competence_lock,
        lock,
    )
    return _publish_json(
        path,
        payload,
        load_existing=lambda existing: load_cut3r_source_competence_v2_lock(
            comparison_lock,
            source_competence_lock,
            existing,
        ),
    )


def load_cut3r_source_competence_v2_lock(
    comparison_lock: Any,
    source_competence_lock: Any,
    path: str | Path,
) -> dict[str, Any]:
    return validate_cut3r_source_competence_v2_lock(
        comparison_lock,
        source_competence_lock,
        _strict_json(path),
    )


def write_cut3r_source_competence_v2_report(
    comparison_lock: Any,
    source_competence_lock: Any,
    common_support_lock: Any,
    records: Any,
    path: str | Path,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    payload = validate_cut3r_source_competence_v2_report(
        comparison_lock,
        source_competence_lock,
        common_support_lock,
        records,
        report,
    )
    return _publish_json(
        path,
        payload,
        load_existing=lambda existing: load_cut3r_source_competence_v2_report(
            comparison_lock,
            source_competence_lock,
            common_support_lock,
            records,
            existing,
        ),
    )


def load_cut3r_source_competence_v2_report(
    comparison_lock: Any,
    source_competence_lock: Any,
    common_support_lock: Any,
    records: Any,
    path: str | Path,
) -> dict[str, Any]:
    return validate_cut3r_source_competence_v2_report(
        comparison_lock,
        source_competence_lock,
        common_support_lock,
        records,
        _strict_json(path),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("comparison_lock")
    freeze.add_argument("source_competence_lock")
    freeze.add_argument("specification")
    freeze.add_argument("--output", required=True)
    verify_lock = commands.add_parser("verify-lock")
    verify_lock.add_argument("comparison_lock")
    verify_lock.add_argument("source_competence_lock")
    verify_lock.add_argument("common_support_lock")
    report = commands.add_parser("report")
    report.add_argument("comparison_lock")
    report.add_argument("source_competence_lock")
    report.add_argument("common_support_lock")
    report.add_argument("records")
    report.add_argument("--output", required=True)
    report.add_argument("--v1-output")
    report.add_argument("--require-pass", action="store_true")
    verify_report = commands.add_parser("verify-report")
    verify_report.add_argument("comparison_lock")
    verify_report.add_argument("source_competence_lock")
    verify_report.add_argument("common_support_lock")
    verify_report.add_argument("records")
    verify_report.add_argument("report")
    verify_report.add_argument("--require-pass", action="store_true")
    gates = commands.add_parser("gates")
    gates.add_argument("comparison_lock")
    gates.add_argument("source_competence_lock")
    gates.add_argument("common_support_lock")
    gates.add_argument("records")
    gates.add_argument("report")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    comparison = load_cut3r_comparison_lock(arguments.comparison_lock)
    source_lock = load_cut3r_source_competence_lock(
        comparison,
        arguments.source_competence_lock,
    )
    if arguments.command == "freeze":
        lock = build_cut3r_source_competence_v2_lock(
            comparison,
            source_lock,
            _strict_json(arguments.specification),
        )
        write_cut3r_source_competence_v2_lock(
            comparison,
            source_lock,
            arguments.output,
            lock,
        )
        print(lock["common_support_lock_id"])
        return 0
    v2_lock = load_cut3r_source_competence_v2_lock(
        comparison,
        source_lock,
        arguments.common_support_lock,
    )
    if arguments.command == "verify-lock":
        print(v2_lock["common_support_lock_id"])
        return 0
    records = _strict_json(arguments.records)
    if arguments.command == "report":
        report = build_cut3r_source_competence_v2_report(
            comparison,
            source_lock,
            v2_lock,
            records,
        )
        write_cut3r_source_competence_v2_report(
            comparison,
            source_lock,
            v2_lock,
            records,
            arguments.output,
            report,
        )
        if arguments.v1_output:
            _, v1_records = _normalize_v2_records(
                comparison,
                source_lock,
                v2_lock,
                records,
            )
            v1_report = build_cut3r_source_competence_report(
                comparison,
                source_lock,
                v1_records,
            )
            write_cut3r_source_competence_report(
                comparison,
                source_lock,
                v1_records,
                arguments.v1_output,
                v1_report,
            )
        print(report["source_competence_report_v2_id"])
        return 0 if report["source_competence_pass"] or not arguments.require_pass else 3
    report = load_cut3r_source_competence_v2_report(
        comparison,
        source_lock,
        v2_lock,
        records,
        arguments.report,
    )
    if arguments.command == "verify-report":
        print(report["source_competence_report_v2_id"])
        return 0 if report["source_competence_pass"] or not arguments.require_pass else 3
    mean_gate, identity_gate = source_competence_gates_v2(report)
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


# Compatibility names used by the checked-in v2 tests and documentation.
CUT3R_SOURCE_COMPETENCE_V2_PROPER_SCORE_SEMANTICS = PROPER_SCORE_SEMANTICS
CUT3R_SOURCE_COMPETENCE_V2_RECORDS_SCHEMA = RECORDS_SCHEMA


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
