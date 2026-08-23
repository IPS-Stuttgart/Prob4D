"""Denominator-safe exact small-sample CUT3R recurrent-state recovery analysis."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from ._cut3r_recovery_v2_exact import (
    _count_vectors,
    _exact_group_bootstrap,
    _leave_one_group_out,
    _triplet_v2,
)
from ._cut3r_recovery_v2_report import (
    _build_v2_from_validated_v1_report,
    build_cut3r_recurrent_state_recovery_v2_report,
    cut3r_recurrent_state_recovery_v2_summary,
    load_cut3r_recurrent_state_recovery_v2_report,
    validate_cut3r_recurrent_state_recovery_v2_report,
    write_cut3r_recurrent_state_recovery_v2_report,
)
from ._cut3r_recovery_v2_spec import (
    CUT3R_RECURRENT_STATE_RECOVERY_V2_CLAIM_BOUNDARY,
    CUT3R_RECURRENT_STATE_RECOVERY_V2_INTERVAL_METHOD,
    CUT3R_RECURRENT_STATE_RECOVERY_V2_PRIMARY_ENDPOINT,
    CUT3R_RECURRENT_STATE_RECOVERY_V2_SCHEMA,
    CUT3R_RECURRENT_STATE_RECOVERY_V2_SECONDARY_ENDPOINT,
    CUT3R_RECURRENT_STATE_RECOVERY_V2_SPEC_CLAIM_BOUNDARY,
    CUT3R_RECURRENT_STATE_RECOVERY_V2_SPEC_SCHEMA,
    CUT3R_RECURRENT_STATE_RECOVERY_V2_SPEC_VERSION,
    CUT3R_RECURRENT_STATE_RECOVERY_V2_VERSION,
    build_cut3r_recurrent_state_recovery_v2_specification,
    load_cut3r_recurrent_state_recovery_v2_specification,
    validate_cut3r_recurrent_state_recovery_v2_specification,
)
from .cut3r_recurrent_state_recovery import (
    CUT3R_RECURRENT_STATE_RECOVERY_METRICS,
    _add_bound_evidence_arguments,
    _load_bound_inputs,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prob4d prediction cut3r-recovery-v2",
        description=__doc__,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="build the bound v2 recovery report")
    _add_bound_evidence_arguments(build)
    build.add_argument("--output", required=True)
    verify = commands.add_parser("verify", help="rebuild and verify a v2 report")
    _add_bound_evidence_arguments(verify)
    verify.add_argument("report")
    summarize = commands.add_parser("summarize", help="summarize a verified v2 report")
    _add_bound_evidence_arguments(summarize)
    summarize.add_argument("report")
    summarize.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(list(argv) if argv is not None else None)
    inputs = _load_bound_inputs(arguments)
    if arguments.command == "build":
        report = build_cut3r_recurrent_state_recovery_v2_report(*inputs)
        write_cut3r_recurrent_state_recovery_v2_report(
            *inputs,
            arguments.output,
            report,
        )
        print(report["recurrent_state_recovery_v2_report_id"])
        return 0
    report = load_cut3r_recurrent_state_recovery_v2_report(
        *inputs,
        arguments.report,
    )
    if arguments.command == "verify":
        print(report["recurrent_state_recovery_v2_report_id"])
        return 0
    summary = cut3r_recurrent_state_recovery_v2_summary(report)
    if arguments.json:
        print(json.dumps(summary, sort_keys=True, indent=2, allow_nan=False))
    else:
        print(f"report_id: {summary['recurrent_state_recovery_v2_report_id']}")
        print(f"evaluable groups: {summary['evaluable_group_count']}")
        for metric, result in summary["metrics"].items():
            print(
                f"{metric}: gain={result['prob4d_gain']} "
                f"recovery={result['recovery_status']} "
                f"fraction={result['recovery_fraction']}"
            )
        print(f"target access: {summary['target_access']}")
    return 0


__all__ = [
    "CUT3R_RECURRENT_STATE_RECOVERY_V2_CLAIM_BOUNDARY",
    "CUT3R_RECURRENT_STATE_RECOVERY_V2_INTERVAL_METHOD",
    "CUT3R_RECURRENT_STATE_RECOVERY_V2_PRIMARY_ENDPOINT",
    "CUT3R_RECURRENT_STATE_RECOVERY_V2_SCHEMA",
    "CUT3R_RECURRENT_STATE_RECOVERY_V2_SECONDARY_ENDPOINT",
    "CUT3R_RECURRENT_STATE_RECOVERY_V2_SPEC_CLAIM_BOUNDARY",
    "CUT3R_RECURRENT_STATE_RECOVERY_V2_SPEC_SCHEMA",
    "CUT3R_RECURRENT_STATE_RECOVERY_V2_SPEC_VERSION",
    "CUT3R_RECURRENT_STATE_RECOVERY_V2_VERSION",
    "_build_v2_from_validated_v1_report",
    "build_cut3r_recurrent_state_recovery_v2_report",
    "build_cut3r_recurrent_state_recovery_v2_specification",
    "cut3r_recurrent_state_recovery_v2_summary",
    "load_cut3r_recurrent_state_recovery_v2_report",
    "load_cut3r_recurrent_state_recovery_v2_specification",
    "main",
    "validate_cut3r_recurrent_state_recovery_v2_report",
    "validate_cut3r_recurrent_state_recovery_v2_specification",
    "write_cut3r_recurrent_state_recovery_v2_report",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
