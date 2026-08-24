#!/usr/bin/env python3
"""Render the README provider table from one pinned strict snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SNAPSHOT = ROOT / "evidence" / "provider_status_snapshot_v1.json"
DEFAULT_README = ROOT / "README.md"
START_MARKER = "<!-- provider-evidence-status:begin -->"
END_MARKER = "<!-- provider-evidence-status:end -->"
ALLOWED_EVIDENCE_STATES = frozenset(
    {
        "controlled_only",
        "diagnostic_only",
        "not_established",
        "retrospective_negative",
        "terminal_support_negative",
    }
)


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def load_snapshot(path: Path) -> dict[str, Any]:
    """Load strict JSON without duplicate keys or non-finite constants."""

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(
            handle,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    if type(value) is not dict:
        raise ValueError("provider status snapshot must be a JSON object")
    return value


def _expect_fields(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    name: str,
) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{name} fields changed; "
            f"expected={sorted(expected)}, observed={sorted(value)}"
        )


def _nonempty_string(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(
            f"{name} must be a nonempty exact string without surrounding whitespace"
        )
    return value


def _git_blob_sha1(path: Path) -> str:
    value = path.read_bytes()
    header = f"blob {len(value)}\0".encode("ascii")
    return hashlib.sha1(header + value).hexdigest()


def _source_path(value: object, *, root: Path) -> Path:
    text = _nonempty_string(value, name="evidence_source.path")
    relative = Path(text)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("evidence_source.path must remain inside the repository")
    path = root / relative
    if not path.is_file() or path.is_symlink():
        raise ValueError("evidence_source.path must name a regular repository file")
    return path


def validate_snapshot(snapshot: dict[str, Any], *, root: Path = ROOT) -> None:
    """Validate structure, source identity, route order, and claim boundaries."""

    _expect_fields(
        snapshot,
        {
            "contract",
            "contract_version",
            "snapshot_date",
            "evidence_source",
            "routes",
            "not_authorized",
        },
        name="provider status snapshot",
    )
    if snapshot["contract"] != "prob4d.provider-status-snapshot":
        raise ValueError("unexpected provider status snapshot contract")
    if type(snapshot["contract_version"]) is not int or snapshot["contract_version"] != 1:
        raise ValueError("unexpected provider status snapshot version")
    try:
        date.fromisoformat(
            _nonempty_string(snapshot["snapshot_date"], name="snapshot_date")
        )
    except ValueError as error:
        raise ValueError("snapshot_date must be an ISO date") from error

    source = snapshot["evidence_source"]
    if type(source) is not dict:
        raise ValueError("evidence_source must be an object")
    _expect_fields(
        source,
        {"path", "git_blob_sha1", "role"},
        name="evidence_source",
    )
    source_path = _source_path(source["path"], root=root)
    expected_sha = _nonempty_string(
        source["git_blob_sha1"],
        name="evidence_source.git_blob_sha1",
    )
    _nonempty_string(source["role"], name="evidence_source.role")
    if len(expected_sha) != 40 or any(
        character not in "0123456789abcdef" for character in expected_sha
    ):
        raise ValueError("evidence_source.git_blob_sha1 must be a lowercase SHA-1 digest")
    actual_sha = _git_blob_sha1(source_path)
    if actual_sha != expected_sha:
        raise ValueError(
            "pinned provider status contract changed: "
            f"expected {expected_sha}, observed {actual_sha}"
        )

    routes = snapshot["routes"]
    if not isinstance(routes, list) or not routes:
        raise ValueError("routes must be a nonempty list")
    seen_ids: set[str] = set()
    seen_routes: set[str] = set()
    observed_orders: set[int] = set()
    for route in routes:
        if type(route) is not dict:
            raise ValueError("each provider route must be an object")
        _expect_fields(
            route,
            {
                "id",
                "table_order",
                "provider_route",
                "implementation_state",
                "evidence_state",
                "evidence_boundary",
            },
            name="provider route",
        )
        route_id = _nonempty_string(route["id"], name="route.id")
        if route_id in seen_ids:
            raise ValueError(f"duplicate provider route id: {route_id}")
        seen_ids.add(route_id)
        provider_route = _nonempty_string(
            route["provider_route"],
            name=f"{route_id}.provider_route",
        )
        if provider_route in seen_routes:
            raise ValueError(f"duplicate provider route label: {provider_route}")
        seen_routes.add(provider_route)
        order = route["table_order"]
        if type(order) is not int or order < 1 or order in observed_orders:
            raise ValueError(f"invalid or duplicate table_order for {route_id}")
        observed_orders.add(order)
        evidence_state = _nonempty_string(
            route["evidence_state"],
            name=f"{route_id}.evidence_state",
        )
        if evidence_state not in ALLOWED_EVIDENCE_STATES:
            raise ValueError(f"unsupported evidence state for {route_id}: {evidence_state}")
        _nonempty_string(
            route["implementation_state"],
            name=f"{route_id}.implementation_state",
        )
        _nonempty_string(
            route["evidence_boundary"],
            name=f"{route_id}.evidence_boundary",
        )
    expected_orders = set(range(1, len(routes) + 1))
    if observed_orders != expected_orders:
        raise ValueError("provider route table_order values must be contiguous from one")

    not_authorized = snapshot["not_authorized"]
    if not isinstance(not_authorized, list) or not not_authorized:
        raise ValueError("not_authorized must be a nonempty list")
    values = [
        _nonempty_string(item, name=f"not_authorized[{index}]")
        for index, item in enumerate(not_authorized)
    ]
    if len(values) != len(set(values)):
        raise ValueError("not_authorized contains duplicate values")


def _escape_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def render_status_block(snapshot: dict[str, Any]) -> str:
    """Render the canonical README status block and provenance note."""

    routes: Sequence[Mapping[str, Any]] = sorted(
        snapshot["routes"],
        key=lambda item: item["table_order"],
    )
    snapshot_date = _escape_cell(snapshot["snapshot_date"])
    lines = [
        START_MARKER,
        f"Snapshot: **{snapshot_date}**. Adapter maturity and scientific evidence are separate.",
        "The authoritative cross-repository claim and data-access status remains in",
        "[`FlorianPfaff/BayesianPhysTwin-Paper`]"
        "(https://github.com/FlorianPfaff/BayesianPhysTwin-Paper).",
        "",
        "| Provider route | Implementation state | Current evidence boundary |",
        "| --- | --- | --- |",
    ]
    for route in routes:
        lines.append(
            "| "
            + " | ".join(
                (
                    _escape_cell(route["provider_route"]),
                    _escape_cell(route["implementation_state"]),
                    _escape_cell(route["evidence_boundary"]),
                )
            )
            + " |"
        )
    lines.extend(
        (
            "",
            "This table is generated from",
            "[`evidence/provider_status_snapshot_v1.json`]"
            "(evidence/provider_status_snapshot_v1.json),",
            "which pins the repository provider-status contract by Git blob identity.",
            "Regenerate it with `python scripts/render_provider_status.py --write`; CI",
            "checks that the contract, snapshot, and README stay synchronized.",
            END_MARKER,
        )
    )
    return "\n".join(lines)


def replace_status_block(readme: str, block: str) -> str:
    """Replace exactly one generated provider-status block."""

    if readme.count(START_MARKER) != 1 or readme.count(END_MARKER) != 1:
        raise ValueError("README must contain exactly one provider-status marker pair")
    start = readme.index(START_MARKER)
    end = readme.index(END_MARKER, start) + len(END_MARKER)
    return readme[:start] + block + readme[end:]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--readme", type=Path, default=DEFAULT_README)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true", help="update README in place")
    mode.add_argument(
        "--check",
        action="store_true",
        help="fail when README is not synchronized (default)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    snapshot = load_snapshot(args.snapshot)
    validate_snapshot(snapshot, root=ROOT)
    current = args.readme.read_text(encoding="utf-8")
    expected = replace_status_block(current, render_status_block(snapshot))
    if args.write:
        args.readme.write_text(expected, encoding="utf-8")
        return 0
    if current != expected:
        raise SystemExit(
            "README provider status is stale; run "
            "python scripts/render_provider_status.py --write"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
