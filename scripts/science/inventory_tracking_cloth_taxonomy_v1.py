#!/usr/bin/env python3
"""Inventory Tracking Cloth recording names without opening CSV contents."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


_GENERIC_TOKENS = {
    "csv",
    "data",
    "dataset",
    "recording",
    "recordings",
    "tracking",
    "cloth",
    "deformation",
    "trial",
    "trials",
}


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.split(r"[^a-z0-9]+", text.lower())
        if token
        and token not in _GENERIC_TOKENS
        and not token.isdigit()
        and len(token) >= 3
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def run(dataset_root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    dataset = protocol["dataset"]
    paths = sorted(dataset_root.rglob("*.csv"))
    expected = int(dataset["expected_csv_files"])
    if len(paths) != expected:
        raise RuntimeError(f"expected {expected} CSV files, found {len(paths)}")

    source_aliases = {
        str(label): [str(alias).lower() for alias in aliases]
        for label, aliases in dataset["source_aliases"].items()
    }
    target_aliases = [str(alias).lower() for alias in dataset["target_aliases"]]

    token_members: dict[str, list[str]] = defaultdict(list)
    directory_counts: Counter[str] = Counter()
    alias_counts: Counter[str] = Counter()
    unassigned: list[str] = []
    multiply_assigned: dict[str, list[str]] = {}
    files: list[dict[str, Any]] = []

    for path in paths:
        relative = path.relative_to(dataset_root).as_posix()
        lower = relative.lower()
        for token in sorted(_tokens(relative)):
            token_members[token].append(relative)
        parts = Path(relative).parts
        for depth, part in enumerate(parts[:-1]):
            directory_counts[f"{depth}:{part}"] += 1

        hits: list[str] = []
        for label, aliases in source_aliases.items():
            if any(alias in lower for alias in aliases):
                hits.append(label)
        if any(alias in lower for alias in target_aliases):
            hits.append("collision")
        hits = sorted(set(hits))
        if len(hits) == 1:
            alias_counts[hits[0]] += 1
        elif not hits:
            unassigned.append(relative)
        else:
            multiply_assigned[relative] = hits

        metadata = path.stat()
        files.append(
            {
                "relative_path": relative,
                "bytes": int(metadata.st_size),
                "alias_hits": hits,
                "tokens": sorted(_tokens(relative)),
            }
        )

    token_counts = [
        {
            "token": token,
            "count": len(members),
            "members": sorted(members),
        }
        for token, members in token_members.items()
    ]
    token_counts.sort(key=lambda row: (-int(row["count"]), str(row["token"])))

    return {
        "schema": "prob4d.tracking-cloth-taxonomy-inventory.v1",
        "dataset_root": str(dataset_root),
        "expected_csv_files": expected,
        "csv_file_count": len(files),
        "total_csv_bytes": sum(int(row["bytes"]) for row in files),
        "alias_counts": dict(sorted(alias_counts.items())),
        "unassigned_count": len(unassigned),
        "unassigned_paths": sorted(unassigned),
        "multiply_assigned_count": len(multiply_assigned),
        "multiply_assigned_paths": dict(sorted(multiply_assigned.items())),
        "directory_counts": dict(sorted(directory_counts.items())),
        "token_membership": token_counts,
        "files": files,
        "information_boundary": {
            "csv_file_contents_opened": False,
            "csv_metadata_read": True,
            "relative_paths_read": True,
            "dataset_mutated": False,
        },
    }


def write_summary(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# Tracking Cloth taxonomy inventory",
        "",
        f"- CSV files: `{result['csv_file_count']}`",
        f"- Alias counts: `{json.dumps(result['alias_counts'], sort_keys=True)}`",
        f"- Unassigned: `{result['unassigned_count']}`",
        f"- Multiply assigned: `{result['multiply_assigned_count']}`",
        "",
        "## Token membership counts",
        "",
    ]
    for row in result["token_membership"]:
        lines.append(f"- `{row['token']}`: {row['count']}")
    lines.extend(["", "## Unassigned paths", ""])
    lines.extend(f"- `{item}`" for item in result["unassigned_paths"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.dataset_root, args.protocol)
    _write_json(args.output_dir / "result.json", result)
    write_summary(result, args.output_dir / "summary.md")
    print(json.dumps({"csv_file_count": result["csv_file_count"]}))


if __name__ == "__main__":
    main()
