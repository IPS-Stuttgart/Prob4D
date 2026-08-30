"""Inventory the public DOT rope archives without decoding trajectory payloads."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import stat
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any
from zipfile import BadZipFile, ZipFile, ZipInfo

EXPECTED_ARCHIVES: tuple[tuple[str, str, int, int], ...] = (
    ("R01-10.zip", "ca546ff5f22c0279123ccb18509858ee", 1, 10),
    ("R11-20.zip", "23ce3e7067465d3edabe20b4c7cfa388", 11, 20),
    ("R21-30.zip", "8aee77f79d1aff6e1f3fd21886b251a0", 21, 30),
    ("R31-40.zip", "8a96081d20af9fa486fafa8e24e54442", 31, 40),
    ("R41-50.zip", "fe1edaf708d0dc55ebf108d24badacce", 41, 50),
    ("R51-60.zip", "aa74ff1f12a898d61e5255abf09680aa", 51, 60),
    ("R61-70.zip", "da7a7a11a7d4e4541c891f3fafbe07b7", 61, 70),
)

REQUEST_SCHEMA = "dot-rope-curvature-inventory-request-v1"
RESULT_SCHEMA = "dot-rope-curvature-inventory-v1"
FIXED_DATASET_ROOT = "/mnt/seagate10tb/florianpfaff/datasets/dot-rope"
FIXED_RUNNER_LABEL = "gpuserver6000"
SEQUENCE_PATTERN = re.compile(r"(?<![A-Za-z0-9])R(\d{2})(?!\d)", re.IGNORECASE)
DIGIT_PATTERN = re.compile(r"\d+")


def split_for_sequence(sequence_number: int) -> str:
    """Return the prospectively frozen split for one DOT rope sequence."""

    if 1 <= sequence_number <= 30:
        return "development"
    if 31 <= sequence_number <= 40:
        return "calibration"
    if 41 <= sequence_number <= 70:
        return "held_out"
    return "outside_protocol"


def sequence_id_from_member(name: str) -> str | None:
    """Extract an ``Rxx`` identifier from an archive member path."""

    match = SEQUENCE_PATTERN.search(name)
    if match is None:
        return None
    return f"R{int(match.group(1)):02d}"


def member_name_is_safe(name: str) -> bool:
    """Reject absolute, traversal, NUL-containing, or backslash paths."""

    if not name or "\x00" in name or "\\" in name:
        return False
    path = PurePosixPath(name)
    return not path.is_absolute() and ".." not in path.parts


def zip_info_is_symlink(info: ZipInfo) -> bool:
    """Return whether a ZIP member advertises a symbolic-link file mode."""

    mode = info.external_attr >> 16
    return stat.S_IFMT(mode) == stat.S_IFLNK


def normalized_member_pattern(name: str) -> str:
    """Collapse numeric path components to expose repeated dataset layouts."""

    normalized = SEQUENCE_PATTERN.sub("R##", name)
    return DIGIT_PATTERN.sub("#", normalized)


def hash_file(path: Path, algorithm: str = "md5") -> str:
    """Hash a file without loading it into memory."""

    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _extension(name: str) -> str:
    suffix = PurePosixPath(name).suffix.lower()
    return suffix if suffix else "<none>"


def inventory_archive(
    path: Path,
    *,
    expected_md5: str,
    sequence_start: int,
    sequence_stop: int,
    source_sample_limit: int = 400,
) -> tuple[dict[str, Any], list[str]]:
    """Inspect one archive using its central directory only.

    Compressed archive bytes are read for the publisher MD5. Member payloads are
    never opened or extracted by this function.
    """

    if not path.is_file():
        raise FileNotFoundError(f"Missing required archive: {path.name}")

    observed_md5 = hash_file(path)
    extension_counts: Counter[str] = Counter()
    prefix_counts: Counter[str] = Counter()
    pattern_counts: Counter[str] = Counter()
    sequence_counts: Counter[str] = Counter()
    member_names: set[str] = set()
    duplicate_names: list[str] = []
    unsafe_names: list[str] = []
    encrypted_names: list[str] = []
    symlink_names: list[str] = []
    source_sample: list[str] = []
    file_count = 0
    directory_count = 0
    compressed_bytes = 0
    uncompressed_bytes = 0
    largest_member_bytes = 0

    try:
        with ZipFile(path) as archive:
            archive_comment_bytes = len(archive.comment)
            for info in archive.infolist():
                name = info.filename
                if name in member_names:
                    duplicate_names.append(name)
                member_names.add(name)
                if not member_name_is_safe(name):
                    unsafe_names.append(name)
                if info.flag_bits & 0x1:
                    encrypted_names.append(name)
                if zip_info_is_symlink(info):
                    symlink_names.append(name)
                if info.is_dir():
                    directory_count += 1
                    continue

                file_count += 1
                compressed_bytes += info.compress_size
                uncompressed_bytes += info.file_size
                largest_member_bytes = max(largest_member_bytes, info.file_size)
                extension_counts[_extension(name)] += 1
                path_parts = PurePosixPath(name).parts
                prefix_counts[path_parts[0] if path_parts else "<root>"] += 1
                pattern_counts[normalized_member_pattern(name)] += 1
                sequence_id = sequence_id_from_member(name)
                if sequence_id is not None:
                    sequence_counts[sequence_id] += 1
                    sequence_number = int(sequence_id[1:])
                    if (
                        split_for_sequence(sequence_number) == "development"
                        and len(source_sample) < source_sample_limit
                    ):
                        source_sample.append(name)
    except BadZipFile as exc:
        raise ValueError(f"Invalid ZIP central directory: {path.name}") from exc

    record: dict[str, Any] = {
        "archive": path.name,
        "size_bytes": path.stat().st_size,
        "expected_md5": expected_md5,
        "observed_md5": observed_md5,
        "publisher_md5_matches": observed_md5 == expected_md5,
        "nominal_sequence_start": sequence_start,
        "nominal_sequence_stop": sequence_stop,
        "nominal_split": split_for_sequence(sequence_start),
        "file_count": file_count,
        "directory_count": directory_count,
        "compressed_member_bytes": compressed_bytes,
        "uncompressed_member_bytes": uncompressed_bytes,
        "largest_member_bytes": largest_member_bytes,
        "archive_comment_bytes": archive_comment_bytes,
        "duplicate_member_count": len(duplicate_names),
        "unsafe_member_count": len(unsafe_names),
        "encrypted_member_count": len(encrypted_names),
        "symlink_member_count": len(symlink_names),
        "sequence_counts": dict(sorted(sequence_counts.items())),
        "extension_counts": dict(extension_counts.most_common()),
        "top_level_prefix_counts": dict(prefix_counts.most_common(30)),
        "common_member_patterns": dict(pattern_counts.most_common(80)),
        "payload_members_opened": 0,
    }
    problems = {
        "duplicate_members": duplicate_names[:20],
        "unsafe_members": unsafe_names[:20],
        "encrypted_members": encrypted_names[:20],
        "symlink_members": symlink_names[:20],
    }
    record["problem_samples"] = problems
    return record, source_sample


def validate_request(
    request_path: Path,
    *,
    expected_source_revision: str | None = None,
) -> dict[str, Any]:
    """Validate the immutable execution request without touching the dataset."""

    request = json.loads(request_path.read_text(encoding="utf-8"))
    required: dict[str, Any] = {
        "schema": REQUEST_SCHEMA,
        "mode": "inventory",
        "dataset_root": FIXED_DATASET_ROOT,
        "runner_label": FIXED_RUNNER_LABEL,
        "development_sequences": "R01-R30",
        "calibration_sequences": "R31-R40",
        "held_out_sequences": "R41-R70",
        "held_out_payload_decode_authorized": False,
        "provider_execution_authorized": False,
        "paper_claim_authorized": False,
        "evidence_class": (
            "public-real-data inventory; central-directory metadata only; "
            "no provider or target outcome claim"
        ),
    }
    for key, expected in required.items():
        if request.get(key) != expected:
            raise ValueError(
                f"Invalid request field {key!r}: expected {expected!r}, "
                f"got {request.get(key)!r}"
            )

    allowed = set(required) | {"request_id", "expected_source_revision"}
    unknown = set(request) - allowed
    if unknown:
        raise ValueError(f"Unknown request fields: {sorted(unknown)}")
    if not request.get("request_id"):
        raise ValueError("request_id must be nonempty")
    revision = request.get("expected_source_revision", "")
    if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
        raise ValueError("expected_source_revision must be a lowercase commit SHA")
    if expected_source_revision is not None and revision != expected_source_revision:
        raise ValueError(
            "request expected_source_revision does not match the authorized source revision"
        )
    return request


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_inventory(
    *,
    dataset_root: Path,
    output_dir: Path,
    request_path: Path,
    source_revision: str,
    execution_revision: str,
) -> dict[str, Any]:
    """Run the target-closed archive inventory and write aggregate evidence."""

    request = validate_request(
        request_path,
        expected_source_revision=source_revision,
    )
    if str(dataset_root) != FIXED_DATASET_ROOT:
        raise ValueError(f"dataset_root must equal {FIXED_DATASET_ROOT}")
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    output_dir.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []
    source_samples: list[str] = []
    for archive_name, expected_md5, sequence_start, sequence_stop in EXPECTED_ARCHIVES:
        record, sample = inventory_archive(
            dataset_root / archive_name,
            expected_md5=expected_md5,
            sequence_start=sequence_start,
            sequence_stop=sequence_stop,
            source_sample_limit=max(0, 600 - len(source_samples)),
        )
        records.append(record)
        source_samples.extend(sample)

    expected_names = {item[0] for item in EXPECTED_ARCHIVES}
    observed_zip_names = {path.name for path in dataset_root.glob("*.zip")}
    extra_zip_names = sorted(observed_zip_names - expected_names)
    missing_zip_names = sorted(expected_names - observed_zip_names)
    valid = (
        not missing_zip_names
        and all(record["publisher_md5_matches"] for record in records)
        and all(record["duplicate_member_count"] == 0 for record in records)
        and all(record["unsafe_member_count"] == 0 for record in records)
        and all(record["encrypted_member_count"] == 0 for record in records)
        and all(record["symlink_member_count"] == 0 for record in records)
    )

    aggregate_extensions: Counter[str] = Counter()
    aggregate_sequences: Counter[str] = Counter()
    for record in records:
        aggregate_extensions.update(record["extension_counts"])
        aggregate_sequences.update(record["sequence_counts"])

    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "decision": "inventory-pass" if valid else "inventory-fail",
        "classification": (
            "public real-data archive inventory; no member payload decoded; "
            "not provider competence or paper evidence"
        ),
        "request": request,
        "source_revision": source_revision,
        "execution_revision": execution_revision,
        "github_run_id": os.environ.get("GITHUB_RUN_ID"),
        "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "runner_name": os.environ.get("RUNNER_NAME"),
        "runner_label_required": FIXED_RUNNER_LABEL,
        "dataset": {
            "persistent_id": "doi:10.13021/ORC2020/XXLVXM",
            "publisher_version": "29.0",
            "license": "CC0-1.0",
            "dataset_root": str(dataset_root),
            "expected_archive_count": len(EXPECTED_ARCHIVES),
            "observed_zip_count": len(observed_zip_names),
            "missing_zip_names": missing_zip_names,
            "extra_zip_names": extra_zip_names,
        },
        "information_boundary": {
            "development_sequences": "R01-R30",
            "calibration_sequences": "R31-R40",
            "held_out_sequences": "R41-R70",
            "archive_bytes_read_for_md5": True,
            "zip_central_directories_read": True,
            "member_payloads_opened": 0,
            "member_payloads_extracted": 0,
            "held_out_payloads_decoded": 0,
            "provider_executed": False,
            "outcome_metrics_computed": False,
        },
        "archives": records,
        "aggregate_extension_counts": dict(aggregate_extensions.most_common()),
        "aggregate_sequence_counts": dict(sorted(aggregate_sequences.items())),
    }

    (output_dir / "inventory.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    archive_rows = [
        {
            key: record[key]
            for key in (
                "archive",
                "size_bytes",
                "expected_md5",
                "observed_md5",
                "publisher_md5_matches",
                "nominal_sequence_start",
                "nominal_sequence_stop",
                "nominal_split",
                "file_count",
                "directory_count",
                "compressed_member_bytes",
                "uncompressed_member_bytes",
                "largest_member_bytes",
                "duplicate_member_count",
                "unsafe_member_count",
                "encrypted_member_count",
                "symlink_member_count",
                "payload_members_opened",
            )
        }
        for record in records
    ]
    _write_csv(output_dir / "archives.csv", list(archive_rows[0]), archive_rows)
    _write_csv(
        output_dir / "extensions.csv",
        ["extension", "member_count"],
        [
            {"extension": extension, "member_count": count}
            for extension, count in aggregate_extensions.most_common()
        ],
    )
    _write_csv(
        output_dir / "sequences.csv",
        ["sequence", "split", "member_count"],
        [
            {
                "sequence": sequence,
                "split": split_for_sequence(int(sequence[1:])),
                "member_count": count,
            }
            for sequence, count in sorted(aggregate_sequences.items())
        ],
    )
    (output_dir / "development_member_sample.txt").write_text(
        "\n".join(source_samples[:600]) + ("\n" if source_samples else ""),
        encoding="utf-8",
    )

    total_size = sum(record["size_bytes"] for record in records)
    total_members = sum(record["file_count"] for record in records)
    report_lines = [
        "# DOT rope target-closed archive inventory",
        "",
        f"- Decision: **{result['decision']}**",
        f"- Runner: `{result['runner_name']}`; required label: `{FIXED_RUNNER_LABEL}`",
        f"- Archive count: `{len(records)}`; total bytes: `{total_size}`",
        f"- File members represented: `{total_members}`",
        "- Publisher MD5 matches: "
        + ("all seven" if all(row["publisher_md5_matches"] for row in records) else "NO"),
        "- Member payloads decoded/extracted: `0/0`",
        "- Frozen split: development `R01-R30`, calibration `R31-R40`, "
        "held-out `R41-R70`",
        "",
        "## Archive summary",
        "",
        "| Archive | Split | Files | Uncompressed bytes | MD5 |",
        "|---|---:|---:|---:|---:|",
    ]
    for record in records:
        report_lines.append(
            "| {archive} | {split} | {files} | {bytes} | {status} |".format(
                archive=record["archive"],
                split=record["nominal_split"],
                files=record["file_count"],
                bytes=record["uncompressed_member_bytes"],
                status="pass" if record["publisher_md5_matches"] else "FAIL",
            )
        )
    report_lines.extend(
        [
            "",
            "## Most common file extensions",
            "",
        ]
    )
    for extension, count in aggregate_extensions.most_common(20):
        report_lines.append(f"- `{extension}`: {count}")
    report_lines.extend(
        [
            "",
            "## Development-member sample",
            "",
            "The first development paths are listed to guide a source-only parser. "
            "No member bytes were opened.",
            "",
            "```text",
            *source_samples[:120],
            "```",
            "",
            "This run is an operational inventory only. It does not establish "
            "real-provider accuracy, calibration, BayesianPhysTwin benefit, "
            "Causal4D benefit, or state of the art.",
        ]
    )
    report = "\n".join(report_lines) + "\n"
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    print(report)

    if not valid:
        raise RuntimeError("DOT rope archive inventory failed; inspect inventory.json")
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-request")
    validate.add_argument("--request", type=Path, required=True)
    validate.add_argument("--expected-source-revision")

    inventory = subparsers.add_parser("inventory")
    inventory.add_argument("--dataset-root", type=Path, required=True)
    inventory.add_argument("--output", type=Path, required=True)
    inventory.add_argument("--request", type=Path, required=True)
    inventory.add_argument("--source-revision", required=True)
    inventory.add_argument("--execution-revision", required=True)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.command == "validate-request":
        request = validate_request(
            args.request,
            expected_source_revision=args.expected_source_revision,
        )
        print(json.dumps(request, sort_keys=True))
        return 0
    if args.command == "inventory":
        run_inventory(
            dataset_root=args.dataset_root,
            output_dir=args.output,
            request_path=args.request,
            source_revision=args.source_revision,
            execution_revision=args.execution_revision,
        )
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
