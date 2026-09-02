from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import zipfile

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "science"
    / "audit_tracking_cloth_orbit_tube_roster.py"
)
SPEC = importlib.util.spec_from_file_location(
    "audit_tracking_cloth_orbit_tube_roster",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _header(*labels: str, include_data: bool = True) -> bytes:
    rows = [
        ["Format Version", "1.23", "Take Name", "synthetic"],
        ["Frame", "Time (Seconds)"]
        + sum((["Marker", "", ""] for _ in labels), []),
        ["", ""] + sum((["Position", "", ""] for _ in labels), []),
        ["", ""] + sum(([f"cloth:{label}", "", ""] for label in labels), []),
        ["", ""] + sum(([label, "", ""] for label in labels), []),
        ["", ""] + sum((["X", "Y", "Z"] for _ in labels), []),
        ["", ""] + sum((["Millimeters", "Millimeters", "Millimeters"] for _ in labels), []),
    ]
    if include_data:
        rows.append(["0", "0.000"] + ["1.0", "2.0", "3.0"] * len(labels))
    stream = io.StringIO(newline="")
    import csv

    writer = csv.writer(stream, lineterminator="\n")
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _hash(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    digest.update(path.read_bytes())
    return digest.hexdigest()


def test_parse_header_labels_ignores_trajectory_values() -> None:
    payload = _header("1", "5", "20")
    first_seven = b"\n".join(payload.splitlines()[:7]) + b"\n"
    assert MODULE._parse_header_labels(first_seven, 7) == ("1", "5", "20")


def test_parse_header_rejects_a_data_row_inside_the_boundary() -> None:
    rows = _header("1", "5", "20").splitlines()
    bad_header = b"\n".join(rows[:6] + [rows[7]]) + b"\n"
    with pytest.raises(ValueError, match="trajectory data row"):
        MODULE._parse_header_labels(bad_header, 7)


def test_safe_member_rejects_path_traversal() -> None:
    with pytest.raises(ValueError, match="unsafe ZIP member"):
        MODULE._safe_member("../outside.csv")


def test_complete_audit_fixes_roles_before_header_read(tmp_path: Path) -> None:
    archive_path = tmp_path / "tracking_dataset.zip"
    fresh = [
        "tracking_dataset/Self-collisions/cotton_A2_a.csv",
        "tracking_dataset/Self-collisions/denim_A2_b.csv",
        "tracking_dataset/Self-collisions/wool_A2_c.csv",
    ]
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for index, member in enumerate(fresh):
            labels = ("1", "5", "20") if index != 1 else ("1", "5", "20", "21")
            archive.writestr(member, _header(*labels))
        archive.writestr("tracking_dataset/Hitting/cotton_A2_hitting.csv", _header("1", "5", "20"))

    protocol = {
        "protocol_id": "synthetic-sealed-audit",
        "dataset": {
            "archive_filename": archive_path.name,
            "archive_md5": _hash(archive_path, "md5"),
            "archive_sha256": _hash(archive_path, "sha256"),
            "expected_csv_files": 4,
            "fresh_category_token": "Self-collisions",
            "expected_fresh_files": 3,
        },
        "split": {
            "sha256_salt": "fixed-before-header-read",
            "source_count": 1,
            "calibration_count": 1,
            "target_count": 1,
        },
        "header_audit": {
            "line_count": 7,
            "maximum_line_bytes": 65536,
        },
        "claim_boundary": ["synthetic test"],
    }

    result = MODULE.run_audit(protocol, tmp_path)
    assert result["counts"] == {
        "all_csv": 4,
        "fresh": 3,
        "source": 1,
        "calibration": 1,
        "target": 1,
    }
    assert result["marker_intersections"]["all_roles"] == ["1", "5", "20"]
    assert sum(record["trajectory_rows_read"] for record in result["records"]) == 0
    assert result["information_boundary"] == {
        "zip_directory_opened": True,
        "fixed_header_rows_opened": True,
        "raw_header_text_published": False,
        "trajectory_rows_read": 0,
        "source_trajectory_values_opened": False,
        "calibration_trajectory_values_opened": False,
        "target_trajectory_values_opened": False,
        "split_fixed_before_header_read": True,
    }
    assert {record["role"] for record in result["records"]} == {
        "source",
        "calibration",
        "target",
    }
    assert len(result["result_id"]) == 64


def test_audit_rejects_wrong_archive_identity(tmp_path: Path) -> None:
    archive_path = tmp_path / "tracking_dataset.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "tracking_dataset/Self-collisions/a.csv",
            _header("1", "5", "20"),
        )
    protocol = {
        "protocol_id": "bad-identity",
        "dataset": {
            "archive_filename": archive_path.name,
            "archive_md5": "0" * 32,
            "archive_sha256": "0" * 64,
            "expected_csv_files": 1,
            "fresh_category_token": "Self-collisions",
            "expected_fresh_files": 1,
        },
        "split": {
            "sha256_salt": "x",
            "source_count": 1,
            "calibration_count": 0,
            "target_count": 0,
        },
        "header_audit": {"line_count": 7, "maximum_line_bytes": 65536},
        "claim_boundary": [],
    }
    with pytest.raises(ValueError, match="MD5 mismatch"):
        MODULE.run_audit(protocol, tmp_path)


def test_protocol_contains_only_header_level_information() -> None:
    protocol_path = (
        Path(__file__).resolve().parents[1]
        / "protocols"
        / "tracking-cloth-orbit-tube-roster-audit-v1.json"
    )
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    assert protocol["split"] == {
        "assignment": "sort by sha256(salt + '|' + ZIP member path), then take 9 source, 9 calibration, and 9 target recordings",
        "calibration_count": 9,
        "sha256_salt": "prob4d-tracking-cloth-orbit-tube-v1",
        "source_count": 9,
        "target_count": 9,
    }
    assert protocol["header_audit"]["trajectory_rows_allowed"] == 0
    assert protocol["header_audit"]["publish_raw_header_text"] is False
