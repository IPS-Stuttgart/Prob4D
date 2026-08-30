"""Tests for the target-closed DOT rope archive inventory."""

from __future__ import annotations

import hashlib
from pathlib import Path
from zipfile import ZipFile, ZipInfo

import pytest

from experiments.dot_rope_curvature_v1.inventory import (
    inventory_archive,
    member_name_is_safe,
    normalized_member_pattern,
    sequence_id_from_member,
    split_for_sequence,
    zip_info_is_symlink,
)


@pytest.mark.parametrize(
    ("sequence_number", "expected"),
    [
        (1, "development"),
        (30, "development"),
        (31, "calibration"),
        (40, "calibration"),
        (41, "held_out"),
        (70, "held_out"),
        (0, "outside_protocol"),
        (71, "outside_protocol"),
    ],
)
def test_split_for_sequence(sequence_number: int, expected: str) -> None:
    assert split_for_sequence(sequence_number) == expected


def test_member_safety_and_sequence_detection() -> None:
    assert member_name_is_safe("R01/camera/000001.png")
    assert not member_name_is_safe("../escape")
    assert not member_name_is_safe("/absolute")
    assert not member_name_is_safe("R01\\windows")
    assert sequence_id_from_member("dataset/R07/track.npy") == "R07"
    assert sequence_id_from_member("dataset/no_sequence/track.npy") is None
    assert normalized_member_pattern("R07/cam12/frame000123.png") == ("R##/cam#/frame#.png")


def test_symlink_detection() -> None:
    regular = ZipInfo("regular.txt")
    regular.external_attr = 0o100644 << 16
    symlink = ZipInfo("link")
    symlink.external_attr = 0o120777 << 16
    assert not zip_info_is_symlink(regular)
    assert zip_info_is_symlink(symlink)


def test_inventory_archive_reads_central_directory_only(tmp_path: Path) -> None:
    archive_path = tmp_path / "R01-10.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("R01/camera/000001.png", b"not-an-image")
        archive.writestr("R02/ground_truth/track.npy", b"not-a-numpy-array")
        archive.writestr("README.txt", b"metadata")
    expected_md5 = hashlib.md5(archive_path.read_bytes()).hexdigest()  # noqa: S324

    record, sample = inventory_archive(
        archive_path,
        expected_md5=expected_md5,
        sequence_start=1,
        sequence_stop=10,
    )

    assert record["publisher_md5_matches"] is True
    assert record["file_count"] == 3
    assert record["payload_members_opened"] == 0
    assert record["sequence_counts"] == {"R01": 1, "R02": 1}
    assert "R01/camera/000001.png" in sample
    assert record["unsafe_member_count"] == 0
