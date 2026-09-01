from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from prob4d.motive_csv import (
    common_marker_labels,
    read_motive_layout,
    read_motive_markers,
)


def _write_motive(
    path: Path,
    labels: list[str],
    *,
    order: list[int] | None = None,
    duplicate_first: bool = False,
) -> None:
    indices = order or list(range(len(labels)))
    ordered = [labels[index] for index in indices]
    if duplicate_first:
        ordered[1] = ordered[0]
    width = 2 + 3 * len(ordered)
    type_row = ["", ""]
    label_row = ["", ""]
    id_row = ["", ""]
    quantity_row = ["", ""]
    axis_row = ["Frame", "Time"]
    for label in ordered:
        type_row.extend(["Marker"] * 3)
        label_row.extend([label] * 3)
        id_row.extend([f"ID-{label}"] * 3)
        quantity_row.extend(["Position"] * 3)
        axis_row.extend(["X", "Y", "Z"])
    assert all(len(row) == width for row in (type_row, label_row, id_row, quantity_row, axis_row))
    rows = [
        [
            "Format Version",
            "1.2",
            "Take Name",
            "fixture",
            "Length Units",
            "Meters",
            "Coordinate Space",
            "Global",
        ],
        [],
        type_row,
        label_row,
        id_row,
        quantity_row,
        axis_row,
    ]
    for frame in range(2):
        row: list[str] = [str(100 + frame), f"{frame / 120.0:.8f}"]
        for label in ordered:
            value = float(int(label)) if label.isdigit() else float(len(label))
            row.extend(
                [
                    f"{0.001 * (value + frame):.9f}",
                    f"{0.001 * (2.0 * value + frame):.9f}",
                    f"{0.001 * (3.0 * value + frame):.9f}",
                ]
            )
        rows.append(row)
    with path.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(rows)


def test_reads_documented_motive_multirow_marker_layout(tmp_path: Path) -> None:
    path = tmp_path / "recording.csv"
    _write_motive(path, ["1", "2", "3"], order=[2, 0, 1])
    layout = read_motive_layout(path)
    assert layout.data_start_row == 7
    assert layout.header_row_count == 7
    assert layout.length_units == "Meters"
    assert layout.marker_labels == ("3", "1", "2")
    assert [marker.columns for marker in layout.markers] == [
        (2, 3, 4),
        (5, 6, 7),
        (8, 9, 10),
    ]


def test_selected_markers_are_label_addressed_and_scaled_to_mm(tmp_path: Path) -> None:
    path = tmp_path / "recording.csv"
    _write_motive(path, ["1", "2", "3"], order=[2, 0, 1])
    values, scale, details = read_motive_markers(path, ["1", "3"])
    assert values.shape == (2, 2, 3)
    assert scale == 1000.0
    np.testing.assert_allclose(values[0, 0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(values[0, 1], [3.0, 6.0, 9.0])
    np.testing.assert_allclose(values[1, 0], [2.0, 3.0, 4.0])
    assert details["parser"] == "strict-motive-multirow-marker-v1"
    assert details["selected_marker_labels"] == ["1", "3"]


def test_common_labels_are_naturally_sorted_across_column_orders(tmp_path: Path) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    labels = [str(index) for index in range(1, 21)]
    _write_motive(first, labels, order=list(reversed(range(20))))
    order = [
        14,
        9,
        4,
        19,
        3,
        13,
        18,
        8,
        1,
        2,
        12,
        17,
        7,
        6,
        11,
        16,
        0,
        10,
        5,
        15,
    ]
    _write_motive(second, labels, order=order)
    assert common_marker_labels([first, second], 20) == labels


def test_missing_or_duplicate_marker_identity_is_rejected(tmp_path: Path) -> None:
    valid = tmp_path / "valid.csv"
    duplicate = tmp_path / "duplicate.csv"
    _write_motive(valid, ["1", "2", "3"])
    _write_motive(duplicate, ["1", "2", "3"], duplicate_first=True)
    with pytest.raises(ValueError, match="absent"):
        read_motive_markers(valid, ["1", "9"])
    with pytest.raises(ValueError, match="duplicate"):
        read_motive_layout(duplicate)


def test_no_global_identity_is_not_silently_replaced_by_column_index(tmp_path: Path) -> None:
    labelled = tmp_path / "labelled.csv"
    unlabeled = tmp_path / "unlabeled.csv"
    _write_motive(labelled, ["1", "2", "3"])
    _write_motive(unlabeled, ["Marker_105", "Marker_108", "Marker_109"])
    with pytest.raises(ValueError, match="fewer than three common"):
        common_marker_labels([labelled, unlabeled], 20)
