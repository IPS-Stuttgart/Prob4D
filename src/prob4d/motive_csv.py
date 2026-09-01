"""Strict parsing of OptiTrack Motive multirow marker CSV exports.

Motive stores type, marker label, unique ID, quantity, and axis on separate
header rows. Treating the first row as a conventional flat CSV header silently
loses every marker identity. This module reconstructs only explicit
``Marker``/``Position``/``X,Y,Z`` triples and rejects ambiguous layouts.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class MotiveMarker:
    """One explicitly labelled 3-D marker in a Motive CSV export."""

    label: str
    unique_id: str
    columns: tuple[int, int, int]


@dataclass(frozen=True)
class MotiveLayout:
    """Validated header layout for one Motive marker CSV."""

    delimiter: str
    data_start_row: int
    length_units: str | None
    markers: tuple[MotiveMarker, ...]
    header_row_count: int

    @property
    def marker_labels(self) -> tuple[str, ...]:
        return tuple(marker.label for marker in self.markers)


def _sniff_delimiter(sample: str) -> str:
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t").delimiter
    except csv.Error:
        return max((",", ";", "\t"), key=sample.count)


def _float_or_nan(value: str | None, delimiter: str) -> float:
    if value is None:
        return math.nan
    text = value.strip()
    if not text:
        return math.nan
    if delimiter == ";" and "," in text and "." not in text:
        text = text.replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return math.nan


def _is_numeric_prefix(row: Sequence[str], delimiter: str) -> bool:
    return (
        len(row) >= 2
        and math.isfinite(_float_or_nan(row[0], delimiter))
        and math.isfinite(_float_or_nan(row[1], delimiter))
    )


def _natural_label_key(value: str) -> tuple[int, int | str]:
    stripped = value.strip()
    if stripped.isdigit():
        return (0, int(stripped))
    return (1, stripped.casefold())


def _length_units(header_rows: Sequence[Sequence[str]]) -> str | None:
    if not header_rows:
        return None
    first = list(header_rows[0])
    for index, value in enumerate(first[:-1]):
        if value.strip().casefold() == "length units":
            result = first[index + 1].strip()
            return result or None
    return None


def _layout_from_rows(rows: Sequence[Sequence[str]], delimiter: str) -> MotiveLayout:
    data_start = next(
        (index for index, row in enumerate(rows) if _is_numeric_prefix(row, delimiter)),
        None,
    )
    if data_start is None:
        raise ValueError("Motive CSV contains no numeric frame/time row")
    if data_start < 7:
        raise ValueError("Motive CSV header is shorter than the documented multirow layout")
    header = rows[:data_start]
    axis_index = next(
        (
            index
            for index in range(data_start - 1, -1, -1)
            if len(header[index]) >= 5
            and header[index][0].strip().casefold().startswith("frame")
            and "time" in header[index][1].strip().casefold()
            and sum(cell.strip().upper() in {"X", "Y", "Z"} for cell in header[index][2:]) >= 3
        ),
        None,
    )
    if axis_index is None:
        raise ValueError("Motive axis header row was not identified")
    type_index = next(
        (
            index
            for index, row in enumerate(header[:axis_index])
            if sum(cell.strip().casefold() == "marker" for cell in row) >= 3
        ),
        None,
    )
    if type_index is None or type_index + 2 >= axis_index:
        raise ValueError("Motive marker type/identity rows were not identified")
    label_index = type_index + 1
    unique_id_index = type_index + 2
    position_index = axis_index - 1
    width = min(
        len(header[type_index]),
        len(header[label_index]),
        len(header[unique_id_index]),
        len(header[position_index]),
        len(header[axis_index]),
    )
    markers: list[MotiveMarker] = []
    column = 2
    while column + 2 < width:
        marker_types = tuple(
            header[type_index][column + offset].strip().casefold() for offset in range(3)
        )
        quantities = tuple(
            header[position_index][column + offset].strip().casefold() for offset in range(3)
        )
        axes = tuple(header[axis_index][column + offset].strip().upper() for offset in range(3))
        if (
            marker_types == ("marker", "marker", "marker")
            and quantities
            == (
                "position",
                "position",
                "position",
            )
            and axes == ("X", "Y", "Z")
        ):
            labels = {header[label_index][column + offset].strip() for offset in range(3)}
            unique_ids = {header[unique_id_index][column + offset].strip() for offset in range(3)}
            if len(labels) != 1 or len(unique_ids) != 1:
                raise ValueError("Motive marker triple has inconsistent identity cells")
            label = next(iter(labels))
            unique_id = next(iter(unique_ids))
            if not label or not unique_id:
                raise ValueError("Motive marker label or unique ID is empty")
            markers.append(
                MotiveMarker(
                    label=label,
                    unique_id=unique_id,
                    columns=(column, column + 1, column + 2),
                )
            )
            column += 3
        else:
            column += 1
    if len(markers) < 3:
        raise ValueError("Motive CSV contains fewer than three explicit 3-D markers")
    labels = [marker.label for marker in markers]
    if len(set(labels)) != len(labels):
        raise ValueError("Motive CSV contains duplicate marker labels")
    return MotiveLayout(
        delimiter=delimiter,
        data_start_row=data_start,
        length_units=_length_units(header),
        markers=tuple(markers),
        header_row_count=len(header),
    )


def read_motive_layout(path: Path) -> MotiveLayout:
    """Read and validate only the header and first numeric row."""

    source = Path(path)
    with source.open("r", encoding="utf-8-sig", errors="strict", newline="") as stream:
        sample = stream.read(65536)
    delimiter = _sniff_delimiter(sample)
    rows = list(csv.reader(sample.splitlines(), delimiter=delimiter))
    return _layout_from_rows(rows, delimiter)


def common_marker_labels(paths: Iterable[Path], maximum: int) -> list[str]:
    """Return naturally ordered labels common to every supplied recording."""

    if maximum < 3:
        raise ValueError("maximum must permit at least three markers")
    common: set[str] | None = None
    recording_count = 0
    for path in paths:
        labels = set(read_motive_layout(Path(path)).marker_labels)
        common = labels if common is None else common & labels
        recording_count += 1
    if recording_count == 0:
        raise ValueError("at least one recording is required")
    result = sorted(common or (), key=_natural_label_key)
    if len(result) < 3:
        raise ValueError(f"fewer than three common 3-D marker labels: {result}")
    return result[:maximum]


def _unit_scale_to_mm(units: str | None, coordinates: FloatArray) -> float:
    normalized = (units or "").strip().casefold()
    if normalized in {"m", "meter", "meters", "metre", "metres"}:
        return 1000.0
    if normalized in {"cm", "centimeter", "centimeters", "centimetre", "centimetres"}:
        return 10.0
    if normalized in {"mm", "millimeter", "millimeters", "millimetre", "millimetres"}:
        return 1.0
    pair_distances: list[float] = []
    sample_count = min(12, coordinates.shape[0])
    frame_indices = np.unique(
        np.linspace(
            0,
            max(coordinates.shape[0] - 1, 0),
            num=sample_count,
            dtype=np.int64,
        )
    )
    for frame_index in frame_indices:
        frame = coordinates[int(frame_index)]
        valid = frame[np.all(np.isfinite(frame), axis=1)]
        if valid.shape[0] < 2:
            continue
        differences = valid[:, None, :] - valid[None, :, :]
        distances = np.linalg.norm(differences, axis=-1)
        pair_distances.extend(distances[np.triu_indices(valid.shape[0], 1)].tolist())
    positive = np.asarray(
        [value for value in pair_distances if value > 0.0 and math.isfinite(value)],
        dtype=np.float64,
    )
    if positive.size == 0:
        raise ValueError("coordinate unit cannot be inferred from degenerate geometry")
    median = float(np.median(positive))
    if 1e-4 <= median < 10.0:
        return 1000.0
    if 10.0 <= median < 10000.0:
        return 1.0
    raise ValueError(f"unsupported coordinate scale; median marker distance was {median:g}")


def read_motive_markers(
    path: Path,
    marker_labels: Sequence[str],
) -> tuple[FloatArray, float, dict[str, object]]:
    """Read selected labelled marker positions as ``(frames, markers, 3)`` in mm."""

    requested = [str(value) for value in marker_labels]
    if len(requested) < 1 or len(set(requested)) != len(requested):
        raise ValueError("marker_labels must be a nonempty unique sequence")
    source = Path(path)
    text = source.read_text(encoding="utf-8-sig", errors="strict")
    delimiter = _sniff_delimiter(text[:65536])
    rows = list(csv.reader(text.splitlines(), delimiter=delimiter))
    layout = _layout_from_rows(rows, delimiter)
    by_label = {marker.label: marker for marker in layout.markers}
    missing = sorted(set(requested) - set(by_label), key=_natural_label_key)
    if missing:
        raise ValueError(f"selected marker labels are absent from {source}: {missing}")
    selected = [by_label[label] for label in requested]
    frames: list[list[list[float]]] = []
    frame_numbers: list[float] = []
    for row in rows[layout.data_start_row :]:
        if not _is_numeric_prefix(row, delimiter):
            continue
        frame_numbers.append(_float_or_nan(row[0], delimiter))
        frame: list[list[float]] = []
        for marker in selected:
            frame.append(
                [
                    _float_or_nan(row[column] if column < len(row) else None, delimiter)
                    for column in marker.columns
                ]
            )
        frames.append(frame)
    if not frames:
        raise ValueError(f"Motive CSV has no numeric trajectory rows: {source}")
    coordinates = np.asarray(frames, dtype=np.float64)
    scale = _unit_scale_to_mm(layout.length_units, coordinates)
    result = coordinates * scale
    return (
        result,
        scale,
        {
            "delimiter": "tab" if delimiter == "\t" else delimiter,
            "rows": int(result.shape[0]),
            "header_rows": layout.header_row_count,
            "data_start_row_zero_based": layout.data_start_row,
            "length_units": layout.length_units,
            "available_marker_count": len(layout.markers),
            "selected_marker_labels": requested,
            "first_frame_number": frame_numbers[0],
            "last_frame_number": frame_numbers[-1],
            "parser": "strict-motive-multirow-marker-v1",
        },
    )
