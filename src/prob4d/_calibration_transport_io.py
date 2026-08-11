"""Strict JSON persistence for calibration-transport artifacts."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ._calibration_transport_evidence import CalibrationTransportEvidenceV1
from ._calibration_transport_model import CalibrationTransportModelV1
from ._immutable_json import plain_json
from ._strict_json import load_json_object


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _atomic_write_json(
    path: Path,
    record: Mapping[str, Any],
    *,
    overwrite: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                plain_json(record),
                stream,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary_name, path)
        else:
            try:
                os.link(temporary_name, path)
            except FileExistsError:
                raise FileExistsError(f"refusing to overwrite {path}") from None
            os.unlink(temporary_name)
        _fsync_directory(path.parent)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def save_calibration_transport_model(
    model: CalibrationTransportModelV1,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> None:
    if not isinstance(model, CalibrationTransportModelV1):
        raise TypeError("model must be a CalibrationTransportModelV1")
    if type(overwrite) is not bool:
        raise TypeError("overwrite must be a Boolean")
    _atomic_write_json(Path(path), model.to_dict(), overwrite=overwrite)


def load_calibration_transport_model(path: str | Path) -> CalibrationTransportModelV1:
    return CalibrationTransportModelV1.from_dict(
        load_json_object(path, name="calibration transport model")
    )


def save_calibration_transport_evidence(
    evidence: CalibrationTransportEvidenceV1,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> None:
    if not isinstance(evidence, CalibrationTransportEvidenceV1):
        raise TypeError("evidence must be a CalibrationTransportEvidenceV1")
    if type(overwrite) is not bool:
        raise TypeError("overwrite must be a Boolean")
    _atomic_write_json(Path(path), evidence.to_dict(), overwrite=overwrite)


def load_calibration_transport_evidence(
    path: str | Path,
    *,
    model: CalibrationTransportModelV1,
) -> CalibrationTransportEvidenceV1:
    return CalibrationTransportEvidenceV1.from_dict(
        load_json_object(path, name="calibration transport evidence"),
        model=model,
    )
