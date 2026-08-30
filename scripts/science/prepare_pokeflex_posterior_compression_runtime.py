#!/usr/bin/env python3
"""Apply hash-bound, pre-data repairs to the PokeFlex diagnostic copy."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

_EXPECTED_SOURCE_GIT_BLOB_SHA1 = "0a4169f95149644fb9d00fca877a67b2672da36e"
_ORIGINAL_IMPORTS = """from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable
"""
_PATCHED_IMPORTS = """from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
"""
_ORIGINAL_MODE_COLUMN = """        columns.append(
            np.concatenate(
                [coefficient * mode_xyz for coefficient in coefficients]
            )
        )
"""
_PATCHED_MODE_COLUMN = """        columns.append(
            np.concatenate(
                [coefficient * mode_xyz for coefficient in coefficients]
            ).reshape(-1)
        )
"""
_ORIGINAL_SOLVER_INITIALIZATION = """        self._variance = float(variance)
        self._factor = z.reshape(self.dimension, z.shape[2]).copy()
        scaled = self._factor / math.sqrt(self._variance)
        core = np.eye(z.shape[2], dtype=np.float64) + scaled.T @ scaled
        self._core = np.linalg.cholesky(0.5 * (core + core.T))
"""
_PATCHED_SOLVER_INITIALIZATION = """        self._variance = float(variance)
        self._factor = z.reshape(self.dimension, z.shape[2]).copy()
        basis, singular_values, _ = np.linalg.svd(
            self._factor,
            full_matrices=False,
        )
        if singular_values.size:
            cutoff = (
                np.finfo(float).eps
                * max(self._factor.shape)
                * singular_values[0]
            )
            retained = singular_values > cutoff
            self._basis = basis[:, retained]
            self._singular_values = singular_values[retained]
        else:
            self._basis = np.empty((self.dimension, 0), dtype=np.float64)
            self._singular_values = np.empty(0, dtype=np.float64)
"""
_ORIGINAL_SOLVE = """        matrix = raw.reshape(self.dimension, -1)
        base = matrix / self._variance
        rhs = self._factor.T @ base
        correction = self._factor @ np.linalg.solve(
            self._core.T,
            np.linalg.solve(self._core, rhs),
        ) / self._variance
        result = (base - correction).reshape(raw.shape)
"""
_PATCHED_SOLVE = """        matrix = raw.reshape(self.dimension, -1)
        coefficients = self._basis.T @ matrix
        projection = self._basis @ coefficients
        residual = matrix - projection
        residual_norm = np.linalg.norm(residual, axis=0)
        matrix_norm = np.linalg.norm(matrix, axis=0)
        roundoff = (
            64.0
            * np.finfo(float).eps
            * max(self._factor.shape)
            * np.maximum(matrix_norm, np.finfo(float).tiny)
        )
        residual[:, residual_norm <= roundoff] = 0.0
        parallel = self._basis @ (
            coefficients
            / (self._variance + self._singular_values[:, None] ** 2)
        )
        result = (parallel + residual / self._variance).reshape(raw.shape)
"""
_ORIGINAL_ARCHIVE_DISCOVERY = """    archives = sorted(dataset_root.glob(str(protocol["archive_glob"])))
"""
_PATCHED_ARCHIVE_DISCOVERY = """    archives = sorted(
        path
        for path in dataset_root.rglob("*")
        if path.is_file() and path.suffix.lower() == ".zip"
    )
"""


def _git_blob_sha1(payload: bytes) -> str:
    framed = f"blob {len(payload)}\0".encode("ascii") + payload
    return hashlib.sha1(framed, usedforsecurity=False).hexdigest()


def _content_id(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_no_clobber(path: Path, payload: dict[str, object]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)
    except FileExistsError:
        if path.read_text(encoding="utf-8") != encoded:
            raise


def _replace_exactly_once(text: str, original: str, patched: str, name: str) -> str:
    if text.count(original) != 1 or patched in text:
        raise RuntimeError(f"PokeFlex diagnostic {name} preimage changed")
    return text.replace(original, patched, 1)


def apply_repair(source_path: Path) -> dict[str, object]:
    if source_path.is_symlink():
        raise RuntimeError("PokeFlex diagnostic source must not be a symbolic link")
    source = source_path.resolve(strict=True)
    if not source.is_file():
        raise RuntimeError("PokeFlex diagnostic source must be a regular file")
    raw = source.read_bytes()
    source_blob = _git_blob_sha1(raw)
    if source_blob != _EXPECTED_SOURCE_GIT_BLOB_SHA1:
        raise RuntimeError("PokeFlex diagnostic source bytes changed")
    text = raw.decode("utf-8")
    text = _replace_exactly_once(
        text,
        _ORIGINAL_IMPORTS,
        _PATCHED_IMPORTS,
        "import-hygiene",
    )
    text = _replace_exactly_once(
        text,
        _ORIGINAL_MODE_COLUMN,
        _PATCHED_MODE_COLUMN,
        "shape-repair",
    )
    text = _replace_exactly_once(
        text,
        _ORIGINAL_SOLVER_INITIALIZATION,
        _PATCHED_SOLVER_INITIALIZATION,
        "solver-initialization",
    )
    text = _replace_exactly_once(
        text,
        _ORIGINAL_SOLVE,
        _PATCHED_SOLVE,
        "stable-solve",
    )
    text = _replace_exactly_once(
        text,
        _ORIGINAL_ARCHIVE_DISCOVERY,
        _PATCHED_ARCHIVE_DISCOVERY,
        "case-insensitive-recursive-archive-discovery",
    )
    patched = text.encode("utf-8")
    source.write_bytes(patched)
    record: dict[str, object] = {
        "schema": "prob4d.pokeflex-posterior-compression-predata-repair.v2",
        "schema_version": 2,
        "status": "applied-before-real-data-access",
        "source_member": (
            "scripts/science/run_pokeflex_posterior_compression_real_geometry.py"
        ),
        "source_git_blob_sha1": source_blob,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "patched_sha256": hashlib.sha256(patched).hexdigest(),
        "repairs": [
            "Move Iterable to collections.abc for current Python lint semantics.",
            (
                "Flatten each learned spatial-mode trajectory column to the same "
                "3N observation coordinate vector used by translation columns."
            ),
            (
                "Solve diagonal-plus-low-rank systems in the factor's orthonormal "
                "singular subspace, retaining a separately solved orthogonal "
                "residual and suppressing only representation-roundoff residuals."
            ),
            (
                "Inventory ZIP files recursively with a case-insensitive suffix "
                "test so the known mirror layout is not mistaken for an empty root."
            ),
        ],
        "numerical_reason": (
            "The first Woodbury implementation subtracted terms of order 1/R from "
            "each other at the registered R=1e-8 floor, producing a theoretically "
            "symmetric query Schur complement with 1e-5 relative antisymmetry."
        ),
        "information_boundary": (
            "All repairs are applied and tested before the self-hosted job reads "
            "any PokeFlex ZIP central directory or member payload."
        ),
    }
    record["artifact_id"] = _content_id(record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record = apply_repair(args.source)
    _write_no_clobber(args.output, record)
    print(record["artifact_id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
