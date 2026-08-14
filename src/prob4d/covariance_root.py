"""Versioned covariance-root implementations for observation export.

The frozen provider-v1 eigendecomposition basis and provider-v2 canonical basis
are exposed as explicit callables.  A task-local mode remains for compatibility,
but importing this module never replaces functions in another module.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Literal

import numpy as np

CovarianceRootMode = Literal["legacy_eigenvectors", "canonical_eigenspaces"]
CovarianceRootFunction = Callable[..., tuple[np.ndarray, float]]
COVARIANCE_ROOT_MODES: tuple[CovarianceRootMode, ...] = (
    "legacy_eigenvectors",
    "canonical_eigenspaces",
)
_ROOT_MODE: ContextVar[CovarianceRootMode] = ContextVar(
    "prob4d_covariance_root_mode",
    default="legacy_eigenvectors",
)


def _validated_mode(mode: CovarianceRootMode) -> CovarianceRootMode:
    if mode not in COVARIANCE_ROOT_MODES:
        raise ValueError(f"mode must be one of {COVARIANCE_ROOT_MODES}")
    return mode


def legacy_covariance_root(
    covariance: np.ndarray,
    *,
    max_rank: int | None = None,
    relative_eigenvalue_floor: float = 1e-12,
    coordinate_normalizer: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Return the frozen provider-v1 deterministic PSD root."""

    matrix = np.asarray(covariance, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("covariance root requires a square matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("covariance root requires finite values")
    if max_rank is not None and max_rank < 1:
        raise ValueError("max_rank must be positive when supplied")
    if not 0.0 <= relative_eigenvalue_floor < 1.0:
        raise ValueError("relative_eigenvalue_floor must lie in [0, 1)")
    if coordinate_normalizer is None:
        normalizer = np.ones(matrix.shape[0], dtype=np.float64)
    else:
        normalizer = np.asarray(coordinate_normalizer, dtype=np.float64)
        if normalizer.shape != (matrix.shape[0],):
            raise ValueError("coordinate_normalizer must match covariance dimension")
        if not np.all(np.isfinite(normalizer)) or np.any(normalizer <= 0.0):
            raise ValueError("coordinate_normalizer must be finite and positive")

    symmetric = 0.5 * (matrix + matrix.T)
    normalized = normalizer[:, None] * symmetric * normalizer[None, :]
    eigenvalues, eigenvectors = np.linalg.eigh(normalized)
    spectral_scale = max(
        float(np.max(np.abs(eigenvalues), initial=0.0)),
        np.finfo(np.float64).tiny,
    )
    if float(np.min(eigenvalues, initial=0.0)) < -(
        1e-14 + 1e-10 * spectral_scale
    ):
        raise ValueError("covariance root requires positive semidefinite input")
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    maximum = float(eigenvalues[0]) if len(eigenvalues) else 0.0
    keep = eigenvalues > maximum * relative_eigenvalue_floor
    indices = np.flatnonzero(keep)
    if max_rank is not None:
        indices = indices[:max_rank]
    total_trace = float(np.sum(eigenvalues))
    retained_trace = float(np.sum(eigenvalues[indices]))
    retained_fraction = 1.0 if total_trace == 0.0 else retained_trace / total_trace
    selected_vectors = eigenvectors[:, indices].copy()
    for column in range(selected_vectors.shape[1]):
        pivot = int(np.argmax(np.abs(selected_vectors[:, column])))
        if selected_vectors[pivot, column] < 0.0:
            selected_vectors[:, column] *= -1.0
    normalized_root = selected_vectors * np.sqrt(eigenvalues[indices])[None]
    root = normalized_root / normalizer[:, None]
    return root, retained_fraction


# Retained for source compatibility with focused tests and diagnostics.
_LEGACY_ROOT = legacy_covariance_root


def _eigenvalues_numerically_equal(
    first: float,
    second: float,
    *,
    spectral_scale: float,
) -> bool:
    tolerance = 1e-14 + 1e-10 * spectral_scale
    return abs(float(first) - float(second)) <= tolerance


def _canonical_projector_basis(vectors: np.ndarray) -> np.ndarray:
    """Construct a basis from an eigenspace projector, not arbitrary eigenvectors."""

    values = np.asarray(vectors, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("eigenspace vectors must form a matrix")
    dimension, rank = values.shape
    if rank == 0:
        return np.empty((dimension, 0), dtype=np.float64)
    projector = values @ values.T
    basis: list[np.ndarray] = []
    tolerance = 256.0 * np.finfo(np.float64).eps * max(1, dimension)
    for axis in range(dimension):
        candidate = projector[:, axis].copy()
        for _ in range(2):
            for previous in basis:
                candidate -= previous * float(previous @ candidate)
        norm = float(np.linalg.norm(candidate))
        if norm <= tolerance:
            continue
        candidate /= norm
        pivot = int(np.argmax(np.abs(candidate)))
        if candidate[pivot] < 0.0:
            candidate *= -1.0
        basis.append(candidate)
        if len(basis) == rank:
            break
    if len(basis) != rank:
        raise RuntimeError("failed to construct a canonical eigenspace basis")
    result = np.column_stack(basis)
    if not np.allclose(
        result.T @ result,
        np.eye(rank),
        atol=1e-11,
        rtol=1e-11,
    ):
        raise RuntimeError("canonical eigenspace basis lost orthogonality")
    return result


def _canonical_selected_eigen_root(
    normalized_covariance: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    indices: np.ndarray,
    *,
    spectral_scale: float,
) -> np.ndarray:
    if not len(indices):
        return np.empty((normalized_covariance.shape[0], 0), dtype=np.float64)
    parts: list[np.ndarray] = []
    position = 0
    while position < len(indices):
        stop = position + 1
        while stop < len(indices) and _eigenvalues_numerically_equal(
            float(eigenvalues[indices[stop - 1]]),
            float(eigenvalues[indices[stop]]),
            spectral_scale=spectral_scale,
        ):
            stop += 1
        group_indices = indices[position:stop]
        group_vectors = eigenvectors[:, group_indices]
        if len(group_indices) == 1:
            vector = group_vectors[:, 0].copy()
            pivot = int(np.argmax(np.abs(vector)))
            if vector[pivot] < 0.0:
                vector *= -1.0
            parts.append(
                vector[:, None] * np.sqrt(float(eigenvalues[group_indices[0]]))
            )
        else:
            basis = _canonical_projector_basis(group_vectors)
            reduced = basis.T @ normalized_covariance @ basis
            reduced = 0.5 * (reduced + reduced.T)
            try:
                reduced_root = np.linalg.cholesky(reduced)
            except np.linalg.LinAlgError as error:
                raise ValueError(
                    "selected covariance eigenspace is not positive definite"
                ) from error
            parts.append(basis @ reduced_root)
        position = stop
    return np.concatenate(parts, axis=1)


def canonical_covariance_root(
    covariance: np.ndarray,
    *,
    max_rank: int | None = None,
    relative_eigenvalue_floor: float = 1e-12,
    coordinate_normalizer: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Return a root with a canonical basis for repeated eigenspaces.

    The function fails closed if the eigenvalue floor or rank cap would split a
    numerically repeated eigenspace. Such a split would make the retained
    covariance subspace depend on an arbitrary eigensolver basis.
    """

    matrix = np.asarray(covariance, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("covariance root requires a square matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("covariance root requires finite values")
    if max_rank is not None and max_rank < 1:
        raise ValueError("max_rank must be positive when supplied")
    if not 0.0 <= relative_eigenvalue_floor < 1.0:
        raise ValueError("relative_eigenvalue_floor must lie in [0, 1)")
    if coordinate_normalizer is None:
        normalizer = np.ones(matrix.shape[0], dtype=np.float64)
    else:
        normalizer = np.asarray(coordinate_normalizer, dtype=np.float64)
        if normalizer.shape != (matrix.shape[0],):
            raise ValueError("coordinate_normalizer must match covariance dimension")
        if not np.all(np.isfinite(normalizer)) or np.any(normalizer <= 0.0):
            raise ValueError("coordinate_normalizer must be finite and positive")

    symmetric = 0.5 * (matrix + matrix.T)
    normalized = normalizer[:, None] * symmetric * normalizer[None, :]
    eigenvalues, eigenvectors = np.linalg.eigh(normalized)
    spectral_scale = max(
        float(np.max(np.abs(eigenvalues), initial=0.0)),
        float(np.finfo(np.float64).tiny),
    )
    if float(np.min(eigenvalues, initial=0.0)) < -(
        1e-14 + 1e-10 * spectral_scale
    ):
        raise ValueError("covariance root requires positive semidefinite input")
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    maximum = float(eigenvalues[0]) if len(eigenvalues) else 0.0
    eligible = np.flatnonzero(eigenvalues > maximum * relative_eigenvalue_floor)

    if 0 < len(eligible) < len(eigenvalues) and _eigenvalues_numerically_equal(
        float(eigenvalues[len(eligible) - 1]),
        float(eigenvalues[len(eligible)]),
        spectral_scale=spectral_scale,
    ):
        raise ValueError(
            "relative eigenvalue floor cuts through a numerically repeated eigenspace"
        )
    if max_rank is not None and len(eligible) > max_rank:
        if _eigenvalues_numerically_equal(
            float(eigenvalues[max_rank - 1]),
            float(eigenvalues[max_rank]),
            spectral_scale=spectral_scale,
        ):
            raise ValueError(
                "max_rank cuts through a numerically repeated covariance eigenspace"
            )

    indices = eligible if max_rank is None else eligible[:max_rank]
    total_trace = float(np.sum(eigenvalues))
    retained_trace = float(np.sum(eigenvalues[indices]))
    retained_fraction = 1.0 if total_trace == 0.0 else retained_trace / total_trace
    normalized_root = _canonical_selected_eigen_root(
        normalized,
        eigenvalues,
        eigenvectors,
        indices,
        spectral_scale=spectral_scale,
    )
    return normalized_root / normalizer[:, None], retained_fraction


def covariance_root_function(mode: CovarianceRootMode) -> CovarianceRootFunction:
    """Return the implementation associated with one declared mode."""

    selected = _validated_mode(mode)
    if selected == "canonical_eigenspaces":
        return canonical_covariance_root
    return legacy_covariance_root


def covariance_root_for_mode(
    mode: CovarianceRootMode,
    covariance: np.ndarray,
    *,
    max_rank: int | None = None,
    relative_eigenvalue_floor: float = 1e-12,
    coordinate_normalizer: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Evaluate one explicitly declared covariance-root mode."""

    return covariance_root_function(mode)(
        covariance,
        max_rank=max_rank,
        relative_eigenvalue_floor=relative_eigenvalue_floor,
        coordinate_normalizer=coordinate_normalizer,
    )


def current_covariance_root_mode() -> CovarianceRootMode:
    """Return the task-local compatibility mode."""

    return _ROOT_MODE.get()


def _dispatch_covariance_root(
    covariance: np.ndarray,
    *,
    max_rank: int | None = None,
    relative_eigenvalue_floor: float = 1e-12,
    coordinate_normalizer: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Compatibility dispatcher used by the stable exporter wrapper."""

    return covariance_root_for_mode(
        current_covariance_root_mode(),
        covariance,
        max_rank=max_rank,
        relative_eigenvalue_floor=relative_eigenvalue_floor,
        coordinate_normalizer=coordinate_normalizer,
    )


def install_covariance_root_dispatch() -> None:
    """Compatibility no-op; dispatch now occurs through a stable wrapper."""


@contextmanager
def covariance_root_mode(mode: CovarianceRootMode) -> Iterator[None]:
    """Select a compatibility mode without process-global function mutation."""

    token = _ROOT_MODE.set(_validated_mode(mode))
    try:
        yield
    finally:
        _ROOT_MODE.reset(token)


__all__ = [
    "COVARIANCE_ROOT_MODES",
    "CovarianceRootFunction",
    "CovarianceRootMode",
    "canonical_covariance_root",
    "covariance_root_for_mode",
    "covariance_root_function",
    "covariance_root_mode",
    "current_covariance_root_mode",
    "install_covariance_root_dispatch",
    "legacy_covariance_root",
]
