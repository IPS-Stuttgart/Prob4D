"""Fail-closed covariance validation and regularized linear algebra."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]


def _validated_eigendecomposition(
    covariance: FloatArray,
    *,
    name: str,
    absolute_negative_tolerance: float,
    relative_negative_tolerance: float,
    symmetry_atol: float,
    symmetry_rtol: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    values = np.asarray(covariance, dtype=np.float64)
    if values.ndim < 2 or values.shape[-1] != values.shape[-2] or values.shape[-1] == 0:
        raise ValueError(f"{name} must contain nonempty square matrices")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be finite")
    if absolute_negative_tolerance < 0.0 or relative_negative_tolerance < 0.0:
        raise ValueError("negative-eigenvalue tolerances must be non-negative")

    transposed = np.swapaxes(values, -1, -2)
    symmetric = 0.5 * (values + transposed)
    if not np.allclose(
        values,
        symmetric,
        atol=symmetry_atol,
        rtol=symmetry_rtol,
    ):
        raise ValueError(f"{name} must be symmetric")

    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    spectral_scale = np.maximum(
        np.max(np.abs(eigenvalues), axis=-1),
        np.finfo(np.float64).tiny,
    )
    tolerance = absolute_negative_tolerance + relative_negative_tolerance * spectral_scale
    minimum = np.min(eigenvalues, axis=-1)
    if np.any(minimum < -tolerance):
        worst = float(np.min(minimum))
        raise ValueError(
            f"{name} must be positive semidefinite; minimum eigenvalue is {worst:.6g}"
        )
    return symmetric, eigenvalues, eigenvectors


def validated_covariance_psd(
    covariance: FloatArray,
    *,
    name: str = "covariance",
    shape: tuple[int, ...] | None = None,
    readonly: bool = True,
    absolute_negative_tolerance: float = 1e-14,
    relative_negative_tolerance: float = 1e-10,
    symmetry_atol: float = 1e-12,
    symmetry_rtol: float = 1e-10,
) -> FloatArray:
    """Return a defensive, validated PSD covariance without inventing uncertainty.

    Materially asymmetric or indefinite input fails closed. Floating-point-scale
    negative eigenvalues are projected to zero, while an exactly zero covariance
    remains exactly zero. This makes the helper suitable for immutable model and
    artifact contracts where inverse-specific regularization would change the
    declared covariance semantics.
    """

    values = np.asarray(covariance, dtype=np.float64)
    if shape is not None and values.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    symmetric, eigenvalues, eigenvectors = _validated_eigendecomposition(
        values,
        name=name,
        absolute_negative_tolerance=absolute_negative_tolerance,
        relative_negative_tolerance=relative_negative_tolerance,
        symmetry_atol=symmetry_atol,
        symmetry_rtol=symmetry_rtol,
    )
    if np.any(eigenvalues < 0.0):
        clipped = np.maximum(eigenvalues, 0.0)
        result = np.einsum(
            "...ij,...j,...kj->...ik",
            eigenvectors,
            clipped,
            eigenvectors,
            optimize=True,
        )
        result = 0.5 * (result + np.swapaxes(result, -1, -2))
    else:
        result = symmetric.copy()
    if readonly:
        result.setflags(write=False)
    return result


def covariance_eigendecomposition(
    covariance: FloatArray,
    *,
    name: str = "covariance",
    eigenvalue_floor: float = 1e-12,
    absolute_negative_tolerance: float = 1e-14,
    relative_negative_tolerance: float = 1e-10,
    symmetry_atol: float = 1e-12,
    symmetry_rtol: float = 1e-10,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Validate symmetric PSD input and return a regularized eigendecomposition.

    Materially negative eigenvalues indicate an invalid covariance and raise.
    Floating-point-scale negative values are clipped only after they pass a
    scale-aware tolerance. The returned eigenvalues are at least
    ``eigenvalue_floor`` so downstream inverse and log-determinant operations are
    well defined.
    """

    if not np.isfinite(eigenvalue_floor) or eigenvalue_floor <= 0.0:
        raise ValueError("eigenvalue_floor must be finite and positive")
    symmetric, eigenvalues, eigenvectors = _validated_eigendecomposition(
        covariance,
        name=name,
        absolute_negative_tolerance=absolute_negative_tolerance,
        relative_negative_tolerance=relative_negative_tolerance,
        symmetry_atol=symmetry_atol,
        symmetry_rtol=symmetry_rtol,
    )
    regularized = np.maximum(eigenvalues, eigenvalue_floor)
    return symmetric, regularized, eigenvectors


def regularized_inverse_psd(
    covariance: FloatArray,
    *,
    name: str = "covariance",
    eigenvalue_floor: float = 1e-12,
) -> FloatArray:
    """Return a regularized inverse after fail-closed PSD validation."""

    _, eigenvalues, eigenvectors = covariance_eigendecomposition(
        covariance,
        name=name,
        eigenvalue_floor=eigenvalue_floor,
    )
    return np.einsum(
        "...ij,...j,...kj->...ik",
        eigenvectors,
        1.0 / eigenvalues,
        eigenvectors,
        optimize=True,
    )


def covariance_statistics(
    covariance: FloatArray,
    *,
    name: str = "covariance",
    eigenvalue_floor: float = 1e-12,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return symmetric covariance, regularized inverse, and log determinant."""

    symmetric, eigenvalues, eigenvectors = covariance_eigendecomposition(
        covariance,
        name=name,
        eigenvalue_floor=eigenvalue_floor,
    )
    inverse = np.einsum(
        "...ij,...j,...kj->...ik",
        eigenvectors,
        1.0 / eigenvalues,
        eigenvectors,
        optimize=True,
    )
    log_determinant = np.sum(np.log(eigenvalues), axis=-1)
    return symmetric, inverse, log_determinant


__all__ = [
    "covariance_eigendecomposition",
    "covariance_statistics",
    "regularized_inverse_psd",
    "validated_covariance_psd",
]
