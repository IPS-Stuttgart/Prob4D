"""Small dependency-free helpers for three-dimensional similarity transforms."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]
_SO3_PI_SIGN_TOLERANCE = 64.0 * np.finfo(np.float64).eps


def _readonly_copy(value: FloatArray, *, shape: tuple[int, ...], name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64).copy()
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    array.setflags(write=False)
    return array


def _canonical_axis_sign(axis: FloatArray) -> FloatArray:
    """Resolve the sign ambiguity of a unit axis deterministically.

    At an angle of exactly pi, ``axis`` and ``-axis`` encode the same rotation.
    Eigenvector routines may return either sign, so the first numerically
    nonzero component is made positive. This lexicographic rule is stable for
    axes with equal-magnitude components.
    """

    result = _readonly_copy(axis, shape=(3,), name="rotation axis").copy()
    nonzero = np.flatnonzero(np.abs(result) > 1e-12)
    if nonzero.size == 0:
        raise ValueError("rotation axis must be nonzero")
    if result[int(nonzero[0])] < 0.0:
        result = -result
    return result


def skew(vector: FloatArray) -> FloatArray:
    """Return the cross-product matrix for a three-vector."""

    x, y, z = _readonly_copy(vector, shape=(3,), name="vector")
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def so3_exp(rotation_vector: FloatArray) -> FloatArray:
    """Map an axis-angle vector to a rotation matrix."""

    vector = _readonly_copy(rotation_vector, shape=(3,), name="rotation_vector")
    theta = float(np.linalg.norm(vector))
    generator = skew(vector)
    if theta < 1e-8:
        return np.eye(3) + generator + 0.5 * generator @ generator
    a = np.sin(theta) / theta
    b = (1.0 - np.cos(theta)) / (theta * theta)
    return np.eye(3) + a * generator + b * generator @ generator


def so3_log(rotation: FloatArray) -> FloatArray:
    """Map a rotation matrix to its shortest, branch-canonical axis-angle vector."""

    matrix = _readonly_copy(rotation, shape=(3, 3), name="rotation")
    cosine = float(np.clip((np.trace(matrix) - 1.0) * 0.5, -1.0, 1.0))
    theta = float(np.arccos(cosine))
    antisymmetric = np.array(
        [
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ]
    )
    if theta < 1e-8:
        return 0.5 * antisymmetric
    if np.pi - theta < 1e-5:
        eigenvalues, eigenvectors = np.linalg.eigh((matrix + np.eye(3)) * 0.5)
        axis = eigenvectors[:, int(np.argmax(eigenvalues))]
        if float(np.linalg.norm(antisymmetric)) > _SO3_PI_SIGN_TOLERANCE:
            if np.dot(axis, antisymmetric) < 0:
                axis = -axis
        else:
            axis = _canonical_axis_sign(axis)
            theta = float(np.pi)
        return theta * axis
    return theta / (2.0 * np.sin(theta)) * antisymmetric


def so3_right_jacobian(rotation_vector: FloatArray) -> FloatArray:
    """Return the right Jacobian of the SO(3) exponential coordinates."""

    vector = _readonly_copy(rotation_vector, shape=(3,), name="rotation_vector")
    angle = float(np.linalg.norm(vector))
    generator = skew(vector)
    if angle < 1e-6:
        return np.eye(3) - 0.5 * generator + generator @ generator / 6.0
    return (
        np.eye(3)
        - (1.0 - np.cos(angle)) / angle**2 * generator
        + (angle - np.sin(angle)) / angle**3 * generator @ generator
    )


@dataclass(frozen=True)
class Sim3:
    """A transform ``y = scale * rotation @ x + translation``.

    The seven-vector convention used by :meth:`as_vector` is
    ``[log_scale, rotation_vector(3), translation(3)]``. Array fields are
    defensively copied and read-only so validated transforms remain immutable.
    """

    scale: float = 1.0
    rotation: FloatArray = field(default_factory=lambda: np.eye(3))
    translation: FloatArray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self) -> None:
        scale = float(self.scale)
        rotation = _readonly_copy(self.rotation, shape=(3, 3), name="Sim3 rotation")
        translation = _readonly_copy(
            self.translation,
            shape=(3,),
            name="Sim3 translation",
        )
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Sim3 scale must be finite and strictly positive")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7):
            raise ValueError("Sim3 rotation must be orthonormal")
        determinant = float(np.linalg.det(rotation))
        if not np.isclose(determinant, 1.0, atol=1e-7, rtol=1e-7):
            raise ValueError("Sim3 rotation must be proper")
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "translation", translation)

    @classmethod
    def identity(cls) -> Sim3:
        return cls()

    @classmethod
    def from_vector(cls, vector: FloatArray) -> Sim3:
        vector = np.asarray(vector, dtype=np.float64)
        if vector.shape != (7,):
            raise ValueError("Sim3 parameter vector must have shape (7,)")
        if not np.all(np.isfinite(vector)):
            raise ValueError("Sim3 parameter vector must be finite")
        return cls(
            scale=float(np.exp(vector[0])),
            rotation=so3_exp(vector[1:4]),
            translation=vector[4:7],
        )

    def as_vector(self) -> FloatArray:
        return np.concatenate(([np.log(self.scale)], so3_log(self.rotation), self.translation))

    def compose(self, other: Sim3) -> Sim3:
        """Return ``self(other(x))``."""

        return Sim3(
            scale=self.scale * other.scale,
            rotation=self.rotation @ other.rotation,
            translation=self.scale * (self.rotation @ other.translation) + self.translation,
        )

    def inverse(self) -> Sim3:
        inverse_rotation = self.rotation.T
        inverse_scale = 1.0 / self.scale
        return Sim3(
            scale=inverse_scale,
            rotation=inverse_rotation,
            translation=-inverse_scale * (inverse_rotation @ self.translation),
        )

    def transform_points(self, points: FloatArray) -> FloatArray:
        points = np.asarray(points, dtype=np.float64)
        return self.scale * np.einsum("ij,...j->...i", self.rotation, points) + self.translation

    def transform_vectors(self, vectors: FloatArray) -> FloatArray:
        vectors = np.asarray(vectors, dtype=np.float64)
        return self.scale * np.einsum("ij,...j->...i", self.rotation, vectors)

    def rotate_directions(self, directions: FloatArray) -> FloatArray:
        directions = np.asarray(directions, dtype=np.float64)
        return np.einsum("ij,...j->...i", self.rotation, directions)

    def transform_covariances(self, covariances: FloatArray) -> FloatArray:
        covariances = np.asarray(covariances, dtype=np.float64)
        return self.scale**2 * np.einsum(
            "ij,...jk,lk->...il", self.rotation, covariances, self.rotation
        )

    def relative_to(self, reference: Sim3) -> Sim3:
        """Return the transform from this local frame into ``reference`` local frame."""

        return reference.inverse().compose(self)
