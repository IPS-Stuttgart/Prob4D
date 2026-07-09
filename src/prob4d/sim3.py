"""Small dependency-free helpers for three-dimensional similarity transforms."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.floating]


def skew(vector: FloatArray) -> FloatArray:
    """Return the cross-product matrix for a three-vector."""

    x, y, z = np.asarray(vector, dtype=np.float64)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def so3_exp(rotation_vector: FloatArray) -> FloatArray:
    """Map an axis-angle vector to a rotation matrix."""

    vector = np.asarray(rotation_vector, dtype=np.float64)
    theta = float(np.linalg.norm(vector))
    generator = skew(vector)
    if theta < 1e-8:
        return np.eye(3) + generator + 0.5 * generator @ generator
    a = np.sin(theta) / theta
    b = (1.0 - np.cos(theta)) / (theta * theta)
    return np.eye(3) + a * generator + b * generator @ generator


def so3_log(rotation: FloatArray) -> FloatArray:
    """Map a rotation matrix to its shortest axis-angle vector."""

    matrix = np.asarray(rotation, dtype=np.float64)
    cosine = float(np.clip((np.trace(matrix) - 1.0) * 0.5, -1.0, 1.0))
    theta = float(np.arccos(cosine))
    antisymmetric = np.array(
        [matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0], matrix[1, 0] - matrix[0, 1]]
    )
    if theta < 1e-8:
        return 0.5 * antisymmetric
    if np.pi - theta < 1e-5:
        eigenvalues, eigenvectors = np.linalg.eigh((matrix + np.eye(3)) * 0.5)
        axis = eigenvectors[:, int(np.argmax(eigenvalues))]
        if np.dot(axis, antisymmetric) < 0:
            axis = -axis
        return theta * axis
    return theta / (2.0 * np.sin(theta)) * antisymmetric


@dataclass(frozen=True)
class Sim3:
    """A transform ``y = scale * rotation @ x + translation``.

    The seven-vector convention used by :meth:`as_vector` is
    ``[log_scale, rotation_vector(3), translation(3)]``.
    """

    scale: float = 1.0
    rotation: FloatArray = field(default_factory=lambda: np.eye(3))
    translation: FloatArray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self) -> None:
        scale = float(self.scale)
        rotation = np.asarray(self.rotation, dtype=np.float64)
        translation = np.asarray(self.translation, dtype=np.float64)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Sim3 scale must be finite and strictly positive")
        if rotation.shape != (3, 3):
            raise ValueError("Sim3 rotation must have shape (3, 3)")
        if translation.shape != (3,):
            raise ValueError("Sim3 translation must have shape (3,)")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7):
            raise ValueError("Sim3 rotation must be orthonormal")
        if np.linalg.det(rotation) < 0.0:
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
        return cls(
            scale=float(np.exp(vector[0])),
            rotation=so3_exp(vector[1:4]),
            translation=vector[4:7],
        )

    def as_vector(self) -> FloatArray:
        return np.concatenate(
            ([np.log(self.scale)], so3_log(self.rotation), self.translation)
        )

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

