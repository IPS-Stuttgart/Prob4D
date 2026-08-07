"""Sparse square-root prior for a causal tree of seven-dimensional gauges.

The current provider-v2 factor bundle deliberately stores a dense joint gauge
covariance.  This module is an additive in-memory execution representation for
the production causal spanning tree.  It stores one parent transition and one
innovation square root per gauge, so retained prior storage grows linearly in
the number of windows while covariance and information actions remain exact.

No portable provider schema or frozen artifact identity is changed by this
module.  Dense admission is fail closed: a covariance is accepted only when the
declared parent tree reconstructs it within the requested numerical tolerance.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from ._immutable_array import immutable_array, immutable_integer_array

FloatArray: TypeAlias = NDArray[np.floating[Any]]
IntArray: TypeAlias = NDArray[np.integer[Any]]

GAUGE_DIMENSION = 7
SPARSE_GAUGE_TREE_PRIOR_MODE = "causal-gauge-tree-square-root-v1"


def _validated_symmetric_psd(
    value: Any,
    *,
    name: str,
    shape: tuple[int, int],
    tolerance: float = 1e-12,
) -> FloatArray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be finite")
    symmetric = 0.5 * (matrix + matrix.T)
    scale = max(float(np.max(np.abs(symmetric), initial=0.0)), 1.0)
    if not np.allclose(matrix, symmetric, atol=tolerance * scale, rtol=1e-10):
        raise ValueError(f"{name} must be symmetric")
    eigenvalues = np.linalg.eigvalsh(symmetric)
    if float(np.min(eigenvalues, initial=0.0)) < -tolerance * scale:
        raise ValueError(f"{name} must be positive semidefinite")
    return symmetric


def _deterministic_psd_root(covariance: FloatArray) -> FloatArray:
    """Return a deterministic square root with exactly seven columns."""

    symmetric = _validated_symmetric_psd(
        covariance,
        name="innovation covariance",
        shape=(GAUGE_DIMENSION, GAUGE_DIMENSION),
    )
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    for column in range(GAUGE_DIMENSION):
        pivot = int(np.argmax(np.abs(eigenvectors[:, column])))
        if eigenvectors[pivot, column] < 0.0:
            eigenvectors[:, column] *= -1.0
    return eigenvectors * np.sqrt(eigenvalues)[None, :]


def _covariance_pseudoinverse(covariance: FloatArray) -> FloatArray:
    """Return a deterministic PSD pseudoinverse without hidden regularization."""

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    scale = max(float(np.max(np.abs(eigenvalues), initial=0.0)), 1.0)
    keep = eigenvalues > 1e-12 * scale
    inverse_values = np.zeros_like(eigenvalues)
    inverse_values[keep] = 1.0 / eigenvalues[keep]
    return (eigenvectors * inverse_values) @ eigenvectors.T


def _gauge_rhs_blocks(
    value: Any,
    *,
    gauge_count: int,
    name: str,
) -> tuple[FloatArray, str]:
    array = np.asarray(value, dtype=np.float64)
    dimension = GAUGE_DIMENSION * gauge_count
    layout: str
    if array.shape == (dimension,):
        blocks = array.reshape(gauge_count, GAUGE_DIMENSION, 1)
        layout = "flat-vector"
    elif array.shape == (gauge_count, GAUGE_DIMENSION):
        blocks = array[:, :, None]
        layout = "block-vector"
    elif array.ndim == 2 and array.shape[0] == dimension:
        blocks = array.reshape(gauge_count, GAUGE_DIMENSION, array.shape[1])
        layout = "flat-matrix"
    elif array.ndim == 3 and array.shape[:2] == (gauge_count, GAUGE_DIMENSION):
        blocks = array
        layout = "block-matrix"
    else:
        raise ValueError(
            f"{name} must have shape ({dimension},), ({gauge_count}, 7), "
            f"({dimension}, N), or ({gauge_count}, 7, N)"
        )
    if not np.all(np.isfinite(blocks)):
        raise ValueError(f"{name} must be finite")
    return np.asarray(blocks, dtype=np.float64), layout


def _restore_gauge_rhs(blocks: FloatArray, layout: str) -> FloatArray:
    gauge_count = blocks.shape[0]
    if layout == "flat-vector":
        return blocks[:, :, 0].reshape(GAUGE_DIMENSION * gauge_count)
    if layout == "block-vector":
        return blocks[:, :, 0]
    if layout == "flat-matrix":
        return blocks.reshape(GAUGE_DIMENSION * gauge_count, blocks.shape[2])
    if layout == "block-matrix":
        return blocks
    raise RuntimeError("unknown gauge right-hand-side layout")


def _observation_rhs_blocks(
    value: Any,
    *,
    observation_count: int,
    name: str,
) -> tuple[FloatArray, str]:
    array = np.asarray(value, dtype=np.float64)
    dimension = 3 * observation_count
    layout: str
    if array.shape == (dimension,):
        blocks = array.reshape(observation_count, 3, 1)
        layout = "flat-vector"
    elif array.shape == (observation_count, 3):
        blocks = array[:, :, None]
        layout = "block-vector"
    elif array.ndim == 2 and array.shape[0] == dimension:
        blocks = array.reshape(observation_count, 3, array.shape[1])
        layout = "flat-matrix"
    elif array.ndim == 3 and array.shape[:2] == (observation_count, 3):
        blocks = array
        layout = "block-matrix"
    else:
        raise ValueError(
            f"{name} must have shape ({dimension},), ({observation_count}, 3), "
            f"({dimension}, N), or ({observation_count}, 3, N)"
        )
    if not np.all(np.isfinite(blocks)):
        raise ValueError(f"{name} must be finite")
    return np.asarray(blocks, dtype=np.float64), layout


def _restore_observation_rhs(blocks: FloatArray, layout: str) -> FloatArray:
    observation_count = blocks.shape[0]
    if layout == "flat-vector":
        return blocks[:, :, 0].reshape(3 * observation_count)
    if layout == "block-vector":
        return blocks[:, :, 0]
    if layout == "flat-matrix":
        return blocks.reshape(3 * observation_count, blocks.shape[2])
    if layout == "block-matrix":
        return blocks
    raise RuntimeError("unknown observation right-hand-side layout")


@dataclass(frozen=True, slots=True)
class SparseGaugeTreePrior:
    """Implicit square root of one ordered linear-Gaussian gauge tree.

    For the root, ``x[0] = R[0] z[0]``.  Every later gauge satisfies

    ``x[i] = A[i] x[parent[i]] + R[i] z[i]``

    with independent standard-normal ``z`` blocks.  ``A[0]`` is exactly zero
    and ``parent_indices[0]`` is ``-1``.
    """

    window_ids: tuple[str, ...]
    parent_indices: IntArray
    transition_matrices: FloatArray
    innovation_roots: FloatArray
    mode: str = SPARSE_GAUGE_TREE_PRIOR_MODE

    def __post_init__(self) -> None:
        window_ids = tuple(str(value) for value in self.window_ids)
        if not window_ids or any(not value for value in window_ids):
            raise ValueError("window_ids must contain nonempty strings")
        if len(set(window_ids)) != len(window_ids):
            raise ValueError("window_ids must be unique")
        parent_indices = immutable_integer_array(
            self.parent_indices,
            name="parent_indices",
        )
        transitions = np.asarray(self.transition_matrices, dtype=np.float64)
        roots = np.asarray(self.innovation_roots, dtype=np.float64)
        count = len(window_ids)
        expected = (count, GAUGE_DIMENSION, GAUGE_DIMENSION)
        if parent_indices.shape != (count,):
            raise ValueError("parent_indices must have shape (K,)")
        if transitions.shape != expected:
            raise ValueError(f"transition_matrices must have shape {expected}")
        if roots.shape != expected:
            raise ValueError(f"innovation_roots must have shape {expected}")
        if not np.all(np.isfinite(transitions)) or not np.all(np.isfinite(roots)):
            raise ValueError("tree matrices must be finite")
        if int(parent_indices[0]) != -1:
            raise ValueError("the first gauge must be the unique root")
        if np.any(transitions[0] != 0.0):
            raise ValueError("the root transition matrix must be exactly zero")
        for index in range(1, count):
            parent = int(parent_indices[index])
            if parent < 0 or parent >= index:
                raise ValueError("every child parent must precede the child")
        if self.mode != SPARSE_GAUGE_TREE_PRIOR_MODE:
            raise ValueError("sparse gauge-tree prior mode changed")
        object.__setattr__(self, "window_ids", window_ids)
        object.__setattr__(self, "parent_indices", parent_indices)
        object.__setattr__(
            self,
            "transition_matrices",
            immutable_array(transitions, dtype=np.float64),
        )
        object.__setattr__(
            self,
            "innovation_roots",
            immutable_array(roots, dtype=np.float64),
        )

    @classmethod
    def from_components(
        cls,
        *,
        window_ids: Sequence[str],
        parent_indices: Any,
        transition_matrices: Any,
        innovation_covariances: Any,
    ) -> SparseGaugeTreePrior:
        """Build from explicit transitions and conditional covariance blocks."""

        identifiers = tuple(str(value) for value in window_ids)
        count = len(identifiers)
        covariances = np.asarray(innovation_covariances, dtype=np.float64)
        expected = (count, GAUGE_DIMENSION, GAUGE_DIMENSION)
        if covariances.shape != expected:
            raise ValueError(f"innovation_covariances must have shape {expected}")
        roots = np.empty_like(covariances)
        for index in range(count):
            covariance = _validated_symmetric_psd(
                covariances[index],
                name=f"innovation covariance {index}",
                shape=(GAUGE_DIMENSION, GAUGE_DIMENSION),
            )
            roots[index] = _deterministic_psd_root(covariance)
        return cls(
            window_ids=identifiers,
            parent_indices=parent_indices,
            transition_matrices=transition_matrices,
            innovation_roots=roots,
        )

    @classmethod
    def from_dense_covariance(
        cls,
        *,
        window_ids: Sequence[str],
        parent_indices: Any,
        covariance: Any,
        parity_atol: float = 1e-10,
        parity_rtol: float = 1e-8,
    ) -> SparseGaugeTreePrior:
        """Factor a dense covariance and reject a non-tree dependence pattern."""

        identifiers = tuple(str(value) for value in window_ids)
        count = len(identifiers)
        dimension = GAUGE_DIMENSION * count
        parents = np.asarray(parent_indices)
        if parents.shape != (count,) or parents.dtype.kind not in {"i", "u"}:
            raise ValueError("parent_indices must contain K genuine integers")
        parents = np.asarray(parents, dtype=np.int64)
        if count < 1 or int(parents[0]) != -1:
            raise ValueError("the first gauge must be the unique root")
        for index in range(1, count):
            if int(parents[index]) < 0 or int(parents[index]) >= index:
                raise ValueError("every child parent must precede the child")
        matrix = _validated_symmetric_psd(
            covariance,
            name="dense joint gauge covariance",
            shape=(dimension, dimension),
        )
        transitions = np.zeros(
            (count, GAUGE_DIMENSION, GAUGE_DIMENSION),
            dtype=np.float64,
        )
        innovations = np.empty_like(transitions)
        innovations[0] = matrix[:GAUGE_DIMENSION, :GAUGE_DIMENSION]
        for child in range(1, count):
            parent = int(parents[child])
            child_slice = slice(
                GAUGE_DIMENSION * child,
                GAUGE_DIMENSION * (child + 1),
            )
            parent_slice = slice(
                GAUGE_DIMENSION * parent,
                GAUGE_DIMENSION * (parent + 1),
            )
            parent_covariance = matrix[parent_slice, parent_slice]
            cross = matrix[child_slice, parent_slice]
            transition = cross @ _covariance_pseudoinverse(parent_covariance)
            innovation = matrix[child_slice, child_slice] - transition @ cross.T
            transitions[child] = transition
            innovations[child] = _validated_symmetric_psd(
                innovation,
                name=f"conditional innovation covariance {child}",
                shape=(GAUGE_DIMENSION, GAUGE_DIMENSION),
                tolerance=max(parity_atol, 1e-12),
            )
        result = cls.from_components(
            window_ids=identifiers,
            parent_indices=parents,
            transition_matrices=transitions,
            innovation_covariances=innovations,
        )
        reconstructed = result.dense_covariance()
        if not np.allclose(
            reconstructed,
            matrix,
            atol=parity_atol,
            rtol=parity_rtol,
        ):
            maximum_error = float(np.max(np.abs(reconstructed - matrix), initial=0.0))
            raise ValueError(
                "dense gauge covariance is not representable by the declared "
                f"causal tree; maximum reconstruction error is {maximum_error:.6e}"
            )
        return result

    @classmethod
    def from_joint_gauge_posterior(
        cls,
        posterior: Any,
        *,
        parity_atol: float = 1e-10,
        parity_rtol: float = 1e-8,
    ) -> SparseGaugeTreePrior:
        """Admit an existing dense production posterior through strict parity."""

        window_ids = tuple(str(value) for value in posterior.window_ids)
        parent_ids = tuple(posterior.parent_window_ids)
        if not bool(posterior.cross_window_covariance_preserved):
            raise ValueError("posterior must preserve complete cross-window covariance")
        if len(parent_ids) != len(window_ids) or parent_ids[0] is not None:
            raise ValueError("posterior does not declare one complete causal parent tree")
        positions = {window_id: index for index, window_id in enumerate(window_ids)}
        parents = np.full(len(window_ids), -1, dtype=np.int64)
        for child in range(1, len(window_ids)):
            parent_id = parent_ids[child]
            if not isinstance(parent_id, str) or parent_id not in positions:
                raise ValueError("posterior contains an unknown causal parent")
            parent = positions[parent_id]
            if parent >= child:
                raise ValueError("posterior parent must precede its child")
            parents[child] = parent
        return cls.from_dense_covariance(
            window_ids=window_ids,
            parent_indices=parents,
            covariance=posterior.joint_covariance,
            parity_atol=parity_atol,
            parity_rtol=parity_rtol,
        )

    @property
    def gauge_count(self) -> int:
        return len(self.window_ids)

    @property
    def dense_dimension(self) -> int:
        return GAUGE_DIMENSION * self.gauge_count

    @property
    def retained_nbytes(self) -> int:
        """Bytes retained by the three numerical tree arrays."""

        return int(
            self.parent_indices.nbytes
            + self.transition_matrices.nbytes
            + self.innovation_roots.nbytes
        )

    @property
    def dense_covariance_nbytes(self) -> int:
        return int(self.dense_dimension * self.dense_dimension * np.dtype(np.float64).itemsize)

    @property
    def dense_to_sparse_storage_ratio(self) -> float:
        return self.dense_covariance_nbytes / self.retained_nbytes

    @property
    def innovation_covariances(self) -> FloatArray:
        result = np.einsum(
            "kij,klj->kil",
            self.innovation_roots,
            self.innovation_roots,
            optimize=True,
        )
        return 0.5 * (result + result.swapaxes(1, 2))

    @property
    def supports_information_actions(self) -> bool:
        for root in self.innovation_roots:
            sign, log_determinant = np.linalg.slogdet(root)
            if sign == 0.0 or not np.isfinite(log_determinant):
                return False
        return True

    def apply_square_root(self, value: Any) -> FloatArray:
        """Apply the implicit block-lower-triangular square root."""

        innovations, layout = _gauge_rhs_blocks(
            value,
            gauge_count=self.gauge_count,
            name="square-root input",
        )
        result = np.empty_like(innovations)
        result[0] = self.innovation_roots[0] @ innovations[0]
        for child in range(1, self.gauge_count):
            parent = int(self.parent_indices[child])
            result[child] = (
                self.transition_matrices[child] @ result[parent]
                + self.innovation_roots[child] @ innovations[child]
            )
        return _restore_gauge_rhs(result, layout)

    def apply_square_root_transpose(self, value: Any) -> FloatArray:
        """Apply the transpose of the implicit square root."""

        right_hand_side, layout = _gauge_rhs_blocks(
            value,
            gauge_count=self.gauge_count,
            name="square-root-transpose input",
        )
        accumulated = right_hand_side.copy()
        result = np.empty_like(accumulated)
        for child in range(self.gauge_count - 1, -1, -1):
            result[child] = self.innovation_roots[child].T @ accumulated[child]
            parent = int(self.parent_indices[child])
            if parent >= 0:
                accumulated[parent] += self.transition_matrices[child].T @ accumulated[child]
        return _restore_gauge_rhs(result, layout)

    def apply_covariance(self, value: Any) -> FloatArray:
        """Apply the exact dense covariance without materializing it."""

        return self.apply_square_root(self.apply_square_root_transpose(value))

    def apply_information(self, value: Any) -> FloatArray:
        """Apply the exact information matrix when every innovation is SPD."""

        blocks, layout = _gauge_rhs_blocks(
            value,
            gauge_count=self.gauge_count,
            name="information input",
        )
        gradient = np.zeros_like(blocks)
        for child in range(self.gauge_count):
            parent = int(self.parent_indices[child])
            residual = blocks[child].copy()
            if parent >= 0:
                residual -= self.transition_matrices[child] @ blocks[parent]
            try:
                whitened = np.linalg.solve(self.innovation_roots[child], residual)
                weighted = np.linalg.solve(
                    self.innovation_roots[child].T,
                    whitened,
                )
            except np.linalg.LinAlgError as error:
                raise ValueError(
                    "information actions require strictly positive-definite "
                    "innovation covariance blocks"
                ) from error
            gradient[child] += weighted
            if parent >= 0:
                gradient[parent] -= self.transition_matrices[child].T @ weighted
        return _restore_gauge_rhs(gradient, layout)

    def solve_covariance(self, value: Any) -> FloatArray:
        """Alias for applying the exact information matrix."""

        return self.apply_information(value)

    def log_determinant(self) -> float:
        """Return ``log(det(Sigma))`` without materializing ``Sigma``."""

        total = 0.0
        for root in self.innovation_roots:
            sign, log_determinant = np.linalg.slogdet(root)
            if sign == 0.0 or not np.isfinite(log_determinant):
                raise ValueError(
                    "log determinant requires strictly positive-definite "
                    "innovation covariance blocks"
                )
            total += 2.0 * float(log_determinant)
        return total

    def dense_square_root(self) -> FloatArray:
        """Materialize the exact dense square root for compatibility tests."""

        identity = np.eye(self.dense_dimension, dtype=np.float64).reshape(
            self.gauge_count,
            GAUGE_DIMENSION,
            self.dense_dimension,
        )
        result = self.apply_square_root(identity)
        return np.asarray(result).reshape(self.dense_dimension, self.dense_dimension)

    def dense_covariance(self) -> FloatArray:
        """Materialize the exact dense covariance for compatibility boundaries."""

        identity = np.eye(self.dense_dimension, dtype=np.float64)
        result = np.asarray(self.apply_covariance(identity), dtype=np.float64)
        return 0.5 * (result + result.T)

    def _resolve_gauges(self, gauges: Sequence[int | str], *, name: str) -> tuple[int, ...]:
        positions = {window_id: index for index, window_id in enumerate(self.window_ids)}
        resolved: list[int] = []
        for value in gauges:
            if isinstance(value, bool):
                raise TypeError(f"{name} must contain gauge indices or IDs")
            if isinstance(value, str):
                if value not in positions:
                    raise ValueError(f"{name} references an unknown gauge ID")
                index = positions[value]
            elif isinstance(value, (int, np.integer)):
                index = int(value)
                if index < 0 or index >= self.gauge_count:
                    raise ValueError(f"{name} references an unknown gauge index")
            else:
                raise TypeError(f"{name} must contain gauge indices or IDs")
            resolved.append(index)
        if not resolved:
            raise ValueError(f"{name} must not be empty")
        if len(set(resolved)) != len(resolved):
            raise ValueError(f"{name} must not contain duplicates")
        return tuple(resolved)

    def cross_covariance(
        self,
        left_gauges: Sequence[int | str],
        right_gauges: Sequence[int | str],
    ) -> FloatArray:
        """Materialize selected ordered ``7 x 7`` covariance blocks."""

        left = self._resolve_gauges(left_gauges, name="left_gauges")
        right = self._resolve_gauges(right_gauges, name="right_gauges")
        basis = np.zeros(
            (self.gauge_count, GAUGE_DIMENSION, GAUGE_DIMENSION * len(right)),
            dtype=np.float64,
        )
        for column, gauge in enumerate(right):
            basis[
                gauge,
                :,
                GAUGE_DIMENSION * column : GAUGE_DIMENSION * (column + 1),
            ] = np.eye(GAUGE_DIMENSION)
        covariance_columns = np.asarray(self.apply_covariance(basis))
        return covariance_columns[np.asarray(left)].reshape(
            GAUGE_DIMENSION * len(left),
            GAUGE_DIMENSION * len(right),
        )

    def marginal_covariance(self, gauges: Sequence[int | str]) -> FloatArray:
        selected = tuple(gauges)
        return self.cross_covariance(selected, selected)

    def apply_observation_covariance(
        self,
        local_gauge_jacobian: Any,
        gauge_indices: Any,
        value: Any,
    ) -> FloatArray:
        """Apply ``H Sigma H.T`` for one local ``3 x 7`` block per row."""

        jacobians = np.asarray(local_gauge_jacobian, dtype=np.float64)
        if jacobians.ndim != 3 or jacobians.shape[1:] != (3, GAUGE_DIMENSION):
            raise ValueError("local_gauge_jacobian must have shape (M, 3, 7)")
        if not np.all(np.isfinite(jacobians)):
            raise ValueError("local_gauge_jacobian must be finite")
        indices = np.asarray(gauge_indices)
        if indices.shape != (len(jacobians),) or indices.dtype.kind not in {"i", "u"}:
            raise ValueError("gauge_indices must contain M genuine integers")
        indices = np.asarray(indices, dtype=np.int64)
        if np.any(indices < 0) or np.any(indices >= self.gauge_count):
            raise ValueError("gauge_indices reference an unknown gauge")
        observation_rhs, layout = _observation_rhs_blocks(
            value,
            observation_count=len(jacobians),
            name="observation covariance input",
        )
        row_contributions = np.einsum(
            "mij,miq->mjq",
            jacobians,
            observation_rhs,
            optimize=True,
        )
        gauge_rhs = np.zeros(
            (self.gauge_count, GAUGE_DIMENSION, observation_rhs.shape[2]),
            dtype=np.float64,
        )
        np.add.at(gauge_rhs, indices, row_contributions)
        gauge_covariance = np.asarray(self.apply_covariance(gauge_rhs))
        result = np.einsum(
            "mij,mjq->miq",
            jacobians,
            gauge_covariance[indices],
            optimize=True,
        )
        return _restore_observation_rhs(result, layout)

    def observation_covariance(
        self,
        local_gauge_jacobian: Any,
        gauge_indices: Any,
    ) -> FloatArray:
        """Materialize ``H Sigma H.T`` for diagnostics and parity tests."""

        jacobians = np.asarray(local_gauge_jacobian)
        if jacobians.ndim != 3:
            raise ValueError("local_gauge_jacobian must have shape (M, 3, 7)")
        dimension = 3 * len(jacobians)
        result = np.asarray(
            self.apply_observation_covariance(
                local_gauge_jacobian,
                gauge_indices,
                np.eye(dimension, dtype=np.float64),
            )
        )
        return 0.5 * (result + result.T)


__all__ = [
    "GAUGE_DIMENSION",
    "SPARSE_GAUGE_TREE_PRIOR_MODE",
    "SparseGaugeTreePrior",
]
