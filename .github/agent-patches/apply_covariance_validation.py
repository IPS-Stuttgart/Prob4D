from __future__ import annotations

from pathlib import Path


def replace_exact(path: str, old: str, new: str, *, count: int = 1) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    found = text.count(old)
    if found != count:
        raise RuntimeError(
            f"{path}: expected {count} exact occurrence(s), found {found}"
        )
    target.write_text(text.replace(old, new), encoding="utf-8")


replace_exact(
    "src/prob4d/fusion.py",
    "from numpy.typing import NDArray\n\nfrom .data import PredictionWindow\n",
    "from numpy.typing import NDArray\n\n"
    "from .covariance import regularized_inverse_psd\n"
    "from .data import PredictionWindow\n",
)
replace_exact(
    "src/prob4d/fusion.py",
    """def _regularized_inverse(covariance: FloatArray, floor: float = 1e-12) -> FloatArray:
    covariance = np.asarray(covariance, dtype=np.float64)
    symmetric = 0.5 * (covariance + np.swapaxes(covariance, -1, -2))
    identity = np.eye(symmetric.shape[-1])
    try:
        return np.linalg.inv(symmetric + floor * identity)
    except np.linalg.LinAlgError:
        eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
        eigenvalues = np.maximum(eigenvalues, floor)
        return np.einsum("...ij,...j,...kj->...ik", eigenvectors, 1.0 / eigenvalues, eigenvectors)
""",
    """def _regularized_inverse(covariance: FloatArray, floor: float = 1e-12) -> FloatArray:
    return regularized_inverse_psd(
        covariance,
        name="fusion covariance",
        eigenvalue_floor=floor,
    )
""",
)
replace_exact(
    "src/prob4d/fusion.py",
    "            _, log_determinant = np.linalg.slogdet(covariance)\n",
    """            sign, log_determinant = np.linalg.slogdet(covariance)
            if np.any(sign <= 0.0) or not np.all(np.isfinite(log_determinant)):
                raise ValueError(
                    "covariance intersection produced a non-positive covariance"
                )
""",
    count=2,
)

replace_exact(
    "src/prob4d/metrics.py",
    "from numpy.typing import NDArray\n\nfrom .fusion import FusedSequence\n",
    "from numpy.typing import NDArray\n\n"
    "from .covariance import covariance_statistics\n"
    "from .fusion import FusedSequence\n",
)
replace_exact(
    "src/prob4d/metrics.py",
    """    if normalizers.shape != (errors.shape[0],):
        raise ValueError("uncertainty_normalizers must have shape (N,)")
    active = (
        np.all(np.isfinite(errors), axis=1)
        & np.all(np.isfinite(covariances), axis=(1, 2))
        & np.isfinite(target_norms)
        & np.isfinite(normalizers)
    )
""",
    """    if normalizers.shape != (errors.shape[0],):
        raise ValueError("uncertainty_normalizers must have shape (N,)")
    if not np.all(np.isfinite(covariances)):
        raise ValueError("uncertainty diagnostic covariances must be finite")
    active = (
        np.all(np.isfinite(errors), axis=1)
        & np.isfinite(target_norms)
        & np.isfinite(normalizers)
    )
""",
)
replace_exact(
    "src/prob4d/metrics.py",
    """    symmetric = 0.5 * (covariances + np.swapaxes(covariances, -1, -2))
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    eigenvalues = np.maximum(eigenvalues, 1e-12)
    inverse = np.einsum("...ij,...j,...kj->...ik", eigenvectors, 1.0 / eigenvalues, eigenvectors)
    mahalanobis_squared = np.einsum("...i,...ij,...j->...", errors, inverse, errors)
    log_determinant = np.sum(np.log(eigenvalues), axis=-1)
""",
    """    symmetric, inverse, log_determinant = covariance_statistics(
        covariances,
        name="uncertainty diagnostic covariance",
        eigenvalue_floor=1e-12,
    )
    mahalanobis_squared = np.einsum("...i,...ij,...j->...", errors, inverse, errors)
""",
)
replace_exact(
    "src/prob4d/metrics.py",
    """    eigenvalues, eigenvectors = np.linalg.eigh(
        0.5 * (active_covariance + np.swapaxes(active_covariance, -1, -2))
    )
    eigenvalues = np.maximum(eigenvalues, 1e-12)
    inverse_covariance = np.einsum(
        "...ij,...j,...kj->...ik", eigenvectors, 1.0 / eigenvalues, eigenvectors
    )
    mahalanobis = np.einsum("...i,...ij,...j->...", active_error, inverse_covariance, active_error)
    log_determinant = np.sum(np.log(eigenvalues), axis=-1)
""",
    """    _, inverse_covariance, log_determinant = covariance_statistics(
        active_covariance,
        name="active sequence covariance",
        eigenvalue_floor=1e-12,
    )
    mahalanobis = np.einsum("...i,...ij,...j->...", active_error, inverse_covariance, active_error)
""",
)
replace_exact(
    "src/prob4d/metrics.py",
    """        predicted_flow = scale * prediction.scene_flow[prediction_indices]
        flow_mask = prediction.deform_mask[prediction_indices] & truth.deform_mask[truth_indices]
        if np.any(flow_mask):
            flow_epe = float(
                np.mean(
                    np.linalg.norm(
                        predicted_flow[flow_mask] - truth.scene_flow[truth_indices][flow_mask],
                        axis=-1,
                    )
                )
            )
""",
    """        predicted_flow = scale * prediction.scene_flow[prediction_indices]
        truth_flow = truth.scene_flow[truth_indices]
        flow_mask = (
            prediction.deform_mask[prediction_indices]
            & truth.deform_mask[truth_indices]
            & predicted_mask
            & truth_mask
            & np.all(np.isfinite(predicted_flow), axis=-1)
            & np.all(np.isfinite(truth_flow), axis=-1)
        )
        if np.any(flow_mask):
            flow_epe = float(
                np.mean(
                    np.linalg.norm(
                        predicted_flow[flow_mask] - truth_flow[flow_mask],
                        axis=-1,
                    )
                )
            )
""",
)
