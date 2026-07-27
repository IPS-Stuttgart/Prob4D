from pathlib import Path


def replace_once(old: str, new: str) -> None:
    path = Path("src/prob4d/gauge.py")
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one gauge.py occurrence, found {count}")
    path.write_text(text.replace(old, new), encoding="utf-8")


replace_once(
    """from .alignment import WindowAlignment
from .sim3 import Sim3
""",
    """from .alignment import WindowAlignment
from .covariance import (
    covariance_eigendecomposition,
    regularized_inverse_psd,
    validated_covariance_psd,
)
from .sim3 import Sim3
""",
)
replace_once(
    """    def __post_init__(self) -> None:
        covariance = np.asarray(self.covariance, dtype=np.float64)
        if covariance.shape != (7, 7):
            raise ValueError("relative gauge covariance must have shape (7, 7)")
        object.__setattr__(self, "covariance", covariance)
""",
    """    def __post_init__(self) -> None:
        reference_id = str(self.reference_id)
        moving_id = str(self.moving_id)
        if not reference_id or not moving_id:
            raise ValueError("relative gauge window IDs must be nonempty")
        if reference_id == moving_id:
            raise ValueError("relative gauge window IDs must be distinct")
        covariance = validated_covariance_psd(
            self.covariance,
            name="relative gauge covariance",
            shape=(7, 7),
        )
        residual_rms = float(self.residual_rms)
        if not np.isfinite(residual_rms) or residual_rms < 0.0:
            raise ValueError("residual_rms must be finite and non-negative")
        num_correspondences = int(self.num_correspondences)
        if num_correspondences != self.num_correspondences or num_correspondences < 0:
            raise ValueError("num_correspondences must be a non-negative integer")
        object.__setattr__(self, "reference_id", reference_id)
        object.__setattr__(self, "moving_id", moving_id)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "residual_rms", residual_rms)
        object.__setattr__(self, "num_correspondences", num_correspondences)
""",
)
replace_once(
    """    def __post_init__(self) -> None:
        covariance = np.asarray(self.covariance, dtype=np.float64)
        if covariance.shape != (7, 7):
            raise ValueError("gauge covariance must have shape (7, 7)")
        object.__setattr__(self, "covariance", covariance)
""",
    """    def __post_init__(self) -> None:
        window_id = str(self.window_id)
        if not window_id:
            raise ValueError("gauge window_id must be nonempty")
        covariance = validated_covariance_psd(
            self.covariance,
            name="gauge covariance",
            shape=(7, 7),
        )
        object.__setattr__(self, "window_id", window_id)
        object.__setattr__(self, "covariance", covariance)
""",
)
replace_once(
    """    def __post_init__(self) -> None:
        covariance = np.asarray(self.covariance, dtype=np.float64)
        if covariance.shape != (7, 7):
            raise ValueError("gauge-anchor covariance must have shape (7, 7)")
        object.__setattr__(self, "covariance", covariance)
""",
    """    def __post_init__(self) -> None:
        window_id = str(self.window_id)
        if not window_id:
            raise ValueError("gauge-anchor window_id must be nonempty")
        covariance = validated_covariance_psd(
            self.covariance,
            name="gauge-anchor covariance",
            shape=(7, 7),
        )
        object.__setattr__(self, "window_id", window_id)
        object.__setattr__(self, "covariance", covariance)
""",
)
replace_once(
    """    def __post_init__(self) -> None:
        if self.scale <= 0 or self.rotation <= 0 or self.translation <= 0:
            raise ValueError("gauge covariance inflation factors must be positive")
        if not 0.0 < self.trim_quantile <= 1.0:
            raise ValueError("trim_quantile must be in (0, 1]")
        if self.count < 0:
            raise ValueError("calibration count must be non-negative")
""",
    """    def __post_init__(self) -> None:
        factors = np.asarray(
            [self.scale, self.rotation, self.translation],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(factors)) or np.any(factors <= 0.0):
            raise ValueError("gauge covariance inflation factors must be finite and positive")
        trim_quantile = float(self.trim_quantile)
        if not np.isfinite(trim_quantile) or not 0.0 < trim_quantile <= 1.0:
            raise ValueError("trim_quantile must be finite and in (0, 1]")
        count = int(self.count)
        if count != self.count or count < 0:
            raise ValueError("calibration count must be a non-negative integer")
        object.__setattr__(self, "scale", float(factors[0]))
        object.__setattr__(self, "rotation", float(factors[1]))
        object.__setattr__(self, "translation", float(factors[2]))
        object.__setattr__(self, "trim_quantile", trim_quantile)
        object.__setattr__(self, "count", count)
""",
)
replace_once(
    """    def apply(self, covariance: FloatArray) -> FloatArray:
        covariance = np.asarray(covariance, dtype=np.float64)
        if covariance.shape != (7, 7):
            raise ValueError("gauge covariance must have shape (7, 7)")
        scaling = self.scaling_matrix
        inflated = scaling @ covariance @ scaling
        return 0.5 * (inflated + inflated.T)
""",
    """    def apply(self, covariance: FloatArray) -> FloatArray:
        covariance = validated_covariance_psd(
            covariance,
            name="gauge covariance",
            shape=(7, 7),
            readonly=False,
        )
        scaling = self.scaling_matrix
        return validated_covariance_psd(
            scaling @ covariance @ scaling,
            name="inflated gauge covariance",
            shape=(7, 7),
            readonly=False,
        )
""",
)
replace_once(
    """        if not np.all(np.isfinite(errors)) or not np.all(np.isfinite(covariances)):
            raise ValueError("gauge calibration inputs must be finite")
""",
    """        if not np.all(np.isfinite(errors)):
            raise ValueError("gauge calibration errors must be finite")
        covariances = validated_covariance_psd(
            covariances,
            name="gauge calibration covariances",
            shape=(errors.shape[0], 7, 7),
            readonly=False,
        )
""",
)
replace_once(
    """                symmetric = 0.5 * (covariance + covariance.T)
                values[index] = (
                    error @ np.linalg.pinv(symmetric, rcond=1e-10) @ error
                ) / error.size
""",
    """                information = regularized_inverse_psd(
                    covariance,
                    name="gauge calibration covariance block",
                    eigenvalue_floor=1e-12,
                )
                values[index] = (error @ information @ error) / error.size
""",
)
replace_once(
    """    def __post_init__(self) -> None:
        if self.scale <= 0 or self.standard_deviation <= 0:
            raise ValueError("scale anchor values must be strictly positive")
""",
    """    def __post_init__(self) -> None:
        window_id = str(self.window_id)
        scale = float(self.scale)
        standard_deviation = float(self.standard_deviation)
        if not window_id:
            raise ValueError("scale-anchor window_id must be nonempty")
        if not np.isfinite(scale) or not np.isfinite(standard_deviation):
            raise ValueError("scale anchor values must be finite")
        if scale <= 0.0 or standard_deviation <= 0.0:
            raise ValueError("scale anchor values must be strictly positive")
        object.__setattr__(self, "window_id", window_id)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "standard_deviation", standard_deviation)
""",
)
replace_once(
    """    def __post_init__(self) -> None:
        local = np.asarray(self.local_point, dtype=np.float64)
        global_point = np.asarray(self.global_point, dtype=np.float64)
        covariance = np.asarray(self.covariance, dtype=np.float64)
        if local.shape != (3,) or global_point.shape != (3,):
            raise ValueError("point-anchor coordinates must have shape (3,)")
        if covariance.shape != (3, 3):
            raise ValueError("point-anchor covariance must have shape (3, 3)")
        object.__setattr__(self, "local_point", local)
        object.__setattr__(self, "global_point", global_point)
        object.__setattr__(self, "covariance", covariance)
""",
    """    def __post_init__(self) -> None:
        window_id = str(self.window_id)
        local = np.asarray(self.local_point, dtype=np.float64).copy()
        global_point = np.asarray(self.global_point, dtype=np.float64).copy()
        if not window_id:
            raise ValueError("point-anchor window_id must be nonempty")
        if local.shape != (3,) or global_point.shape != (3,):
            raise ValueError("point-anchor coordinates must have shape (3,)")
        if not np.all(np.isfinite(local)) or not np.all(np.isfinite(global_point)):
            raise ValueError("point-anchor coordinates must be finite")
        covariance = validated_covariance_psd(
            self.covariance,
            name="point-anchor covariance",
            shape=(3, 3),
        )
        local.setflags(write=False)
        global_point.setflags(write=False)
        object.__setattr__(self, "window_id", window_id)
        object.__setattr__(self, "local_point", local)
        object.__setattr__(self, "global_point", global_point)
        object.__setattr__(self, "covariance", covariance)
""",
)
replace_once(
    """def _whitener(covariance: FloatArray, floor: float = 1e-10) -> FloatArray:
    symmetric = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    eigenvalues = np.maximum(eigenvalues, floor)
    return (eigenvectors * (1.0 / np.sqrt(eigenvalues))) @ eigenvectors.T
""",
    """def _whitener(covariance: FloatArray, floor: float = 1e-10) -> FloatArray:
    _, eigenvalues, eigenvectors = covariance_eigendecomposition(
        covariance,
        name="gauge factor covariance",
        eigenvalue_floor=floor,
    )
    return (eigenvectors * (1.0 / np.sqrt(eigenvalues))) @ eigenvectors.T
""",
)
