import numpy as np
import pytest

from prob4d.covariance import covariance_eigendecomposition, validated_covariance_psd


@pytest.mark.parametrize(
    ("keyword", "value", "exception"),
    [
        ("absolute_negative_tolerance", np.nan, ValueError),
        ("relative_negative_tolerance", np.inf, ValueError),
        ("symmetry_atol", -1.0, ValueError),
        ("symmetry_rtol", True, TypeError),
    ],
)
def test_covariance_validation_rejects_invalid_tolerances(
    keyword: str,
    value: object,
    exception: type[Exception],
) -> None:
    arguments = {keyword: value}

    with pytest.raises(exception):
        validated_covariance_psd(np.eye(2), **arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize("invalid", [np.nan, np.inf, 0.0, -1.0, True])
def test_covariance_eigenvalue_floor_must_be_finite_positive(invalid: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        covariance_eigendecomposition(
            np.eye(2),
            eigenvalue_floor=invalid,  # type: ignore[arg-type]
        )


def test_covariance_validation_still_projects_tiny_negative_roundoff() -> None:
    covariance = np.diag([1.0, -1e-15])

    validated = validated_covariance_psd(covariance)

    np.testing.assert_allclose(validated, np.diag([1.0, 0.0]))
    assert not validated.flags.writeable
