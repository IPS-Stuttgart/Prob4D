from __future__ import annotations

import numpy as np
import pytest

from prob4d.diagnostics.covariance_support import covariance_support_diagnostic


def test_full_rank_diagnostic_matches_standard_nees() -> None:
    diagnostic = covariance_support_diagnostic(
        np.array([2.0, 3.0]),
        np.diag([4.0, 9.0]),
    )

    assert diagnostic.dimension == 2
    assert diagnostic.rank == 2
    assert diagnostic.observable_normalized_squared_error == pytest.approx(2.0)
    assert diagnostic.rank_normalized_observable_squared_error == pytest.approx(1.0)
    assert diagnostic.nullspace_error_norm == pytest.approx(0.0)
    assert diagnostic.support_consistent


def test_singular_covariance_is_valid_when_error_lies_in_its_range() -> None:
    diagnostic = covariance_support_diagnostic(
        np.array([2.0, 0.0]),
        np.diag([4.0, 0.0]),
    )

    assert diagnostic.rank == 1
    assert diagnostic.observable_normalized_squared_error == pytest.approx(1.0)
    assert diagnostic.rank_normalized_observable_squared_error == pytest.approx(1.0)
    assert diagnostic.nullspace_error_norm == pytest.approx(0.0)
    assert diagnostic.support_consistent


def test_nullspace_error_is_reported_instead_of_silently_zero_weighted() -> None:
    diagnostic = covariance_support_diagnostic(
        np.array([2.0, 0.25]),
        np.diag([4.0, 0.0]),
    )

    assert diagnostic.rank == 1
    assert diagnostic.observable_normalized_squared_error == pytest.approx(1.0)
    assert diagnostic.nullspace_error_norm == pytest.approx(0.25)
    assert not diagnostic.support_consistent


def test_zero_covariance_accepts_only_numerically_zero_error() -> None:
    zero = covariance_support_diagnostic(np.zeros(3), np.zeros((3, 3)))
    nonzero = covariance_support_diagnostic(
        np.array([0.0, 0.0, 1e-4]),
        np.zeros((3, 3)),
    )

    assert zero.rank == 0
    assert zero.observable_normalized_squared_error == 0.0
    assert zero.support_consistent
    assert nonzero.rank == 0
    assert not nonzero.support_consistent
    assert nonzero.nullspace_error_norm == pytest.approx(1e-4)


def test_invalid_covariance_fails_closed() -> None:
    with pytest.raises(ValueError, match="positive semidefinite"):
        covariance_support_diagnostic(
            np.ones(2),
            np.diag([1.0, -1e-3]),
        )

    with pytest.raises(ValueError, match="symmetric"):
        covariance_support_diagnostic(
            np.ones(2),
            np.array([[1.0, 0.5], [0.0, 1.0]]),
        )
