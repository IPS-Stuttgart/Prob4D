from __future__ import annotations

import numpy as np
import pytest

from prob4d._immutable_array import immutable_array, immutable_integer_array


def test_immutable_array_copies_values_and_cannot_be_reenabled() -> None:
    source = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
    frozen = immutable_array(source)

    source[...] = -1.0
    np.testing.assert_array_equal(
        frozen,
        np.arange(12, dtype=np.float32).reshape(2, 2, 3),
    )
    assert not frozen.flags.writeable
    with pytest.raises(ValueError, match="cannot set WRITEABLE flag"):
        frozen.setflags(write=True)
    with pytest.raises(ValueError, match="read-only"):
        frozen[0, 0, 0] = 99.0


def test_immutable_array_preserves_empty_and_scalar_shapes() -> None:
    scalar = immutable_array(np.asarray(3.5, dtype=np.float64))
    empty = immutable_array(np.empty((0, 3), dtype=np.float32))

    assert scalar.shape == ()
    assert scalar.dtype == np.dtype(np.float64)
    assert float(scalar) == 3.5
    assert empty.shape == (0, 3)
    assert empty.dtype == np.dtype(np.float32)
    assert not scalar.flags.writeable
    assert not empty.flags.writeable


def test_immutable_array_rejects_object_storage() -> None:
    with pytest.raises(ValueError, match="must not contain Python objects"):
        immutable_array(np.asarray([object()], dtype=object))


def test_immutable_integer_array_rejects_coercion_and_preserves_values() -> None:
    frozen = immutable_integer_array(np.asarray([1, 2], dtype=np.uint8), name="ids")

    np.testing.assert_array_equal(frozen, [1, 2])
    assert frozen.dtype == np.dtype(np.int64)
    with pytest.raises(ValueError, match="must contain integers"):
        immutable_integer_array(np.asarray([1.0]), name="ids")
    with pytest.raises(ValueError, match="must contain integers"):
        immutable_integer_array(np.asarray([True]), name="ids")
