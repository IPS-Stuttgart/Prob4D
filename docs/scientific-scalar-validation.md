# Scientific scalar validation

Prob4D's scientific configuration and report boundaries reject values whose
meaning depends on Python coercion rather than the declared contract.

## Integer fields

Counts, iteration budgets, ranks, and index-like controls require genuine Python
or NumPy integer scalars. Boolean values, integral floats, numeric strings, and
fractional values are rejected before numerical work. This prevents cases such
as `True == 1` or `int(1.5) == 1` from silently changing an experiment.

## Real-valued fields

Continuous scientific controls require finite Python or NumPy real scalars.
Booleans, strings, complex values, NaN, and infinities are rejected. Bounds are
checked without first coercing an inadmissible value to another numerical type.

## Canonical grouped calibration identifiers

`GroupBalancedCalibrationReport.group_ids` is a canonical tuple of non-empty,
unique strings in sorted order. Lists, sets, a bare string, and non-string
entries are rejected rather than normalized. Group counts require genuine
positive integers and must sum exactly to the aggregate count.

These rules affect malformed or coercion-dependent inputs only. Valid Python and
NumPy scalar inputs retain their estimator and artifact semantics, and frozen
historical artifacts are not rewritten.
