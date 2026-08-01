# Immutable fused-sequence contract

`prob4d.fusion.FusedSequence` is the in-memory contract for a dense sequence in
one global gauge. It is also exported from the package root as
`prob4d.FusedSequence`.

## Public construction

The public dataclass constructor treats every supplied array as externally
owned. It therefore:

- creates defensive copies;
- normalizes frame indices to `int64`, point/flow means and covariances to
  `float64`, masks to `bool`, and contributor counts to `uint16`;
- validates nonempty, nonnegative, strictly increasing frame indices;
- validates shapes, finite active geometry, integer contributor counts, and at
  least one contributor for every valid point;
- requires `scene_flow`, `deform_mask`, and `flow_covariance` to be present or
  absent together; and
- makes every retained array read-only.

Mutating constructor inputs after validation therefore cannot alter the fused
sequence. Inactive point or flow payload entries are not interpreted; historical
artifacts may retain arbitrary sentinels outside their masks.

## Covariance admission

Point covariance is validated wherever `valid_mask` is true. Flow covariance is
validated wherever both `deform_mask` and `valid_mask` are true. Active matrices
must be finite, symmetric, and positive semidefinite.

The validation is processed in bounded chunks. For a symmetric `3 x 3` matrix,
positive semidefiniteness is equivalent to nonnegative principal minors. The
common well-conditioned path checks those minors using vectorized scalar
operations. Only a matrix with a negative principal minor enters the repository's
scale-aware eigendecomposition, which either rejects material indefiniteness or
projects tolerated floating-point-scale negative eigenvalues to zero.

This validates the contract without allocating a full-field eigensystem. It is
not a substitute for uncertainty calibration: a mathematically valid covariance
can still be too narrow or too wide on held-out data.

## Internal ownership transfer

`fuse_windows` and the fused-artifact loader allocate or decode their arrays
privately. They use the private `_from_owned_arrays` path to transfer those arrays
into the same validated contract without making a second dense defensive copy.
Accepted arrays are frozen in place and no mutable alias is returned. Ordinary
callers should use the public constructor.

This distinction is part of execution semantics only. It does not change fused
means, covariance meanings, correlation assumptions, provider schemas, or
claim-bearing artifact identities.

## Regression coverage

Tests cover defensive-copy isolation, canonical dtypes, read-only fields, the
internal no-second-copy path, complete flow triples, contributor admission,
finite active geometry, covariance symmetry and PSD rejection, tolerated
roundoff projection, artifact-load immutability, and the principal-minor fast
path.
