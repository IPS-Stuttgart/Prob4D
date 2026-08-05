# Joint covariance diagnostics

Prob4D observations separate conditional point covariance from a shared low-rank
gauge factor. Marginal coverage alone cannot show whether the exported
cross-observation dependence is calibrated. The grouped command

```bash
prob4d diagnostic joint-covariance matched-residuals.npz \
  --output joint-covariance-report.json
```

evaluates matched residuals under

```text
C = blockdiag(D_1, ..., D_N) + U U^T.
```

The input NPZ contains exactly:

| Field | Shape | Meaning |
| --- | --- | --- |
| `residual_xyz_m` | `N x 3` | prediction-minus-truth residuals in metres |
| `local_covariance_m2` | `N x 3 x 3` | positive-definite conditional covariance |
| `low_rank_factor_m` | `N x 3 x R` | shared covariance factor |
| `factor_group_ids` | `N` | optional independent integer or string group IDs |

Unknown members fail closed. Rows with different `factor_group_ids` are evaluated
as independent groups. A single coherent factor must not be split across IDs.

## Reported quantities

For every group, the report includes:

- joint Mahalanobis energy and normalized NEES;
- Gaussian negative log likelihood and log determinant;
- the effective shared rank;
- normalized residual energy in the shared gauge-induced subspace; and
- normalized residual energy in its conditional orthogonal complement.

Group summaries are averaged with equal group weight. This prevents a dense
sequence from dominating solely through row count.

The implementation whitens each 3-D conditional block and uses the Woodbury
identity and matrix determinant lemma. It never constructs the dense
`3N x 3N` covariance. Singular structure is recovered from the `R x R` matrix
`U^T D^-1 U`, avoiding a tall `3N x R` SVD and its left-singular-vector storage.
The Woodbury correction reuses the Cholesky factor already computed for the Gram
matrix.

The command reads the NPZ bytes once, evaluates those exact bytes, and records
their SHA-256. Its report is published with create-if-absent semantics, so a
concurrent writer cannot be overwritten. The numerical rank tolerance and
explicit claim boundary are retained in the JSON report.

This is a calibration diagnostic, not a provider promotion decision. Provider
competence and guarded BayesianPhysTwin benefit remain separate frozen gates.
