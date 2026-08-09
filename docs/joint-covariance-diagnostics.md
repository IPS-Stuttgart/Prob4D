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

## Paired dependence ablation

The ordinary diagnostic answers whether one supplied joint covariance is
calibrated. A separate target-free ablation asks the narrower mechanism question:
does retaining the cross-row dependence improve a proper score over alternatives
with the same residuals and grouping?

Run it from an installed package or repository checkout:

```bash
python -m prob4d.joint_covariance_ablation matched-residuals.npz \
  --output joint-covariance-ablation.json \
  --bootstrap-replicates 2000 \
  --bootstrap-seed 7 \
  --confidence-level 0.95
```

It evaluates three arms without constructing a dense joint covariance:

1. **joint:** `blockdiag(D_i) + U U^T`;
2. **marginal-preserving independence:** `blockdiag(D_i + U_i U_i^T)`, which
   preserves every 3-D row marginal while deleting all cross-row covariance; and
3. **conditional only:** `blockdiag(D_i)`, which also removes the shared marginal
   uncertainty.

The marginal-preserving arm is the primary dependence ablation because it cannot
win merely by changing individual-row variance. For each independent group, the
report records:

- the Gaussian NLL-per-dimension advantage of the joint arm;
- the improvement in absolute normalized-NEES error relative to one; and
- the corresponding joint and ablation values.

Positive advantages favor the full joint covariance. Summaries use equal group
weight and include the fraction of groups favoring the joint arm. Percentile
intervals use a deterministic paired bootstrap over complete
`factor_group_ids`; rows, points, pixels, and covariance coordinates are never
resampled as independent units. Bootstrap index generation is chunked and the
replicate count is bounded, so the diagnostic does not allocate a
`replicates x groups` array without limit.

The ablation reads and hashes the same strict NPZ schema as the ordinary
joint-covariance command, evaluates those exact bytes, and publishes its report
with the same create-if-absent behavior. It is controlled mechanism evidence,
not permission to reuse or retune an opened target cohort.

This is a calibration diagnostic, not a provider promotion decision. Provider
competence and guarded BayesianPhysTwin benefit remain separate frozen gates.
