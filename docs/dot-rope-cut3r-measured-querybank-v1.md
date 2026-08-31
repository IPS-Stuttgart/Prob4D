# Measured DOT/CUT3R query-bank and cache experiment

This experiment tests the specific systems claim left open by the Tracking Cloth
studies: whether a posterior-preserving shared factor remains useful when the
factor is measured by the actual DOT/CUT3R Prob4D route rather than synthesized
or fitted as an unrestricted empirical covariance.

The workflow reuses the immutable marker-free provider artifact from run
`33329701704`. It opens only the already-authorized R01--R03 2-D/3-D markers,
reconstructs the clustered-bootstrap relative-Sim(3) covariance, and propagates
that covariance through the real provider geometry of eight registered off-axis
probes. Provider inference is not rerun.

The primary covariance uses the source-frozen dependence strength `alpha=0.85`:

- `sqrt(alpha) U` is the common low-rank factor;
- `(1-alpha) diag(U U^T)` is an independent diagonal remainder; and
- the registered `0.02 * provider_span` noise floor is added independently.

This representation preserves each marginal variance while applying the frozen
cross-query dependence tempering. Query banks contain 1, 2, 4, and 8 probes.

For every sequence and bank, the experiment reports:

- measured parameter and propagated factor ranks;
- exact full/compressed gain, posterior covariance, and realized-mean parity;
- materialized factor, latent projection, direct `(K, P)` cache, and
  self-contained factor payloads in raw and compressed NPZ bytes;
- covariance acquisition, compression, cache construction, serialization,
  deserialization, consumer materialization, and repeated-query update costs;
- 10, 100, and 1000 Mbit/s transmission scenarios; and
- a positive-definite complete-joint spectral comparator under the same raw
  factor-plus-projection byte budget.

The primary payload comparison assumes the observation diagonal, query prior,
query/observation cross covariance, and means are already resident at the
consumer. The self-contained comparison is reported separately. A direct query
cache remains the preferred implementation for one immutable query or when the
common model blocks are not resident.

This is source-development representation and systems evidence. R04--R70 remain
unopened. It does not establish held-out transfer, BayesianPhysTwin or Causal4D
benefit, deployment calibration, safety, or state of the art.
