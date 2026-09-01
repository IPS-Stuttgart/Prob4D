### Added

- Add prior-anchored Gaussian query messages that exactly reproduce a fixed
  correlation-aware query posterior.
- Add covariance-intersection composition that counts the common prior once,
  remains idempotent for duplicated messages, and fails closed for incompatible
  query/prior identities or invalid weights.
- Add deterministic pairwise log-determinant/trace weight selection, focused
  algebraic tests, and the source-only DOT evaluation boundary.
