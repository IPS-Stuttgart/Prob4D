Add an experimental exact registered-query covariance decomposition. It computes
the minimum numerical latent subspace preserving every registered query covariance
and cross-covariance, returns the complete query-orthogonal uncertainty factor,
fails closed when a rank cap is too small, and explicitly forbids dropping the
orthogonal factor from the observation likelihood.