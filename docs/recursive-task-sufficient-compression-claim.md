# Candidate claim wording (not yet manuscript-approved)

For a registered linear-Gaussian task-state closure `z = T x` satisfying `L = B T`, `H = D T`,
and `T F = A T`, exact one-step posterior-preserving compression of a supplied correlated
measurement-noise factor for `z` composes recursively: the compressed and full filters have the
same posterior over `z`, and therefore over every registered task `q = B z`, at every update for
every measurement sequence. The minimum retained factor rank at an update remains
`rank(U.T @ solve(S, Cov(z,y).T))`, bounded by the recursive task-state dimension.

This wording is conditional on the registered closure and exogenous Gaussian-noise model. It does
not claim novelty of functional observers or state aggregation, does not preserve observation
evidence, and does not extend automatically to changed task families, nonlinear linearizations,
or unregistered robust-weight changes.
