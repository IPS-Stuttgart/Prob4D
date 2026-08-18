Add a content-addressed source-validation certificate for calibration after a
frozen Bayesian update-selection guard. It separates the all-group candidate,
accepted subset, exact fallback, and deployed policy; requires disjoint guard-fit,
guard-calibration, and validation object/session groups; evaluates coverage,
width, proper score, harmful accepted updates, support, and worst-group gates; and
fails closed on non-exact deployment or target-outcome semantics. This is
source-validation infrastructure and does not authorize target access.
