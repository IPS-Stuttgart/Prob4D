# Joint Sim(3) linearization-closure diagnostic

Prob4D propagates complete joint gauge uncertainty through causal chains of
`Sim(3)` transforms. The production representation uses analytic first-order
Jacobians, which are exact local derivatives but can become an inaccurate
Gaussian approximation when the gauge distribution is broad, compositions are
long, or rotations approach the `SO(3)` logarithm branch cut.

`prob4d.gauge_linearization_closure` compares that first-order propagation with a
deterministic nonlinear sigma-point reference before a residual failure is
attributed to conditional point covariance. It is a source-only diagnostic and
does not change provider-v2, the production gauge tree, any calibration, or any
downstream BayesianPhysTwin decision.

## Compared representations

For a chain

```text
T = T_0 compose T_1 compose ... compose T_(K-1)
```

and a complete joint Gaussian over all `7K` transform coordinates, the analytic
path computes the chain Jacobian, transforms the declared local points, and
propagates the full covariance into point space. An optional caller-owned query
Jacobian projects the same point uncertainty into a registered downstream query.

The nonlinear reference forms a deterministic positive-semidefinite covariance
root and evaluates spherical-radial sigma points through exact `Sim(3)`
composition and point transformation. For rank `r`, the rule evaluates the
`2r` points

```text
g_mean +/- sqrt(r) * L[:, j]
```

with weight `1 / (2r)`. Rank-zero covariance is handled exactly. No Monte Carlo
seed, sample-order tolerance, or target outcome enters the result.

## Reported closure metrics

Each case reports, separately for transformed points and the optional query:

- normalized nonlinear mean shift;
- relative covariance Frobenius error;
- maximum directional variance-ratio deviation on the analytic supported
  subspace;
- nonlinear variance outside the analytic supported subspace;
- retained rank of the complete joint gauge covariance; and
- minimum clearance of the mean and every sigma-point transform chain from the
  `SO(3)` logarithm branch cut.

The branch-cut check is fail-closed. A chain whose analytic derivative is
undefined, or whose nonlinear sigma points enter the frozen unsafe clearance,
is not converted into a numerical covariance result.

## Independent-group aggregation

Complete physical objects or independently acquired object sessions are the
independent groups. Frames, windows, points, pixels, and query coordinates are
nested observations only. A group passes only when every registered case in that
group passes. Groups then receive equal mass in the final pass fraction.

The terminal decisions are:

- `linearization-closure-adequate`;
- `linearization-closure-negative`; or
- `insufficient-independent-groups`.

A negative result localizes the problem to first-order gauge propagation or
nonlinear query projection. It does **not** authorize a richer conditional
point-uncertainty model. A passing result establishes only numerical closure
under the declared Gaussian coordinate model; it does not establish empirical
calibration.

## Build and verify

Start from the checked-in example:

```bash
python -m prob4d.gauge_linearization_closure build \
  docs/examples/gauge-linearization-closure-input.json \
  --output outputs/gauge-linearization-closure.json \
  --require-pass

python -m prob4d.gauge_linearization_closure verify \
  outputs/gauge-linearization-closure.json \
  --require-pass
```

A valid negative artifact is still written. With `--require-pass`, the command
returns exit status 3 for a valid negative or insufficient-group decision.
Publication is content-addressed, atomic, and no-clobber; an identical rewrite is
idempotent, while different retained bytes at the same path are rejected.

The thresholds in the example are illustrative only. A real provider study must
freeze its policy from source/calibration design before inspecting the relevant
source residual outcomes.

## Position in provider readiness

Run this diagnostic after source means and identities are useful and before
conditional point-covariance development is authorized. A practical order is:

```text
support
  -> means
  -> identity
  -> gauge/dependence
  -> Sim(3) linearization closure
  -> conditional point covariance
  -> physical-query value
```

If closure fails, retain the existing point model and evaluate a separately
versioned gauge remedy such as iterated relinearization or nonlinear query
projection on source groups only. If closure passes but conditional residual
energy remains miscalibrated after the other upstream gates pass, the existing
source-covariance localization rule may then authorize point-uncertainty work.

## Scientific boundary

This diagnostic uses no target outcomes and makes no provider-competence,
calibration, BayesianPhysTwin-benefit, Causal4D-benefit, deployment-safety, or
state-of-the-art claim. It is a failure-localization tool for deciding which
upstream uncertainty representation should be investigated next.
