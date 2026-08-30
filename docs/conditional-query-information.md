# Conditional query information for correlated partial gauges

## Scientific question

Which additional 4-D prediction window is useful **after** the already consumed
windows, for one declared downstream physical query?

An overlap's standalone covariance is not its incremental information. Repeated
source evidence can look precise without adding information; a geometrically
complementary window can be more useful; reducing total gauge uncertainty can be
irrelevant to the query. This experimental module joins the existing observable
Sim(3) factor and query-observability work with explicit cross-window dependence.
It changes no stable exporter, production guard, evidence gate, or target access.

## Model and conditional likelihood

Work in one fixed, centroid-normalized, seven-dimensional local gauge chart:

\[
x\sim\mathcal N(m_0,P_0),\qquad
\begin{bmatrix}y_H\\y_c\end{bmatrix}
=\begin{bmatrix}H_H\\H_c\end{bmatrix}x+
\begin{bmatrix}\epsilon_H\\\epsilon_c\end{bmatrix},\qquad
\operatorname{Cov}(\epsilon)=
\begin{bmatrix}R_{HH}&R_{Hc}\\R_{cH}&R_{cc}\end{bmatrix}.
\]

The complete prior is independent of this observation noise. The joint noise
covariance is known/frozen for this calculation; it is **not** a predictive
covariance that already contains state uncertainty. For nonsingular history
noise, define

\[
L=R_{cH}R_{HH}^{-1},\quad
\widetilde y_c=y_c-Ly_H,\quad
\widetilde H_c=H_c-LH_H,\quad
\widetilde R_c=R_{cc}-LR_{Hc}.
\]

Then the new likelihood is

\[
p(y_c\mid x,y_H)
=\mathcal N(\widetilde y_c;\widetilde H_cx,\widetilde R_c).
\]

The implementation whitens only the supported noise subspace. It neither
inverts a geometric nullspace nor fills it with a ridge. Singular noise is
accepted only when its zero-noise directions carry no state information and the
observations satisfy their deterministic identities. Informative noiseless
constraints require a constrained solver and are explicitly rejected, rather
than silently discarded by a pseudoinverse. Numerical support is relative to
the declared covariance scale, with `rtol=1e-10` by default.

### Proposition: exact incremental query value

Let the actual history posterior be `(m_H, P_H)`, and let `q=Jx` be the fixed
local query with positive-definite output metric `W`. For positive-definite
conditional innovation covariance
`C = H_tilde P_H H_tilde.T + R_tilde`,

\[
\Delta_c=\operatorname{tr}\!\left[
 WJ P_H\widetilde H_c^T C^{-1}\widetilde H_c P_H J^T\right]\geq0.
\]

This is the expected reduction of squared query error when using the posterior
mean, conditional on the existing history and under the stated Gaussian model.
It is zero exactly when `J P_H H_tilde.T = 0`. It can be computed before reading
any candidate outcome. The supported-subspace implementation extends the
calculation to compatible, redundant singular observations.

**Proof.** Gaussian conditioning gives
`P_new = P_H - P_H H_tilde.T C^-1 H_tilde P_H`. The posterior mean's conditional
Bayes risk for the metric-squared query loss is `trace(W J P J.T)`. Subtracting
risks gives the displayed trace, equivalently the squared Frobenius norm of
`W^(1/2) J P_H H_tilde.T C^(-1/2)`. Positive definiteness proves the zero-gain
criterion. This is an application of standard Gaussian conditioning and
quantity-of-interest design, not a claim to have invented those identities.

### Corollary: source-replay invariance

If a candidate is a deterministic replay `y_c=T y_H`, with matching designs and
noise covariance blocks, both its conditional design and conditional noise are
zero. Its incremental information and query value are zero. The session returns
the identical `GaussianGaugeBelief` object, not a numerically reconstructed copy.
An inconsistent replay is rejected before session state is changed.

Information nullspaces and prior correlations remain distinct. A factor does
not add precision along its nullspace, but a correlated prior can legitimately
transmit information to that direction. The implementation does not incorrectly
force nullspace marginal variances to stay constant for every prior.

### Counterexample: standalone filtering and greedy guarantees

Consider `x ~ N(0,1)`, `y_A=x+b`, and `y_B=b+eta`, with independent
`b ~ N(0,1)` and `eta ~ N(0,0.1)`. Window B has no standalone state information.
Yet after A, it changes posterior variance from `1/2` to `1/12`: a gain of
`5/12`. It estimates the shared nuisance rather than directly measuring x.

Thus discarding every candidate with zero standalone query information is not
valid under dependence. Also, variance-reduction set utility is not generally
submodular: B's gain increases after A. The implemented selector is an exact
**one-step** choice among the supplied candidates, not a globally optimal
multi-window scheduler or a greedy approximation guarantee. This analytic
reference-channel control does not claim that a particular provider supplies
such a channel.

## API and integration

`CorrelatedGaugeDesign` holds window row partitions, common-chart design rows,
and the full externally specified joint noise covariance. `ConditionalGaugeSession`
owns the actually consumed history. `preview_query` has no candidate-outcome
argument. `select_query_window` ranks expected reduction per positive cost and
returns `None` if no candidate clears the frozen minimum.

For an existing `ObservableGaugeFactor` in **the same chart**, use
`factor.observable_basis.T` as the observation design,
`factor.observable_covariance` as its marginal noise covariance, and zero as its
factor-centered observation. The integration test verifies exact parity with
`fuse_local_gaussian` and `evaluate_query_observability`, then verifies that
replaying the same source evidence changes nothing. Cross-window covariance
blocks still need independent justification; diagonal blocks alone are not
sufficient. Factors from different linearization charts must first be transported
to the declared common chart; merely assigning the same text identifier is not a
coordinate transformation.

A session must begin at a prior that has not consumed any of its windows. Replay
its admitted history through the session rather than wrapping an already updated
BayesianPhysTwin belief as an untouched prior. Downstream BayesianPhysTwin still
owns complete physical-belief construction and its exact caller-owned fallback.
The identity guarantee here applies to the local gauge-belief object only.
Causal4D query Jacobians can define a future endpoint or contrast objective, but
this module does not execute Causal4D or establish causal benefit.

The kernel is dense in the compressed window rows. It is a small research
implementation, not a demonstrated large-stream sparse solver. Known chart,
state-independent Gaussian covariance, first-order query validity, provider
support, identity and lineage, calibration, physical consistency, and downstream
guards remain separate requirements. A positive utility does not authorize a
real update or a target evaluation.

## Reproducible controlled study

```bash
python -m pytest -q tests/test_conditional_gauge_design.py \
  tests/test_conditional_gauge_integration.py
python -m prob4d.conditional_gauge_study --output /tmp/conditional-query-result.json
```

The frozen numerical design is in
`protocols/conditional-query-design-study-v1.json`. The study generates 10,000
independent Gaussian episodes, a rank-six line-like history, a 0.999-correlated
repeat, complementary rank-six support, and a precise but weakly query-relevant
rotation measurement. Each policy receives one additional window at equal cost.
Selections are made before any episode is drawn. The point-position query uses
a fixed local Sim(3) Jacobian, not a nonlinear scene renderer or physical solver.

Five arms separate selection from covariance accounting: history only, marginal
query selection with independent updates, marginal selection with correct
updates, global gauge-variance selection with correct updates, and conditional
query selection with correct updates. Report Euclidean query RMSE, Gaussian NLL,
normalized query NEES, 90% ellipsoid coverage, harmful-episode fraction, and paired
episode-bootstrap squared-loss differences. An exact-replay sweep and the
reference-channel counterexample are analytic controls. An independently computed
dense joint posterior checks the conditional implementation.

Paper-facing generated results and exact source manifests belong in
`FlorianPfaff/BayesianPhysTwin-Paper`, not in the public release claim table.
This is a constructed mechanism study, not a fresh real-provider evaluation,
independent calibration, safety result, or proof of state of the art. Individual
episodes may become worse despite a positive expected value; those fractions
must be reported rather than suppressed.

## Relation to prior work and the larger paper

Partial-factor/localizability treatment already exists in registration and
SLAM. Quantity-of-interest optimal design and sensor selection with correlated
noise also predate this work. The proposed Prob4D contribution is the narrower
combination: partial learned-window gauge likelihoods, conditional source-evidence
accounting, decision-relevant query value, and replay/selection failure controls
at the physical-twin interface. A literature review and a fresh real-provider
study are still needed before claiming a broadly novel or empirically validated
system. This work does not reopen any terminal MotionCrafter, CUT3R, or Deform360
cohort and does not require collecting new physical data.

Primary references:

- Tuna et al., *X-ICP: Localizability-Aware LiDAR Registration for Robust
  Localization in Extreme Environments*, IEEE T-RO, 2024;
  arXiv:2211.16335, https://arxiv.org/abs/2211.16335.
- Attia, Alexanderian, and Saibaba, *Goal-Oriented Optimal Design of Experiments
  for Large-Scale Bayesian Linear Inverse Problems*, Inverse Problems, 2018;
  arXiv:1802.06517, https://arxiv.org/abs/1802.06517.
- Liu et al., *Sensor Selection for Estimation with Correlated Measurement
  Noise*, IEEE T-SP 64(13):3509--3522, 2016;
  DOI:10.1109/TSP.2016.2550005, https://arxiv.org/abs/1508.03690.
