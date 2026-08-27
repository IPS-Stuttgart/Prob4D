# Observable-subspace Sim(3) gauge factors

Status: **experimental scientific kernel**. This path is not admitted to the
claim-bearing provider-v2 export.

## Motivation

Prob4D currently estimates a full seven-dimensional relative Sim(3) gauge from
an overlap and rejects the alignment when its Gauss--Newton information has rank
below seven. That is a sound fail-closed policy for a covariance-only interface,
but it discards useful information in geometries that are only *partially*
observable.

A deformable linear object is the canonical case. Corresponding centerline
points determine scale, centroid translation, and two rotational directions, but
they do not determine twist around the transformed centerline. Rejecting the
whole overlap loses six valid directions. Adding a numerical ridge has the
opposite failure: it turns an arbitrary representative twist into fabricated
measurement information.

The experimental kernel in `prob4d.observable_gauge` instead returns a
rank-deficient Gaussian information factor. A full-rank prior supplies the
missing direction, so the posterior remains complete without pretending that
the visual overlap measured twist.

## Origin-invariant local chart

Let the fitted transform be \(\widehat T\), the weighted source centroid be
\(\bar x\), the transformed centroid be \(\bar y=\widehat T(\bar x)\), and the
weighted RMS radius be \(\rho\). The local coordinate is

\[
\boldsymbol\zeta =
[\delta\ell,\;\delta\boldsymbol\phi^\top,\;
 \delta\boldsymbol\tau^\top]^\top,
\]

where log-scale and left rotation act around \(\bar y\), and
\(\delta\boldsymbol\tau\) is centroid translation normalized by \(\rho\). For a
fitted centered point \(\boldsymbol q_i=\widehat T(\boldsymbol x_i)-\bar y\), the
first-order point Jacobian is

\[
\boldsymbol J_i =
\begin{bmatrix}
\boldsymbol q_i & -[\boldsymbol q_i]_\times & \rho\boldsymbol I_3
\end{bmatrix}.
\]

Centering removes dependence on the arbitrary coordinate origin. Normalizing
centroid translation by \(\rho\) also avoids comparing raw translation units to
log-scale and rotation units when the observability spectrum is thresholded.
The weighted geometry information is

\[
\boldsymbol H = \sum_i w_i\boldsymbol J_i^\top\boldsymbol J_i.
\]

With eigendecomposition
\(\boldsymbol H=\boldsymbol U\operatorname{diag}(\lambda_i)\boldsymbol U^\top\),
only directions satisfying
\(\lambda_i/\lambda_{\max}\geq\epsilon_{\mathrm{obs}}\) are retained. The IID
factor information is

\[
\boldsymbol\Lambda_{\mathrm{obs}} =
\boldsymbol U_r
\operatorname{diag}(\lambda_1/\widehat\sigma^2,\ldots,
                    \lambda_r/\widehat\sigma^2)
\boldsymbol U_r^\top.
\]

The module also supports a cluster-robust sandwich covariance restricted to the
same observable subspace.

## Fusion with a complete prior

For a full-rank local prior
\(\mathcal N(\boldsymbol\mu^-,\boldsymbol P^-)\), the factor is centered at the
fitted chart origin and gives

\[
\boldsymbol P^+ =
\left[(\boldsymbol P^-)^{-1}+\boldsymbol\Lambda_{\mathrm{obs}}\right]^{-1},
\qquad
\boldsymbol\mu^+ =
\boldsymbol P^+(\boldsymbol P^-)^{-1}\boldsymbol\mu^-.
\]

A nullspace direction receives no direct information. It can still change when
the prior correlates it with observed directions, which is the correct Bayesian
behavior; an isotropic prior leaves it exactly unchanged. Existing Prob4D gauge
priors in standard `Sim3.as_vector()` coordinates can be transported through
`CentroidGaugeChart.transport_vector_gaussian` and fused with
`ObservableGaugeFactor.fuse_vector_gaussian`.

## Frozen controlled result

The checked-in study uses 1,000 independent DLO-like trials with 48 collinear
correspondences, 10 mm point noise, and a 0.6 rad true twist that cannot be seen
from the centerline. Three methods are compared:

1. exact physical-prior fallback, representing the current rank-rejection path;
2. the rank-six observable-subspace factor; and
3. an intentionally invalid full-rank control that assigns the missing direction
   the weakest observable precision.

| Method | Centerline RMSE | Probe RMSE | 90% coverage | Normalized NEES | Harmful trials |
|---|---:|---:|---:|---:|---:|
| Exact prior fallback | 94.246 mm | 100.475 mm | 90.7% | 0.980 | -- |
| Observable subspace | **1.955 mm** | **17.764 mm** | 88.0% | 1.028 | **0.0%** |
| Full-rank completion control | 1.956 mm | 93.843 mm | 0.0% | 10611.347 | 43.7% |

The observable factor improves centerline RMSE by 97.93% and off-axis probe RMSE
by 82.32% relative to fallback while retaining calibrated full-state uncertainty.
The full-rank completion looks equally good on the observed centerline but is
catastrophically overconfident and harmful on 437 of 1,000 off-axis trials. This
is precisely the distinction that point-only reconstruction metrics miss.

Reproduce the result with

```bash
PYTHONPATH=src python -m prob4d.observable_gauge_study \
  --output evidence/observable-gauge-control-v1/result.json
```

## Relationship to prior work

Localizability-aware ICP methods such as X-ICP and LP-ICP already diagnose weak
SE(3) directions and constrain or preserve their updates. ICP covariance work
also shows that registration uncertainty cannot be reduced safely to a generic
closed-form covariance. The paper contribution should therefore not be framed
as the discovery of geometric degeneracy.

The defensible extension is narrower and specific to this stack:

- a centroid-normalized **Sim(3)** information factor rather than a hard pose
  update;
- exact retention of a rank-deficient gauge likelihood from learned overlapping
  4D windows;
- fusion with a complete correlated gauge prior rather than arbitrary damping;
- propagation toward point covariance and a guarded Bayesian physical-twin
  candidate; and
- decision-relevant evaluation that exposes harm outside the observed support.

Relevant context includes:

- T. Tuna et al., *X-ICP: Localizability-Aware LiDAR Registration for Robust
  Localization in Extreme Environments*, 2022, arXiv:2211.16335.
- H. Yue et al., *LP-ICP: General Localizability-Aware Point Cloud Registration
  for Robust Localization in Extreme Unstructured Environments*, 2025,
  arXiv:2501.02580.
- D. Landry et al., *CELLO-3D: Estimating the Covariance of ICP in the Real
  World*, 2018, arXiv:1810.01470.
- T. Ding et al., *LASER: Layer-wise Scale Alignment for Training-Free Streaming
  4D Reconstruction*, CVPR 2026.
- M. Kim et al., *GP-4DGS: Probabilistic 4D Gaussian Splatting from Monocular
  Video via Variational Gaussian Processes*, CVPR 2026.

## Claim boundary and promotion path

The current result is controlled mechanism evidence only. It does **not** show
real-provider competence, target calibration, BayesianPhysTwin benefit, or
Causal4D benefit. The existing claim-bearing provider-v2 path remains unchanged.

A paper-bearing promotion requires a prospective protocol with these stages:

1. freeze the observability threshold on source/calibration sequences;
2. audit the rank spectrum on a fresh real provider without opening target
   outcomes;
3. transport the existing complete gauge prior into the centroid chart and form
   complete posterior candidates;
4. evaluate point means, full covariance, off-support physical queries, accepted
   update harm, and exact fallback on held-out targets; and
5. expose the posterior to Causal4D only after the BayesianPhysTwin guard passes.

This should be developed as a Prob4D-focused companion contribution. The bounded
BayesianPhysTwin paper should absorb it only after a fresh real-provider result
shows a material gain; otherwise the current paper remains cleaner and stronger
without it.
