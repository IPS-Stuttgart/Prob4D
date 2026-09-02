# Continuous axial-symmetry certificates with calibrated support

Status: **experimental mechanism and real-trajectory evaluation**.

This extension separates three questions that the finite-orbit work deliberately
left open:

1. Can the registered axial ambiguity be optimized over the complete continuous
   group rather than a finite angle roster?
2. Can the size of the admitted ambiguity support be calibrated from real
   source trajectories rather than supplied as a controlled hidden angle?
3. Can harmful accepted updates be controlled directly rather than inferred
   from compatibility-set coverage?

The answers are conditional. The geometry below is exact for a declared
`SO(2)` axial orbit. The support statement is split-conformal marginal coverage
for an exchangeable recording-level score. The direct utility primitive in
`group_risk_control.py` instead controls the expected bounded loss of the next
exchangeable group for a fixed nested policy family. Neither statement is
conditional coverage, acceptance-conditional risk, provider competence, or
deployment safety.

## Exact continuous vector-query diameter

For a vector first-harmonic query

\[
q(\theta)=c+a\cos\theta+b\sin\theta ,
\]

let \(M=[a\ b]\). For any linear weighting matrix \(W\),

\[
\operatorname{diam}_{W}\bigl(q(SO(2))\bigr)
=
\sup_{\theta,\phi}
\|W(q(\theta)-q(\phi))\|_2
=
2\sigma_{\max}(WM).
\]

The upper bound follows because the difference between two unit-circle points
has norm at most two. Equality is attained by antipodal points in the right
singular direction of \(WM\). For a rotating 3-D point this gives twice the
largest weighted orbit radius without discretizing the angle.

If the query is additionally perturbed by an arbitrary Euclidean ball of radius
\(r_q\), the generic Minkowski/Lipschitz calculation gives the certified bound

\[
\operatorname{diam}_{W}(q(SO(2))\oplus B_{r_q})
\le 2\sigma_{\max}(WM)+2\|W\|_2r_q.
\]

The first term is exact. The additive expression is an exact worst-case bound
for the declared unstructured ball, but it need not equal the actual diameter
of a more structured remainder set. It must not be described as an exact orbit
diameter when \(r_q>0\). The implementation rejects when the declared model
scope is not admitted or when this certified diameter bound exceeds the
registered tolerance.

## Group-conformal axial tube

A causal predictor supplies a representative point \(\widehat p\) on an axial
orbit. For real point \(p\), let \(\widehat\theta\) be its closest continuous
orbit angle and let \(d\) be the remaining Euclidean distance to that orbit.
With radial scale \(r>0\) and a predeclared angular scale \(\theta_0>0\), use

\[
s(p)=\max\left\{
\frac{|\widehat\theta|}{\theta_0},
\frac{d}{r}
\right\}.
\]

One complete recording contributes one score, here the conservative empirical
90th percentile of its nested case scores. Given \(n\) exchangeable
calibration-group scores, the finite threshold is their

\[
k=\left\lceil(n+1)(1-\alpha)\right\rceil
\]

order statistic. When \(k>n\), the implementation returns no finite bound and
the caller must reject. Otherwise, the next exchangeable recording score is
below the threshold with marginal probability at least \(1-\alpha\).

For calibrated threshold \(\widehat q\), a case receives the continuous support

\[
|\theta|\leq \min(\pi,\widehat q\theta_0),
\qquad
\|e\|_2\leq \widehat q r .
\]

The guarantee concerns the registered recording score. It does not imply
conditional coverage for a selected subgroup, selective coverage after
admission, or future support completeness under distribution shift.

## Exact query and action bounds

A scalar affine query remains a first harmonic over the angle arc. Its exact
continuous-orbit extrema are computed analytically. Enlarging those extrema by
\(L\widehat q r\) is a certified Lipschitz enclosure for the additive tube; the
result is not asserted to be the minimum possible enclosure for a structured
remainder.

For an orbiting point \(x(\theta)\) and fixed target \(y\),
\(\|x(\theta)-y\|_2^2\) is also a first harmonic. Comparing candidate point
\(c\) and fallback point \(f\) under the same latent angle gives

\[
D(x)=\|x-f\|_2^2-\|x-c\|_2^2 .
\]

If \(x=x(\theta)+e\), the orbit-model error in this advantage is exactly linear
in \(e\) and satisfies

\[
|D(x)-D(x(\theta))|
\leq 2\|c-f\|_2\|e\|_2.
\]

This supplies a deterministic error envelope for the calibrated tube. The
candidate is admitted only when the minimum continuous-arc advantage minus this
envelope and the registered decision margin is strictly positive. Rejection
returns the caller-owned complete fallback.

## Direct group-level risk control

Compatibility support and selective utility are not the same property. For a
fixed policy family indexed from least to most conservative, let
\(L_g(\lambda_j)\in[0,B]\) be the complete-recording loss and require that it be
nonincreasing with conservatism for every calibration recording. Define

\[
\widehat R_n^+(\lambda_j)
=
\frac{n}{n+1}\widehat R_n(\lambda_j)+\frac{B}{n+1}.
\]

`select_group_conformal_risk_control` selects the least conservative candidate
whose corrected empirical risk is at most \(\alpha\). Under exchangeability of
the calibration recordings and the next recording, and provided that the
model, fallback, score, candidate grid, loss, and grouping were all frozen
before calibration, this controls expected next-recording loss at level
\(\alpha\).

For selective belief revision, a useful bounded loss is the fraction of cases
in a recording for which the candidate is both accepted and worse than the
registered fallback. A nested robust-advantage margin makes that loss
nonincreasing because accepted sets can only shrink.

The correction has a visible finite-sample floor \(B/(n+1)\). With 12 risk
calibration recordings and \(B=1\), no level below \(1/13\) can be certified,
even after observing zero calibration loss. This is fail-closed behavior, not a
numerical defect.

## Executed cloth-only Tracking Cloth study

The completed workflow used the public Tracking Cloth Deformation collection on
the `gpuserver4090` runner. It admitted 80 cloth-only recordings identified by
explicit Motive layouts:

- A2: 20 markers;
- A3: 12 markers.

At each case the current axis was observed and one probe was hidden. A
transported constant-angular-velocity construction based on earlier frames
supplied the representative. Five recording-disjoint folds were evaluated
separately by cloth size at horizons 3, 6, 12, and 24 frames.

The support-calibration result was positive, but the preregistered utility result
was negative. The local predictor was already almost never harmful, while the
axis-center fallback was much weaker. At 24 frames, valid rejections therefore
increased policy RMSE. This establishes an important boundary:

```text
valid calibrated compatibility support does not imply useful abstention.
```

A competent fallback or direct calibration of decision loss is separately
required.

## Prospective augmented-layout reserve

The first study parsed full trajectories only for the 80 cloth-only layouts.
A later fail-closed header audit inspected metadata and Motive headers for the
remaining 40 layouts without parsing or hashing any marker trajectory values.
It found:

- four labelled A2 Hitting recordings suitable for source-only method
  development;
- 36 A2 Self-collision recordings with 22 unlabeled markers each;
- a balanced four-material, three-condition, three-repetition structure.

The intended prospective order is therefore:

1. develop and freeze an identity-free physical-axis rule using only the four
   labelled Hitting recordings;
2. use Self-collision `rep1` recordings for compatibility-support calibration;
3. use `rep2` recordings for the nested group-risk policy calibration;
4. evaluate the frozen policy exactly once on untouched `rep3` recordings.

There are 12 recordings in each self-collision stage: one for every material and
condition combination. The three repetitions are not interchangeable after any
trajectory values are opened. No `rep3` trajectory may inform a model,
threshold, score, fallback, candidate grid, or criterion.

## Claim boundary

The trajectories and residuals in the completed cloth-only experiment are real.
The missing-probe condition is simulated, and its representative is a fixed
kinematic predictor rather than a learned visual provider. The augmented reserve
can provide a prospective physical-axis and decision-risk test, but it still
cannot establish learned-provider competence, unique physical-state recovery,
BayesianPhysTwin or Causal4D benefit, deployment safety, or state of the art.
