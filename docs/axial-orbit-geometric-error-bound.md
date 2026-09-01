# From geometric uncertainty to an axial orbit certificate

Status: **conservative deterministic error propagation**.

`robust_orbit_query.certify_axial_linear_query` requires a bound

\[
\|B-\widehat B\|_2\leq\eta
\]

on the two-column coefficient matrix of the physical query orbit. This module
provides a sufficient value of \(\eta\) from separately supplied Euclidean
bounds on the estimated unit axis, pivot, point positions, and linear query
map. It does not estimate or calibrate those geometric bounds.

## Geometry

Let \(u\) be the true unit axis, \(c\) the true pivot, and \(p_i\) one point.
With \(r_i=p_i-c\), the axial orbit coefficients are

\[
a_i=(I-uu^\top)r_i,
\qquad
b_i=u\times a_i.
\]

For estimated quantities, suppose

\[
\|p_i-\widehat p_i\|_2\leq\epsilon_i,
\quad
\|c-\widehat c\|_2\leq\epsilon_c,
\quad
\|u-\widehat u\|_2\leq\epsilon_u.
\]

Define \(\delta_i=\epsilon_i+\epsilon_c\). For unit axes,

\[
\|uu^\top-\widehat u\widehat u^\top\|_2
\leq 2\epsilon_u.
\]

Using the estimated offset \(\widehat r_i=\widehat p_i-\widehat c\), a sufficient
cosine-coefficient bound is

\[
\|a_i-\widehat a_i\|_2
\leq
\delta_i+ho\|\widehat r_i\|_2,
\qquad
\rho=\min(2,2\epsilon_u).
\]

Writing this bound as \(\alpha_i\), the true cosine coefficient satisfies
\(\|a_i\|\leq\|\widehat a_i\|+\alpha_i\). Hence

\[
\|b_i-\widehat b_i\|_2
\leq
\alpha_i+\epsilon_u(\|\widehat a_i\|_2+\alpha_i).
\]

Stack the pointwise cosine and sine errors into a matrix \(E\in\mathbb R^{3N\times2}\).
The per-point bounds imply

\[
\|E\|_2\leq\|E\|_F
\leq
\sqrt{\sum_i\alpha_i^2+\sum_i\beta_i^2},
\]

where \(\beta_i\) is the sine-coefficient bound above. For a linear query
\(q=Wp\), the projected coefficient error is therefore bounded by

\[
\|WE\|_2
\leq
\|W\|_2
\sqrt{\sum_i\alpha_i^2+\sum_i\beta_i^2}
=: \eta_{\rm geom}.
\]

`bound_axial_query_coefficient_error` returns this value together with every
intermediate bound. It can be passed directly to
`certify_axial_linear_query`.

## Interpretation

The bound is deliberately conservative. Its role is to preserve the safety
semantics of the query gate:

- a small validated geometry budget can certify useful invariant queries;
- a clearly variant query can be certified non-identifiable;
- a loose budget yields an undetermined decision and exact fallback;
- a loose budget must not be hidden by replacing it with a nominal orbit.

Conservatism affects acceptance and fallback frequency, not the guarantee that
a certified invariant query lies below the declared diameter tolerance.

The axis error is the Euclidean distance between the true and estimated unit
axis vectors. Axis sign is part of the angular convention: if the sign is
unresolved, the caller must canonicalize the axis and angular law consistently
or use a bound that covers the ambiguity. The point, pivot, and axis bounds must
hold simultaneously for the case being certified.

## Validation

The focused test suite includes 400 random multi-point, multi-query adversarial
instances. In every instance it constructs a true axis, pivot, and point cloud
inside the supplied bounds and verifies that the actual spectral-norm error of
the projected coefficient matrix does not exceed the returned value. It also
checks independent Rodrigues propagation, similarity-frame equivariance,
zero-error exactness, immutable outputs, and malformed contracts.

The permanent CI composes the geometric budget with the three-way orbit
certificate on an off-axis vector query. The resulting query is certified
variant without reading a physical outcome.

## Claim boundary

This is a deterministic implication from supplied geometric bounds. It is not
a learned uncertainty estimator, a confidence interval obtained from repeated
data, or evidence that any particular provider satisfies the bounds. A future
learned-provider experiment must estimate and validate the point, axis, and
pivot budgets on source groups before applying the certificate to held-out
queries. The recording-disjoint Tracking Cloth experiment remains a controlled
hidden-gauge real-trajectory result.