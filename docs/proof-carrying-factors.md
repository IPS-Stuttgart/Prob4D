# Proof-carrying physical factors

## Status

This document specifies the first executable slice of **Proof4D**: portable,
content-addressed certificates that authorize only explicitly declared physical
consequences.

The implementation now contains two complementary certificate families:

1. **Local linear-query support.** A factor may be consumed for one declared
   first-order query only when that query is sufficiently insensitive to the
   factor nullspace.
2. **Global axial-orbit action advantage.** A candidate action may replace the
   caller-owned fallback only when it has uniformly positive robust advantage
   over every angle in one declared continuous axial-rotation ambiguity arc.

Both families have independent fail-closed verifiers. The global action
certificate is not a sampled-orbit approximation: its verifier analytically
checks extrema over the complete declared angle arc, including interior
stationary points.

This remains narrower than the full Proof4D research program. It does not yet
establish arbitrary-group support, sequential composition, active sensing, or a
target-distribution harm guarantee.

## Motivation

A numerical factor normally carries an estimate and covariance. Downstream code
must trust that the factor contains enough information for whatever query or
action it performs. That trust is unsafe for rank-deficient factors: a factor
may support one physical query while leaving another unresolved. A local test
can also be insufficient when a nonlinear query has zero derivative at one
representative but varies elsewhere on the unresolved orbit.

Proof4D changes the interface. A provider may propose a result, but a small
independent verifier determines whether the exact declared consequence follows
from the carried witness. Malformed evidence, unsupported consequences, and
failed model scope all return the exact caller-owned fallback.

# Certificate family 1: local linear-query support

## Carried witness

The local certificate binds:

- observable and nullspace bases;
- observable information and reconstructed full information matrix;
- the declared coordinate chart;
- a query identifier and digest of the query program;
- the local query Jacobian and output metric;
- observable query coordinates as a row-space witness;
- a source-frozen maximum nullspace-sensitivity threshold;
- the producer admission decision;
- the caller-owned fallback identifier;
- provider, input, source-factor, assumption, and calibration provenance; and
- a content-derived certificate ID.

## Verified statement

Let \(U\) and \(N\) be orthonormal bases of the factor's observable and
unobservable subspaces. Let \(B\) be the declared local query Jacobian and
\(W\succ0\) its output metric. The producer supplies

\[
C = W^{1/2} B U
\]

as a witness. The independent verifier checks

\[
\Lambda \approx U H U^\top,\qquad
\Lambda N \approx 0,\qquad
C \approx W^{1/2}BU
\]

and recomputes the worst-case relative local reconstruction residual

\[
\eta_{\max}
=
\frac{
  \left\|W^{1/2}B-CU^\top\right\|_2
}{
  \max\!\left(\left\|W^{1/2}B\right\|_2,\operatorname{tiny}_{64}\right)
}.
\]

Because \([U\;N]\) is orthonormal, this equals the relative spectral norm of the
query component acting through the declared nullspace. The verifier admits the
factor for the query only when

\[
\eta_{\max}\le \tau_q,
\]

where \(\tau_q\) is carried in the certificate and must be frozen outside the
target case.

For \(\tau_q=0\) in exact arithmetic, this is the local identifiability
condition \(BN=0\), equivalently
\(\ker(\Lambda)\subseteq\ker(B)\).

## Local usage

```python
from prob4d.proof_carrying_factor import (
    build_observable_gauge_query_certificate,
    write_proof_carrying_factor,
)

certificate = build_observable_gauge_query_certificate(
    factor,
    query_jacobian_local=query_jacobian,
    query_id="rope:endpoint-position-v1",
    query_program_digest="sha256:...",
    fallback_id="belief:physical-prior:42",
    input_digest="sha256:...",
    maximum_relative_nullspace_sensitivity=1e-6,
)
write_proof_carrying_factor("certificate.json", certificate)
```

Verify without importing `prob4d`:

```bash
python -m prob4d_independent_verifier.proof_carrying certificate.json
```

# Certificate family 2: global axial-orbit action advantage

## Why a second family is necessary

The local condition certifies a first-order query at one chart point. It cannot
by itself prove that an action remains preferable across a finite physical
ambiguity. For one unresolved shared axial rotation, affine point queries and
action losses have the harmonic form

\[
L_a(\theta)=c_a+u_a\cos\theta+v_a\sin\theta.
\]

A zero derivative at one representative does not imply constant loss over the
orbit. The global certificate therefore compares the candidate and fallback on
the complete declared arc rather than sampling a few angles or trusting a local
Jacobian.

## Verified statement

Let

\[
\Delta(\theta)
= L_{\mathrm{fb}}(\theta)-L_{\mathrm{cand}}(\theta)
= c+u\cos\theta+v\sin\theta
\]

be fallback-minus-candidate advantage on the shared ambiguity arc
\(\mathcal A\). Let \(e\ge0\) be a declared uniform error bound on this
advantage, \(m\ge0\) the required action margin, and \(s\ge0\) an explicit
numerical slack. The verifier analytically recomputes

\[
\underline\Delta
=
\min_{\theta\in\mathcal A}\Delta(\theta)-e,
\qquad
\overline\Delta
=
\max_{\theta\in\mathcal A}\Delta(\theta)+e.
\]

It evaluates both arc endpoints and every contained stationary point. The
candidate is admitted exactly when

\[
\text{scope admitted},\qquad
\mathcal A\ne\varnothing,\qquad
\underline\Delta > m+s.
\]

Thus an admitted certificate proves a **global robust action preference over
every state in the declared continuous orbit arc**, conditional on the supplied
orbit model and uniform error bound.

## Additional bindings

The action certificate binds more than the algebraic coefficients. It includes:

- the shared gauge identity, axis, origin, and complete angle arc;
- a digest of the external support receipt required for positive scope;
- candidate and fallback action identities;
- digests of both action-loss programs;
- the complete fallback belief identifier and digest;
- the registered admission-policy digest;
- the sealed input digest, assumptions, producer identity, and optional
  calibration receipt;
- the harmonic difference witness, nominal extrema, robust extrema, producer
  decision, reason codes, and content ID.

The verifier recomputes all extrema and reason codes. Merely resealing a false
admission, altering the orbit, changing a loss coefficient, widening claim
scope, removing the support receipt, or modifying a reported bound fails
closed.

## Global action usage

```python
from prob4d.proof_carrying_orbit import (
    build_axial_orbit_action_certificate,
    write_proof_carrying_orbit,
)

certificate = build_axial_orbit_action_certificate(
    fallback_loss=fallback_loss,
    candidate_loss=candidate_loss,
    scope_admitted=True,
    support_receipt_digest="sha256:...",
    fallback_action_id="action:physical-fallback-v1",
    candidate_action_id="action:low-force-grasp-v1",
    fallback_loss_program_digest="sha256:...",
    candidate_loss_program_digest="sha256:...",
    fallback_id="belief:complete-physical-fallback-v1",
    fallback_digest="sha256:...",
    input_digest="sha256:...",
    admission_policy_digest="sha256:...",
    advantage_error_bound=0.02,
    required_margin=0.05,
)
write_proof_carrying_orbit("action-certificate.json", certificate)
```

Verify independently:

```bash
python -m prob4d_independent_verifier.orbit_advantage action-certificate.json
```

Checked examples are available at:

```text
examples/proof4d/linear-query-supported.json
examples/proof4d/linear-query-rejected.json
examples/proof4d/axial-action-supported.json
examples/proof4d/axial-action-rejected.json
```

# Fail-closed outcomes

Both verifier families use the same execution semantics:

| Result | Meaning | Execution |
|---|---|---|
| `verified-admit` | The artifact is internally valid and its declared consequence passes. | Consume only that declared query or candidate action, subject to its assumptions. |
| `verified-reject` | The artifact is internally valid, but the consequence is unsupported. | Return the exact caller-owned fallback. |
| `invalid-fail-closed` | Content identity, schema, witness, decision, scope, or provenance is inconsistent. | Return the exact caller-owned fallback and record the verifier reason. |

Module commands use exit status 0, 2, and 3 for these outcomes, respectively.
The package intentionally keeps Prob4D's single installed `prob4d` console
entry point; independent verification is invoked with `python -m`.

# Threat model

The independent verifiers distrust the producing process and reject:

- content changes without a matching content ID;
- duplicate JSON keys and non-finite numbers;
- widened claim scope;
- malformed subspaces or orbit geometry;
- information outside a declared observable subspace;
- mismatched local row-space or global harmonic witnesses;
- false producer admission or inconsistent reason codes;
- altered analytic extrema or bounded-error margins;
- an admitted orbit scope without an external support receipt digest;
- a non-exact fallback policy; and
- malformed provenance identifiers or digests.

The verifiers intentionally do **not** establish that:

- sensor data are truthful or complete;
- a learned provider is competent on the current object;
- declared assumptions or external support receipts are true;
- supplied program digests denote the intended query or loss implementation;
- the local Jacobian captures a nonlinear query globally;
- a declared axial orbit exhausts the physical compatibility set;
- factor covariance or the advantage error bound is calibrated;
- the target episode is exchangeable with calibration data; or
- executing an admitted query or action is deployment-safe.

Those facts require additional certificate families rather than a broader
interpretation of the current ones.

# Relation to existing Prob4D diagnostics

`prob4d.query_observability` remains the richer local diagnostic layer. It
reports direct support separately from prior-mediated variance reduction and
downstream posterior contraction. `prob4d.axial_query_certificate` supplies the
analytic continuous-orbit model used by the global action producer.

The portable proof artifacts are intentionally more structural. Prior
information may bound an unresolved query, but it is not misreported as evidence
supplied by a factor. Likewise, the orbit support receipt is carried explicitly:
the action verifier proves the consequence conditional on that support instead
of pretending to prove support exhaustiveness.

# Research roadmap

The next certificate families should be added without weakening either current
family:

1. **General finite/group-orbit certificate.** Carry arbitrary registered group
   actions or finite compatibility supports with independently reproducible
   query images.
2. **Finite-decision certificate.** Carry hypothesis classes, class masses,
   bounded losses, and classwise maximizers proving a worst-case regret bound.
3. **Compositional certificate calculus.** Define sound rules for joining,
   marginalizing, transporting, and invalidating certificates across a factor
   graph.
4. **Sequential task-belief certificate.** Prove that a recursive quotient state
   preserves a registered family of future queries and actions.
5. **Active measurement certificate.** Prove that an information-gathering
   action splits unresolved decision classes at lower cost than fallback.
6. **Target-valid harm certificate.** Add trajectory-blocked finite-sample
   bounds on harmful departures from fallback.
7. **Authenticity and execution-context layer.** Bind certificates to trusted
   signing identities and caller-supplied execution contexts, preventing valid
   artifacts from being replayed for a different task.

The scientific objective is not to make every world model correct. It is to
ensure that every consumed physical consequence is paired with an explicit,
independently checkable statement of what the evidence actually identifies.
