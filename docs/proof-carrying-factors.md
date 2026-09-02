# Proof-carrying physical factors

## Status

This document specifies the first executable slice of **Proof4D**: a portable,
content-addressed certificate proving that one declared local linear query is
supported by the observable subspace of one partial physical factor.

The implementation is deliberately narrower than the full Proof4D research
program. It establishes the certificate envelope, independent verifier,
fail-closed execution semantics, and a binding to Prob4D's existing
`ObservableGaugeFactor`. It does not yet certify finite ambiguity orbits,
decision loss, sequential composition, active sensing, or target-distribution
risk.

## Motivation

A numerical factor normally carries an estimate and covariance. Downstream code
must trust that the factor has enough information for whatever query or action
it performs. That trust is unsafe for rank-deficient factors: a factor may
support one physical query while leaving another query unresolved.

A proof-carrying factor adds a machine-checkable witness. Version 1 carries:

- the observable and nullspace bases;
- the observable information and reconstructed full information matrix;
- the declared coordinate chart;
- a query identifier and digest of the query program;
- the local query Jacobian and output metric;
- observable query coordinates as a row-space witness;
- a source-frozen maximum nullspace-sensitivity threshold;
- the producer's admission decision;
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

## Three outcomes

The verifier never converts malformed input into permission:

| Result | Meaning | Execution |
|---|---|---|
| `verified-admit` | The certificate is internally valid and the declared local query passes. | The caller may consume the factor for that query, subject to the declared assumptions. |
| `verified-reject` | The certificate is internally valid, but the query depends too strongly on the nullspace. | Return the exact caller-owned fallback. |
| `invalid-fail-closed` | The content ID, schema, subspaces, information witness, query witness, decision, or provenance is inconsistent. | Return the exact caller-owned fallback and record the verifier reason. |

The command-line verifier uses exit status 0, 2, and 3 for these outcomes,
respectively.

## Usage

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
proof4d-verify certificate.json
```

The checked examples are:

```bash
proof4d-verify examples/proof4d/linear-query-supported.json
proof4d-verify examples/proof4d/linear-query-rejected.json
```

The second command intentionally exits with status 2.

## Threat model

The independent verifier distrusts the producing process and rejects:

- content changes without a matching content ID;
- duplicate JSON keys and non-finite numbers;
- widened claim scope;
- malformed or non-orthonormal subspace bases;
- information outside the declared observable subspace;
- a mismatched query row-space witness;
- false producer admission;
- a non-exact fallback policy; and
- malformed provenance identifiers or digests.

The verifier intentionally does **not** establish that:

- sensor data are truthful or complete;
- a learned provider is competent on the current object;
- the declared assumptions are true;
- the supplied query program digest denotes the intended query;
- the local Jacobian captures a nonlinear query globally;
- a finite ambiguity support is exhaustive;
- the factor covariance is calibrated;
- the target episode is exchangeable with calibration data; or
- executing the admitted query is deployment-safe.

Those facts require additional certificate families rather than a broader
interpretation of version 1.

## Relation to existing Prob4D diagnostics

`prob4d.query_observability` remains the richer diagnostic layer. It reports
direct support separately from prior-mediated variance reduction and downstream
posterior contraction.

The proof-carrying certificate is intentionally more structural. Its admission
decision is based only on the declared query's worst-case local nullspace
component. Prior information may bound an unresolved query, but it is not
misreported as evidence supplied by the factor.

## Research roadmap

The next certificate families should be added without weakening version 1:

1. **Finite-orbit certificate.** Carry a registered set or group action,
   evaluator-independent query values, and an orbit-diameter witness.
2. **Finite-decision certificate.** Carry hypothesis classes, class masses,
   bounded losses, and classwise maximizers proving a worst-case regret bound.
3. **Compositional certificate calculus.** Define sound rules for joining,
   marginalizing, transporting, and invalidating certificates across a factor
   graph.
4. **Sequential task-belief certificate.** Prove that a recursive quotient state
   preserves a registered family of future queries and actions.
5. **Active measurement certificate.** Prove that an information-gathering
   action splits the unresolved decision classes at lower cost than fallback.
6. **Target-valid harm certificate.** Add trajectory-blocked finite-sample
   bounds on harmful departures from fallback.

The scientific objective is not to make every world model correct. It is to
ensure that every consumed physical consequence is paired with an explicit,
independently checkable statement of what the evidence actually identifies.
