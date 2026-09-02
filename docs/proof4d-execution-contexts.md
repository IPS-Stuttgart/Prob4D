# Proof4D caller-owned execution contexts

## Problem

Independent verification establishes that a certificate is internally valid and
that its declared consequence follows from its carried witness. That alone does
not prove that the certificate belongs to the **current execution**. A valid
certificate for yesterday's sensor input, a different query implementation, a
different action policy, or another fallback could otherwise be replayed.

Proof4D therefore separates two trust domains:

1. the provider produces a content-addressed physical certificate; and
2. the caller independently declares a content-addressed execution context.

The execution gate authorizes consumption only when both artifacts verify and
every caller binding equals the corresponding certificate field exactly.

## Context schema

A context uses schema `prob4d.proof4d-execution-context`, version 1:

```json
{
  "schema": "prob4d.proof4d-execution-context",
  "schema_version": 1,
  "certificate_kind": "...",
  "bindings": {
    "certificate_id": "sha256:..."
  },
  "context_id": "sha256:..."
}
```

`context_id` is the SHA-256 digest of the canonical JSON object after removing
the `context_id` field. Unknown, omitted, duplicate, malformed, or non-finite
fields fail closed.

## Required local-query bindings

For `observable-gauge-linear-query-v1`, the caller binds:

- the exact certificate ID;
- the sealed input digest;
- the source-factor digest;
- the query ID;
- the query-program digest; and
- the caller-owned fallback ID.

A certificate cannot therefore be reused for a different input, factor, query,
query implementation, or fallback without producing a context mismatch.

## Required global-action bindings

For `shared-axial-orbit-robust-advantage-v1`, the caller binds:

- the exact certificate ID and sealed input digest;
- the shared-gauge identity and external support-receipt digest;
- candidate and fallback action identities;
- both action-loss program digests;
- the complete fallback belief ID and digest; and
- the registered admission-policy digest.

A globally valid action certificate cannot therefore authorize another action,
loss implementation, ambiguity support, policy, input, or fallback.

## Execution

The certificate and context remain separate files:

```bash
python -m prob4d_independent_verifier.execution_gate \
  certificate.json \
  caller-context.json
```

Exit status has the same fail-closed semantics as the underlying verifiers:

| Exit | Decision | Meaning |
|---:|---|---|
| 0 | `verified-admit` | Certificate verifies, its consequence is admitted, and every caller binding matches. |
| 2 | `verified-reject` | Certificate and context match, but the declared consequence is unsupported. |
| 3 | `invalid-fail-closed` | Certificate, context, or equality binding is invalid. |

The gate reports the underlying certificate decision separately from the final
execution decision. A context mismatch is always an invalid execution, never a
valid rejection that downstream code might accidentally reinterpret.

Checked examples are available at:

```text
examples/proof4d/linear-query-supported.context.json
examples/proof4d/axial-action-supported.context.json
```

## Security boundary

Caller binding closes **cross-context replay**, but it is not authentication.
The current version does not prove who created the context, that the declared
input digest corresponds to truthful sensor data, or that the caller did not
intentionally reuse an old context. Deployment-grade use additionally requires:

- trusted signing identities or an authenticated transport;
- a caller-generated nonce, monotonic counter, or transaction identity;
- protected construction of the expected context from the actual execution;
- audit retention linking the context, certificate, verifier report, and action;
- the existing exact-fallback behavior for any missing or failed binding.

Those operational controls are deliberately not hidden inside the mathematical
certificate. Proof4D reports exactly which part is algebraically verified and
which part remains an external trust obligation.
