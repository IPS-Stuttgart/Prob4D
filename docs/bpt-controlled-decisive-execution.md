# Controlled Prob4D-to-BayesianPhysTwin execution

This repository hosts the authoritative execution boundary for the frozen
`prob4d-bpt-controlled-decisive-v1` study. The scientific protocol and estimator
implementation remain owned by BayesianPhysTwin pull request 109; this workflow
changes only how the two private exact source revisions reach `workstation2`.

## Why the execution boundary is here

The original BayesianPhysTwin workflow could verify its own source but its
repository-scoped token no longer had read access to the transferred private
`IPS-Stuttgart/Prob4D` repository. It therefore failed before running any
calibration or target group. Reversing an ordinary cross-repository checkout
would merely exchange that problem for private BayesianPhysTwin access.

The replacement workflow uses two independently verified transports:

1. Prob4D checks out its own frozen producer revision
   `aa8ffc6541011d044561e09870569a14ab3f586f` through the current repository
   token.
2. The registered self-hosted runner resolves BayesianPhysTwin revision
   `76d4aba20dd386e1f8583e501781d702d7937566` from its local Git object store,
   which is populated by the BayesianPhysTwin PR workflow. A bounded SSH clone
   is permitted only when the runner already has a read-only identity. The job
   fails if the exact object cannot be resolved.

Both sources are checked out detached, required to be clean, verified with Git,
and installed into a fresh virtual environment. The BayesianPhysTwin revision
must descend from the preregistered base
`b2da5df5eddd5437d444b60b11130262d115e264`.

## Frozen scientific identities

The execution verifies before opening target outcomes:

- protocol SHA-256
  `921da8a6f14f9430b3f4861d68326d904f61b922e3aedd2b35882ea97bc63111`;
- Prob4D producer revision
  `aa8ffc6541011d044561e09870569a14ab3f586f`;
- BayesianPhysTwin base revision
  `b2da5df5eddd5437d444b60b11130262d115e264`;
- decoded controlled-runner SHA-256
  `16beddf036d797ad16868a4b45596b11b2f9617ac6f39f609b5a1b9ce6de3a63`;
- independent verifier SHA-256
  `1b07e9b9c0b3f31c1700d1e4f97ae43467ce01a05b9e04108b4b2c434efb5eda`.

The source/calibration groups, target groups, seeds, scenarios, methods,
endpoints, guard-selection rule, bootstrap unit, and acceptance thresholds are
unchanged.

## Valid outcomes

The runner returns:

- exit code `0` for a completed registered pass;
- exit code `3` for a completed valid negative result;
- any other code for an execution, contract, or infrastructure failure.

A valid negative result is uploaded and retained rather than converted into a
failed or retuned experiment. The independent verifier reparses every target
trial, recomputes guard decisions, aggregate metrics, paired bootstrap
intervals, and the registered final decision before the workflow accepts either
scientific outcome.

## Evidence bundle

The uploaded 90-day artifact contains:

- the frozen protocol and result report;
- all calibration and target trial rows;
- report and result checksums;
- console output;
- exact source-resolution diagnostics;
- runner and numerical-runtime information;
- infrastructure checksums; and
- the decision exit code and exact repository revisions.

This remains controlled synthetic evidence. Even a pass authorizes only the
next genuinely fresh physical-object or acquisition-session experiment; it does
not establish a real-world provider, calibration, physical-benefit, or Causal4D
intervention claim.
