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
2. The BayesianPhysTwin integrity workflow stages revision
   `59256a124c4df1d780b79d1c31d6c1d01e63d10f` on the registered self-hosted
   runner. The Prob4D workflow resolves that exact commit from the runner-local
   Git object store. A bounded SSH clone is permitted only when the runner
   already has a read-only identity. The job fails if the exact object cannot be
   resolved.

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
- BayesianPhysTwin execution revision
  `59256a124c4df1d780b79d1c31d6c1d01e63d10f`;
- BayesianPhysTwin preregistered base revision
  `b2da5df5eddd5437d444b60b11130262d115e264`;
- decoded controlled-runner SHA-256
  `83af55e5744531110df5744031ab30b570bb5a0b9aa0bbb246961db783e166f5`;
- independent verifier SHA-256
  `4a206f1bf15b85e47cbe1f13c3095d7531b17c8d32c9ddb68f26ca0d099778ad`.

The source/calibration groups, target groups, seeds, scenarios, methods,
endpoints, guard-selection rule, bootstrap unit, and acceptance thresholds are
unchanged.

## Independently reproducible guard fitting

The evidence bundle includes `calibration_trials.csv`, which retains every
method-by-source-group candidate used to fit a deployment threshold. Each row
records the group and scenario identity, solver admissibility, risk score,
baseline and raw RMSE, harmful-update flag, uncertainty summaries, nominal
probability, identifiable fraction, query sensitivity, and fixed-point status.

The independent verifier does not trust the calibration summary in
`report.json`. For each method it reconstructs the exact frozen search over the
reject-all threshold and every finite admissible observed risk score. It applies
the registered minimum accepted-group requirement, harmful accepted-rate
ceiling, and deterministic tie-breaking over deployed RMSE, accepted support,
harmful count, threshold, and harmful rate. Target rows are admitted only after
every reconstructed calibration record matches the report.

## Valid outcomes

The runner returns:

- exit code `0` for a completed registered pass;
- exit code `3` for a completed valid negative result;
- any other code for an execution, contract, or infrastructure failure.

A valid negative result is uploaded and retained rather than converted into a
failed or retuned experiment. After independently reconstructing all guard
thresholds, the verifier reparses every target trial and recomputes guard
decisions, exact fallback, aggregate metrics, paired bootstrap intervals, and
the registered final decision before the workflow accepts either scientific
outcome.

## Evidence bundle

The uploaded 90-day artifact contains:

- the frozen protocol and content-addressed result report;
- all calibration candidate rows;
- all fresh target trial rows;
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
