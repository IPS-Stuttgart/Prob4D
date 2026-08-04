# Controlled Prob4D-to-BayesianPhysTwin execution

This repository hosts the authoritative execution boundary for the frozen
`prob4d-bpt-controlled-decisive-v1` study. The protocol and estimator remain
owned by BayesianPhysTwin pull request 109. This workflow changes only how the
exact private source revisions reach `workstation2`, how a known verifier
transport defect is repaired, and how the completed evidence is independently
validated.

## Exact source boundary

The workflow binds all inputs before opening calibration or target outcomes:

- Prob4D producer revision:
  `aa8ffc6541011d044561e09870569a14ab3f586f`;
- BayesianPhysTwin scientific implementation:
  `76d4aba20dd386e1f8583e501781d702d7937566`;
- preregistered BayesianPhysTwin base:
  `b2da5df5eddd5437d444b60b11130262d115e264`;
- protocol SHA-256:
  `921da8a6f14f9430b3f4861d68326d904f61b922e3aedd2b35882ea97bc63111`;
- decoded controlled-runner SHA-256:
  `16beddf036d797ad16868a4b45596b11b2f9617ac6f39f609b5a1b9ce6de3a63`;
- decoded independent-verifier SHA-256:
  `1b07e9b9c0b3f31c1700d1e4f97ae43467ce01a05b9e04108b4b2c434efb5eda`.

The source groups, disjoint target groups, methods, seeds, scenarios,
endpoints, threshold fitting, bootstrap unit, and acceptance criteria are
unchanged.

## Why execution is owned by Prob4D

The original BayesianPhysTwin workflow could verify its own source but its
repository-scoped token could not read the transferred private Prob4D
repository. Prob4D can check out its own frozen producer using the current
repository token. The job resolves the exact BayesianPhysTwin commit from the
registered runner's Git object store and creates a detached shared clone. A
bounded SSH clone is permitted only when the runner already has a read-only
identity. Missing source, an unexpected revision, failed ancestry, or a dirty
checkout fails closed.

Both source snapshots are installed into a fresh Python 3.12 environment. The
workflow verifies repository identities, protocol bytes, encoded payload
inventories, decoded source hashes, and source compilation before running the
study.

## Registered verifier transport repair

BayesianPhysTwin revision `76d4aba…` contains the registered verifier source,
but its two-file Base64 transport has one extraneous character in
`part01.txt`. This is a packaging defect, not a change to the estimator,
protocol, trial generation, thresholds, or registered verifier source.
BayesianPhysTwin pull request 109 removes that character in its repaired head.

The Prob4D execution remains pinned to the preregistered scientific revision and
applies only the following fail-closed materialization:

1. require the exact two verifier files and their frozen SHA-256 identities;
2. require the exact 8,069-character transport and the character `s` at index
   7,570;
3. delete that one character;
4. Base64-decode with strict validation and decompress;
5. require the registered decoded verifier SHA-256 above; and
6. compile and execute the materialized verifier from the evidence directory.

The artifact records the source revision, both transport hashes, repair index,
removed character, and decoded verifier hash. Any other transport bytes or
repair location fail before an outcome is admitted.

## Execution and independent verification

After focused cross-repository regressions, the runner executes all 48
calibration groups and all 384 fresh target groups. Only two scientific exit
codes are accepted:

- `0`: completed registered pass;
- `3`: completed valid negative result.

Any other exit code is an execution, contract, or infrastructure failure. A
valid negative result is retained rather than retuned or converted into a green
claim.

Before accepting either registered outcome, the independent verifier checks the
result checksums, requires nonempty protocol, report, calibration-trial, and
target-trial files, reparses every trial, reconstructs guard fitting, recomputes
exact fallback, aggregate metrics, paired bootstrap intervals, acceptance
criteria, and the final decision, and checks the reported repository revisions.

## Evidence bundle

The 90-day artifact contains the frozen protocol, content-addressed report,
calibration candidates, fresh target trials, result checksums, console output,
source-resolution diagnostics, runtime information, the materialized verifier
and its repair record, infrastructure checksums, the scientific exit code, and
both exact repository revisions.

This remains controlled synthetic evidence. Even a registered pass authorizes
only progression to a genuinely fresh physical-object or acquisition-session
experiment. It does not establish real-world provider competence, deployment
calibration, physical benefit, or Causal4D intervention benefit.
