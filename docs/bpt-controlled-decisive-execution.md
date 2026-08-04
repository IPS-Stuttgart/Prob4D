# Controlled Prob4D-to-BayesianPhysTwin execution

This repository hosts the authoritative reproduction boundary for the frozen
`prob4d-bpt-controlled-decisive-v1` study. The scientific protocol, complete
estimator source, and retained result remain owned by BayesianPhysTwin. Prob4D
provides the exact private producer checkout and the registered self-hosted
runner.

## Source transport

The first orchestration attempt encoded the BayesianPhysTwin runner and verifier
as compressed text fragments. That transport was incomplete and failed during
`zlib.decompress` before any calibration or target group ran. It has been
removed from the execution path.

The repaired workflow uses complete Git sources:

1. Prob4D checks out producer revision
   `aa8ffc6541011d044561e09870569a14ab3f586f` through this repository's
   read-only token.
2. BayesianPhysTwin stages complete revision
   `db0f0119a3a4220f5489566829846681e844627d` outside the ephemeral Actions
   workspace. That revision contains the full uncompressed generator, runner,
   focused tests, frozen protocol, and retained checksummed evidence.
3. The Prob4D workflow clones that exact runner-local Git object into an
   isolated checkout. A preconfigured read-only SSH identity is only a bounded
   fallback. Missing source, a dirty tree, a revision mismatch, or an ancestry
   mismatch fails closed.

The orchestration has read-only repository permissions and contains no branch
mutation or result-publication step. Both repositories are installed into a
fresh Python 3.12 virtual environment. The BayesianPhysTwin execution revision
must descend from preregistered base
`b2da5df5eddd5437d444b60b11130262d115e264`.

## Frozen identities

Before execution the workflow verifies:

- protocol SHA-256
  `921da8a6f14f9430b3f4861d68326d904f61b922e3aedd2b35882ea97bc63111`;
- Prob4D producer revision
  `aa8ffc6541011d044561e09870569a14ab3f586f`;
- BayesianPhysTwin complete-source revision
  `db0f0119a3a4220f5489566829846681e844627d`;
- BayesianPhysTwin preregistered base
  `b2da5df5eddd5437d444b60b11130262d115e264`; and
- retained registered report identity
  `c592807d62e9f5121acf85747432574601264160de67b15e9a1c8e48a12cc040`.

The methods, calibration and target seeds, scenarios, endpoints, guard search,
bootstrap unit, and acceptance thresholds are unchanged.

## Reproduction and independent checks

The workflow runs focused Ruff and pytest checks on the complete uncompressed
study implementation, then reruns all 48 calibration groups and 384 target
groups. Exit code `0` is a registered pass and exit code `3` is a completed
valid negative result; every other exit code is an infrastructure or contract
failure.

After execution, a separate inline verifier:

- checks all generated SHA-256 entries;
- reparses all 2,304 target method rows;
- recomputes the six registered primary criteria from the report;
- verifies the decision exit code;
- verifies exact fallback for rejected updates;
- requires exact equality between the regenerated deterministic target table
  and the retained target table; and
- confirms that neither source checkout was modified.

The complete 90-day artifact contains environment and source-resolution records,
pinned package versions, console output, generated protocol/report/trials,
scientific checksums, infrastructure checksums, and the decision exit code.

## Claim boundary

This is controlled calibration/target-separated synthetic evidence. The retained
registered result is positive, but it authorizes only progression to a genuinely
fresh physical-object or acquisition-session experiment. It does not establish
real-world Prob4D provider competence, calibrated deployment uncertainty,
BayesianPhysTwin benefit on an independent physical cohort, or Causal4D
intervention benefit.
