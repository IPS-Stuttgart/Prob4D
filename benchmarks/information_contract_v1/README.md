# Information-contract benchmark v1

This directory contains the provider-neutral protocol, schemas, and
deterministic conformance controls for the Prob4D information-contract
benchmark.

- `protocol.json` freezes the independent-unit, multi-axis, information-order,
  negative-control, release, and claim-boundary rules.
- `suite.schema.json` defines the compact self-contained replay format.
- `challenge.schema.json` and `submission.schema.json` define the preferred
  truth-separated public-evaluation interface.
- `controlled_suite.json` exercises accuracy/calibration rank reversal,
  dependence preservation, finite-support query identification, exact decision
  fallback, payload parity, unsupported specificity, and shared provider bias.

The controlled suite is **not** a public-data benchmark result. It opens no
trajectory, invokes no learned provider, and defines no scalar overall ranking.

Run the self-contained and truth-separated controls:

```bash
python -m pytest \
  tests/test_information_contract_benchmark.py \
  tests/test_information_contract_sealed.py

python -m prob4d.information_contract_benchmark smoke \
  /tmp/prob4d-information-contract

python -m prob4d.information_contract_sealed smoke \
  /tmp/prob4d-information-contract-sealed
```

The sealed interface prevents a submission from supplying target truth,
registered losses, ambiguity classes, or fallback. It also distinguishes
retrospective open-target replay from a prospectively sealed challenge and adds
a finite-quotient query test that can disagree with the local nullspace test.

The public-data promotion gate requires at least two distinct provider
contracts and two independently collected public datasets, with manifests,
payload hashes, statistical units, queries, actions, loss semantics, dependence
groups, target opening, and claim boundaries frozen before held outcomes are
scored.

See:

- [`docs/information-contract-benchmark.md`](../../docs/information-contract-benchmark.md)
  for scientific scope and the self-contained replay format;
- [`docs/information-contract-sealed-submissions.md`](../../docs/information-contract-sealed-submissions.md)
  for the truth-separated challenge/submission contract; and
- [`STATUS.md`](STATUS.md) for the current empirical promotion boundary.
