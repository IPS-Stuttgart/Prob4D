# Information-contract benchmark v1

This directory contains the provider-neutral protocol and deterministic
conformance suite for the Prob4D information-contract benchmark.

- `protocol.json` freezes the independent-unit, multi-axis, information-order,
  negative-control, release, and claim-boundary rules.
- `controlled_suite.json` exercises accuracy/calibration rank reversal,
  dependence preservation, finite-support query identification, exact decision
  fallback, payload parity, unsupported specificity, and shared provider bias.

The controlled suite is **not** a public-data benchmark result. It opens no
trajectory, invokes no learned provider, and defines no scalar overall ranking.

Run:

```bash
python -m pytest tests/test_information_contract_benchmark.py
python -m prob4d.information_contract_benchmark controlled_suite.json
```

The public-data promotion gate requires at least two distinct provider contracts
and two independently collected public datasets, with manifests, payload hashes,
statistical units, queries, actions, loss semantics, dependence groups, target
opening, and claim boundaries frozen before the held outcomes are scored.

See [`docs/information-contract-benchmark.md`](../../docs/information-contract-benchmark.md)
for the scientific scope and panel format.
