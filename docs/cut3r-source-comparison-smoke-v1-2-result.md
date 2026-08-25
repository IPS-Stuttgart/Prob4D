# CUT3R source-comparison smoke v1.2 result

The one registered distinct development smoke consumed its write-once attempt
ledger and reached the CUT3R provider forward pass. The frozen model and
checkpoint loaded, 58 causal source frames were decoded, and the first native
continuous inference terminated with a CUDA device-side index assertion.

The retained execution record reports:

- 0 ordinary successes;
- 1 retained technical failure;
- 0 prediction products;
- no source truth, candidate-reference contents, targets, BayesianPhysTwin, or
  Causal4D access; and
- no retry authorization.

This is a real provider-runtime negative result, not an accuracy result. It
closes the registered schema-v3 smoke and does not authorize the two frozen
source shards.

## Custody defect

The independent artifact verifier then correctly rejected the retained case:
the failure path had moved all 58 decoded PNG frames into the case directory.
The source-only server tree remains quarantined and unmodified; it must not be
published as an artifact. The compact metadata-only result is preserved at
`evidence/cut3r-source-comparison-smoke-v1-2/summary.json`.

The runner now removes its decoded-frame directory in a `finally` block before
publishing either a success or a technical-failure manifest. This cleanup fix
is prospective implementation hardening only. It does not reopen or retry the
consumed smoke and does not convert the failed artifact into valid custody.

## Claim boundary

The result establishes that the repaired import path reaches actual CUT3R GPU
inference, and that this frozen 58-frame invocation is not runtime-compatible
with the pinned provider/checkpoint path. It establishes no provider competence,
point accuracy, uncertainty calibration, transfer, or downstream physical-twin
benefit.
