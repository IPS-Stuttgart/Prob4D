# CUT3R source-comparison smoke v1.2 result

The one registered distinct development smoke consumed its write-once attempt
ledger and entered the CUT3R provider forward call. The frozen model and
checkpoint loaded, and 58 causal source frames were decoded. The call did not
return successfully; the exact attempt ended after the runtime reported a CUDA
device-side index assertion.

The retained execution record reports:

- 0 ordinary successes;
- 1 retained technical failure;
- 0 prediction products;
- no source truth, candidate-reference contents, targets, BayesianPhysTwin, or
  Causal4D access; and
- no retry authorization.

The compact summary records `cut3r_inference_started=true` and
`cut3r_inference_completed=false`. The quarantined raw case manifest retains
the legacy `cut3r_inference_executed=false` field. The frozen runner sets that
legacy field to true only after the provider forward call returns, so false here
denotes non-completion rather than non-entry.

This is a retained provider-runtime failure for this exact attempt, not an
accuracy result. It closes the registered schema-v3 smoke and does not authorize
the two frozen source shards.

## Custody defect

The independent artifact verifier then correctly rejected the retained case:
the failure path had moved all 58 decoded PNG frames into the case directory.
Under the no-raw-access review boundary, the operator attests that the
source-only server tree remains quarantined and unmodified; it must not be
published as an artifact. The compact metadata-only result is preserved at
`evidence/cut3r-source-comparison-smoke-v1-2/summary.json`.

The runner now removes its decoded-frame directory in a `finally` block before
publishing either a success or a technical-failure manifest. This cleanup fix
is prospective implementation hardening only. It does not reopen or retry the
consumed smoke and does not convert the failed artifact into valid custody.

## Claim boundary

The result establishes that the repaired import path loaded the pinned model
and checkpoint and that this exact frozen attempt reached provider inference
but did not complete after a CUDA device-side index assertion was reported. A
single no-retry attempt does not establish deterministic incompatibility of the
invocation, provider, or checkpoint. It establishes no provider competence,
point accuracy, uncertainty calibration, transfer, or downstream physical-twin
benefit.
