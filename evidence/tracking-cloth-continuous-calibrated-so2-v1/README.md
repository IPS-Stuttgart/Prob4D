# Tracking Cloth continuous SO(2) calibration v1

This directory retains the compact evidence for Prob4D's first real-trajectory
continuous-symmetry calibration study.

## Terminal result

The registered result is `evaluated-continuous-calibrated-so2-negative`.

The continuous geometry and calibration behaved as intended:

- exact full-circle vector-query diameter, without an angle grid;
- 90.00%--93.75% held-fold coverage of the registered recording-level score
  at nominal 90%;
- 97.62%--98.20% case-level support coverage;
- 99.88%--99.95% coverage of analytic scalar-query intervals;
- nonzero admission at every horizon; and
- exact fallback whenever the certificate rejected.

The preregistered utility result is negative. The transported
constant-angular-velocity local predictor was already essentially never harmful,
so the calibrated certificate could not strictly reduce harmful acceptance. At
24 frames it rejected 27.5% of cases, and the deliberately invariant axis-center
fallback increased mean recording-level RMSE from 32.480 to 117.621 mm.

This negative result must not be reclassified as a successful selective-risk
certificate. It supports exact continuous-group computation and marginal
recording-score calibration, while showing that support validity alone does not
make abstention useful when the fallback is weak.

## Evidence identities

- Protocol ID: `930ad490206f8b16a113fbc221af9f0d35c906318625fc661695508b2e03da55`
- Executed source: `0b0233edf33d55ae8cb0d67660be0307301eccb3`
- Workflow run: `33591643438`
- Preflight job: `100126681594`
- Evaluation job: `100126922548`
- Artifact ID: `9831982020`
- Artifact digest: `sha256:4baf9f4d64193a9993f2e93f6973c4897d07f08c5bef023cf0e76fbe3cced513`
- Result SHA-256: `5bfb93c4a6e3bc2ba20054e992511331f8b70b8fd16b8a32daea7081d44e6382`
- Protocol SHA-256: `cbe6e55a6785dbf3df54fab60753e56dfcce3aac1189e532b5153e33306cf17f`
- Inventory SHA-256: `daf8f0c9eaffd67caa3fa44d4129d0b574c154a8a9072f9bbe54752d5eda94b4`

`summary.json` contains the bounded numerical result. The complete result,
inventory, recording scores, protocol and manifest are retained in the bound
workflow artifact. No raw trajectory payload is retained.
