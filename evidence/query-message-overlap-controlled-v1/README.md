# Query-message overlap controlled evidence

This directory retains the compact byte-identical summary emitted by the
registered query-message overlap study. The full per-correlation result remains
in the immutable Actions artifact and is hash-bound by `manifest.json`.

## Result

The study evaluates 393,216 complete three-dimensional Gaussian query cases
across cross-window correlations `0`, `0.25`, `0.5`, `0.75`, `0.9`, and `0.99`.

- decision: `controlled-overlap-passed`;
- protocol ID: `7425758b5fbe7c0ae7f22bf3cfaa207746bf60881beb59a0c421ff362651b6cf`;
- result ID: `551e93bbf503909114177cd9607ceaa9743a8abf7dc5ad07e91a065bf9932da9`;
- workflow run: `33565333587`;
- artifact: `9822877862`;
- artifact SHA-256:
  `c5a1a911c87701b57ef0eeb7002a3b69b474867c4a6842b05e80f8f560659b34`.

At correlation at least `0.75`, naive independent message addition has
normalized NEES at least `1.597637` and nominal-90% coverage at most `72.9279%`.
At correlation `0.99`, these become `1.773671` and `68.2297%`.
Query-space covariance intersection has maximum normalized NEES `0.914409`,
minimum coverage `92.1402%`, and lower coordinate RMSE than either single
window at every registered correlation. Duplicate-message mean and covariance
errors are exactly zero.

## Custody and claim boundary

The workflow executed the registered study twice and required byte-identical
outputs before publishing the artifact. `manifest.json` binds the exact
evaluated revision, artifact, result, protocol, compact summary, and the
SHA-256 of the full retained result. The permanent workflow regenerates both
outputs, byte-compares the summary, and verifies the full-result hash.

This is controlled linear-Gaussian evidence. It does not establish learned
provider calibration, real-data utility, nonlinear-query validity, robot
safety, or state of the art. No dataset payload was opened, and no DOT source or
confirmation execution was authorized by this study.
