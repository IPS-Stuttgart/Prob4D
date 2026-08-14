### Added

- A runtime-neutral recurrent-online CUT3R importer that converts official
  depth, confidence, and camera outputs into a content-addressed provider-neutral
  prediction manifest with prefix-growing causal lineage.
- Randomized `Sim(3)` group-action and covariance-preservation regression tests.
- A repository documentation-surface check against the grouped command registry
  and public API manifest.

### Fixed

- Active documentation no longer teaches removed Prob4D 0.4 standalone
  executables or the retired broad package-root Python facade.

### Scientific boundary

The CUT3R adapter establishes byte-level interoperability, provenance, and
causal-lineage validation only. It does not establish provider competence,
metric scale, covariance calibration, BayesianPhysTwin benefit, Causal4D benefit,
deployment safety, or state of the art.
