### Changed

- The recurrent-online CUT3R adapter now canonicalizes frames through temporary
  NPY memory maps instead of retaining and stacking the complete reconstructed
  sequence in memory.
- CUT3R imports fail closed on configurable frame-count, spatial-size,
  source-byte, and uncompressed dense-array budgets, and reject singular camera
  intrinsics with a contextual validation error. The loader identity now binds
  every implementation module introduced by the bounded-memory split.

### Scientific boundary

The change preserves the portable prediction-window schema, provider-manifest
semantics, causal lineage, confidence support rule, and all downstream readiness
and promotion gates. It is an ingestion scalability and hardening change, not new
provider-competence or physical-twin evidence.
