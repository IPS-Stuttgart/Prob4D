# Concise review summary

The branch resolves the PointWorld representation mismatch by adding a sparse
persistent-point artifact instead of altering dense `PredictionWindow` semantics.
It also adds an outcome-blind Flat'n'Fold support gate that reuses the existing
provider-support feasibility contract.

The branch does not yet execute PointWorld or use Flat'n'Fold target data. Its
scientific value is that the next real source run now has explicit identities,
stop rules, and a non-rasterized artifact boundary rather than an ad hoc adapter.
