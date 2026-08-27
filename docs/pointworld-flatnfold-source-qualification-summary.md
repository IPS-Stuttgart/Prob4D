# PointWorld--Flat'n'Fold source qualification summary

The branch chooses a sparse persistent-point representation for PointWorld and
adds a strict, outcome-blind support path for Flat'n'Fold. It intentionally does
not execute the provider or inspect target outcomes.

The next source execution must freeze exact model/runtime and dataset bytes,
retain all three cameras and one action lineage per demonstration, pass
`ProviderSupportFeasibilityV1`, and produce one canonical sparse source artifact.
Only then may source mean/identity and uncertainty calibration be evaluated.
