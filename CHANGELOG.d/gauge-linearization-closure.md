Add a source-only deterministic sigma-point closure diagnostic for complete joint
`Sim(3)` composition, transformed point covariance, and optional downstream query
covariance. The artifact aggregates complete source object/session groups equally,
fails closed near `SO(3)` logarithm branch cuts, and explicitly cannot authorize a
richer conditional point-uncertainty model.
