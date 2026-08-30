# Deform360 persistent-trajectory query study

Status: **prospective public-data study in two information-separated stages**.

## Scientific question

Can correlation-aware fusion of overlapping real deformable-object trajectory forecasts improve a declared physical query without fabricating independence across windows, while an outcome-blind guard retains the exact latest-window fallback when the candidate is unsupported?

The study uses the official Deform360 `pcd_clean.tar` products. Each archive contains one persistent set of reconstructed three-dimensional points across frames, with per-frame positions, velocities, camera provenance, and visibility. This gives a real-geometry test of Prob4D's recursive overlap and dependence semantics without claiming that the source is a learned 4-D provider.

## Information order

The study is intentionally split.

1. **Tar-header inventory.** Inspect only filesystem metadata and tar headers under
   `/mnt/seagate10tb/florianpfaff/datasets/deform360/processed-repository/processed`.
   Archive member payloads and NumPy arrays remain unopened. The result records complete object/episode identities, frame-member support, unstable or malformed archives, and an explicit exclusion union for prior Deform360 evidence.
2. **Frozen numerical evaluation.** After the inventory is retained, commit one exact object/episode roster, archive identities, development/calibration/evaluation roles, trajectory query, prediction horizons, estimators, guard, statistical units, and success rules. Only that later request may open selected `pcd_clean` arrays.

A failed or insufficient inventory is a complete first-stage result. It cannot be repaired by silently dropping objects or reading arrays to choose an easier cohort. A successful inventory authorizes protocol design only; it is not an accuracy or calibration result.

## Planned evaluation after inventory

For one source-defined persistent point region and terminal query, each causal window supplies a constant-velocity forecast from the official point position and velocity at its window endpoint. The comparison will keep a small frozen arm set:

1. latest-window forecast;
2. uniform overlap average;
3. naive marginal precision fusion that treats windows as independent, retained as a dependence failure control;
4. full correlation-aware generalized least-squares fusion using source-fitted cross-window residual covariance;
5. one query-directed subset selected from conditional incremental value; and
6. guarded correlation-aware fusion with exact latest-window fallback.

Development/calibration objects will fit covariance and the guard. Evaluation objects will be disjoint physical object identities. Frames, points, coordinates, and episodes are nested observations, not independent statistical units.

Primary outputs will include object-balanced physical-query RMSE, a proper distributional score, 90% coverage with width, accepted/rejected/fallback counts, materially harmful accepted updates, and worst-object regret. Provider competence, learned-model value, BayesianPhysTwin value, Causal4D benefit, deployment safety, and state of the art remain outside this bounded study.

## Current stage boundary

`deform360-pcd-query-inventory-v1` authorizes only the first stage. It does not extract archive members, call `numpy.load`, execute a provider, score a query, open a target outcome, or mutate the mounted dataset.
