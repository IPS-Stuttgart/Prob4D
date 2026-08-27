# PointWorld--Flat'n'Fold source qualification stop rules

Stop the current provider version and retain a negative result when any of the
following occurs after its corresponding identity has been frozen:

1. the exact checkpoint or auxiliary model cannot be loaded in the bound runtime;
2. the complete source roster lacks the frozen three-camera/action/geometry
   support;
3. Baxter/gripper action poses cannot be converted into PointWorld robot point
   flow without using future target state;
4. PointWorld scene-point order is not persistent over the declared horizon;
5. the output cannot satisfy the sparse source-export schema without
   interpolation or truth-based repair;
6. source mean or identity competence fails the frozen comparator gate;
7. native uncertainty cannot be calibrated without unacceptable width or
   worst-garment undercoverage;
8. calibration transport rejects the frozen target prefix;
9. the source-selected recursive candidate does not improve or safely dominate
   the simple comparator; or
10. the downstream BayesianPhysTwin guard cannot bound harmful accepted updates.

After a stop, do not change the checkpoint, point representation, mask, action
conversion, camera subset, garment roster, covariance family, or downstream query
under the same protocol identity. A new attempt requires a new version and a new
unopened target cohort where applicable.
