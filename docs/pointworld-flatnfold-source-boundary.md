# Source-only information boundary

Allowed before the support and representation decisions:

- released repository code and documentation;
- exact model, loader, calibration, action, and dataset-byte identities;
- complete inventory/source garment metadata;
- camera and action geometry;
- source-only PointWorld predictions after support passes; and
- source/calibration residuals only after their roster is frozen.

Prohibited:

- any target-garment provider residual;
- target truth used to create point identities or rasterization;
- BayesianPhysTwin innovations used to choose representation or covariance;
- target-driven camera, frame, mask, checkpoint, or garment exclusions;
- Causal4D outcomes; and
- relabelling native PointWorld log variance as calibrated metric covariance.
