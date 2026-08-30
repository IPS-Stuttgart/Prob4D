# DOT rope shared-curvature validation

This experiment is the public real-data follow-up to the controlled shared-gauge
curvature result. The initial stage inventories the seven publisher ZIP archives
at the fixed `gpuserver6000` path. It verifies the publisher MD5 values and reads
ZIP central directories, but it does not open or extract any member payload.

The split is frozen before trajectory inspection:

- development: `R01-R30`;
- calibration: `R31-R40`;
- held out: `R41-R70`.

The first stage answers only whether the installed release is complete and what
source-side parser is required. A later source-only stage may decode development
payloads to construct a real-camera/oracle-correspondence geometry experiment.
Calibration and held-out payload access require separate request records and
cannot be authorized by the inventory result.

The intended scientific comparison holds provider means fixed and compares
first-order shared gauge, axis spherical-radial propagation, scalar inflation,
pointwise curvature, shared quadratic curvature, and higher-degree cubature.
Primary endpoints will be sequence-clustered proper score, coverage with width,
and the non-averaging behavior of shared uncertainty across rope query points.

No workflow in this directory may upload videos, images, meshes, dense
trajectories, or raw correspondences. Only aggregate, sanitized evidence is
allowed.
