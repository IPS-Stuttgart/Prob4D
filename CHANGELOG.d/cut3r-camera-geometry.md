### Fixed

- The recurrent-online CUT3R importer now preserves camera-origin unit viewing
  rays in the sequence-local frame instead of forcing downstream covariance code
  to infer rays from the arbitrary common-frame origin.
- Added a fail-closed camera-relative depth model that recovers one camera centre
  per frame from the imported points and rays and uses translation-invariant
  camera-to-point range for anisotropic uncertainty.
- CUT3R dense-memory limits now include the retained ray field, while manifest
  metadata reports legacy point/mask/index bytes, ray bytes, and their total
  separately.

### Scientific boundary

This corrects provider geometry and uncertainty-coordinate semantics. It opens no
source residual, target outcome, calibration, BayesianPhysTwin update, or
Causal4D result and does not establish provider competence or downstream benefit.
