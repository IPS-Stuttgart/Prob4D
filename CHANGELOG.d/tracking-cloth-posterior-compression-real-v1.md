### Experimental real-data validation

- Add a recording-disjoint real-motion-capture evaluation of
  posterior-preserving shared-noise compression on the complete 120-recording
  Tracking Cloth Deformation dataset.
- Compare exact full covariance, posterior-preserving compression, matched-rank
  covariance PCA, conditional-only, prior-only, and cached-query baselines.
- Trigger the protected `gpuserver4090` run only through a fixed execution
  request, retain compact hash-bound evidence, and never upload raw trajectories.
- This study does not promote a learned provider, establish deployment
  calibration, or claim BayesianPhysTwin/Causal4D physical benefit.
