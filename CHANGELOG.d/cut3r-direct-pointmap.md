- Added a provider-neutral recurrent-online CUT3R route that preserves the original
  `pts3d_in_self_view` XYZ point maps instead of reconstructing XYZ from saved
  depth and fitted intrinsics.
- Added a content-addressed, target-free audit that quantifies direct-versus-depth
  geometry loss and verifies causal-prefix closure against a longer recurrent run.
- Retained `prediction import-cut3r-online` as the explicit depth-reprojected
  compatibility route; the new representation and audit change no frozen provider
  artifact, target-access decision, BayesianPhysTwin guard, or scientific result.
