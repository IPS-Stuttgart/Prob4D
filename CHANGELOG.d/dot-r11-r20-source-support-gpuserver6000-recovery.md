### Changed

- Add a fail-closed `gpuserver6000` recovery lane for the frozen DOT R11–R20
  source-support qualification. It retires the untouched queued
  `gpuserver4090` provider, materializes only the publisher-verified
  `R11-20.zip`, applies the audited CUT3R runtime compatibility patch, and
  repairs only the dataset-free smoke workspace. The protocol, request
  identity, source-support objective, and unopened R21–R70 boundary are
  unchanged.
