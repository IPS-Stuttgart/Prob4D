# Tracking Cloth finite-orbit hosted recovery v1

The existing Tracking Cloth finite-orbit request was authorized on `main`, but its data job remained unassigned on the `gpuserver4090` selector. It has no executable steps, runner assignment, or artifacts.

This recovery evaluates the unchanged protocol from the checksum-verified official Zenodo release. Before downloading the release, the hosted authorization job requires the predecessor run, head SHA, job name, runner labels, empty step list, zero runner assignment, and zero artifacts to match the frozen record, then retires the queued predecessor.

The experiment retains the original information order:

1. select the anchor/probe triplet from the 64 shaking and twisting source recordings;
2. write the source seal;
3. evaluate once on the 56 table-collision, stick-hitting, and self-collision recordings;
4. publish compact evidence only, never raw CSV or ZIP payloads.

This is public real-trajectory evidence under a controlled rank-deficient factor. It does not test a learned visual provider, identify a physical state, establish deployment calibration or safety, or claim state of the art.
