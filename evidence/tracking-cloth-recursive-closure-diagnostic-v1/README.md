# Tracking Cloth recursive-closure diagnostic — retained negative result

This record binds the preregistered retrospective structural diagnostic on the checksum-verified
public Tracking Cloth Deformation release. It was designed to answer one question before promoting
the controlled recursive theorem: **does the strict exact task-state closure remain materially
smaller on real deformable motion?**

The answer is **no for this registered model**.

## Result

All 80 compatible public recordings were used only to fit structural linear maps; no held-out
prediction score or target-performance metric was optimized. The fitted displacement states have
60 dimensions for A2 and 36 for A3.

For all three frozen task families—centroid, one central marker, and four evenly spaced markers—and
for all three frozen relative rank tolerances (`1e-12`, `1e-6`, `1e-3`):

- A2 observation-map rank is `60/60`;
- A3 observation-map rank is `36/36`;
- every strict task-plus-observation closure is the complete state; and
- even the task-only dynamics closure grows to the complete state.

Thus the exact infinite-horizon LTI closure mechanism that is strongly positive in the controlled
20D example does **not** yield a useful lower-dimensional exact recursive state on this real-cloth
linearization. This opened cohort must not be tuned into a replacement positive result.

## Provenance

- source revision: `075d75c6ab545bf7d5d6e3a9766678edac503e0d`;
- workflow run: `33590480190`;
- hosted job: `100123391684`;
- artifact: `9831566817`;
- artifact SHA-256: `a924728572037569b3ecabdbd9d64f46cb11aada73be879fe91162ba1895bda1`;
- protocol SHA-256: `da1112652c06e5a302d35ceeaa66ddc0b031bfc32319c7135beeb5465a7d5909`;
- result SHA-256: `b6c981e43691f1b0c7fd9607aad688144ac6d24cf983a0dfa7cda0d1c88240d9`;
- official release MD5: `b4868b702f8a42b2ea1069d0f1a3b8f6`;
- accepted recordings: A2 `48`, A3 `32`;
- Python `3.12.14`, NumPy `2.2.6`.

The workflow deleted the downloaded raw release before uploading the compact artifact.

## Consequence for the large-contribution path

Do not promote the strict recursive closure as the ICRA paper's major new robotics claim. The
controlled theorem remains useful as an exact boundary and negative control, but real full-field
cloth dynamics show why exact invariant closure is too strong.

The next research target must weaken the requirement while retaining a certificate. The most
promising directions are finite-horizon task/decision sufficiency or a certified recursive
rank--distortion bound. Either must beat the full-state closure without post-outcome retuning and
must retain the direct fixed-query cache as a comparator.

## Claim boundary

This is retrospective real-geometry structural evidence, not independent confirmation. It does not
establish predictive improvement, useful recursive compression, a learned visual provider,
closed-loop control benefit, deployment calibration, or state of the art.
