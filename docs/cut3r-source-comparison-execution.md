# CUT3R source-comparison execution

The retained source preflight authorizes no inference.  Before any source RGB is
decoded, build a separate execution plan with
`build_cut3r_source_comparison_execution_plan.py`.  The plan binds the exact
Prob4D commit and source files, CUT3R commit and callable files, checkpoint,
runtime inventory, 40 source cases, and the following three-arm comparison:

- `native-continuous`: one official recurrent CUT3R call over frames `[0,58)`;
- `restarted-newest`: fresh calls over `[0,25)`, `[17,42)`, and `[33,58)`,
  retaining the latest-starting valid prediction at each frame and pixel; and
- `restarted-prob4d-fused`: the same restarted windows, gauges, and uncertainty
  fields, combined by decoded uniform fusion.

The restarted windows are aligned only to their immediate predecessor using
same-frame, same-pixel valid correspondences and robust Sim(3).  The first gauge
is the identity and later gauges are propagated sequentially with
correlation-aware covariance.  CUT3R direct `pts3d_in_self_view` maps are used;
the historical depth-reprojection compatibility path is not used.  The three
registered seeds affect only deterministic correspondence subsampling in the
gauge fit.  Raw CUT3R inference is executed once per arm/window with the official
demo seed of 42, so the primary contrast does not spend three times as much model
inference on the treatment.

The runner publishes each case by atomic directory rename.  A completed case is
never overwritten.  A failure is retained once with its stage flags and is not
retried or replaced.  Two shards partition the lexicographically sorted source
roster.  A one-case smoke is allowed only on a frozen development case and never
counts toward the source barrier.

This stage may decode the frozen source RGB and write source predictions.  It may
not open source truth, source physical residuals, candidate reference contents,
target payloads, or target outcomes, and it may not run BayesianPhysTwin or
Causal4D.  Its outputs establish implementation and source-provider evidence
only.  They do not establish held-out competence, physical-twin benefit,
deployment safety, or state of the art.
