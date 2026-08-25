# CUT3R source-comparison smoke v1

The one authorized frozen development-case smoke ran once on physical GPU 1 and
terminated while initializing the CUT3R Python runtime. CUT3R's repository root
was importable, but its internal `src` directory was not exposed as a top-level
package path, so `src.dust3r.inference` failed on CUT3R's own absolute
`dust3r.*` import.

This is a pre-science technical failure. The runner had not verified or decoded
the selected video, executed CUT3R inference, written a prediction, opened source
truth, or accessed any target. The output directory contained zero files after
termination. The registered no-retry rule remains in force, so this case was not
rerun.

The follow-up implementation adds both the repository root and `CUT3R/src` to
the import surface, imports the runtime consistently through `dust3r`, and
retains future shared-runtime initialization failures as explicit zero-progress
case artifacts. These repairs are implementation evidence only and do not
authorize another smoke or the frozen source shards.
