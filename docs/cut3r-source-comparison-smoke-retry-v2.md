# CUT3R source-comparison smoke replacement v2

The first authorized development smoke for the frozen CUT3R source comparison
terminated while initializing CUT3R's Python runtime. Its retained artifact
`ef4e5bf187570e918df1d7d14434b4ae55f983c347104b9c6f7ad52b42f7a7bf`
records that no source frame was decoded, no CUT3R inference was executed, no
prediction was written, and no source truth, target data, BayesianPhysTwin, or
Causal4D result was opened.

This protocol permits exactly one replacement smoke on the same frozen
development case. It is justified only by that zero-progress state. The
replacement:

- keeps source-freeze ID
  `5e739b92c2628c61fa99ae68da61d5814ca94d4b6de5720b75c4552de82d1b2c`;
- keeps CUT3R revision
  `8bc15dc92a6d7fd92920b4ec81540d3dec7d3ecf` and checkpoint bytes unchanged;
- keeps the 40-case roster, windows, seeds, alignment, covariance, fusion,
  comparator, and information boundary unchanged;
- corrects only the recorded/imported callable from
  `src.dust3r.inference.inference` to `dust3r.inference.inference`;
- requires the exact Python/CUDA package inventory retained by the first smoke;
- executes on physical GPU 1 of the labelled `workstation2` runner; and
- independently rehashes every retained case byte and emits a custody receipt.

The GitHub issue command is accepted only once. A second command, a rerun after a
terminal receipt, or a run with an existing replacement artifact fails before
source decoding. The self-hosted job has read-only Actions, contents, and issue
permissions.

An ordinary-success replacement smoke permits preparation of the separately
reviewed two-shard source execution. It does not itself authorize those shards.
A retained technical failure is terminal for this replacement route and may not
be silently retried.

Neither outcome opens source residuals or truth, confirmation or target
payloads, BayesianPhysTwin, or Causal4D. The result is implementation and custody
evidence only, not provider competence, downstream benefit, deployment safety,
or a state-of-the-art claim.
