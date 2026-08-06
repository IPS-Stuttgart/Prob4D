# Zero-copy prediction-window truth views

`TruthSequence` is the ordinary external truth contract. Its constructor
canonicalizes floating fields to `float64`, defensively copies every array, and
makes the retained values read-only. That is the correct default for independent
truth supplied by a caller.

Some engineering checks intentionally compare a fused result against another
already validated Prob4D prediction artifact. Examples include backend parity,
storage-path profiling, and ablations that use the same MotionCrafter disjoint
baseline as a common internal reference. Copying a full-resolution
`PredictionWindow` through `TruthSequence` can duplicate hundreds of megabytes of
point maps and scene flow even though the source artifact is already immutable.

`PredictionWindowTruthView` makes this special case explicit:

```python
from prob4d.metrics import evaluate_sequence
from prob4d.truth_view import prediction_window_truth_view

reference = prediction_window_truth_view(disjoint_prediction_window)
metrics = evaluate_sequence(fused_prediction, reference)
```

The view:

- accepts only a validated `PredictionWindow`, including a verified read-only
  memory-mapped execution-store window;
- requires every retained source array to be non-writeable;
- preserves array identity and storage dtype without conversion or copying;
- remains a `TruthSequence` subtype, so existing metric and evaluation-mode
  functions require no alternate estimator path; and
- records the exact source window ID for audit output.

The adapter changes ownership and retains the source storage dtype. Mixed
prediction/reference arithmetic is still evaluated in floating-point NumPy
operations, but a float32 reference is not first materialized as a complete
float64 array. Consequently, derived operations performed solely on the
reference—such as a reference-vector norm—may differ from the ordinary copied
path by storage-rounding at the last few floating-point bits. The focused
contract requires exact structural/count equality and bounds every scalar metric
difference by `1e-20` on the adversarial float32 regression. Reports that require
byte-identical metrics across storage dtypes should continue to use the ordinary
`TruthSequence` copy.

## Claim boundary

A prediction-window truth view is suitable only where the prediction artifact is
predeclared as a common internal reference. It does **not** convert a model output
into external ground truth, establish reconstruction accuracy, calibrate
uncertainty, or justify a downstream BayesianPhysTwin or Causal4D claim. Reports
using this adapter should name the reference artifact and state that the result
is an engineering or ablation comparison.
