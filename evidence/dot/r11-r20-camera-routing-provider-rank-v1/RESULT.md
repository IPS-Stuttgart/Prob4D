# DOT R11–R20 camera-routing CUT3R source-rank result

## Outcome

**Decision:** `camera-routing-provider-rank-negative`

The corrected routed-camera experiment completed successfully on `gpuserver4090` (`workstation1`) in GitHub Actions run `33552798863`. The target-closed synthetic CUT3R smoke passed, all R11–R20 marker-free predictions were sealed, and the source-only rank gate was evaluated.

Only **1 of 10** source sequences passed the fixed rank/support criterion; the promotion threshold was **9 of 10**. The only supported sequence was `R18` on `cam001`:

- factor rank: `6` (expected `6`)
- observable condition ratio: `0.9984359824719174`
- normalized support margin: `3.375`
- fit-A / fit-B / common overlap support: `27 / 27 / 27`

The remaining nine sequences produced full-rank (`7`) factors rather than the required one-dimensional gauge defect. Camera routing therefore did not recover a broadly applicable rank-6 ambiguity class for the fixed candidate `expanded__overlap-345`.

## Reproducibility identities

- evaluated commit: `c64765ea766e667a566e1b565e8ed01ffd734e53`
- result ID: `a1fc018dc7fb504b35f6fbfc422e7a59edaeb71009c6f29c512019d42f949ced`
- provider seal ID: `38ea78e8bf44cbeaedeeadaee862af3cc6369d35d7e3b5a2b5fac0f020c7145b`
- provider artifact: `dot-r11-r20-routed-provider-fast-33552798863-1`
- provider artifact ID: `9818146750`
- provider artifact SHA-256: `70c2cec1cf33b65ae6653a3839fb4f74d57023d1d49cced6c61e6948f4d7b8a6`
- rank-result artifact: `dot-r11-r20-routed-provider-rank-fast-33552798863-1`
- rank-result artifact ID: `9818149548`
- rank-result artifact SHA-256: `d6f81bbb2fbe8af7cf574fe4bdad313fe0ac05d2882a3673c832b2ce2e79add0`

The three routed provider components were:

| Camera | Sequences | Provider bundle ID |
|---|---|---|
| `cam001` | R13, R14, R15, R16, R18, R19 | `14d78d100a62e11e0e51d76491e72f9bb34d30e141a8a510d9bdb69bacb35dff` |
| `cam002` | R17 | `e0c476a039e9610a2849c6dba3996936fbf7f648b68b61d93bc0519fff389dae` |
| `cam005` | R11, R12, R20 | `60c56f8c1cc2b07e418d67b01d81c3dc7dd7b369dbeaac9899f029af8f79b818` |

## Scientific interpretation

This is a valid negative source-feasibility result. The learned provider and camera routing worked technically, but the intended rank-deficient ambiguity structure was present in only one sequence. Consequently, this fixed DOT route should not be promoted to R21–R30 confirmation or used as the flagship positive paper result.

The result points to a different formulation: treat the prevalent full-rank cross-window uncertainty as a general correlated belief and test query-level value directly, rather than requiring a rank-6 gauge factor in every sequence. That is a new experiment and should not be represented as a rescue of this fixed rank-gate result.

No BayesianPhysTwin or Causal4D run was executed, and no R21–R70 payload was opened by this experiment.
