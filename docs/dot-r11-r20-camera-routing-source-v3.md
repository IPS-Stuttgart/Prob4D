# DOT R11-R20 camera-routing source v3

This source-development lane follows the terminal fixed-camera R04-R10 `heldout-support-negative` result without retuning or reinterpreting it.

The source split is R11-R20. R21-R30 remains a one-shot confirmation cohort and R31-R70 remains reserve.

A source-only raw marker audit on run `33527652881` found that the frozen `expanded__overlap-345` geometry supports all 10 R11-R20 sequences when a deterministic per-sequence single-camera rule is allowed. The rule chooses the lexicographically smallest camera that meets the frozen pixel-zero-based 2-D support thresholds. It selected `cam005` for R11, R12, and R20; `cam002` for R17; and `cam001` for the remaining six sequences. The minimum selected-camera support margin was 1.125 times the registered threshold. No provider prediction, reconstruction error, NLL, or proper score entered that routing decision.

The next registered gate runs the unchanged CUT3R revision and checkpoint only on those routed source cameras. Provider outputs are sealed and uploaded before source marker qualification. Source qualification may use marker support and the rank-six observable-factor mechanism, but not reconstruction error or proper scores. Promotion requires at least 9 of 10 routed source sequences to support the fixed `expanded__overlap-345` geometry with factor rank six.

If the source provider-rank gate qualifies, a separate future protocol/request may open R21-R30 exactly once. That confirmation must predict all five normal-view cameras before target marker access, choose the lexicographically smallest camera meeting the frozen raw 2-D support rule, refuse camera switching after a factor-rank failure, seal all query decisions/predictions before 3-D outcome access, and retain R31-R70 unopened.
