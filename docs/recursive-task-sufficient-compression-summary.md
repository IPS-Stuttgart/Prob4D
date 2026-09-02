# Recursive task-sufficient compression branch summary

This branch turns the fixed-query compression theorem into a testable recursive hypothesis.

Core mechanism: preserve the smallest exact task-state closure rather than only the task reported
at the current instant. In the controlled 20D design, the current task is 3D, the recursive closure
is 4D, and the supplied correlated-noise factor has rank 8. Rank-3 compression is exact at the
first task update but fails later; rank-4 closure-aware compression remains recursively identical to
the full task-state filter. A deliberately violated transition expands the closure to 5D.

No manuscript or robotics claim should be changed until CI, immutable controlled execution, and
prior-art review are complete.
