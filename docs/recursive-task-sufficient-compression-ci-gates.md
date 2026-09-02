# CI gates

The research branch is acceptable for merge only if:

- repository lint/type/test checks pass;
- the recursive closure test returns dimension 4 for the registered design;
- a one-term closure violation expands the dimension to 5;
- the full shared factor has rank 8;
- closure-aware compression retains rank 4 at every registered update;
- current-task-only compression retains rank 3 and is exact at the first update;
- current-task-only recursion subsequently exceeds 1e-3 task-mean error; and
- closure-aware recursive mean and covariance errors remain below 1e-12.

These are controlled-mechanism gates only and do not establish real-data or robotics benefit.
