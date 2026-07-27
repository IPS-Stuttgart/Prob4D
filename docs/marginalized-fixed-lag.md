# Marginalized Fixed-Lag Gauge Smoothing

The original fixed-lag reconstruction control optimized the recent gauges while
holding expired gauges at their posterior means. That discarded boundary
uncertainty and could make later gauge marginals overconfident.

`MarginalizedFixedLagGaugeSmoother` carries a quadratic information prior across
the moving boundary. Before the oldest active gauge leaves the lag, Prob4D
linearizes the previous boundary prior and every factor touching that gauge,
then eliminates its seven `Sim(3)` coordinates with a Schur complement:

```text
H_new = H_rr - H_rm H_mm^-1 H_mr

g_new = g_r - H_rm H_mm^-1 g_m.
```

The resulting prior is retained on the remaining active gauges. Relative factors
whose temporal span is at least the configured lag fail closed because one of
their endpoints would otherwise be eliminated before the factor arrives.

This fixes the zero-uncertainty boundary approximation. The portable fixed-lag
observation export still contains block-diagonal historical marginals rather
than one exact all-window covariance, so fixed-lag mode remains an explicitly
acknowledged reconstruction control and still requires opt-in.
