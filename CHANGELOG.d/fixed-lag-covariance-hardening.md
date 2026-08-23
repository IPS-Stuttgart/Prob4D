Hardened the legacy fixed-lag gauge smoother's covariance reporting. Optimization
damping now affects steps only; covariance is recomputed from the accepted final
undamped Gauss--Newton Jacobian, and rank-deficient active gauge systems fail
closed instead of receiving a silent pseudoinverse. The production marginalized
joint-gauge observation path and frozen provider protocols are unchanged.
