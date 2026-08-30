### Experimental scientific kernel

- Add prior-aware shared-noise-factor compression preserving the complete
  posterior of a frozen local Gaussian query, with a necessary/sufficient range
  condition, minimum-rank proof and numerical downdate audit.
- Return the exact original factor when a rank cap cannot preserve the posterior.
- Add independent dense-reference tests, existing-operator integration coverage,
  and a controlled Sim(3)-linearized study including a cached-query baseline.
- No stable API/export change, provider promotion, calibration change, physical
  update authorization, or access to closed/terminal data. The reduced factor
  does not preserve observation evidence and must not be used to score it.
