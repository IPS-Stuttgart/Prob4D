# Claim-bearing factor-bundle integration gates

The strict envelope in `prob4d.provider_v2_factor_bundle` is the producer-side
admission boundary for prospective explicit-gauge experiments. Before a factor
bundle is used for a Bayesian-PhysTwin update, integration tests should verify:

1. the claim-bearing envelope, neutral manifest, and NPZ payload load through
   `load_claim_bearing_observation_factor_bundle`;
2. the physical linearization identifies the same case, row identities, causal
   cutoff, and immutable observation artifact;
3. the consumer uses conditional point covariance, the sparse gauge Jacobian,
   and the complete joint gauge prior exactly once;
4. association probability, source-side prior reliability, nominal-component
   probability, and composite information weight remain separate quantities;
5. the accepted or fallback Bayesian-PhysTwin belief binds the envelope artifact
   ID, provider manifest ID, calibration IDs, runtime revision source, and
   physical-linearization ID; and
6. Causal4D consumes only that accepted belief rather than reopening the raw
   Prob4D factors.

A successful contract test establishes interoperability and provenance, not
physical-prediction benefit. Accuracy, harmful accepted-update frequency,
uncertainty coverage, and exact fallback remain prospective empirical gates.
