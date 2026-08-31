### Experimental real-data evaluation

- Add a recording-disjoint Tracking Cloth query-portfolio study that measures
  exact posterior-preserving rank, resident-model payload, and structured update
  cost against a direct cached Gaussian query message.
- Add a PSD-consistent complete-joint spectral comparator rather than relying on
  covariance truncations that can produce an invalid posterior.
- Preserve the explicit boundary that direct caching is preferable for a single
  immutable query unless a factor-level consumer or joint-query portfolio is
  required.
