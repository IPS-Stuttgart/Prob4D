# Cycle-guard normalization evidence

This directory preserves the compact, machine-readable result of the frozen
`prob4d-cycle-guard-normalization-v1` experiment.

The complete raw evidence remains the checksummed GitHub Actions artifact from
workflow run `30887486572`:

- artifact ID: `8883841829`;
- ZIP SHA-256: `8d8015e1bfb8aedca3f3609f5a8c2ba7674d2adaaf4d29cba400d97845430e1c`;
- report ID: `d51e4cd30115d774cd59f09fff03fe920f56ff4ecf4cacfee974822a5a88d162`;
- raw-trial CSV SHA-256: `c1684a801cae7d37729b9d1b80f2689f9d1c0ef89bddd765ecc45218dfdd7c65`.

The normalized guard retained detection of all 89 injected precise-looking biased
edges and reduced worst-clean false fallback from 29.7% to 13.3%. It failed the
preregistered absolute ceiling of 10%, so the result is retained as a valid partial
negative and the method remains experimental.

See `summary.json` for exact machine-readable values and
`docs/cycle-guard-normalization-results-v1.md` for interpretation. The production
spanning tree is unchanged.
