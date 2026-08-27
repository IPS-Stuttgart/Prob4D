# CUT3R compiled-RoPE runtime contract

## Purpose

The terminal `cut3r-source-comparison-smoke-v1-2` result must not be retried. It entered CUT3R inference and terminated with a CUDA device-side assertion before writing a prediction. This contract defines a **new prospective provider-runtime identity** that fails closed unless the pinned CUT3R checkout imports its compiled CUDA RoPE kernel.

CUT3R's installation instructions require building `src/croco/models/curope` with `python setup.py build_ext --inplace`. When that extension is unavailable, CUT3R permits a Python RoPE fallback. The fallback is not admissible for Prob4D provider qualification because the same failure signature has been reported upstream around its positional embedding lookup.

The historical `run_cut3r_source_comparison.py` entry point is intentionally left unchanged. Any separately preregistered successor route must use `run_cut3r_source_comparison_native_rope.py`, so the terminal v1.2 implementation remains byte-for-byte attributable to its frozen result.

## Contract

`require_compiled_cut3r_rope()` is called before the delegated executor constructs `dust3r.model`. It requires all of the following:

1. the native extension is importable;
2. the imported module is a platform extension, not `curope.py`;
3. the extension resides inside the pinned CUT3R `src/croco/models/curope` tree;
4. it exposes the required `rope_2d` symbol;
5. the extension binary and all relevant source members are hashed into a self-authenticating receipt.

The receipt contains only checkout-relative paths. Its `artifact_id` must be frozen into any successor source-comparison plan together with the CUT3R revision, checkpoint digest, PyTorch/CUDA inventory, compiler identity, and Prob4D implementation revision.

## Preparation

From an installed Prob4D checkout:

```bash
python scripts/science/prepare_cut3r_runtime.py \
  --cut3r-checkout /path/to/CUT3R \
  --build \
  --output /sealed/runtime/cut3r-compiled-rope-receipt.json
```

Run the provider process freshly after compilation. For a frozen receipt, repeat verification without `--build` and with `--expected-artifact-id`.

A separately registered successor execution uses:

```bash
python scripts/science/run_cut3r_source_comparison_native_rope.py \
  <the exact arguments frozen by the successor plan>
```

This entry point delegates to the established source-comparison executor only after replacing its runtime constructor with one that verifies and content-binds the native RoPE extension. It does not make the consumed v1.2 smoke executable again and does not itself authorize source or target access.

## Admissible next experiment

The successor route is not a retry of v1.2. It must use a new plan version and a fresh registered development case. Before opening any scientific outcomes:

1. freeze the compiled-RoPE receipt and runtime inventory;
2. run a technical-only CUT3R example capsule to establish forward-pass compatibility;
3. discard example predictions and retain only stage/provenance evidence;
4. prospectively register the fresh source-only cohort and exact continuous-versus-restarted comparison;
5. permit target/BayesianPhysTwin/Causal4D execution only after ordinary source success and custody verification.

The scientific endpoint should not be another isolated reconstruction score. The paper-strength experiment compares, on identical frames and provider weights:

- native continuous CUT3R state;
- restarted CUT3R windows with Prob4D gauge-aware probabilistic fusion;
- exact BayesianPhysTwin fallback.

Primary outcomes are downstream physical action error or regret, harmful accepted-update frequency with the already-frozen exact certificate, fallback rate, and support/identifiability diagnostics. Point error and calibration are mechanism diagnostics, not the headline claim.
