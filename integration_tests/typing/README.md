# Installed-wheel typing fixtures

These files validate Prob4D from the perspective of a downstream package after
an actual wheel has been installed into an isolated environment.

- `consumer_v2.py` must pass strict MyPy checking with expression-level `Any`
  rejected. It exercises the concrete return type of the claim-bearing loader and
  the annotated `Sim3` method surface.
- `consumer_v2_invalid.py` must fail with an `arg-type` diagnostic. It is a
  negative control proving that the installed `py.typed` package does not accept
  an integer where a path is required.

The fixtures are type-check inputs, not runtime examples. They import only the
stable `prob4d.api.v2` façade and must not reach into a source checkout or open
scientific data.
