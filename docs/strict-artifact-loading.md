# Strict artifact loading

Prob4D's current content-addressed calibration artifacts and schema-v4
observation-factor manifests use portable JSON types as part of their contract.
Loaders reject Python-style coercion aliases rather than normalizing them after
parsing.

The strict boundary requires:

- schema versions, counts, frame indices, dimensions, and window settings to be
  JSON integers, excluding Booleans and integral floating-point values;
- probabilities, scales, and calibration coefficients to be finite JSON numbers,
  excluding strings and Booleans;
- revisions, identifiers, repository names, array keys, and SHA-256 digests to
  already be strings;
- metadata object keys to be strings at every nesting level;
- exact schema-v1 calibration and schema-v4 manifest field sets;
- literal JSON Booleans for security- and covariance-semantics flags; and
- unique JSON object keys. Duplicate keys and non-finite `NaN`/`Infinity` tokens
  fail before any artifact is constructed.

Valid existing calibration descriptors, canonical bytes, and artifact IDs are
unchanged. Observation-factor schema v2 and v3 remain explicit compatibility
inputs; their historical value upgrading is not promoted into the schema-v4
contract.
