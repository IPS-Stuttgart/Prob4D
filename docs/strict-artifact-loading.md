# Strict artifact loading

Prob4D's current content-addressed calibration artifacts, portable
observation-belief archives, and schema-v4 observation-factor manifests use
portable JSON types as part of their contract. Loaders reject Python-style
coercion aliases rather than normalizing them after parsing.

The strict boundary requires:

- schema versions, counts, frame indices, dimensions, and window settings to be
  JSON integers, excluding Booleans and integral floating-point values;
- probabilities, scales, and calibration coefficients to be finite JSON numbers,
  excluding strings and Booleans;
- revisions, identifiers, repository names, array keys, and SHA-256 digests to
  already be strings;
- metadata object keys to be strings at every nesting level;
- exact schema-v1 calibration, observation-belief, and schema-v4 manifest field
  sets;
- literal JSON Booleans for security- and covariance-semantics flags;
- scalar UTF-8 JSON descriptor members in portable NPZ archives; and
- unique JSON object keys. Duplicate keys and non-finite `NaN`/`Infinity` tokens
  fail before any artifact is constructed.

Observation-belief loading validates the raw descriptor before constructing the
normalized Python value. A numeric `case_id`, Boolean `schema_version`, numeric
view name, or similar value therefore cannot be accepted merely because a
Python `str(...)` or `int(...)` conversion would produce a valid-looking
artifact.

Valid existing calibration and observation descriptors, canonical bytes,
numeric arrays, and artifact IDs are unchanged. Observation-factor schema v2
and v3 remain explicit compatibility inputs; their historical value upgrading
is not promoted into the schema-v4 contract.
