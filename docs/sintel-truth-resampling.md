# Sintel truth resampling

The Sintel uncertainty diagnostic sometimes resizes camera-space 3-D truth maps
to the MotionCrafter output resolution. Invalid points must not be converted to
zero coordinates and then mixed into valid bilinear samples.

## Mask-normalized interpolation

For a point field `x` and validity mask `m`, Prob4D now computes

```text
numerator = bilinear_resize(m * x)
support   = bilinear_resize(m)
x_resized = numerator / support
```

where division occurs only for positive support. An output row is valid only
when `support >= minimum_resize_support`; rejected rows are stored as zero and
remain excluded by the returned validity mask. The default threshold is `0.5`
and is an explicit argument of `load_sintel_truth`.

This prevents invalid zero or non-finite coordinates from biasing otherwise
supported 3-D points while retaining an auditable support rule. Same-resolution
loads preserve valid values exactly and zero invalid storage rows.

## Scope

The change affects only Sintel truth loading for the diagnostic uncertainty
analysis. It does not change MotionCrafter predictions, provider-v2 export,
gauge fitting, the predeclared family split, or any frozen evidence artifact.
Previously generated Sintel result tables should retain their original code
revision; new runs must record the updated revision and support threshold.
