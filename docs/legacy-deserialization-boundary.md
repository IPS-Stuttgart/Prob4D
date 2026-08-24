# Legacy deserialization boundary

Portable Prob4D artifacts do not use pickle. Historical PhysTwin inputs still
contain NumPy arrays and primitive containers serialized with pickle, so the
legacy dataset adapter admits only the minimal NumPy reconstruction globals
through a restricted unpickler. Arbitrary imported Python globals, symbolic-link
substitution, malformed payloads, and unexpected final-data container types fail
closed.

The PhysTwin experiment and state diagnostics use the same restricted loader for
manual tracks, simulator trajectories, and final-data metadata. They are
retrospective diagnostic paths, not an alternate provider artifact format.

A repository policy test scans current runtime and script sources for direct
`pickle.load`, `pickle.loads`, and NumPy loads that can enable pickle. It admits
one exact exception:

```text
scripts/science/build_cut3r_deform360_source_freeze.py
```

That exception belongs to the frozen CUT3R Deform360 source-freeze-v1 protocol.
The historical revision and its terminal execution record remain immutable. The
exception must not be copied, generalized, or silently repaired under the same
protocol identity.

A future source-freeze version must convert the official camera-name mappings in
an outcome-blind, credential-free preprocessing boundary into non-object NPZ
arrays plus canonical JSON names, bind the input and output byte digests in a
conversion receipt, and consume the converted files with `allow_pickle=False`.
Changing that input representation requires a separately reviewed protocol and
execution identity before source outcomes are opened.
