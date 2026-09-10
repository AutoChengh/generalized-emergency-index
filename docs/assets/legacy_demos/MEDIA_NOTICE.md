# Historical SIND and CIMSS-TA GIFs

These two animations were downloaded unchanged on 2026-09-10 from the
[original public GEI repository](https://github.com/AutoChengh/generalized-emergency-index/tree/497b959f579321b04f178a323c4745b10a685868),
commit `497b959f579321b04f178a323c4745b10a685868`.

## Sources

- [SIND Tianjin intersection GIF](sind-tianjin-intersection-vehicle-ptw-strong-interaction.gif):
  a naturalistic vehicle-PTW interaction at a Tianjin intersection.
  [Original file](https://github.com/AutoChengh/generalized-emergency-index/blob/497b959f579321b04f178a323c4745b10a685868/assets/demos/sind-tianjin-intersection-vehicle-ptw-strong-interaction.gif).
  Upstream Git blob: `66160929ff43c681c985644d33360d01c5bcca57`.
- [CIMSS-TA Hunan collision GIF](cimss-ta-hunan-ptw-cut-in-collision.gif):
  a reconstructed PTW cut-in collision in Hunan.
  [Original file](https://github.com/AutoChengh/generalized-emergency-index/blob/497b959f579321b04f178a323c4745b10a685868/assets/demos/cimss-ta-hunan-ptw-cut-in-collision.gif).
  Upstream Git blob: `78afbe2e64f9b7e1c52638bb297fca97dae05ba7`.

The dataset and scene descriptions follow the
[original README](https://github.com/AutoChengh/generalized-emergency-index/blob/497b959f579321b04f178a323c4745b10a685868/README.md#visual-examples).
The animations are copied byte-for-byte: no frames, timing, annotations or
numerical values have been changed. The upstream Git blob hashes provide an
integrity check against the archived originals.

## Interpretation

These are historical visualizations, not outputs recomputed or validated
against the refactored package in this repository. The original labels,
formula overlays and metric conventions remain visible for archival fidelity.
They should not be taken as evidence of numerical equivalence with the current
solver, or as a new performance comparison. Consult the current README and
[numerical conventions](../../numerics.md) for the current API and definitions.
In particular, trajectory-conditioned EI and aggregated GEI are distinct;
an illustrative effective TEM/InDepth pair is not a unique aggregate output.

## Attribution and reuse

Retain the GEI project source links and the SIND/CIMSS-TA dataset attribution
when redistributing these files. The upstream repository provides an
[MIT software license](https://github.com/AutoChengh/generalized-emergency-index/blob/497b959f579321b04f178a323c4745b10a685868/LICENSE).
This archive does not assert a new or separate license for the underlying
dataset-derived content; applicable source-data rights and permissions remain
with their respective holders. The AV2-specific media notice in the sibling
`demos` directory does not apply to these two files.
