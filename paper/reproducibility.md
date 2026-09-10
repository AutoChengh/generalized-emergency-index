# Paper-to-code guide

The design reference is **Generalized Emergency Index: A Multi-Trajectory Risk
Quantification Framework for Vehicle-Powered Two-Wheeler Interactions**,
GEI_TITS_v1_6 (13-page snapshot checked on 2026-09-09). Source PDF SHA-256:
`e7850f61e6ee497ef7aa1e4e7ca3b8f2e41460e7caef68c50e1cd1104e36ac97`.
Equation, section and figure numbers below refer to that snapshot.

## Architecture follows the definition

There is one risk operator, not separate implementations of default GEI,
GEI (EMP-D), and GEI (QCNet). Only the supplied future bodies and weights differ.

| Paper component | Implementation | Verification |
|---|---|---|
| Equations (1)-(6): prescribed centre/heading paths and rectangular bodies | `Trajectory`, `BodySize`; analytic default paths | Interpolation, dimensions, and rigid-transform tests |
| Equations (7)-(10): first contact and boundaries | Contact search in `src/gei/_solver.py` | Analytic translation, between-knot contact, rotation-only contact, and dense-polygon comparison |
| Equations (11)-(16): contact-witness material velocity | `_geometry.py` and `_solver.py` | Rotation contribution, actor exchange, and degeneracy tests |
| Equations (17)-(22): fixed-axis pre-contact maximum and EI | Shared trajectory EI evaluator | Known InDepth/EI and additive-clearance tests |
| Equations (23)-(25): weighting and decomposition | `compute_gei_from_futures` | Weight checks, zero-weight handling, and decomposition tests |
| Equations (26)-(27): default CV/CTRV | `compute_gei`, `default_futures` | Stable analytic CTRV and zero-yaw collapse |
| Section III.G: learned future inputs | EMP-D/QCNet output adapters | Coordinates, six-mode retention, logits, and heading construction |

The input adapter is outside the risk kernel. The kernel must not infer labels,
choose a predictor, alter weights to obtain a desired score, or silently append
CV/CTRV modes. Direct joint futures may encode dependence; the product of actor
marginals is an explicit optional assumption.

The default horizon is 10 s (Section III.A). Stochastic and learned experiments
use 3 s. These are different protocols, not interchangeable default settings.

## Experiment correspondence

| Main-text evidence | Protocol to preserve | Release status |
|---|---|---|
| Fig. 2 / Table I: default discrimination | 1766 negative event maxima; up to 1515 positive frames per window | Original baseline suite/data not bundled |
| Fig. 3 / Table II: warning timeliness | Percentile thresholds and final sustained-warning segment | Supplement S4; not the generic discrimination script |
| Table III: identical-future aggregation | 1716 negative event maxima; five pooled frames per crash/window; 128 joint futures; 3 s | Bank evaluator and paired cluster-bootstrap analyzer supplied; historical scores not regenerated |
| Figs. 4-5: conditional-demand separation | Frozen held-out pool, independent event matching, separate Monte Carlo reruns | Supplement S5; fitting/matching pipeline not bundled |
| Fig. 6: retained outcome information | Aligned 303/1605 cohort; training-fold isotonic regression | Supplement S6; distinct from classification AUPRC |
| Fig. 7: AV2 deployment statistics; Fig. 8: case AV2S-PTW-0275 | Same 584-event / 53674-pair-frame registry; six modes per actor; partial-history masks | Output adapters supplied; full scene encoders and archived timing engine not bundled |

The P-MC-inspired and Schreier-inspired samplers in Table III are method-level
implementations, not exact reproductions of the cited predictors. The fitted
five-mode sampler in Section III.E is another experiment; it must not be
confused with the Table III generator or either learned predictor. See the
paper's references [30] (P-MC), [31] (Schreier), [33] (EMP) and [34] (QCNet).
The README additionally shows case AV2S-PTW-0334 from the archived AV2 results;
it is not a second main-text case figure in this 13-page snapshot.

## Accompanying supplement

[Supplementary material](supplementary_material.pdf) contains the extended
definitions, consistency proofs, numerical procedures and experimental details.
It is preserved from the earlier v1.6 manuscript snapshot. PDF SHA-256:
`dcc0194861529e58254238de3a9ceee40e67422b722168a8170519b8453fcd82`.

This 10-page supplement retains its original scientific text and results.
Its printed main-text figure references use the earlier 14-page snapshot
(`34f9e3cc1e1cde17b4921e11533765b8ba93e4327e1bff71966b46857421e821`),
not the current numbering above. Match references by subject: old Figs. 5-6
correspond to current Figs. 4-5, old Fig. 7 to current Fig. 6, and old Fig. 8
to current Fig. 7. The case figures were also reorganized. Embedded links to
`GEI_TITS.pdf` require that earlier manuscript, which is not bundled here.
The supplement must not be described as a cross-reference-synchronized export
of the current 13-page manuscript.

| Supplement | Role in this codebase |
|---|---|
| S1-S2 | Weighted-future definition; default CV/CTRV; aggregation consistency |
| S3 | Numerical definitions implemented and documented in docs/numerics.md |
| S4-S6 | Historical warning, probabilistic and retained-information experiments |
| S7-S8 | Historical timing and AV2 log-replay evidence, not new-version benchmarks |

## Reproducibility boundaries

The refactored core is a numerical revision. Passing analytical tests is not
evidence that it reproduces every historical dataset result bit for bit.
Do not attach the paper's AUPRC or runtime values to this version without
rerunning the corresponding frozen inputs.

The repository currently provides:

- Deterministic, self-contained examples and numerical regression tests.
- EMP-D/QCNet **output** adapters, with no bundled pretrained network.
- An immutable future-bank evaluator returning both contact probability and GEI.
- Paired cluster-bootstrap AUPRC analysis with the Table III pooled-frame protocol.

The dataset-specific full paper pipelines, trained predictor states and
original future banks are not bundled. This package is a reusable reference
implementation and protocol scaffold, not a complete paper-reproduction archive.
Historical experiments must retain their own solver versions, seeds, splits
and manifests; replacing their imports with the new core changes the experiment.

## Required evidence for full paper reproduction

For each published figure or table, record:

1. Source dataset and access instructions; event selection and exclusion rules.
2. Immutable event/frame registry and train/validation/test partition.
3. Future-generator configuration, checkpoint fingerprint and joint weights.
4. Heading construction, coordinate transforms and historical observation masks.
5. GEI source fingerprint, numerical controls and prediction horizon.
6. Scoring/aggregation protocol, primary metric and uncertainty procedure.
7. Output completeness audit, numerical failures, and figure-generation command.

For the archived AV2 replay, the registry contains 584 events and 53674
pair-frames in 278 motorcycle-containing logs. The moving-PTW mean-speed filter
(>0.5 m/s) is a visualization-selection rule, not a scoring exclusion.
The original full replay and its GPU/CPU timings have not been rerun as part
of this package refactoring.

The generic experiment scripts do not verify a caller's omitted raw frames.
Compare their keys against the source registry before making completeness
claims. Likewise, a synthetic adapter example is not evidence of neural
inference accuracy or model deployment.
