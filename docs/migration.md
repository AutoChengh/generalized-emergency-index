# Migration from the older public repository

This is an API and numerical revision, not a cosmetic metric rename.

| Earlier entry/output | New entry/output |
|---|---|
| compute_single_frame / 14 positional values | ActorState + compute_gei |
| Separate probability scripts | JointFuture + compute_gei_from_futures |
| TEM_eff / InDepth_eff | Per-future TEM / InDepth; no aggregate counterpart |
| GEI_2x2 and GEI_prob labels in research scripts | GEI; label the future generator separately |
| gei frame / csv / batch | gei input.json; explicit upstream dataset processing |
| gei-gif / gif_maker.py | Not carried over; old renderer used the old score semantics |

The new core uses contact-witness material velocities, including rotation.
Default CV/CTRV uses continuous contact search and honours the same prediction
horizon for every combination. External pose input uses the same kernel.
Old numerical outputs and speed measurements must not be presented as a
validation or benchmark of this version.

The revision is based on public commit
`497b959f579321b04f178a323c4745b10a685868`. Earlier source and assets remain in
the Git history. Legacy caches, renderers and data samples are not part of the
new package. The README's archived AV2 replays are separately labelled and
licensed; they are not benchmarks of this numerical revision.

## Numerical source lineage

The geometry and interpolated-trajectory solver were refactored from the
research package GEI code v1.2. Source fingerprints at import:

- geometry.py: `7eab881eb5399fec555ad6ef66f18231f32c566787bece234bd109ffd8f312f8`
- trajectory_ei.py: `ad3de05a8335a0c40a3d439bb4c675b0fe9174fca56dc3b82c9fb7f6f7f50b0e`

Changes include a common analytic/interpolated path interface, horizon-limited
continuous translation contact, stable analytic CTRV propagation, material-point
finite-difference fallback, numerical input validation, explicit search-budget
failure, and an additive intrusion margin rather than body inflation.
Regression tests document supported behaviours; they do not substitute for
re-running historical dataset experiments.
