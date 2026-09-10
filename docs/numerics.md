# Numerical conventions

## One operator

For a prescribed joint future h, the solver finds the earliest oriented-body
contact within [0,T]. For 0 < TEM < infinity it returns EI = InDepth / TEM.
GEI is the weighted sum, GEI = sum_h w_h EI_h; only positive-weight futures
contribute to the sum. The contact probability uses the
same future evaluations, not an independent grid-collision implementation.

The public entry points share the solver in `src/gei/_solver.py`:

- `compute_gei` uses exact constant-speed CV/CTRV paths, four weights of 1/4,
  and a default 10 s horizon, matching main-text Section III.A.
- `compute_gei_from_futures` accepts external joint futures.
- `compute_trajectory_ei` exposes a single paired-trajectory evaluation.

## Motion and contact

External poses use linear x/y interpolation and linear unwrapped heading.
Adjacent heading changes must encode the intended shortest arc; undersampled
turns exceeding pi cannot be recovered. Derivatives at knots are taken from the
left for the pre-contact witness calculation. The solver does not estimate
headings or apply vehicle-dynamics constraints to supplied poses.

Nonrotating, piecewise-linear relative motion uses continuous separating-axis
interval intersection. Curved motion uses interval subdivision and a material-
point displacement bound. An interval may be rejected if its separating margin
exceeds the maximum possible relative displacement. Small unresolved intervals
use guard samples and bounded local minimization, followed by bisection.

This is a finite-tolerance numerical method, not an exact symbolic collision
detector or a formal no-missed-contact certificate for arbitrary curved motion.
Very short/grazing contact remains tolerance-sensitive. Decrease the search
step and time tolerance, and check convergence for demanding cases.

## Contact-witness direction and InDepth

At the final pre-contact separated pose, canonical closest points are obtained
from vertex-edge candidates. Numerically tied candidates are deduplicated and
averaged for a deterministic contact-set representative.

Each witness velocity includes centre translation and the rotational term
omega J(q - P). Their relative velocity defines the approach direction. If
degenerate, the solver tries stable backward differences of the same material
points, then a separating-axis normal. The chosen source is returned.

The perpendicular projection direction remains fixed over [0,TEM]. At each
instant, intrusion is the sum of projected half-extents minus projected centre
separation, plus `safety_margin`. InDepth is its non-negative maximum.
The search includes trajectory knots, the contact time, a refined time grid
and local refinement of sampled maxima.

`safety_margin` is an additive intrusion term. It does **not** inflate the
bodies or change TEM/contact probability. The paper's primary setting is zero.

## Boundaries and errors

- Current geometric contact/overlap: TEM = 0, EI = infinity, InDepth undefined.
- No future contact: TEM = infinity, InDepth = EI = 0.
- Finite future contact: EI must be finite; nonfinite results raise
  `NumericalError` and are not capped, discarded or replaced by zero.
- Positive initial separation below contact tolerance raises an error rather
  than fabricating current overlap.
- If contact probability is zero, conditional EI is undefined (`None`).
- Zero-weight futures are input-validated but not solved.

All futures must have the same current poses, body sizes and time grid.
Weights are not silently normalized; normalization requires an explicit option.
Use coordinates local to the scene to avoid subtracting unnecessarily large
numbers. Geometric tolerance does not address uncertainty in measured states.

## Default numerical controls

| Control | Default | Purpose |
|---|---:|---|
| contact_tolerance | 1e-10 m | Future-contact numerical band |
| min_search_step | 1e-4 s | Curved-contact leaf interval |
| time_tolerance | 1e-9 s | Contact-time bisection |
| indepth_max_step | 0.01 s | Initial intrusion grid spacing |
| safety_margin | 0 m | Additive projected intrusion |
| max_contact_evaluations | 250000 | Explicit failure instead of unbounded search |

No general convergence rate, hard real-time bound or prediction-accuracy
guarantee is claimed. These controls must be recorded with experiment outputs.
External forecasts define their own horizon through their final time sample;
the paper's stochastic and learned forecasts use 3 s. Horizon is part of the
risk definition, not a hidden performance knob.
