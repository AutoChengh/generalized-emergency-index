"""The common GEI operator and its current-state convenience entry point."""

import math
from dataclasses import dataclass

import numpy as np

from ._solver import TrajectoryEIResult, _size, evaluate_pair
from .futures import JointFuture, default_futures
from .models import ActorState, SolverOptions


@dataclass(frozen=True)
class FutureEvaluation:
    label: str
    weight: float
    result: TrajectoryEIResult


@dataclass(frozen=True)
class GEIResult:
    """GEI in m/s, model-implied contact probability, and per-future diagnostics.

    conditional_ei is undefined (None) when no future has positive contact
    probability. No aggregate 'effective TEM' or 'effective InDepth' is defined.
    """

    gei: float
    contact_probability: float
    conditional_ei: float | None
    status: str
    evaluations: tuple[FutureEvaluation, ...]


def _weights(values, normalize):
    weights = np.array(values, dtype=float, copy=True)
    if weights.ndim != 1 or len(weights) == 0:
        raise ValueError("weights must be a nonempty vector")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("weights must be finite and non-negative")
    total = float(np.sum(weights))
    if not math.isfinite(total) or total <= 0:
        raise ValueError("weights must have a positive finite sum")
    if normalize:
        weights /= total
    elif not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError("weights must sum to 1; set normalize_weights=True explicitly if intended")
    return weights


def compute_gei_from_futures(
    futures: list[JointFuture],
    size_a,
    size_b,
    *,
    normalize_weights: bool = False,
    options: SolverOptions = SolverOptions(),
) -> GEIResult:
    """Compute GEI = sum(w_h EI_h) from externally supplied joint futures.

    All futures must share the current poses and a common time grid. Zero-weight
    hypotheses are validated but not solved. Numerical failures raise rather
    than being silently discarded, zero-filled or reweighted.
    """
    futures = tuple(futures)
    weights = _weights([f.weight for f in futures], normalize_weights)
    parsed_a, parsed_b = _size(size_a), _size(size_b)
    reference = futures[0]
    for f in futures:
        for path, ref in (
            (f.trajectory_a, reference.trajectory_a),
            (f.trajectory_b, reference.trajectory_b),
        ):
            if not np.array_equal(path.times, ref.times):
                raise ValueError("all futures must share a common time grid")
            p, r = path.pose(0.0), ref.pose(0.0)
            angle_error = math.atan2(
                math.sin(p.heading - r.heading), math.cos(p.heading - r.heading)
            )
            if not np.allclose(p.center, r.center, rtol=0, atol=1e-8) or abs(angle_error) > 1e-8:
                raise ValueError("all futures must share the current pose of each actor")
    evaluations = []
    for index, (future, weight) in enumerate(zip(futures, weights)):
        if weight == 0:
            continue
        result = evaluate_pair(
            future.trajectory_a, future.trajectory_b, parsed_a, parsed_b, options
        )
        evaluations.append(
            FutureEvaluation(future.label or f"future-{index}", float(weight), result)
        )
    gei = math.fsum(e.weight * e.result.ei for e in evaluations)
    probability = math.fsum(
        e.weight for e in evaluations if e.result.contact_status != "no_contact"
    )
    conditional = gei / probability if probability > 0 else None
    status = (
        "current_overlap"
        if math.isinf(gei)
        else ("finite_contact" if probability else "no_contact")
    )
    return GEIResult(gei, probability, conditional, status, tuple(evaluations))


def compute_gei(
    actor_a: ActorState,
    actor_b: ActorState,
    *,
    horizon: float = 10.0,
    options: SolverOptions = SolverOptions(),
) -> GEIResult:
    """Default GEI: four equal-weight CV/CTRV combinations over 10 s.

    Set horizon explicitly for other protocols, such as a 3 s forecast bank.
    The common solver is independent of the choice of future generator.
    """
    return compute_gei_from_futures(
        default_futures(actor_a, actor_b, horizon), actor_a.size, actor_b.size, options=options
    )
