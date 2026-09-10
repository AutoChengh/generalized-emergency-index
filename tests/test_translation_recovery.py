"""Independent closed-form nonrotating reference for Proposition 2.

No package geometry, propagation, or pair-level solver functions are used to
construct the reference. The tests also vary the common rigid coordinate frame.
"""

import math

import numpy as np
import pytest

from gei import ActorState, SolverOptions, compute_gei


def translation_reference(a, b, horizon, margin):
    def orientation(actor):
        c, s = math.cos(actor.heading), math.sin(actor.heading)
        return np.array([[c, s], [-s, c]])

    aa, bb = orientation(a), orientation(b)
    delta = np.array([b.x - a.x, b.y - a.y])
    velocity = b.speed * bb[0] - a.speed * aa[0]

    def extent(axis):
        return (
            a.length * abs(aa[0] @ axis)
            + a.width * abs(aa[1] @ axis)
            + b.length * abs(bb[0] @ axis)
            + b.width * abs(bb[1] @ axis)
        ) / 2

    low, high = 0.0, horizon
    for axis in np.vstack([aa, bb]):
        radius, offset, rate = extent(axis), float(delta @ axis), float(velocity @ axis)
        if abs(rate) < 1e-14:
            if abs(offset) > radius:
                return math.inf, 0.0, 0.0
        else:
            bounds = sorted([(-radius - offset) / rate, (radius - offset) / rate])
            low, high = max(low, bounds[0]), min(high, bounds[1])
            if low > high:
                return math.inf, 0.0, 0.0
    if low == 0:
        return 0.0, math.nan, math.inf
    tangent = np.array([-velocity[1], velocity[0]]) / np.linalg.norm(velocity)
    # In pure relative translation its tangent displacement is constant.
    depth = max(0.0, extent(tangent) - abs(delta @ tangent) + margin)
    return low, depth, depth / low


@pytest.mark.parametrize("seed", range(24))
def test_zero_yaw_recovers_independent_translation_reference(seed):
    rng = np.random.default_rng(seed)
    a = ActorState(0, 0, rng.uniform(1, 10), rng.uniform(-math.pi, math.pi), 0, 4.5, 1.9)
    b = ActorState(
        *rng.uniform(-10, 10, 2),
        rng.uniform(0, 8),
        rng.uniform(-math.pi, math.pi),
        0,
        2.1,
        0.8,
    )
    margin = 0.2 if seed % 2 else 0.0
    tem, depth, ei = translation_reference(a, b, 10, margin)
    result = compute_gei(a, b, options=SolverOptions(safety_margin=margin))
    assert result.gei == pytest.approx(ei, rel=2e-7, abs=2e-7)
    for future in result.evaluations:
        assert future.result.tem == pytest.approx(tem, rel=2e-7, abs=2e-7)
        if math.isfinite(depth):
            assert future.result.indepth == pytest.approx(depth, rel=2e-7, abs=2e-7)


def test_default_horizon_covers_contact_after_three_seconds():
    a = ActorState(0, 0, 1, 0, 0, 2, 2)
    b = ActorState(7, 0, 0, 0, 0, 2, 2)
    assert compute_gei(a, b).gei == pytest.approx(2 / 5)
    assert compute_gei(a, b, horizon=3).gei == 0
