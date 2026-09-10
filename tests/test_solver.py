"""Analytic fixtures and numerical regression tests, independent of crash labels."""

import math
from dataclasses import replace

import numpy as np
import pytest

from gei import (
    ActorState,
    NumericalError,
    SolverOptions,
    Trajectory,
    compute_gei,
    compute_trajectory_ei,
)
from gei._geometry import pose_at


def following(**kwargs):
    return compute_trajectory_ei(
        [0, 3], [[0, 0, 0], [3, 0, 0]], [[5, 0, 0], [5, 0, 0]], (2, 2), (2, 2), **kwargs
    )


def test_analytic_contact_at_horizon():
    r = following()
    assert r.tem == pytest.approx(3)
    assert r.indepth == pytest.approx(2)
    assert r.ei == pytest.approx(2 / 3)


def test_safety_margin_changes_depth_not_contact():
    base = following()
    r = following(options=SolverOptions(safety_margin=0.3))
    assert r.tem == base.tem
    assert r.indepth == pytest.approx(base.indepth + 0.3)


def test_no_contact():
    r = compute_trajectory_ei(
        [0, 2], [[0, 0, 0], [2, 0, 0]], [[5, 0, 0], [5, 0, 0]], (2, 2), (2, 2)
    )
    assert r.contact_status == "no_contact"
    assert math.isinf(r.tem) and r.ei == r.indepth == 0


def test_overlap_and_touch():
    for distance in (0, 2):
        r = compute_trajectory_ei([0, 1], [[0, 0, 0]] * 2, [[distance, 0, 0]] * 2, (2, 2), (2, 2))
        assert r.contact_status == "current_overlap" and r.tem == 0 and math.isinf(r.ei)


def test_positive_separation_never_becomes_infinity():
    with pytest.raises(NumericalError, match="initial separation"):
        compute_trajectory_ei([0, 1], [[0, 0, 0]] * 2, [[2 + 1e-11, 0, 0]] * 2, (2, 2), (2, 2))


def test_contact_between_supplied_knots():
    # Both endpoint configurations are separated; contact lasts only 0.002 s.
    r = compute_trajectory_ei(
        [0, 1], [[0, 0, 0]] * 2, [[-50, 0, 0], [50, 0, 0]], (0.1, 0.1), (0.1, 0.1)
    )
    assert r.tem == pytest.approx(0.499)
    assert math.isfinite(r.ei) and r.ei > 0


def test_rotation_only_contact_uses_material_velocity():
    r = compute_trajectory_ei(
        [0, 1], [[0, 0, 0], [0, 0, math.pi / 2]], [[0, 1.5, 0]] * 2, (4, 1), (0.5, 0.5)
    )
    assert 0 < r.tem < 1
    assert math.isfinite(r.ei) and r.ei > 0
    assert r.anchor_source == "witness_instant"


@pytest.mark.parametrize("angle", [0.0, 0.7, -2.2])
def test_swap_and_rigid_transform_invariance(angle):
    a = np.array([[0.0, 0, 0], [3, 0, 0]])
    b = np.array([[5.0, 0, 0], [5, 0, 0]])
    c, s = math.cos(angle), math.sin(angle)
    rot = np.array([[c, s], [-s, c]])
    for path in (a, b):
        path[:, :2] = path[:, :2] @ rot + [123, -45]
        path[:, 2] += angle
    first = compute_trajectory_ei([0, 3], a, b, (2, 2), (2, 2))
    second = compute_trajectory_ei([0, 3], b, a, (2, 2), (2, 2))
    assert first.ei == pytest.approx(2 / 3, abs=1e-8)
    assert second.ei == pytest.approx(first.ei, abs=1e-8)


def test_current_state_and_supplied_translation_agree():
    a, b = ActorState(0, 0, 1, 0, 0, 2, 2), ActorState(5, 0, 0, 0, 0, 2, 2)
    r = compute_gei(a, b)
    assert r.gei == pytest.approx(following().ei)
    assert len(r.evaluations) == 4
    assert len({e.result.ei for e in r.evaluations}) == 1
    # Historical CV-CV branch incorrectly reported contact outside this horizon.
    assert compute_gei(a, replace(b, x=25), horizon=10).gei == 0


def test_exact_ctrv_and_zero_yaw_limit():
    a = ActorState(0, 0, 2, 0, 1, 2, 1)
    p = pose_at(a, math.pi / 2)
    assert [p.x, p.y, p.heading] == pytest.approx([2, 2, math.pi / 2])
    p = pose_at(replace(a, yaw_rate=1e-14), 1)
    assert p.x == pytest.approx(2) and abs(p.y) < 1e-12


def test_knot_heading_unwrap_and_left_derivative():
    p = Trajectory([0, 1, 2], [[0, 0, 3.1], [1, 0, -3.1], [4, 0, -3]])
    assert abs(p.pose(0.5).heading - math.pi) < 1e-12
    assert p.left_derivative(1)[0][0] == 1
    with pytest.raises(ValueError):
        p.pose(-1)
    with pytest.raises(ValueError):
        p.poses(np.array([np.nan]))
    with pytest.raises(ValueError):
        p.times[0] = 5


@pytest.mark.parametrize(
    "times,poses",
    [
        ([0, 0], [[0, 0, 0]] * 2),
        ([1, 2], [[0, 0, 0]] * 2),
        ([0, 1], [[0, 0, np.nan]] * 2),
        ([0, 1], [[0, 0]] * 2),
    ],
)
def test_bad_trajectories_rejected(times, poses):
    with pytest.raises(ValueError):
        Trajectory(times, poses)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"time_tolerance": float("nan")},
        {"min_search_step": 0},
        {"safety_margin": -1},
        {"contact_tolerance": float("inf")},
        {"max_contact_evaluations": 2},
        {"time_tolerance": 1},
    ],
)
def test_invalid_options(kwargs):
    with pytest.raises(ValueError):
        SolverOptions(**kwargs)


def test_resource_limit_raises_not_zero():
    with pytest.raises(NumericalError):
        compute_trajectory_ei(
            [0, 1],
            [[0, 0, 0], [0, 0, math.pi / 2]],
            [[0, 1.5, 0]] * 2,
            (4, 1),
            (0.5, 0.5),
            options=SolverOptions(max_contact_evaluations=10),
        )


def test_rotating_contact_refinement():
    args = ([0, 1], [[0, 0, 0], [0, 0, 1.4]], [[0, 1.5, 0]] * 2, (4, 1), (0.5, 0.5))
    a = compute_trajectory_ei(*args)
    b = compute_trajectory_ei(
        *args,
        options=SolverOptions(min_search_step=1e-5, time_tolerance=1e-10, indepth_max_step=0.002),
    )
    assert a.tem == pytest.approx(b.tem, abs=1e-8)
    assert a.ei == pytest.approx(b.ei, abs=1e-5)
