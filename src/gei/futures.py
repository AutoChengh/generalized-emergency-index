"""Analytic default futures and explicit weighted future containers."""

import math
from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from ._geometry import center_velocity, pose_at
from ._solver import Trajectory, _Pose
from .models import ActorState


class _KinematicTrajectory(Trajectory):
    """Exact CV/CTRV; knots partition search, not the motion itself."""

    def __init__(self, actor, times):
        self.actor = actor
        samples = [pose_at(actor, float(t)) for t in times]
        super().__init__(times, [[p.x, p.y, p.heading] for p in samples])

    def pose(self, time_s, segment=None):
        self._segment(time_s)
        p = pose_at(self.actor, time_s)
        return _Pose(np.array([p.x, p.y]), p.heading)

    def poses(self, times):
        states = [self.pose(float(t)) for t in times]
        return np.array([s.center for s in states]), np.array([s.heading for s in states])

    def left_derivative(self, time_s):
        self._segment(time_s)
        return center_velocity(self.actor, time_s), self.actor.yaw_rate

    def motion_bound(self, segment, duration, radius):
        return (self.actor.speed + radius * abs(self.actor.yaw_rate)) * duration

    def is_translation(self, segment):
        return self.actor.yaw_rate == 0.0


@dataclass(frozen=True)
class JointFuture:
    """One prescribed pair and its non-negative joint weight."""

    trajectory_a: Trajectory
    trajectory_b: Trajectory
    weight: float
    label: str = ""

    def __post_init__(self):
        if not isinstance(self.trajectory_a, Trajectory) or not isinstance(
            self.trajectory_b, Trajectory
        ):
            raise TypeError("joint futures require Trajectory objects")
        if not math.isfinite(self.weight) or self.weight < 0:
            raise ValueError("joint weight must be finite and non-negative")
        if not np.array_equal(self.trajectory_a.times, self.trajectory_b.times):
            raise ValueError("both actors must use the same time grid")


def independent_joint_futures(
    trajectories_a: Sequence[Trajectory],
    weights_a: Sequence[float],
    trajectories_b: Sequence[Trajectory],
    weights_b: Sequence[float],
) -> list[JointFuture]:
    """Cartesian product with w_ij = q_i p_j; explicitly assumes independence.

    Each marginal distribution must already sum to one. Interaction-aware
    marginal predictions do not, by themselves, supply a joint distribution.
    """
    from .api import _weights

    if len(trajectories_a) != len(weights_a) or len(trajectories_b) != len(weights_b):
        raise ValueError("one weight is required for each trajectory")
    qa, qb = _weights(weights_a, False), _weights(weights_b, False)
    return [
        JointFuture(a, b, float(qa[i] * qb[j]), f"A{i}/B{j}")
        for i, a in enumerate(trajectories_a)
        for j, b in enumerate(trajectories_b)
    ]


def default_futures(a: ActorState, b: ActorState, horizon: float) -> list[JointFuture]:
    """Four equal-weight combinations of exact CV and CTRV motion."""
    if not math.isfinite(horizon) or horizon <= 0:
        raise ValueError("horizon must be finite and positive")
    # Bound angular change per search interval, without changing the yaw rate.
    step = min(0.1, 0.2 / max(abs(a.yaw_rate), abs(b.yaw_rate), 1e-12))
    count = math.ceil(horizon / step)
    if count > 1_000_000:
        raise ValueError("requested horizon/yaw rate creates too many search intervals")
    times = np.linspace(0.0, horizon, count + 1)
    modes_a = [
        _KinematicTrajectory(replace(a, yaw_rate=0.0), times),
        _KinematicTrajectory(a, times),
    ]
    modes_b = [
        _KinematicTrajectory(replace(b, yaw_rate=0.0), times),
        _KinematicTrajectory(b, times),
    ]
    names = ("CV", "CTRV")
    return [
        JointFuture(pa, pb, 0.25, f"{names[i]}/{names[j]}")
        for i, pa in enumerate(modes_a)
        for j, pb in enumerate(modes_b)
    ]
