"""Model-agnostic trajectory-conditioned EI.

The kernel evaluates pose trajectories through a common path interface. It has
no knowledge of predictors, maneuver classes, or probabilities; analytic paths
can override the sampled path's interpolation and motion-bound operations.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Optional, Sequence

import numpy as np
from scipy.optimize import minimize_scalar

from ._geometry import (
    active_separating_direction,
    body_corners,
    closest_witness_pair,
    indepth_along_direction,
    rotation,
    sat_gap,
    unit,
)
from .models import BodySize, NumericalError, SolverOptions


@dataclass(frozen=True)
class TrajectoryEIResult:
    tem: float
    indepth: float
    ei: float
    contact_status: str
    anchor_source: str
    anchor_direction_x: float
    anchor_direction_y: float
    witness_distance: float
    contact_search_evaluations: int
    indepth_evaluations: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class _Pose:
    center: np.ndarray
    heading: float


class Trajectory:
    """Future poses on a shared metric frame, including the current pose at t=0.

    Input shape is (T, 3), columns x, y, heading. Positions and unwrapped
    headings are interpolated linearly. Adjacent heading changes must represent
    the intended shortest arc (less than pi). Arrays are copied and read-only.
    """

    def __init__(self, times: Sequence[float], trajectory: Sequence[Sequence[float]]):
        self.times = np.array(times, dtype=float, copy=True)
        values = np.asarray(trajectory, dtype=float)
        if self.times.ndim != 1 or self.times.size < 2:
            raise ValueError("times must be one-dimensional with at least two entries")
        if values.shape != (self.times.size, 3):
            raise ValueError(f"trajectory must have shape ({self.times.size}, 3)")
        if not np.all(np.isfinite(self.times)) or not np.all(np.isfinite(values)):
            raise ValueError("times and trajectory values must be finite")
        if np.any(np.diff(self.times) <= 0.0):
            raise ValueError("times must be strictly increasing")
        if abs(float(self.times[0])) > 1.0e-12:
            raise ValueError("times must start at zero")
        self.centers = values[:, :2].copy()
        self.headings = np.unwrap(values[:, 2]).astype(float)
        duration = np.diff(self.times)
        self.center_slopes = np.diff(self.centers, axis=0) / duration[:, None]
        self.heading_slopes = np.diff(self.headings) / duration
        if not np.all(np.isfinite(self.center_slopes)) or not np.all(
            np.isfinite(self.heading_slopes)
        ):
            raise ValueError("trajectory derivatives must be finite")
        for array in (
            self.times,
            self.centers,
            self.headings,
            self.center_slopes,
            self.heading_slopes,
        ):
            array.setflags(write=False)

    @property
    def end_time(self) -> float:
        return float(self.times[-1])

    def _segment(self, time_s: float, *, left: bool = False) -> tuple[int, float]:
        if not math.isfinite(time_s) or time_s < -1e-12 or time_s > self.end_time + 1e-12:
            raise ValueError("query time lies outside the trajectory")
        time_s = float(np.clip(time_s, self.times[0], self.times[-1]))
        side = "left" if left else "right"
        index = int(np.searchsorted(self.times, time_s, side=side) - 1)
        index = min(max(index, 0), self.times.size - 2)
        alpha = (time_s - self.times[index]) / (self.times[index + 1] - self.times[index])
        return index, float(np.clip(alpha, 0.0, 1.0))

    def pose(self, time_s: float, segment: Optional[int] = None) -> _Pose:
        if segment is None:
            segment, alpha = self._segment(time_s)
        else:
            alpha = (float(time_s) - self.times[segment]) / (
                self.times[segment + 1] - self.times[segment]
            )
            alpha = float(np.clip(alpha, 0.0, 1.0))
        center = (1.0 - alpha) * self.centers[segment] + alpha * self.centers[segment + 1]
        heading = (1.0 - alpha) * self.headings[segment] + alpha * self.headings[segment + 1]
        return _Pose(np.asarray(center, dtype=float), float(heading))

    def poses(self, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        query = np.asarray(times, dtype=float)
        if query.ndim != 1 or not np.all(np.isfinite(query)):
            raise ValueError("query times must be a finite vector")
        if np.any(query < 0) or np.any(query > self.end_time):
            raise ValueError("query time lies outside the trajectory")
        indices = np.searchsorted(self.times, query, side="right") - 1
        indices = np.clip(indices, 0, self.times.size - 2)
        alpha = (query - self.times[indices]) / (self.times[indices + 1] - self.times[indices])
        centers = (1.0 - alpha[:, None]) * self.centers[indices] + alpha[:, None] * self.centers[
            indices + 1
        ]
        headings = (1.0 - alpha) * self.headings[indices] + alpha * self.headings[indices + 1]
        return centers, headings

    def left_derivative(self, time_s: float) -> tuple[np.ndarray, float]:
        index, _ = self._segment(time_s, left=True)
        return self.center_slopes[index].copy(), float(self.heading_slopes[index])

    def body_point(self, local_point: np.ndarray, time_s: float) -> np.ndarray:
        pose = self.pose(time_s)
        return pose.center + rotation(pose.heading) @ np.asarray(local_point, dtype=float)

    def material_point_velocity(self, local_point: np.ndarray, time_s: float) -> np.ndarray:
        pose = self.pose(time_s)
        center_velocity, yaw_rate = self.left_derivative(time_s)
        offset = self.body_point(local_point, time_s) - pose.center
        return center_velocity + np.asarray([-yaw_rate * offset[1], yaw_rate * offset[0]])

    def motion_bound(self, segment: int, duration: float, radius: float) -> float:
        translation = float(np.linalg.norm(self.center_slopes[segment])) * duration
        rotation_motion = radius * abs(float(self.heading_slopes[segment])) * duration
        return translation + rotation_motion

    def is_translation(self, segment: int) -> bool:
        return self.heading_slopes[segment] == 0.0


@dataclass(frozen=True)
class _Contact:
    tem: float
    tau_minus: float
    tau_plus: float
    status: str
    evaluations: int


class _ContactSearch:
    def __init__(
        self,
        path_a: Trajectory,
        path_b: Trajectory,
        size_a: BodySize,
        size_b: BodySize,
        contact_tol: float,
        min_step: float,
        bisect_tol: float,
        max_evaluations: int,
    ):
        self.path_a = path_a
        self.path_b = path_b
        self.size_a = size_a
        self.size_b = size_b
        self.contact_tol = float(contact_tol)
        self.min_step = float(min_step)
        self.bisect_tol = float(bisect_tol)
        self.max_evaluations = max_evaluations
        self.evaluations = 0
        self._cache: dict[float, float] = {}

    def gap(self, time_s: float, segment: Optional[int] = None) -> float:
        key = float(time_s)
        if key not in self._cache:
            if self.evaluations >= self.max_evaluations:
                raise NumericalError("contact search exceeded its evaluation budget")
            pose_a = self.path_a.pose(key, segment)
            pose_b = self.path_b.pose(key, segment)
            self._cache[key] = sat_gap(
                pose_a.center,
                pose_a.heading,
                self.size_a,
                pose_b.center,
                pose_b.heading,
                self.size_b,
            )
            if not math.isfinite(self._cache[key]):
                raise NumericalError("nonfinite separating-axis geometry")
            self.evaluations += 1
        return self._cache[key]

    def overlap(self, time_s: float, segment: Optional[int] = None) -> bool:
        return self.gap(time_s, segment) <= self.contact_tol

    def _bisect_entry(self, segment: int, low: float, high: float) -> tuple[float, float]:
        if self.overlap(low, segment):
            return low, low
        for _ in range(80):
            if high - low <= self.bisect_tol:
                break
            middle = 0.5 * (low + high)
            if self.overlap(middle, segment):
                high = middle
            else:
                low = middle
        return low, high

    def _motion_certificate(
        self, segment: int, low: float, middle: float, high: float, middle_gap: float
    ) -> bool:
        move_a = max(
            self.path_a.motion_bound(segment, middle - low, self.size_a.radius),
            self.path_a.motion_bound(segment, high - middle, self.size_a.radius),
        )
        move_b = max(
            self.path_b.motion_bound(segment, middle - low, self.size_b.radius),
            self.path_b.motion_bound(segment, high - middle, self.size_b.radius),
        )
        return middle_gap > move_a + move_b + self.contact_tol

    def _leaf(self, segment: int, low: float, high: float) -> Optional[tuple[float, float]]:
        points = np.linspace(low, high, 9)
        gaps = [self.gap(float(point), segment) for point in points]
        for index in range(1, len(points)):
            if gaps[index] <= self.contact_tol:
                return self._bisect_entry(segment, float(points[index - 1]), float(points[index]))
        minimum = minimize_scalar(
            lambda value: self.gap(float(value), segment),
            bounds=(low, high),
            method="bounded",
            options={"xatol": max(1.0e-12, 0.05 * self.bisect_tol)},
        )
        if minimum.success and float(minimum.fun) <= self.contact_tol:
            return self._bisect_entry(segment, low, float(minimum.x))
        return None

    def _search(self, segment: int, low: float, high: float) -> Optional[tuple[float, float]]:
        if self.overlap(low, segment):
            return low, low
        middle = 0.5 * (low + high)
        gap_low = self.gap(low, segment)
        gap_middle = self.gap(middle, segment)
        gap_high = self.gap(high, segment)
        if gap_middle <= self.contact_tol:
            if middle - low <= self.min_step:
                return self._bisect_entry(segment, low, middle)
            return self._search(segment, low, middle)
        if all(gap > self.contact_tol for gap in (gap_low, gap_middle, gap_high)):
            if self._motion_certificate(segment, low, middle, high, gap_middle):
                return None
        if high - low <= self.min_step:
            return self._leaf(segment, low, high)
        found = self._search(segment, low, middle)
        return found if found is not None else self._search(segment, middle, high)

    def run(self) -> _Contact:
        initial_gap = self.gap(0.0)
        if initial_gap <= 0.0:
            return _Contact(0.0, 0.0, 0.0, "current_overlap", self.evaluations)
        if initial_gap <= self.contact_tol:
            raise NumericalError("positive initial separation is below contact_tolerance")
        for segment, (low, high) in enumerate(zip(self.path_a.times[:-1], self.path_a.times[1:])):
            if self.path_a.is_translation(segment) and self.path_b.is_translation(segment):
                found = self._translation_entry(segment, float(low), float(high))
            else:
                found = self._search(segment, float(low), float(high))
            if found is not None:
                tau_minus, tau_plus = found
                return _Contact(tau_plus, tau_minus, tau_plus, "finite_contact", self.evaluations)
        return _Contact(math.inf, math.nan, math.nan, "no_contact", self.evaluations)

    def _translation_entry(self, segment, low, high):
        """Continuous SAT interval intersection for nonrotating linear motion."""
        from ._geometry import axes, projection_radius

        pa, pb = self.path_a.pose(low), self.path_b.pose(low)
        va, _ = self.path_a.left_derivative((low + high) / 2)
        vb, _ = self.path_b.left_derivative((low + high) / 2)
        entry, leave = 0.0, high - low
        for axis in axes(pa.heading, pb.heading):
            distance = float((pb.center - pa.center) @ axis)
            velocity = float((vb - va) @ axis)
            extent = (
                projection_radius(self.size_a, pa.heading, axis)
                + projection_radius(self.size_b, pb.heading, axis)
                + self.contact_tol
            )
            if not all(math.isfinite(v) for v in (distance, velocity, extent)):
                raise NumericalError("nonfinite translation-contact geometry")
            if velocity == 0.0:
                if abs(distance) > extent:
                    return None
                continue
            t1, t2 = sorted(((-extent - distance) / velocity, (extent - distance) / velocity))
            entry, leave = max(entry, t1), min(leave, t2)
            if entry > leave:
                return None
        hit = low + entry
        before = max(0.0, hit - self.bisect_tol)
        return before, hit


def _body_local(point: np.ndarray, pose: _Pose) -> np.ndarray:
    return rotation(pose.heading).T @ (point - pose.center)


def _contact_anchor(
    path_a: Trajectory,
    path_b: Trajectory,
    size_a: BodySize,
    size_b: BodySize,
    contact: _Contact,
) -> tuple[Optional[np.ndarray], str, float]:
    anchor_time = contact.tau_minus
    pose_a, pose_b = path_a.pose(anchor_time), path_b.pose(anchor_time)
    witness = closest_witness_pair(
        body_corners(pose_a.center, pose_a.heading, size_a),
        body_corners(pose_b.center, pose_b.heading, size_b),
    )
    local_a = _body_local(witness.point_a, pose_a)
    local_b = _body_local(witness.point_b, pose_b)
    relative_velocity = path_b.material_point_velocity(
        local_b, anchor_time
    ) - path_a.material_point_velocity(local_a, anchor_time)
    direction = unit(relative_velocity)
    if direction is not None:
        return direction, "witness_instant", witness.distance

    # Follow the same material points backwards, not moving closest-point IDs.
    previous = None
    duration = min(1e-6, anchor_time)
    while duration > 0 and duration <= min(0.05, anchor_time):
        now = path_b.body_point(local_b, anchor_time) - path_a.body_point(local_a, anchor_time)
        old = path_b.body_point(local_b, anchor_time - duration) - path_a.body_point(
            local_a, anchor_time - duration
        )
        direction = unit((now - old) / duration)
        if direction is not None:
            if previous is not None and abs(float(previous @ direction)) > 0.996:
                return direction, "witness_finite_difference", witness.distance
            previous = direction
        if duration >= min(0.05, anchor_time):
            break
        duration = min(2 * duration, 0.05, anchor_time)

    direction, _ = active_separating_direction(
        pose_a.center,
        pose_a.heading,
        size_a,
        pose_b.center,
        pose_b.heading,
        size_b,
    )
    return direction, "sat_fallback", witness.distance


def _peak_indepth(
    path_a: Trajectory,
    path_b: Trajectory,
    size_a: BodySize,
    size_b: BodySize,
    direction: np.ndarray,
    tem: float,
    max_step: float,
) -> tuple[float, int]:
    pieces = max(1, int(math.ceil(tem / max_step)))
    grid = np.unique(
        np.r_[np.linspace(0.0, tem, pieces + 1), path_a.times[path_a.times < tem], tem]
    )
    centers_a, headings_a = path_a.poses(grid)
    centers_b, headings_b = path_b.poses(grid)
    values = np.asarray(
        [
            indepth_along_direction(
                centers_a[index],
                headings_a[index],
                size_a,
                centers_b[index],
                headings_b[index],
                size_b,
                direction,
            )
            for index in range(len(grid))
        ]
    )
    candidates = [float(np.max(values))]
    evaluations = len(grid)
    # Refine every sampled local maximum, including intervals next to endpoints.
    peaks = [
        i
        for i in range(len(grid))
        if (i == 0 or values[i] >= values[i - 1])
        and (i == len(grid) - 1 or values[i] >= values[i + 1])
    ]
    for best in peaks:
        if np.ptp(values) < 1e-14:
            break

        def negative_depth(time_s: float) -> float:
            pose_a, pose_b = path_a.pose(time_s), path_b.pose(time_s)
            return -indepth_along_direction(
                pose_a.center,
                pose_a.heading,
                size_a,
                pose_b.center,
                pose_b.heading,
                size_b,
                direction,
            )

        optimized = minimize_scalar(
            negative_depth,
            bounds=(float(grid[max(0, best - 1)]), float(grid[min(len(grid) - 1, best + 1)])),
            method="bounded",
        )
        evaluations += int(getattr(optimized, "nfev", 0))
        if optimized.success:
            candidates.append(float(-optimized.fun))
    return max(candidates), evaluations


def compute_trajectory_ei(
    times: Sequence[float],
    trajectory_a: Sequence[Sequence[float]],
    trajectory_b: Sequence[Sequence[float]],
    size_a: Sequence[float],
    size_b: Sequence[float],
    *,
    options: SolverOptions = SolverOptions(),
) -> TrajectoryEIResult:
    """Evaluate one joint future; sizes are (length, width) in metres.

    This is the same solver used by both GEI entry points. A finite contact
    has EI = max(projected intrusion + safety_margin, 0) / TEM.
    """
    return evaluate_pair(
        Trajectory(times, trajectory_a),
        Trajectory(times, trajectory_b),
        _size(size_a),
        _size(size_b),
        options,
    )


def _size(value):
    if isinstance(value, BodySize):
        return value
    array = np.asarray(value, dtype=float)
    if array.shape != (2,):
        raise ValueError("size must be (length, width)")
    return BodySize(*map(float, array))


def evaluate_pair(path_a, path_b, parsed_a, parsed_b, options):
    """Internal common operator for analytic and interpolated paths."""
    if not np.array_equal(path_a.times, path_b.times):
        raise ValueError("both trajectories must use identical times")
    contact = _ContactSearch(
        path_a,
        path_b,
        parsed_a,
        parsed_b,
        options.contact_tolerance,
        options.min_search_step,
        options.time_tolerance,
        options.max_contact_evaluations,
    ).run()
    if contact.status == "no_contact":
        return TrajectoryEIResult(
            math.inf,
            0.0,
            0.0,
            contact.status,
            "no_contact",
            math.nan,
            math.nan,
            math.nan,
            contact.evaluations,
            0,
        )
    if contact.status == "current_overlap":
        return TrajectoryEIResult(
            0.0,
            math.nan,
            math.inf,
            contact.status,
            "current_overlap",
            math.nan,
            math.nan,
            math.nan,
            contact.evaluations,
            0,
        )
    direction, anchor_source, witness_distance = _contact_anchor(
        path_a, path_b, parsed_a, parsed_b, contact
    )
    if direction is None:
        raise NumericalError("finite contact has no stable approach direction")
    depth, evaluations = _peak_indepth(
        path_a,
        path_b,
        parsed_a,
        parsed_b,
        direction,
        contact.tem,
        options.indepth_max_step,
    )
    depth = max(0.0, depth + options.safety_margin)
    if contact.tem <= 0 or not math.isfinite(depth / contact.tem):
        raise NumericalError("nonfinite EI in a currently separated interaction")
    return TrajectoryEIResult(
        contact.tem,
        depth,
        depth / contact.tem,
        contact.status,
        anchor_source,
        float(direction[0]),
        float(direction[1]),
        witness_distance,
        contact.evaluations,
        evaluations,
    )
