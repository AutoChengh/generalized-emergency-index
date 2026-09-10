"""SE(2) propagation and oriented-rectangle geometry used by GEI."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .models import ActorState, BodySize, Pose

EPS = 1.0e-10


def unit(vector: np.ndarray, eps: float = EPS) -> Optional[np.ndarray]:
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= eps:
        return None
    return vector / norm


def rotation(heading: float) -> np.ndarray:
    cosine, sine = math.cos(float(heading)), math.sin(float(heading))
    return np.asarray([[cosine, -sine], [sine, cosine]], dtype=float)


def pose_at(actor: ActorState, time_s: float) -> Pose:
    """Exact constant-turn-rate, constant-speed propagation."""
    time_s = float(time_s)
    if time_s < 0.0:
        raise ValueError("time must be non-negative")
    turn = actor.yaw_rate * time_s
    heading = actor.heading + turn
    # Stable analytic integral, including exactly zero yaw rate.
    distance = actor.speed * time_s * float(np.sinc(turn / (2 * math.pi)))
    x = actor.x + distance * math.cos(actor.heading + turn / 2)
    y = actor.y + distance * math.sin(actor.heading + turn / 2)
    return Pose(float(x), float(y), float(heading))


def center_velocity(actor: ActorState, time_s: float) -> np.ndarray:
    heading = actor.heading + actor.yaw_rate * float(time_s)
    return np.asarray(
        [actor.speed * math.cos(heading), actor.speed * math.sin(heading)],
        dtype=float,
    )


def body_corners(center: np.ndarray, heading: float, size: BodySize) -> np.ndarray:
    half_length, half_width = 0.5 * size.length, 0.5 * size.width
    local = np.asarray(
        [
            [half_length, half_width],
            [half_length, -half_width],
            [-half_length, -half_width],
            [-half_length, half_width],
        ],
        dtype=float,
    )
    return local @ rotation(heading).T + np.asarray(center, dtype=float)


def axes(heading_a: float, heading_b: float) -> tuple[np.ndarray, ...]:
    return (
        np.asarray([math.cos(heading_a), math.sin(heading_a)], dtype=float),
        np.asarray([-math.sin(heading_a), math.cos(heading_a)], dtype=float),
        np.asarray([math.cos(heading_b), math.sin(heading_b)], dtype=float),
        np.asarray([-math.sin(heading_b), math.cos(heading_b)], dtype=float),
    )


def projection_radius(size: BodySize, heading: float, axis: np.ndarray) -> float:
    longitudinal = np.asarray([math.cos(heading), math.sin(heading)], dtype=float)
    lateral = np.asarray([-math.sin(heading), math.cos(heading)], dtype=float)
    return 0.5 * size.length * abs(float(longitudinal @ axis)) + 0.5 * size.width * abs(
        float(lateral @ axis)
    )


def sat_gap(
    center_a: np.ndarray,
    heading_a: float,
    size_a: BodySize,
    center_b: np.ndarray,
    heading_b: float,
    size_b: BodySize,
) -> float:
    """Largest separating-axis gap; contact/overlap iff non-positive."""
    delta = np.asarray(center_b, dtype=float) - np.asarray(center_a, dtype=float)
    return float(
        max(
            abs(float(delta @ axis))
            - projection_radius(size_a, heading_a, axis)
            - projection_radius(size_b, heading_b, axis)
            for axis in axes(heading_a, heading_b)
        )
    )


def active_separating_direction(
    center_a: np.ndarray,
    heading_a: float,
    size_a: BodySize,
    center_b: np.ndarray,
    heading_b: float,
    size_b: BodySize,
) -> tuple[np.ndarray, float]:
    delta = np.asarray(center_b, dtype=float) - np.asarray(center_a, dtype=float)
    candidates = []
    for axis in axes(heading_a, heading_b):
        projection = float(delta @ axis)
        radius = projection_radius(size_a, heading_a, axis) + projection_radius(
            size_b, heading_b, axis
        )
        candidates.append((abs(projection) - radius, axis if projection >= 0.0 else -axis))
    gap, direction = max(candidates, key=lambda item: item[0])
    normalized = unit(direction)
    if normalized is None:  # pragma: no cover - axes are always unit vectors
        raise RuntimeError("failed to construct a separating direction")
    return normalized, float(gap)


@dataclass(frozen=True)
class WitnessPair:
    point_a: np.ndarray
    point_b: np.ndarray
    distance: float
    feature_type: str
    multi_contact: bool
    candidate_count: int


def _point_segment_witness(
    point: np.ndarray, start: np.ndarray, end: np.ndarray
) -> tuple[np.ndarray, float, str]:
    segment = end - start
    denominator = float(segment @ segment)
    if denominator <= EPS:
        closest, feature = start.copy(), "vertex"
    else:
        alpha = float((point - start) @ segment / denominator)
        alpha = min(1.0, max(0.0, alpha))
        closest = start + alpha * segment
        feature = "edge" if 1.0e-8 < alpha < 1.0 - 1.0e-8 else "vertex"
    return closest, float(np.linalg.norm(point - closest)), feature


def closest_witness_pair(corners_a: np.ndarray, corners_b: np.ndarray) -> WitnessPair:
    """Canonical closest witness pair for two separated rectangles."""
    candidates: list[tuple[float, np.ndarray, np.ndarray, str]] = []
    for point_a in corners_a:
        for index in range(4):
            point_b, distance, feature = _point_segment_witness(
                point_a, corners_b[index], corners_b[(index + 1) % 4]
            )
            candidates.append((distance, point_a.copy(), point_b, f"A-vertex/B-{feature}"))
    for point_b in corners_b:
        for index in range(4):
            point_a, distance, feature = _point_segment_witness(
                point_b, corners_a[index], corners_a[(index + 1) % 4]
            )
            candidates.append((distance, point_a, point_b.copy(), f"A-{feature}/B-vertex"))

    minimum = min(item[0] for item in candidates)
    tolerance = max(1.0e-9, 1.0e-7 * max(1.0, minimum))
    tied = [item for item in candidates if item[0] <= minimum + tolerance]
    unique: list[tuple[float, np.ndarray, np.ndarray, str]] = []
    for item in tied:
        duplicate = any(
            np.linalg.norm(item[1] - previous[1]) <= tolerance
            and np.linalg.norm(item[2] - previous[2]) <= tolerance
            for previous in unique
        )
        if not duplicate:
            unique.append(item)

    point_a = np.mean([item[1] for item in unique], axis=0)
    point_b = np.mean([item[2] for item in unique], axis=0)
    multi_contact = len(unique) > 1
    feature = "multi/edge-edge" if multi_contact else unique[0][3]
    return WitnessPair(
        point_a=np.asarray(point_a, dtype=float),
        point_b=np.asarray(point_b, dtype=float),
        distance=float(np.linalg.norm(point_b - point_a)),
        feature_type=feature,
        multi_contact=multi_contact,
        candidate_count=len(unique),
    )


def indepth_along_direction(
    center_a: np.ndarray,
    heading_a: float,
    size_a: BodySize,
    center_b: np.ndarray,
    heading_b: float,
    size_b: BodySize,
    direction: np.ndarray,
) -> float:
    perpendicular = np.asarray([-direction[1], direction[0]], dtype=float)
    center_distance = abs(float((np.asarray(center_b) - np.asarray(center_a)) @ perpendicular))
    return float(
        projection_radius(size_a, heading_a, perpendicular)
        + projection_radius(size_b, heading_b, perpendicular)
        - center_distance
    )
