"""Dense polygon-projection oracle for curved default trajectories.

This is a regression comparison at 0.0003 s resolution, not a formal proof.
It intentionally avoids the package's geometry and propagation functions.
"""

import math
from dataclasses import replace

import numpy as np
import pytest

from gei import ActorState, compute_gei


def polygons(actor, times):
    h = actor.heading + actor.yaw_rate * times
    if actor.yaw_rate == 0:
        x = actor.x + actor.speed * np.cos(h) * times
        y = actor.y + actor.speed * np.sin(h) * times
    else:
        x = actor.x + actor.speed / actor.yaw_rate * (np.sin(h) - math.sin(actor.heading))
        y = actor.y - actor.speed / actor.yaw_rate * (np.cos(h) - math.cos(actor.heading))
    corners = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]]) * [actor.length / 2, actor.width / 2]
    rot = np.stack(
        [np.stack([np.cos(h), -np.sin(h)], axis=-1), np.stack([np.sin(h), np.cos(h)], axis=-1)],
        axis=-2,
    )
    return np.einsum("tij,kj->tki", rot, corners) + np.stack([x, y], axis=-1)[:, None]


def dense_first_contact(a, b, times):
    ca, cb = polygons(a, times), polygons(b, times)
    edges = np.concatenate([ca[:, 1:3] - ca[:, :2], cb[:, 1:3] - cb[:, :2]], axis=1)
    normals = np.stack([-edges[..., 1], edges[..., 0]], axis=-1)
    normals /= np.linalg.norm(normals, axis=-1)[..., None]
    pa = np.einsum("tki,tai->tka", ca, normals)
    pb = np.einsum("tki,tai->tka", cb, normals)
    separated = ((pa.max(axis=1) < pb.min(axis=1)) | (pb.max(axis=1) < pa.min(axis=1))).any(axis=1)
    hits = np.flatnonzero(~separated)
    return float(times[hits[0]]) if len(hits) else math.inf


@pytest.mark.parametrize("seed", list(range(8)))
def test_curved_contact_against_dense_polygon_oracle(seed):
    rng = np.random.default_rng(seed)
    a = ActorState(0, 0, 8, 0, rng.uniform(-0.5, 0.5), 4.6, 1.9)
    b = ActorState(
        rng.uniform(6, 14),
        rng.uniform(-5, 5),
        rng.uniform(1, 5),
        rng.uniform(-math.pi, math.pi),
        rng.uniform(-0.5, 0.5),
        2.2,
        0.8,
    )
    result = compute_gei(a, b, horizon=3.0)
    times = np.linspace(0, 3, 10001)
    for evaluation in result.evaluations:
        ma, mb = evaluation.label.split("/")
        aa = replace(a, yaw_rate=0) if ma == "CV" else a
        bb = replace(b, yaw_rate=0) if mb == "CV" else b
        expected = dense_first_contact(aa, bb, times)
        actual = evaluation.result.tem
        if math.isinf(expected):
            assert math.isinf(actual)
        else:
            assert actual == pytest.approx(expected, abs=0.000301)


def test_default_and_dense_sampled_path_agree():
    from gei import compute_trajectory_ei

    a = ActorState(0, 0, 8, 0, 0.08, 4.6, 1.9)
    b = ActorState(16, -12, 7, math.pi / 2, -0.10, 2.2, 0.8)
    analytic = compute_gei(a, b, horizon=4)
    times = np.linspace(0, 4, 401)
    for e in analytic.evaluations:
        ma, mb = e.label.split("/")
        aa, bb = (
            replace(a, yaw_rate=0) if ma == "CV" else a,
            replace(b, yaw_rate=0) if mb == "CV" else b,
        )
        paths = []
        for actor in (aa, bb):
            centers = polygons(actor, times).mean(axis=1)
            paths.append(np.column_stack([centers, actor.heading + actor.yaw_rate * times]))
        sampled = compute_trajectory_ei(times, *paths, (a.length, a.width), (b.length, b.width))
        assert sampled.ei == pytest.approx(e.result.ei, abs=2e-3)
