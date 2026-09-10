"""Lightweight output adapters; no model, checkpoint or GPU dependency.

These functions consume selected EMP-D/QCNet outputs. They do not construct
dataset inputs or run a neural network. The caller owns model inference,
history masks, coordinate transforms and mode-probability calibration.
"""

import math

import numpy as np

from ._solver import Trajectory


def _array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float)


def trajectories_from_positions(
    positions,
    future_times,
    current_pose,
    *,
    origin=(0.0, 0.0),
    frame_heading=0.0,
    max_yaw_rate: float,
    stationary_speed: float = 0.1,
):
    """Convert (K,T,2) future positions to K world-frame pose trajectories.

    current_pose is (x,y,heading) in the world frame. Input positions are in
    the frame given by origin/frame_heading; use defaults for world positions.
    future_times excludes t=0. Heading follows displacement for forward motion,
    holds the previous heading below stationary_speed, and is rate-limited.
    This is an explicit adapter assumption, not an inference by the GEI core.
    Supply Trajectory directly when reliable body headings are available.
    """
    xy, times = _array(positions), _array(future_times)
    current, offset = _array(current_pose), _array(origin)
    if xy.ndim != 3 or xy.shape[0] == 0 or xy.shape[2] != 2:
        raise ValueError("positions must have shape (modes, future_steps, 2)")
    if times.ndim != 1 or len(times) != xy.shape[1] or len(times) == 0:
        raise ValueError("future_times must match the future_steps dimension")
    if not np.all(np.isfinite(times)) or times[0] <= 0 or np.any(np.diff(times) <= 0):
        raise ValueError("future_times must be positive, finite and strictly increasing")
    if current.shape != (3,) or offset.shape != (2,):
        raise ValueError("current_pose must be (x,y,heading), origin must be (x,y)")
    if not all(np.all(np.isfinite(x)) for x in (xy, current, offset)):
        raise ValueError("positions and poses must be finite")
    if not math.isfinite(frame_heading):
        raise ValueError("frame_heading must be finite")
    if not math.isfinite(max_yaw_rate) or max_yaw_rate <= 0:
        raise ValueError("max_yaw_rate must be finite and positive")
    if not math.isfinite(stationary_speed) or stationary_speed < 0:
        raise ValueError("stationary_speed must be finite and non-negative")
    c, s = math.cos(frame_heading), math.sin(frame_heading)
    world = xy @ np.array([[c, s], [-s, c]]) + offset
    all_times = np.r_[0.0, times]
    paths = []
    for points in world:
        centers = np.vstack([current[:2], points])
        headings = [float(current[2])]
        for displacement, dt in zip(np.diff(centers, axis=0), np.diff(all_times)):
            desired = headings[-1]
            if np.linalg.norm(displacement) > stationary_speed * dt:
                desired = math.atan2(displacement[1], displacement[0])
            change = math.atan2(math.sin(desired - headings[-1]), math.cos(desired - headings[-1]))
            headings.append(
                headings[-1] + float(np.clip(change, -max_yaw_rate * dt, max_yaw_rate * dt))
            )
        paths.append(Trajectory(all_times, np.column_stack([centers, headings])))
    return paths


def _prediction(output, position_key, actor_index, future_times, current_pose, **kwargs):
    positions, logits = _array(output[position_key]), _array(output["pi"])
    if positions.ndim != 4 or logits.shape != positions.shape[:2] or positions.shape[-1] < 2:
        raise ValueError("expected positions (actors,modes,steps,channels) and pi (actors,modes)")
    if type(actor_index) is not int or not 0 <= actor_index < positions.shape[0]:
        raise ValueError("actor_index out of range")
    selected_logits = logits[actor_index]
    if not np.all(np.isfinite(selected_logits)):
        raise ValueError("mode logits must be finite")
    weights = np.exp(selected_logits - np.max(selected_logits))
    weights /= weights.sum()
    paths = trajectories_from_positions(
        positions[actor_index, :, :, :2], future_times, current_pose, **kwargs
    )
    return paths, weights


def empd_futures(output, batch_index, future_times, current_pose, **kwargs):
    """EMP-D y_hat/pi adapter. Keep the full mode bank; pi contains logits."""
    return _prediction(output, "y_hat", batch_index, future_times, current_pose, **kwargs)


def qcnet_futures(output, actor_index, future_times, current_pose, **kwargs):
    """QCNet loc_refine_pos/pi adapter. Coordinates must match the supplied frame."""
    return _prediction(output, "loc_refine_pos", actor_index, future_times, current_pose, **kwargs)
