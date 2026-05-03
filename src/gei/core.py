#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import math
import numpy as np


EPS = 1e-12
COLLISION_EI_VALUE = np.inf
D_SAFE = 0.0
K_2DTTC = 1.0
GAMMA = 0.01396

# Use positive infinity for invalid or unreachable time-based metrics.
LARGE_POSITIVE = np.inf


# =========================================================
# Kinematics and geometry helpers
# =========================================================
def ctrv_state_at_t(x0, y0, v, h0, yaw_rate, t):
    """
    Extrapolate a CTRV state at time t.

    Scalar math functions are used here to reduce numpy scalar-call overhead.
    """
    if abs(yaw_rate) < 1e-10:
        c = math.cos(h0)
        s = math.sin(h0)
        x = x0 + v * c * t
        y = y0 + v * s * t
        h = h0
    else:
        ht = h0 + yaw_rate * t
        x = x0 + (v / yaw_rate) * (math.sin(ht) - math.sin(h0))
        y = y0 - (v / yaw_rate) * (math.cos(ht) - math.cos(h0))
        h = ht
    return x, y, h


def get_rect_corners(x, y, h, l, w):
    """
    Return the four corners of an oriented rectangle.
    """
    c = math.cos(h)
    s = math.sin(h)

    local = np.array([
        [ l / 2.0,  w / 2.0],
        [ l / 2.0, -w / 2.0],
        [-l / 2.0, -w / 2.0],
        [-l / 2.0,  w / 2.0],
    ], dtype=float)

    rot = np.array([
        [c, -s],
        [s,  c],
    ], dtype=float)

    return local @ rot.T + np.array([x, y], dtype=float)


def get_state_and_corners(x, y, v, h, yaw, l, w, t):
    xt, yt, ht = ctrv_state_at_t(x, y, v, h, yaw, t)
    corners = get_rect_corners(xt, yt, ht, l, w)
    return xt, yt, ht, corners


def get_rel_velocity_vec(vA, hA_t, vB, hB_t):
    return np.array([
        vB * math.cos(hB_t) - vA * math.cos(hA_t),
        vB * math.sin(hB_t) - vA * math.sin(hA_t)
    ], dtype=float)


def get_unit_rel_velocity_dir(vA, hA_t, vB, hB_t):
    v_rel = get_rel_velocity_vec(vA, hA_t, vB, hB_t)
    n = np.linalg.norm(v_rel)
    if n < EPS:
        return None
    return v_rel / n


# =========================================================
# Oriented bounding-box collision checks
# =========================================================
def _get_axes_from_corners(corners):
    axes = []
    for i in range(4):
        p1 = corners[i]
        p2 = corners[(i + 1) % 4]
        edge = p2 - p1
        axis = np.array([-edge[1], edge[0]], dtype=float)
        norm = np.linalg.norm(axis)
        if norm > EPS:
            axes.append(axis / norm)
    return axes


def _project_polygon(corners, axis):
    vals = corners @ axis
    return np.min(vals), np.max(vals)


def check_collision_obb(corners_a, corners_b):
    """
    Vertex-projection SAT implementation kept for existing callers.
    """
    axes = _get_axes_from_corners(corners_a)[:2] + _get_axes_from_corners(corners_b)[:2]
    for axis in axes:
        min_a, max_a = _project_polygon(corners_a, axis)
        min_b, max_b = _project_polygon(corners_b, axis)
        if max_a < min_b or max_b < min_a:
            return False
    return True


def check_collision_obb_fast_params(xA, yA, hA, lA, wA, xB, yB, hB, lB, wB):
    """
    Fast OBB-SAT collision check.

    This is equivalent to the vertex-projection SAT check, but avoids
    generating corners and derives projection radii directly from each
    rectangle center, heading, length, and width.
    """
    ca = math.cos(hA)
    sa = math.sin(hA)
    cb = math.cos(hB)
    sb = math.sin(hB)

    # Local longitudinal and lateral axes for A and B.
    A_long = (ca, sa)
    A_lat = (-sa, ca)

    B_long = (cb, sb)
    B_lat = (-sb, cb)

    dx = xB - xA
    dy = yB - yA

    hlA = 0.5 * lA
    hwA = 0.5 * wA
    hlB = 0.5 * lB
    hwB = 0.5 * wB

    axes = (A_long, A_lat, B_long, B_lat)

    for ax, ay in axes:
        center_dist = abs(dx * ax + dy * ay)

        rA = (
            hlA * abs(A_long[0] * ax + A_long[1] * ay) +
            hwA * abs(A_lat[0] * ax + A_lat[1] * ay)
        )

        rB = (
            hlB * abs(B_long[0] * ax + B_long[1] * ay) +
            hwB * abs(B_lat[0] * ax + B_lat[1] * ay)
        )

        if center_dist > rA + rB + EPS:
            return False

    return True


def collision_at_t(
    xA, yA, vA, hA, yawA, lA, wA,
    xB, yB, vB, hB, yawB, lB, wB,
    t
):
    """
    Return whether the two OBBs collide at time t using fast OBB-SAT.
    """
    xA_t, yA_t, hA_t = ctrv_state_at_t(xA, yA, vA, hA, yawA, t)
    xB_t, yB_t, hB_t = ctrv_state_at_t(xB, yB, vB, hB, yawB, t)

    return check_collision_obb_fast_params(
        xA_t, yA_t, hA_t, lA, wA,
        xB_t, yB_t, hB_t, lB, wB
    )


def find_ttc_ctrv(
    xA, yA, vA, hA, yawA, lA, wA,
    xB, yB, vB, hB, yawB, lB, wB,
    T_total=10.0,
    dt=0.05,
    bisect_iters=30
):
    """
    Find TTC by coarse scanning and then bisecting the first collision interval.
    """
    if collision_at_t(
        xA, yA, vA, hA, yawA, lA, wA,
        xB, yB, vB, hB, yawB, lB, wB,
        0.0
    ):
        return 0.0

    t_grid = np.arange(dt, T_total + 0.5 * dt, dt)
    prev_t = 0.0

    for t in t_grid:
        if collision_at_t(
            xA, yA, vA, hA, yawA, lA, wA,
            xB, yB, vB, hB, yawB, lB, wB,
            t
        ):
            lo, hi = prev_t, t
            for _ in range(bisect_iters):
                mid = 0.5 * (lo + hi)
                if collision_at_t(
                    xA, yA, vA, hA, yawA, lA, wA,
                    xB, yB, vB, hB, yawB, lB, wB,
                    mid
                ):
                    hi = mid
                else:
                    lo = mid
            return hi

        prev_t = t

    return np.nan


# =========================================================
# InDepth helpers
# =========================================================
def compute_indepth_given_dir(
    centerA, cornersA,
    centerB, cornersB,
    e_parallel,
    d_safe=0.0
):
    perp = np.array([-e_parallel[1], e_parallel[0]], dtype=float)

    delta = centerB - centerA
    D_t1 = abs(np.dot(delta, perp))

    dA_max = np.max(np.abs((cornersA - centerA) @ perp))
    dB_max = np.max(np.abs((cornersB - centerB) @ perp))

    MFD = D_t1 - (dA_max + dB_max)
    indepth = d_safe - MFD
    return indepth


def get_collision_anchor_dir(
    xA, yA, vA, hA, yawA,
    xB, yB, vB, hB, yawB,
    ttc_ctrv,
    dt=0.05,
    anchor_backoff=1e-3,
    max_back_steps=20
):
    """
    Anchor direction for CA-InDepth.

    The reference direction is the relative velocity shortly before the first
    predicted collision.
    """
    if np.isnan(ttc_ctrv):
        return None

    t_anchor = max(ttc_ctrv - anchor_backoff, 0.0)

    for k in range(max_back_steps + 1):
        t_try = max(t_anchor - k * dt, 0.0)

        _, _, hA_t = ctrv_state_at_t(xA, yA, vA, hA, yawA, t_try)
        _, _, hB_t = ctrv_state_at_t(xB, yB, vB, hB, yawB, t_try)

        e_parallel = get_unit_rel_velocity_dir(vA, hA_t, vB, hB_t)
        if e_parallel is not None:
            return e_parallel

    return None


def scan_peak_indepth_ca(
    xA, yA, vA, hA, yawA, lA, wA,
    xB, yB, vB, hB, yawB, lB, wB,
    T_scan, e_parallel_anchor, dt=0.05, d_safe=0.0
):
    """
    CA-InDepth:
    Use the relative-velocity direction shortly before the first predicted
    collision as a fixed reference direction over the prediction horizon.
    """
    if e_parallel_anchor is None:
        return np.nan

    t_grid = np.arange(0.0, T_scan + 0.5 * dt, dt)
    best_val = np.nan

    for t in t_grid:
        xA_t, yA_t, _, cornersA = get_state_and_corners(xA, yA, vA, hA, yawA, lA, wA, t)
        xB_t, yB_t, _, cornersB = get_state_and_corners(xB, yB, vB, hB, yawB, lB, wB, t)

        indepth = compute_indepth_given_dir(
            centerA=np.array([xA_t, yA_t], dtype=float),
            cornersA=cornersA,
            centerB=np.array([xB_t, yB_t], dtype=float),
            cornersB=cornersB,
            e_parallel=e_parallel_anchor,
            d_safe=d_safe
        )

        if np.isnan(best_val) or indepth > best_val:
            best_val = indepth

    return best_val


# =========================================================
# Geometry helpers
# =========================================================
def distance(p1, p2):
    return float(math.hypot(p1[0] - p2[0], p1[1] - p2[1]))


def point_to_segment_distance(p, v1, v2):
    line_len = distance(v1, v2)
    if line_len < EPS:
        return distance(p, v1), np.array(v1, dtype=float)

    v1 = np.array(v1, dtype=float)
    v2 = np.array(v2, dtype=float)
    p = np.array(p, dtype=float)

    t = np.dot(p - v1, v2 - v1) / (line_len ** 2)
    t = max(0.0, min(1.0, t))
    closest = v1 + t * (v2 - v1)

    return distance(p, closest), closest


def get_shortest_distance(corners_A, corners_B):
    min_distance = float("inf")
    closest_A = None
    closest_B = None

    for i in range(4):
        p1 = corners_A[i]
        for k in range(4):
            p2 = corners_B[k]
            p3 = corners_B[(k + 1) % 4]
            dist, closest = point_to_segment_distance(p1, p2, p3)
            if dist < min_distance:
                min_distance = dist
                closest_A = np.array(p1, dtype=float)
                closest_B = np.array(closest, dtype=float)

        p1 = corners_B[i]
        for k in range(4):
            p2 = corners_A[k]
            p3 = corners_A[(k + 1) % 4]
            dist, closest = point_to_segment_distance(p1, p2, p3)
            if dist < min_distance:
                min_distance = dist
                closest_A = np.array(closest, dtype=float)
                closest_B = np.array(p1, dtype=float)

    return min_distance, closest_A, closest_B


def compute_v_closest_cv(xA, yA, vA, hA, lA, wA, xB, yB, vB, hB, lB, wB):
    corners_A = get_rect_corners(xA, yA, hA, lA, wA)
    corners_B = get_rect_corners(xB, yB, hB, lB, wB)

    min_distance, closest_A, closest_B = get_shortest_distance(corners_A, corners_B)

    delta = closest_B - closest_A
    norm_delta = np.linalg.norm(delta)

    if norm_delta < EPS:
        return min_distance, 0.0

    unit_vector = delta / norm_delta
    velocity_diff = np.array([
        vB * math.cos(hB) - vA * math.cos(hA),
        vB * math.sin(hB) - vA * math.sin(hA)
    ], dtype=float)

    v_closest = -np.dot(unit_vector, velocity_diff)
    return min_distance, float(v_closest)


def is_ray_intersect_segment(ray_origin, ray_direction, segment_start, segment_end):
    ray_origin = np.array(ray_origin, dtype=float)
    ray_direction = np.array(ray_direction, dtype=float)
    segment_start = np.array(segment_start, dtype=float)
    segment_end = np.array(segment_end, dtype=float)

    ray_norm = np.linalg.norm(ray_direction)
    if ray_norm < EPS:
        return None

    v1 = ray_origin - segment_start
    v2 = segment_end - segment_start
    v3 = np.array([-ray_direction[1], ray_direction[0]], dtype=float)

    v3_norm = np.linalg.norm(v3)
    if v3_norm < EPS:
        return None
    v3 = v3 / v3_norm

    dot = np.dot(v2, v3)

    if abs(dot) < 1e-10:
        if abs(np.cross(v1, v2)) < 1e-10:
            t0 = np.dot(segment_start - ray_origin, ray_direction) / (ray_norm ** 2)
            t1 = np.dot(segment_end - ray_origin, ray_direction) / (ray_norm ** 2)
            if t0 >= 0 and t1 >= 0:
                return min(t0, t1) * ray_norm
            if t0 < 0 and t1 < 0:
                return None
            return 0.0
        return None

    t1 = np.cross(v2, v1) / dot
    t2 = np.dot(v1, v3) / dot

    if 0 <= t2 <= 1:
        return t1

    return None


def compute_ttc2d_cv(xA, yA, vA, hA, lA, wA, xB, yB, vB, hB, lB, wB):
    """
    Return:
    - TTC2D
    - DTC
    - v_rel_norm
    """
    bbox_A = get_rect_corners(xA, yA, hA, lA, wA)
    bbox_B = get_rect_corners(xB, yB, hB, lB, wB)

    vA_vec = np.array([vA * math.cos(hA), vA * math.sin(hA)], dtype=float)
    vB_vec = np.array([vB * math.cos(hB), vB * math.sin(hB)], dtype=float)
    v_rel = vA_vec - vB_vec

    DTC = np.nan

    for i in range(4):
        neg_flag = False
        for j in range(4):
            dist = is_ray_intersect_segment(
                bbox_A[i], v_rel,
                bbox_B[j], bbox_B[(j + 1) % 4]
            )
            if dist is not None:
                if np.isnan(DTC) or (0 < dist < DTC):
                    DTC = dist
                if dist < 0:
                    neg_flag = True
                if neg_flag and dist > 0:
                    return 0.0, 0.0, float(np.linalg.norm(v_rel))

    for i in range(4):
        neg_flag = False
        for j in range(4):
            dist = is_ray_intersect_segment(
                bbox_B[i], -v_rel,
                bbox_A[j], bbox_A[(j + 1) % 4]
            )
            if dist is not None:
                if np.isnan(DTC) or (0 <= dist < DTC):
                    DTC = dist
                if dist < 0:
                    neg_flag = True
                if neg_flag and dist > 0:
                    return 0.0, 0.0, float(np.linalg.norm(v_rel))

    v_rel_norm = np.linalg.norm(v_rel)

    if not np.isnan(DTC) and v_rel_norm > EPS:
        ttc2d = DTC / v_rel_norm
        if ttc2d < 0:
            return np.inf, np.nan, np.nan
        return float(ttc2d), float(DTC), float(v_rel_norm)

    return np.inf, np.nan, float(v_rel_norm)


def _point_to_obb_local(px, py, cx, cy, cos_h, sin_h):
    dx = px - cx
    dy = py - cy
    local_x = dx * cos_h + dy * sin_h
    local_y = -dx * sin_h + dy * cos_h
    return local_x, local_y


def _is_point_in_obb(px, py, cx, cy, h, length, width, eps=1e-12):
    cos_h = math.cos(h)
    sin_h = math.sin(h)
    local_x, local_y = _point_to_obb_local(px, py, cx, cy, cos_h, sin_h)
    return (
        abs(local_x) <= 0.5 * length + eps and
        abs(local_y) <= 0.5 * width + eps
    )


def _segments_intersect_geom(p1, p2, q1, q2, eps=1e-12):
    def cross(ax, ay, bx, by):
        return ax * by - ay * bx

    def on_segment(a, b, p):
        return (
            min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps and
            min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps
        )

    ax, ay = p1
    bx, by = p2
    cx, cy = q1
    dx, dy = q2

    abx, aby = bx - ax, by - ay
    acx, acy = cx - ax, cy - ay
    adx, ady = dx - ax, dy - ay
    cdx, cdy = dx - cx, dy - cy
    cax, cay = ax - cx, ay - cy
    cbx, cby = bx - cx, by - cy

    d1 = cross(abx, aby, acx, acy)
    d2 = cross(abx, aby, adx, ady)
    d3 = cross(cdx, cdy, cax, cay)
    d4 = cross(cdx, cdy, cbx, cby)

    if ((d1 > eps and d2 < -eps) or (d1 < -eps and d2 > eps)) and \
       ((d3 > eps and d4 < -eps) or (d3 < -eps and d4 > eps)):
        return True

    if abs(d1) <= eps and on_segment(p1, p2, q1):
        return True
    if abs(d2) <= eps and on_segment(p1, p2, q2):
        return True
    if abs(d3) <= eps and on_segment(q1, q2, p1):
        return True
    if abs(d4) <= eps and on_segment(q1, q2, p2):
        return True

    return False


def compute_bbox_distance(xA, yA, hA, lA, wA, xB, yB, hB, lB, wB, eps=1e-12):
    """
    Compute the shortest distance between two OBBs.

    Return 0 when the boxes overlap or touch.
    """
    if check_collision_obb_fast_params(xA, yA, hA, lA, wA, xB, yB, hB, lB, wB):
        return 0.0

    ca, sa = math.cos(hA), math.sin(hA)
    cb, sb = math.cos(hB), math.sin(hB)
    hlA, hwA = 0.5 * lA, 0.5 * wA
    hlB, hwB = 0.5 * lB, 0.5 * wB

    Ax0 = xA - hlA * ca - hwA * sa
    Ay0 = yA - hlA * sa + hwA * ca
    Ax1 = xA + hlA * ca - hwA * sa
    Ay1 = yA + hlA * sa + hwA * ca
    Ax2 = xA + hlA * ca + hwA * sa
    Ay2 = yA + hlA * sa - hwA * ca
    Ax3 = xA - hlA * ca + hwA * sa
    Ay3 = yA - hlA * sa - hwA * ca

    Bx0 = xB - hlB * cb - hwB * sb
    By0 = yB - hlB * sb + hwB * cb
    Bx1 = xB + hlB * cb - hwB * sb
    By1 = yB + hlB * sb + hwB * cb
    Bx2 = xB + hlB * cb + hwB * sb
    By2 = yB + hlB * sb - hwB * cb
    Bx3 = xB - hlB * cb + hwB * sb
    By3 = yB - hlB * sb - hwB * cb

    A_pts = ((Ax0, Ay0), (Ax1, Ay1), (Ax2, Ay2), (Ax3, Ay3))
    B_pts = ((Bx0, By0), (Bx1, By1), (Bx2, By2), (Bx3, By3))

    min_d2 = float("inf")

    for px, py in A_pts:
        for i in range(4):
            qx, qy = B_pts[i]
            rx, ry = B_pts[(i + 1) & 3]

            wx, wy = rx - qx, ry - qy
            vx, vy = px - qx, py - qy
            wlen2 = wx * wx + wy * wy

            t = 0.0 if wlen2 == 0.0 else (vx * wx + vy * wy) / wlen2
            t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t

            cx = qx + t * wx
            cy = qy + t * wy

            dx = px - cx
            dy = py - cy
            d2 = dx * dx + dy * dy

            if d2 < min_d2:
                min_d2 = d2

    for px, py in B_pts:
        for i in range(4):
            qx, qy = A_pts[i]
            rx, ry = A_pts[(i + 1) & 3]

            wx, wy = rx - qx, ry - qy
            vx, vy = px - qx, py - qy
            wlen2 = wx * wx + wy * wy

            t = 0.0 if wlen2 == 0.0 else (vx * wx + vy * wy) / wlen2
            t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t

            cx = qx + t * wx
            cy = qy + t * wy

            dx = px - cx
            dy = py - cy
            d2 = dx * dx + dy * dy

            if d2 < min_d2:
                min_d2 = d2

    return math.sqrt(min_d2)


# =========================================================
# Traditional SSM metrics
# =========================================================
def compute_TTC_lon_1(xA, yA, vA, hA, lA, wA, xB, yB, vB, hB, lB, wB):
    theta_A = np.array([math.cos(hA), math.sin(hA)], dtype=float)
    theta_A_perp = np.array([-math.sin(hA), math.cos(hA)], dtype=float)

    xB_prime = np.array([xB - xA, yB - yA], dtype=float)
    S0lon = np.dot(xB_prime, theta_A)
    S0lat = np.dot(xB_prime, theta_A_perp)

    vB_vector = np.array([vB * math.cos(hB), vB * math.sin(hB)], dtype=float)
    vB_lon = np.dot(vB_vector, theta_A)
    vB_lat = np.dot(vB_vector, theta_A_perp)

    if S0lon * (vA - vB_lon) > 0:
        gap = abs(S0lon) - (lA + lB) / 2.0
        if gap > 0:
            ttc_lon_1 = gap / abs(vA - vB_lon)
            drac_1 = (vA - vB_lon) ** 2 / (2.0 * gap)
            if abs(S0lat + vB_lat * ttc_lon_1) <= K_2DTTC * ((wA + wB) / 2.0) and ttc_lon_1 >= 0:
                return float(ttc_lon_1), float(drac_1)

    return np.nan, np.nan


def compute_TTC_lon_2(xA, yA, vA, hA, lA, wA, xB, yB, vB, hB, lB, wB):
    theta_B = np.array([math.cos(hB), math.sin(hB)], dtype=float)
    theta_B_perp = np.array([-math.sin(hB), math.cos(hB)], dtype=float)

    xA_prime = np.array([xA - xB, yA - yB], dtype=float)
    S0lon = np.dot(xA_prime, theta_B)
    S0lat = np.dot(xA_prime, theta_B_perp)

    vA_vector = np.array([vA * math.cos(hA), vA * math.sin(hA)], dtype=float)
    vA_lon = np.dot(vA_vector, theta_B)
    vA_lat = np.dot(vA_vector, theta_B_perp)

    if S0lon * (vB - vA_lon) > 0:
        gap = abs(S0lon) - (lA + lB) / 2.0
        if gap > 0:
            ttc_lon_2 = gap / abs(vB - vA_lon)
            drac_2 = (vB - vA_lon) ** 2 / (2.0 * gap)
            if abs(S0lat + vA_lat * ttc_lon_2) <= K_2DTTC * ((wA + wB) / 2.0) and ttc_lon_2 >= 0:
                return float(ttc_lon_2), float(drac_2)

    return np.nan, np.nan


def compute_TTC_lat_1(xA, yA, vA, hA, lA, wA, xB, yB, vB, hB, lB, wB):
    theta_A = np.array([math.cos(hA), math.sin(hA)], dtype=float)
    theta_A_perp = np.array([-math.sin(hA), math.cos(hA)], dtype=float)

    xB_prime = np.array([xB - xA, yB - yA], dtype=float)
    S0lon = np.dot(xB_prime, theta_A)
    S0lat = np.dot(xB_prime, theta_A_perp)

    vB_vector = np.array([vB * math.cos(hB), vB * math.sin(hB)], dtype=float)
    vB_lon = np.dot(vB_vector, theta_A)
    vB_lat = np.dot(vB_vector, theta_A_perp)

    if S0lat * vB_lat < 0:
        gap = abs(S0lat) - (wA + wB) / 2.0
        if gap > 0:
            ttc_lat_1 = gap / abs(vB_lat)
            if abs(S0lon - (vA - vB_lon) * ttc_lat_1) <= K_2DTTC * ((lA + lB) / 2.0) and ttc_lat_1 >= 0:
                return float(ttc_lat_1)

    return np.nan


def compute_TTC_lat_2(xA, yA, vA, hA, lA, wA, xB, yB, vB, hB, lB, wB):
    theta_B = np.array([math.cos(hB), math.sin(hB)], dtype=float)
    theta_B_perp = np.array([-math.sin(hB), math.cos(hB)], dtype=float)

    xA_prime = np.array([xA - xB, yA - yB], dtype=float)
    S0lon = np.dot(xA_prime, theta_B)
    S0lat = np.dot(xA_prime, theta_B_perp)

    vA_vector = np.array([vA * math.cos(hA), vA * math.sin(hA)], dtype=float)
    vA_lon = np.dot(vA_vector, theta_B)
    vA_lat = np.dot(vA_vector, theta_B_perp)

    if S0lat * vA_lat < 0:
        gap = abs(S0lat) - (wA + wB) / 2.0
        if gap > 0:
            ttc_lat_2 = gap / abs(vA_lat)
            if abs(S0lon - (vB - vA_lon) * ttc_lat_2) <= K_2DTTC * ((lA + lB) / 2.0) and ttc_lat_2 >= 0:
                return float(ttc_lat_2)

    return np.nan


def calculate_TAdv(xA, yA, vA, hA, lA, xB, yB, vB, hB, lB):
    angle_difference = abs(hA - hB)
    if angle_difference > math.pi:
        angle_difference = 2.0 * math.pi - angle_difference

    delta_x = xB - xA
    delta_y = yB - yA
    norm_delta = math.sqrt(delta_x ** 2 + delta_y ** 2)

    if 0 <= angle_difference <= GAMMA:
        if np.dot([delta_x, delta_y], [math.cos(hA), math.sin(hA)]) > 0:
            TAdv = (norm_delta - lB / 2.0 - lA / 2.0) / vA if abs(vA) > EPS else np.nan
        else:
            TAdv = (norm_delta - lB / 2.0 - lA / 2.0) / vB if abs(vB) > EPS else np.nan
    else:
        denominator_ac = vA * math.sin(hB - hA)
        t_ac = np.nan if abs(denominator_ac) < EPS else (
            delta_x * math.sin(hB) - delta_y * math.cos(hB)
        ) / denominator_ac

        denominator_bc = vB * math.sin(hA - hB)
        t_bc = np.nan if abs(denominator_bc) < EPS else (
            (-delta_x) * math.sin(hA) - (-delta_y) * math.cos(hA)
        ) / denominator_bc

        TAdv = abs(t_ac - t_bc)

    if np.isfinite(TAdv) and TAdv >= 0:
        return float(TAdv)

    return np.nan


def compute_TDM_InDepth_old(x_A, y_A, v_A, h_A, l_A, w_A, x_B, y_B, v_B, h_B, l_B, w_B):
    v_diff = np.array([
        v_B * math.cos(h_B) - v_A * math.cos(h_A),
        v_B * math.sin(h_B) - v_A * math.sin(h_A)
    ], dtype=float)

    v_diff_norm = np.linalg.norm(v_diff)
    if v_diff_norm < EPS:
        return None, None

    theta_B_prime = v_diff / v_diff_norm
    delta = np.array([x_B - x_A, y_B - y_A], dtype=float)
    D_t1 = np.linalg.norm(delta - np.dot(delta, theta_B_prime) * theta_B_prime)

    AA1 = np.array([ l_A / 2 * math.cos(h_A) - w_A / 2 * -math.sin(h_A),  l_A / 2 * math.sin(h_A) - w_A / 2 * math.cos(h_A)])
    AA2 = np.array([ l_A / 2 * math.cos(h_A) + w_A / 2 * -math.sin(h_A),  l_A / 2 * math.sin(h_A) + w_A / 2 * math.cos(h_A)])
    AA3 = np.array([-l_A / 2 * math.cos(h_A) - w_A / 2 * -math.sin(h_A), -l_A / 2 * math.sin(h_A) - w_A / 2 * math.cos(h_A)])
    AA4 = np.array([-l_A / 2 * math.cos(h_A) + w_A / 2 * -math.sin(h_A), -l_A / 2 * math.sin(h_A) + w_A / 2 * math.cos(h_A)])

    d_A_max = np.max(np.array([
        np.linalg.norm(AA1 - np.dot(AA1, theta_B_prime) * theta_B_prime),
        np.linalg.norm(AA2 - np.dot(AA2, theta_B_prime) * theta_B_prime),
        np.linalg.norm(AA3 - np.dot(AA3, theta_B_prime) * theta_B_prime),
        np.linalg.norm(AA4 - np.dot(AA4, theta_B_prime) * theta_B_prime)
    ]))

    BB1 = np.array([ l_B / 2 * math.cos(h_B) - w_B / 2 * -math.sin(h_B),  l_B / 2 * math.sin(h_B) - w_B / 2 * math.cos(h_B)])
    BB2 = np.array([ l_B / 2 * math.cos(h_B) + w_B / 2 * -math.sin(h_B),  l_B / 2 * math.sin(h_B) + w_B / 2 * math.cos(h_B)])
    BB3 = np.array([-l_B / 2 * math.cos(h_B) - w_B / 2 * -math.sin(h_B), -l_B / 2 * math.sin(h_B) - w_B / 2 * math.cos(h_B)])
    BB4 = np.array([-l_B / 2 * math.cos(h_B) + w_B / 2 * -math.sin(h_B), -l_B / 2 * math.sin(h_B) + w_B / 2 * math.cos(h_B)])

    d_B_max = np.max(np.array([
        np.linalg.norm(BB1 - np.dot(BB1, theta_B_prime) * theta_B_prime),
        np.linalg.norm(BB2 - np.dot(BB2, theta_B_prime) * theta_B_prime),
        np.linalg.norm(BB3 - np.dot(BB3, theta_B_prime) * theta_B_prime),
        np.linalg.norm(BB4 - np.dot(BB4, theta_B_prime) * theta_B_prime)
    ]))

    MFD = D_t1 - (d_A_max + d_B_max)
    TDM = -np.dot(delta, theta_B_prime) / v_diff_norm if v_diff_norm != 0 else None
    InDepth = D_SAFE - MFD

    return TDM, InDepth


def compute_extra_ssm_metrics(
    xA, yA, vA, hA, lA, wA,
    xB, yB, vB, hB, lB, wB,
    precomputed_ttc2d_result=None
):
    """
    Compute traditional SSM metrics.

    The CVCV TTC2D result can be reused to avoid duplicate computation.
    """
    collision_now = check_collision_obb_fast_params(
        xA, yA, hA, lA, wA,
        xB, yB, hB, lB, wB
    )

    bbox_distance = compute_bbox_distance(xA, yA, hA, lA, wA, xB, yB, hB, lB, wB)

    if collision_now:
        return {
            "DRAC": 0.0,
            "DRAC2D": LARGE_POSITIVE,
            "TTC": LARGE_POSITIVE,
            "D2TTC": LARGE_POSITIVE,
            "TAdv": np.nan,
            "ACT": LARGE_POSITIVE,
            "EI": 0.0,
            "TTC2D": LARGE_POSITIVE,
            "BBox distance (m)": 0.0,
        }

    ttc_lon_1, drac_1 = compute_TTC_lon_1(xA, yA, vA, hA, lA, wA, xB, yB, vB, hB, lB, wB)
    ttc_lon_2, drac_2 = compute_TTC_lon_2(xA, yA, vA, hA, lA, wA, xB, yB, vB, hB, lB, wB)
    ttc_lat_1 = compute_TTC_lat_1(xA, yA, vA, hA, lA, wA, xB, yB, vB, hB, lB, wB)
    ttc_lat_2 = compute_TTC_lat_2(xA, yA, vA, hA, lA, wA, xB, yB, vB, hB, lB, wB)

    TTC_lon = np.nan if np.isnan(ttc_lon_1) and np.isnan(ttc_lon_2) else np.nanmin([ttc_lon_1, ttc_lon_2])
    D2TTC = np.nan if all(np.isnan(v) for v in [ttc_lon_1, ttc_lon_2, ttc_lat_1, ttc_lat_2]) else np.nanmin([ttc_lon_1, ttc_lon_2, ttc_lat_1, ttc_lat_2])
    DRAC = np.nan if np.isnan(drac_1) and np.isnan(drac_2) else np.nanmin([drac_1, drac_2])
    TTC = TTC_lon

    TAdv = calculate_TAdv(xA, yA, vA, hA, lA, xB, yB, vB, hB, lB)

    shortest_distance, v_closest = compute_v_closest_cv(
        xA, yA, vA, hA, lA, wA,
        xB, yB, vB, hB, lB, wB
    )

    if precomputed_ttc2d_result is None:
        TTC2D_old, DTC_2d, v_rel_norm_2d = compute_ttc2d_cv(
            xA, yA, vA, hA, lA, wA,
            xB, yB, vB, hB, lB, wB
        )
    else:
        TTC2D_old, DTC_2d, v_rel_norm_2d = precomputed_ttc2d_result

    if np.isnan(DTC_2d) or np.isnan(v_rel_norm_2d) or (not np.isfinite(DTC_2d)) or (not np.isfinite(v_rel_norm_2d)):
        DRAC2D = 0.0
    elif DTC_2d == 0:
        DRAC2D = LARGE_POSITIVE
    else:
        DRAC2D = (v_rel_norm_2d ** 2) / (2.0 * DTC_2d)

    if not np.isfinite(TTC2D_old) or TTC2D_old < 0:
        TTC2D_old = LARGE_POSITIVE

    if v_closest > 0:
        TDM_old, InDepth_old = compute_TDM_InDepth_old(
            xA, yA, vA, hA, lA, wA,
            xB, yB, vB, hB, lB, wB
        )
        TDM_old = np.nan if TDM_old is None or TDM_old < 0 else TDM_old
        InDepth_old = np.nan if InDepth_old is None else InDepth_old

        if np.isfinite(InDepth_old) and InDepth_old >= 0:
            EI_old = InDepth_old / TDM_old if np.isfinite(TDM_old) and abs(TDM_old) > EPS else np.nan
            ACT = shortest_distance / v_closest
            MEI_old_unused = InDepth_old / TTC2D_old if np.isfinite(TTC2D_old) and TTC2D_old > 0 else np.nan
        else:
            EI_old = np.nan
            ACT = np.nan
            MEI_old_unused = np.nan
    else:
        EI_old = np.nan
        ACT = np.nan
        MEI_old_unused = np.nan

    DRAC = 0.0 if np.isnan(DRAC) else float(DRAC)
    DRAC2D = 0.0 if np.isnan(DRAC2D) else float(DRAC2D)
    EI_old = 0.0 if np.isnan(EI_old) else float(EI_old)

    TTC = LARGE_POSITIVE if np.isnan(TTC) else float(TTC)
    D2TTC = LARGE_POSITIVE if np.isnan(D2TTC) else float(D2TTC)

    if np.isnan(ACT) or ACT < 0:
        ACT = LARGE_POSITIVE
    else:
        ACT = float(ACT)

    return {
        "DRAC": DRAC,
        "DRAC2D": DRAC2D,
        "TTC": TTC,
        "D2TTC": D2TTC,
        "TAdv": TAdv,
        "ACT": ACT,
        "EI": EI_old,
        "TTC2D": TTC2D_old,
        "BBox distance (m)": float(bbox_distance),
    }


# =========================================================
# GEI core computation
# =========================================================
def compute_mei_cvcv(
    xA, yA, vA, hA, lA, wA,
    xB, yB, vB, hB, lB, wB,
    d_safe=0.0,
    precomputed_tem_cvcv=None
):
    """
    Compute MEI under the CVCV mode.

    A precomputed TEM_CVCV value can be reused to avoid duplicate
    compute_ttc2d_cv calls.
    """
    collision_now = check_collision_obb_fast_params(
        xA, yA, hA, lA, wA,
        xB, yB, hB, lB, wB
    )

    if collision_now:
        return np.inf

    _, v_closest = compute_v_closest_cv(
        xA, yA, vA, hA, lA, wA,
        xB, yB, vB, hB, lB, wB
    )

    if not np.isfinite(v_closest) or v_closest <= 0:
        return 0.0

    e_parallel = get_unit_rel_velocity_dir(vA, hA, vB, hB)
    if e_parallel is None:
        return 0.0

    cornersA = get_rect_corners(xA, yA, hA, lA, wA)
    cornersB = get_rect_corners(xB, yB, hB, lB, wB)

    indepth = compute_indepth_given_dir(
        centerA=np.array([xA, yA], dtype=float),
        cornersA=cornersA,
        centerB=np.array([xB, yB], dtype=float),
        cornersB=cornersB,
        e_parallel=e_parallel,
        d_safe=d_safe
    )

    if not np.isfinite(indepth) or indepth < 0:
        return 0.0

    if precomputed_tem_cvcv is None:
        tem_cvcv, _, _ = compute_ttc2d_cv(
            xA, yA, vA, hA, lA, wA,
            xB, yB, vB, hB, lB, wB
        )
    else:
        tem_cvcv = precomputed_tem_cvcv

    if tem_cvcv == 0:
        return np.inf

    if not np.isfinite(tem_cvcv) or tem_cvcv < 0:
        return 0.0

    return float(indepth) / float(tem_cvcv)


def compute_mode_ttc_and_indepth_ca(
    xA, yA, vA, hA, yawA, lA, wA,
    xB, yB, vB, hB, yawB, lB, wB,
    T_total=10.0,
    dt=0.05,
    d_safe=0.0,
    anchor_backoff=1e-3
):
    """
    Compute one mode:
    - TEM_mode
    - InDepth_CA
    """
    ttc_mode = find_ttc_ctrv(
        xA, yA, vA, hA, yawA, lA, wA,
        xB, yB, vB, hB, yawB, lB, wB,
        T_total=T_total,
        dt=dt
    )

    if ttc_mode == 0:
        return {
            "TTC_mode": 0.0,
            "InDepth_CA": np.nan,
        }

    if np.isnan(ttc_mode) or ttc_mode < 0:
        return {
            "TTC_mode": np.nan,
            "InDepth_CA": np.nan,
        }

    scan_end = min(T_total, ttc_mode)

    e_anchor = get_collision_anchor_dir(
        xA, yA, vA, hA, yawA,
        xB, yB, vB, hB, yawB,
        ttc_ctrv=ttc_mode,
        dt=dt,
        anchor_backoff=anchor_backoff
    )

    indepth_ca = scan_peak_indepth_ca(
        xA, yA, vA, hA, yawA, lA, wA,
        xB, yB, vB, hB, yawB, lB, wB,
        T_scan=scan_end,
        e_parallel_anchor=e_anchor,
        dt=dt,
        d_safe=d_safe
    )

    return {
        "TTC_mode": ttc_mode,
        "InDepth_CA": indepth_ca,
    }


def _sanitize_tem(v):
    try:
        v = float(v)
    except Exception:
        return np.inf

    if np.isnan(v) or v < 0:
        return np.inf

    return v


def _sanitize_indepth(v):
    try:
        v = float(v)
    except Exception:
        return 0.0

    if np.isnan(v) or v < 0:
        return 0.0

    return v


def indepth_over_ttc(indepth, ttc):
    """
    EI = InDepth / TEM.
    """
    if ttc == 0:
        return COLLISION_EI_VALUE

    if np.isnan(ttc) or ttc < 0:
        return 0.0

    if np.isposinf(ttc):
        return 0.0

    if np.isnan(indepth) or indepth <= 0:
        return 0.0

    return float(indepth) / float(ttc)


def _compute_tem_eff(indepth_eff, gei):
    try:
        indepth_eff = float(indepth_eff)
        gei = float(gei)
    except Exception:
        return np.nan

    if np.isnan(gei):
        return np.nan

    if gei == 0.0:
        return np.inf

    if np.isposinf(gei):
        return 0.0

    if gei > 0.0 and np.isfinite(gei):
        return indepth_eff / gei

    return np.inf


def compute_4mode_gei_core_metrics(
    xA, yA, vA, hA, yawA, lA, wA,
    xB, yB, vB, hB, yawB, lB, wB,
    indepth_cvcv_analytical=np.nan,
    ttc_cvcv_analytical=np.nan,
    T_total=10.0,
    dt=0.05,
    d_safe=0.0,
    anchor_backoff=1e-3
):
    """
    Compute only GEI core metrics without traditional SSM metrics.
    """
    _ = indepth_cvcv_analytical
    _ = ttc_cvcv_analytical

    collision_now = check_collision_obb_fast_params(
        xA, yA, hA, lA, wA,
        xB, yB, hB, lB, wB
    )

    cornersA_now = get_rect_corners(xA, yA, hA, lA, wA)
    cornersB_now = get_rect_corners(xB, yB, hB, lB, wB)

    # Compute CVCV TEM once so MEI and extra metrics can reuse it.
    tem_cvcv_raw, dtc_cvcv, vrel_cvcv = compute_ttc2d_cv(
        xA, yA, vA, hA, lA, wA,
        xB, yB, vB, hB, lB, wB
    )

    mei = compute_mei_cvcv(
        xA, yA, vA, hA, lA, wA,
        xB, yB, vB, hB, lB, wB,
        d_safe=d_safe,
        precomputed_tem_cvcv=tem_cvcv_raw
    )

    mode_cvct = compute_mode_ttc_and_indepth_ca(
        xA, yA, vA, hA, 0.0, lA, wA,
        xB, yB, vB, hB, yawB, lB, wB,
        T_total=T_total,
        dt=dt,
        d_safe=d_safe,
        anchor_backoff=anchor_backoff
    )

    mode_ctcv = compute_mode_ttc_and_indepth_ca(
        xA, yA, vA, hA, yawA, lA, wA,
        xB, yB, vB, hB, 0.0, lB, wB,
        T_total=T_total,
        dt=dt,
        d_safe=d_safe,
        anchor_backoff=anchor_backoff
    )

    mode_ctct = compute_mode_ttc_and_indepth_ca(
        xA, yA, vA, hA, yawA, lA, wA,
        xB, yB, vB, hB, yawB, lB, wB,
        T_total=T_total,
        dt=dt,
        d_safe=d_safe,
        anchor_backoff=anchor_backoff
    )

    tem_cvcv = tem_cvcv_raw
    tem_cvct = mode_cvct["TTC_mode"]
    tem_ctcv = mode_ctcv["TTC_mode"]
    tem_ctct = mode_ctct["TTC_mode"]

    if collision_now:
        tem_cvcv = 0.0
        tem_cvct = 0.0
        tem_ctcv = 0.0
        tem_ctct = 0.0
    else:
        tem_cvcv = _sanitize_tem(tem_cvcv)
        tem_cvct = _sanitize_tem(tem_cvct)
        tem_ctcv = _sanitize_tem(tem_ctcv)
        tem_ctct = _sanitize_tem(tem_ctct)

    e_parallel_cvcv = get_unit_rel_velocity_dir(vA, hA, vB, hB)

    if e_parallel_cvcv is None:
        indepth_cvcv = 0.0
    else:
        indepth_cvcv = compute_indepth_given_dir(
            centerA=np.array([xA, yA], dtype=float),
            cornersA=cornersA_now,
            centerB=np.array([xB, yB], dtype=float),
            cornersB=cornersB_now,
            e_parallel=e_parallel_cvcv,
            d_safe=d_safe
        )

    indepth_cvcv = _sanitize_indepth(indepth_cvcv)
    indepth_cvct_ca = _sanitize_indepth(mode_cvct["InDepth_CA"])
    indepth_ctcv_ca = _sanitize_indepth(mode_ctcv["InDepth_CA"])
    indepth_ctct_ca = _sanitize_indepth(mode_ctct["InDepth_CA"])

    if collision_now:
        mei = np.inf
        ei_cvct_ca = np.inf
        ei_ctcv_ca = np.inf
        ei_ctct_ca = np.inf
        gei = np.inf
    else:
        mei = 0.0 if np.isnan(mei) or mei < 0 else float(mei)

        ei_cvct_ca = indepth_over_ttc(indepth_cvct_ca, tem_cvct)
        ei_ctcv_ca = indepth_over_ttc(indepth_ctcv_ca, tem_ctcv)
        ei_ctct_ca = indepth_over_ttc(indepth_ctct_ca, tem_ctct)

        ei_cvct_ca = 0.0 if np.isnan(ei_cvct_ca) or ei_cvct_ca < 0 else float(ei_cvct_ca)
        ei_ctcv_ca = 0.0 if np.isnan(ei_ctcv_ca) or ei_ctcv_ca < 0 else float(ei_ctcv_ca)
        ei_ctct_ca = 0.0 if np.isnan(ei_ctct_ca) or ei_ctct_ca < 0 else float(ei_ctct_ca)

        gei = np.mean([mei, ei_cvct_ca, ei_ctcv_ca, ei_ctct_ca])

    indepth_eff = np.mean([
        indepth_cvcv,
        indepth_cvct_ca,
        indepth_ctcv_ca,
        indepth_ctct_ca
    ])

    tem_eff = _compute_tem_eff(indepth_eff, gei)

    return {
        "TEM_CVCV": tem_cvcv,
        "TEM_CVCT": tem_cvct,
        "TEM_CTCV": tem_ctcv,
        "TEM_CTCT": tem_ctct,

        "InDepth_CVCV": indepth_cvcv,
        "InDepth_CVCT_CA": indepth_cvct_ca,
        "InDepth_CTCV_CA": indepth_ctcv_ca,
        "InDepth_CTCT_CA": indepth_ctct_ca,

        "MEI": mei,
        "EI_CVCT_CA": ei_cvct_ca,
        "EI_CTCV_CA": ei_ctcv_ca,
        "EI_CTCT_CA": ei_ctct_ca,
        "GEI": gei,

        "InDepth_eff": indepth_eff,
        "TEM_eff": tem_eff,

        # Reusable internal object; it is not part of the public CSV schema.
        "_precomputed_ttc2d_result": (tem_cvcv_raw, dtc_cvcv, vrel_cvcv),
    }


def compute_4mode_ca_metrics(
    xA, yA, vA, hA, yawA, lA, wA,
    xB, yB, vB, hB, yawB, lB, wB,
    indepth_cvcv_analytical=np.nan,
    ttc_cvcv_analytical=np.nan,
    T_total=10.0,
    dt=0.05,
    d_safe=0.0,
    anchor_backoff=1e-3,
    compute_extra_metrics=True
):
    """
    Full metric entry point.

    By default this computes both GEI core metrics and traditional SSM metrics.
    """
    core = compute_4mode_gei_core_metrics(
        xA, yA, vA, hA, yawA, lA, wA,
        xB, yB, vB, hB, yawB, lB, wB,
        indepth_cvcv_analytical=indepth_cvcv_analytical,
        ttc_cvcv_analytical=ttc_cvcv_analytical,
        T_total=T_total,
        dt=dt,
        d_safe=d_safe,
        anchor_backoff=anchor_backoff
    )

    precomputed_ttc2d_result = core.get("_precomputed_ttc2d_result", None)

    result = dict(core)
    result.pop("_precomputed_ttc2d_result", None)

    if compute_extra_metrics:
        extra_metrics = compute_extra_ssm_metrics(
            xA, yA, vA, hA, lA, wA,
            xB, yB, vB, hB, lB, wB,
            precomputed_ttc2d_result=precomputed_ttc2d_result
        )

        result.update({
            "DRAC": extra_metrics.get("DRAC", 0.0),
            "DRAC2D": extra_metrics.get("DRAC2D", 0.0),
            "TTC": extra_metrics.get("TTC", LARGE_POSITIVE),
            "D2TTC": extra_metrics.get("D2TTC", LARGE_POSITIVE),
            "TAdv": extra_metrics.get("TAdv", np.nan),
            "ACT": extra_metrics.get("ACT", LARGE_POSITIVE),
            "EI": extra_metrics.get("EI", 0.0),
            "TTC2D": extra_metrics.get("TTC2D", LARGE_POSITIVE),
            "BBox distance (m)": extra_metrics.get("BBox distance (m)", np.nan),
        })

    return result


def compute_4mode_ei_variants(
    xA, yA, vA, hA, yawA, lA, wA,
    xB, yB, vB, hB, yawB, lB, wB,
    indepth_cvcv_analytical=np.nan,
    ttc_cvcv_analytical=np.nan,
    T_total=10.0,
    dt=0.05,
    d_safe=0.0,
    anchor_backoff=1e-3
):
    """
    Backward-compatible legacy function name.
    """
    result = compute_4mode_gei_core_metrics(
        xA, yA, vA, hA, yawA, lA, wA,
        xB, yB, vB, hB, yawB, lB, wB,
        indepth_cvcv_analytical=indepth_cvcv_analytical,
        ttc_cvcv_analytical=ttc_cvcv_analytical,
        T_total=T_total,
        dt=dt,
        d_safe=d_safe,
        anchor_backoff=anchor_backoff
    )

    return {
        "MEI": result["MEI"],
        "EI_CVCT_CA": result["EI_CVCT_CA"],
        "EI_CTCV_CA": result["EI_CTCV_CA"],
        "EI_CTCT_CA": result["EI_CTCT_CA"],
        "GEI": result["GEI"],
        "InDepth_eff": result["InDepth_eff"],
        "TEM_eff": result["TEM_eff"],
    }


def compute_4mode_ir_ca_metrics(*args, **kwargs):
    """
    Backward-compatible legacy entry point.

    IR logic has been removed; this delegates to compute_4mode_ca_metrics.
    """
    return compute_4mode_ca_metrics(*args, **kwargs)


def compute_single_case_report(
    xA, yA, vA, hA, yawA, lA, wA,
    xB, yB, vB, hB, yawB, lB, wB,
    T_total=10.0,
    dt=0.05,
    d_safe=0.0,
    anchor_backoff=1e-3
):
    """
    Compute one case and return detailed metrics plus elapsed time in ms.
    """
    t0 = time.perf_counter()

    result = compute_4mode_ca_metrics(
        xA, yA, vA, hA, yawA, lA, wA,
        xB, yB, vB, hB, yawB, lB, wB,
        T_total=T_total,
        dt=dt,
        d_safe=d_safe,
        anchor_backoff=anchor_backoff,
        compute_extra_metrics=True
    )

    t1 = time.perf_counter()
    result["compute_time_ms"] = (t1 - t0) * 1000.0
    return result


def format_metric(v):
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return "nan"
        if np.isposinf(v):
            return "inf"
        if np.isneginf(v):
            return "-inf"
        return f"{float(v):.4f}"
    return str(v)


def main():
    xA, yA, vA, hA, yawA, lA, wA = 504.0451, -271.9787, 22.9184, 2.5530, 0.0, 17.0237, 2.5907
    xB, yB, vB, hB, yawB, lB, wB = 501.8724, -278.5692, 24.9702, 2.4877, -0.0, 16.3289, 2.5973

    result = compute_single_case_report(
        xA, yA, vA, hA, yawA, lA, wA,
        xB, yB, vB, hB, yawB, lB, wB,
        T_total=10.0,
        dt=0.05,
        d_safe=0.0,
        anchor_backoff=1e-3
    )

    ordered_keys = [
        "TEM_CVCV", "TEM_CVCT", "TEM_CTCV", "TEM_CTCT",
        "InDepth_CVCV", "InDepth_CVCT_CA", "InDepth_CTCV_CA", "InDepth_CTCT_CA",
        "MEI", "EI_CVCT_CA", "EI_CTCV_CA", "EI_CTCT_CA", "GEI",
        "InDepth_eff", "TEM_eff",
        "DRAC", "DRAC2D", "TTC", "D2TTC", "TAdv", "ACT", "EI", "TTC2D", "BBox distance (m)",
        "compute_time_ms",
    ]

    print("===== metrics =====")
    for k in ordered_keys:
        if k in result:
            print(f"{k}: {format_metric(result[k])}")


if __name__ == "__main__":
    main()
