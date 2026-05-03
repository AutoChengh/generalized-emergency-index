#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pure CSV-based GEI visualization.

This script reads already-enriched accident CSV files and directly generates GIF
visualizations. It does NOT compute GEI, TEM, InDepth, DRAC, TTC, or any other
risk metrics. All visualization quantities are read directly from the CSV.

Expected key columns include:
    Time (s)
    Position X (m), Position Y (m), Heading, Length (m), Width (m), Velocity (m/s)
    2_Position X (m), 2_Position Y (m), 2_Heading, 2_Length (m), 2_Width (m), 2_Velocity (m/s)
    TEM_CVCV, TEM_CVCT, TEM_CTCV, TEM_CTCT
    InDepth_CVCV, InDepth_CVCT_CA, InDepth_CTCV_CA, InDepth_CTCT_CA
    MEI, EI_CVCT_CA, EI_CTCV_CA, EI_CTCT_CA
    GEI, InDepth_eff, TEM_eff
    DRAC, DRAC2D, TTC, 2D-TTC, TAdv, ACT, EI, TTC2D, BBox distance (m)

Map logic:
    - Only CSV files whose stem starts with "GEI_SIND" are treated as SIND files.
    - For SIND files, the OSM map under assets/maps is drawn when available.
    - For other GEI_*.csv files, no map is drawn.
"""

import argparse
import glob
import math
import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyBboxPatch
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import MultipleLocator, MaxNLocator, FuncFormatter
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.gridspec import GridSpec
from PIL import Image


# ============================================================
# Basic configuration
# ============================================================

FRAME_STEP = 1
FIGSIZE = (15.4, 10.4)
FIG_DPI = 220

REPO_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_GIF_DIR = Path("gif_visualizations")

VISUALIZE_TIME_RANGE = [0.0, 1.0]

X_MARGIN_LEFT = 10.0
X_MARGIN_RIGHT = 5.0
Y_MARGIN_BOTTOM = 5.0
Y_MARGIN_TOP = 10.0

GIF_LOOP = 0
GIF_OPTIMIZE = False
GIF_DISPOSAL = 2
MIN_GIF_DURATION_MS = 20
FALLBACK_DURATION_MS = 100

DRAW_CENTER_TRAJECTORY = True
HISTORY_TRAJ_COLOR = "#7F7F7F"
HISTORY_TRAJ_LINEWIDTH = 1.1
HISTORY_TRAJ_ALPHA = 0.65

EDGE_COLOR = "#333333"
EDGE_WIDTH = 0.75

DRAW_SPEED_ARROW = True
SHOW_SPEED_TEXT = False
ARROW_COLOR = "#8C8C8C"
ARROW_ALPHA = 0.58
ARROW_LINEWIDTH = 1.1
ARROW_HEAD_WIDTH = 0.48
ARROW_HEAD_LENGTH = 0.68
ARROW_LENGTH_SCALE = 0.38
SPEED_TEXT_OFFSET = 0.65
SPEED_TEXT_BBOX_ALPHA = 0.55

GRID_ON = False
GRID_COLOR = "#D0D0D0"
GRID_LINESTYLE = "--"
GRID_LINEWIDTH = 0.7
GRID_ALPHA = 0.40

ROUND_DECIMALS = 4


# ============================================================
# Map configuration
# ============================================================

MAP_OSM_PATH = REPO_ROOT / "assets" / "maps" / "map_relink_law_save.osm"
MAP_OFFSET = (15.5, 16.5)

MAP_FACE_COLOR = "#DCDCDC"
MAP_GLOBAL_ALPHA = 0.90

DRAW_VIRTUAL_LINES = False
DRAW_WAIT_LINE = True
DRAW_ZEBRA = True
DRAW_STOP_LINE = True

EARTH_R = 6378137.0
GLOBAL_MAP_DATA = None


# ============================================================
# Matplotlib style
# ============================================================

rcParams["font.sans-serif"] = ["Arial"]
rcParams["font.family"] = "sans-serif"
rcParams["axes.unicode_minus"] = False
rcParams["axes.linewidth"] = 1.1
rcParams["xtick.major.width"] = 1.1
rcParams["ytick.major.width"] = 1.1
rcParams["xtick.direction"] = "in"
rcParams["ytick.direction"] = "in"
rcParams["axes.labelsize"] = 13
rcParams["xtick.labelsize"] = 10
rcParams["ytick.labelsize"] = 10
rcParams["axes.titlesize"] = 13
rcParams["figure.titlesize"] = 13


# ============================================================
# Risk colors
# ============================================================

RISK_YELLOW = "#F2D97C"
RISK_ORANGE = "#ECA24B"
RISK_RED = "#D14D1F"

RISK_CMAP = LinearSegmentedColormap.from_list(
    "Risk_YellowOrangeRed",
    [RISK_YELLOW, RISK_ORANGE, RISK_RED],
    N=256,
)

GEI_MIN = 0.0
GEI_MAX = 1.5
GEI_NORM = Normalize(vmin=GEI_MIN, vmax=GEI_MAX)


# ============================================================
# Curve panel style
# ============================================================

CURVE_PRE_ALPHA = 0.95
CURVE_POST_ALPHA = 0.35
CURVE_PRE_LW = 1.8
CURVE_POST_LW = 1.4
CURVE_POST_LS = "--"
LEFT_LINEWIDTH = 2.0

COL_GREEN = "#80B197"
COL_RED = "#D67769"
COL_SOFTRED = "#E7907A"
CURVE_MAIN_COLOR = "#222222"

CURVE_GRID_ON = True
CURVE_GRID_ALPHA = 0.18
CURVE_GRID_LS = "--"
CURVE_GRID_LW = 0.5

REF_VLINE_LW = 1.0
REF_VLINE_LS = "--"
REF_VLINE_ALPHA = 0.60
REF_VLINE_COLOR = "#7D7D7D"


# ============================================================
# GEI principle panel style
# ============================================================

PRINCIPLE_SAFE = "#5FBC78"
PRINCIPLE_MID = "#F2D36B"
PRINCIPLE_DANGER = "#D85B4A"

PRINCIPLE_CMAP = LinearSegmentedColormap.from_list(
    "PrincipleGreenYellowRed",
    [PRINCIPLE_SAFE, PRINCIPLE_MID, PRINCIPLE_DANGER],
    N=256,
)

PRINCIPLE_EI_MAX = 1.5
PRINCIPLE_EI_NORM = Normalize(vmin=0.0, vmax=PRINCIPLE_EI_MAX)

PRINCIPLE_PANEL_FACE = "#FFFFFF"
PRINCIPLE_PANEL_EDGE = "#D5CEC4"
PRINCIPLE_CELL_FACE = "#FFFFFF"
PRINCIPLE_CELL_EDGE = "#D9D2C8"
PRINCIPLE_AXIS_COLOR = "#8A837C"
PRINCIPLE_TEXT_COLOR = "#2B2B2B"
PRINCIPLE_SUBTEXT_COLOR = "#666666"
PRINCIPLE_GRID_COLOR = "#DED7D0"

PRINCIPLE_TEM_CAP_DEFAULT = 4.0
PRINCIPLE_TEM_CAP_RANGE = (2.0, 8.0)
PRINCIPLE_INDEPTH_CAP_DEFAULT = 2.5
PRINCIPLE_INDEPTH_CAP_RANGE = (1.0, 6.0)

EPS_TEM_ZERO = 1e-12
EPS_INDEPTH_POSITIVE = 1e-12
INF_DISPLAY_THRESHOLD = 1e4

MODE_PRINCIPLE_SPECS = [
    ("CVCV", "TEM_CVCV", "InDepth_CVCV", "MEI"),
    ("CVCT", "TEM_CVCT", "InDepth_CVCT_CA", "EI_CVCT_CA"),
    ("CTCV", "TEM_CTCV", "InDepth_CTCV_CA", "EI_CTCV_CA"),
    ("CTCT", "TEM_CTCT", "InDepth_CTCT_CA", "EI_CTCT_CA"),
]


# ============================================================
# Required visualization columns
# ============================================================

REQUIRED_VIS_COLS = [
    "Time (s)",

    "Position X (m)",
    "Position Y (m)",
    "Velocity (m/s)",
    "Heading",
    "Length (m)",
    "Width (m)",

    "2_Position X (m)",
    "2_Position Y (m)",
    "2_Velocity (m/s)",
    "2_Heading",
    "2_Length (m)",
    "2_Width (m)",

    "GEI",
]


# ============================================================
# Utility functions
# ============================================================

def safe_float(v, default=np.nan):
    try:
        if pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default


def choose_col(df_cols, candidates):
    for c in candidates:
        if c in df_cols:
            return c
    return None


def clip_val(x, xmin, xmax):
    return max(xmin, min(xmax, x))


def finite_series(series):
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def finite_series_keep_posinf(series):
    return pd.to_numeric(series, errors="coerce").replace([-np.inf], np.nan)


def gei_to_rgba(gei_value, alpha=1.0):
    gei_raw = safe_float(gei_value, 0.0)
    if not np.isfinite(gei_raw):
        gei = GEI_MAX if gei_raw > 0 else GEI_MIN
    else:
        gei = clip_val(gei_raw, GEI_MIN, GEI_MAX)

    rgba = RISK_CMAP(GEI_NORM(gei))
    return rgba[0], rgba[1], rgba[2], alpha


def build_vehicle_polygon(x, y, heading, length, width):
    dx = length / 2.0
    dy = width / 2.0

    c = math.cos(heading)
    s = math.sin(heading)

    corners = np.array(
        [
            [-dx * c - dy * s, -dx * s + dy * c],
            [ dx * c - dy * s,  dx * s + dy * c],
            [ dx * c + dy * s,  dx * s - dy * c],
            [-dx * c + dy * s, -dx * s - dy * c],
        ],
        dtype=float,
    ) + np.array([x, y], dtype=float)

    return corners


# ============================================================
# Map data structure and drawing logic
# ============================================================

class Way:
    def __init__(self, way_type, subtype, x, y):
        self.way_type = way_type
        self.subtype = subtype
        self.x = x
        self.y = y


class MapData:
    def __init__(self, lat0, lon0, ways):
        self.lat0 = lat0
        self.lon0 = lon0
        self.ways = ways


def latlon2xy(lat, lon, lat0, lon0):
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    lat0 = float(lat0)
    lon0 = float(lon0)

    dlat = np.deg2rad(lat - lat0)
    dlon = np.deg2rad(lon - lon0)
    lat0_rad = math.radians(lat0)

    x = EARTH_R * dlon * math.cos(lat0_rad)
    y = EARTH_R * dlat
    return x, y


def parse_osm(osm_path):
    tree = ET.parse(osm_path)
    root = tree.getroot()

    node_lat = {}
    node_lon = {}
    lats = []
    lons = []

    for node in root.findall("node"):
        nid = node.attrib.get("id")
        lat = node.attrib.get("lat")
        lon = node.attrib.get("lon")

        if nid is None or lat is None or lon is None:
            continue

        lat_f = float(lat)
        lon_f = float(lon)

        node_lat[nid] = lat_f
        node_lon[nid] = lon_f
        lats.append(lat_f)
        lons.append(lon_f)

    if not lats or not lons:
        raise ValueError(f"No nodes found in OSM: {osm_path}")

    lat0 = (min(lats) + max(lats)) / 2.0
    lon0 = (min(lons) + max(lons)) / 2.0

    all_ids = list(node_lat.keys())
    lat_arr = np.array([node_lat[i] for i in all_ids], dtype=float)
    lon_arr = np.array([node_lon[i] for i in all_ids], dtype=float)
    x_arr, y_arr = latlon2xy(lat_arr, lon_arr, lat0, lon0)

    node_x = {}
    node_y = {}
    for i, nid in enumerate(all_ids):
        node_x[nid] = float(x_arr[i])
        node_y[nid] = float(y_arr[i])

    ways = []

    for way in root.findall("way"):
        way_type = ""
        subtype = ""

        for tag in way.findall("tag"):
            k = tag.attrib.get("k", "")
            v = tag.attrib.get("v", "")

            if k == "type":
                way_type = v
            elif k == "subtype":
                subtype = v

        xs = []
        ys = []

        for nd in way.findall("nd"):
            ref = nd.attrib.get("ref")

            if ref is None:
                continue

            if ref in node_x and ref in node_y:
                xs.append(node_x[ref])
                ys.append(node_y[ref])

        if len(xs) >= 2:
            ways.append(
                Way(
                    way_type=way_type,
                    subtype=subtype,
                    x=np.asarray(xs, dtype=float),
                    y=np.asarray(ys, dtype=float),
                )
            )

    return MapData(lat0=lat0, lon0=lon0, ways=ways)


def get_map_style(way_type, subtype):
    way_type = way_type or ""
    subtype = subtype or ""

    color = None
    width = 1.5
    style = "-"
    alpha = 1.0
    draw_flag = True

    if way_type == "curbstone":
        color = "#222222"
        width = 2.0
        alpha = 0.92

    elif way_type == "road_border":
        color = "#303030"
        width = 1.8
        alpha = 0.88

    elif way_type == "guard_rail":
        color = "#3A3A3A"
        width = 1.8
        alpha = 0.85

    elif way_type == "line_thin":
        color = "#F1F1F1"
        width = 1.6
        alpha = 0.72
        if subtype == "dashed":
            style = (0, (6, 5))

    elif way_type == "line_thick":
        color = "#F7F7F7"
        width = 2.6
        alpha = 0.76
        if subtype == "dashed":
            style = (0, (7, 5))

    elif way_type == "pedestrian_marking":
        color = "#F3F3F3"
        width = 1.8
        alpha = 0.62
        style = ":"

    elif way_type == "stop_line":
        color = "#FAFAFA"
        width = 3.0
        alpha = 0.82
        draw_flag = DRAW_STOP_LINE

    elif way_type == "zebra_marking":
        color = "#F8F8F8"
        width = 2.6
        alpha = 0.68
        draw_flag = DRAW_ZEBRA

    elif way_type == "wait_line":
        color = "#E8E2A8"
        width = 2.2
        alpha = 0.62
        style = (0, (5, 4))
        draw_flag = DRAW_WAIT_LINE

    elif way_type == "virtual":
        color = "#4A6CFF"
        width = 1.0
        alpha = 0.18
        style = ":"
        draw_flag = DRAW_VIRTUAL_LINES

    else:
        draw_flag = False

    return color, width, style, alpha, draw_flag


def load_global_map_if_needed():
    global GLOBAL_MAP_DATA

    if GLOBAL_MAP_DATA is not None:
        return GLOBAL_MAP_DATA

    if not os.path.exists(MAP_OSM_PATH):
        print(f"[WARN] Map file not found: {os.path.abspath(MAP_OSM_PATH)}")
        print("[WARN] SIND files will be visualized without map background.")
        GLOBAL_MAP_DATA = None
        return None

    try:
        GLOBAL_MAP_DATA = parse_osm(MAP_OSM_PATH)
        print(f"[INFO] Loaded map: {os.path.abspath(MAP_OSM_PATH)}")
        print(f"[INFO] Parsed map ways: {len(GLOBAL_MAP_DATA.ways)}")
        print(f"[INFO] Map offset used: {MAP_OFFSET}")
    except Exception as e:
        print(f"[WARN] Failed to parse map file: {e}")
        print("[WARN] SIND files will be visualized without map background.")
        GLOBAL_MAP_DATA = None

    return GLOBAL_MAP_DATA


def draw_map_background(ax, map_data, offset=(0.0, 0.0)):
    if map_data is None:
        return

    dx, dy = float(offset[0]), float(offset[1])

    ax.set_facecolor(MAP_FACE_COLOR)
    ax.set_aspect("equal", adjustable="box")

    for w in map_data.ways:
        color, lw, ls, alpha, draw_flag = get_map_style(w.way_type, w.subtype)

        if (color is None) or (not draw_flag):
            continue

        ax.plot(
            w.x + dx,
            w.y + dy,
            color=color,
            linewidth=lw,
            linestyle=ls,
            alpha=alpha * MAP_GLOBAL_ALPHA,
            zorder=1,
            solid_capstyle="round",
            dash_capstyle="round",
        )


def is_SIND_csv_by_name(csv_path_or_stem):
    stem = Path(csv_path_or_stem).stem
    return stem.startswith("GEI_SIND")


def sanitize_tem_for_principle(v):
    """
    TEM visual semantics:
    - TEM = 0 is valid and must be shown exactly at the origin.
    - positive finite TEM is shown normally.
    - missing / negative / very large TEM is treated as +inf.
    """
    x = safe_float(v, np.nan)

    if np.isposinf(x):
        return np.inf

    if not np.isfinite(x):
        return np.inf

    if x < 0:
        return np.inf

    if x > INF_DISPLAY_THRESHOLD:
        return np.inf

    return x


def sanitize_indepth_for_principle(v, default=0.0):
    """
    InDepth visual semantics:
    - InDepth must be non-negative.
    - positive finite value is shown directly.
    - +inf is not meaningful for InDepth drawing, so fallback to default.
    """
    x = safe_float(v, np.nan)

    if not np.isfinite(x):
        return default

    if x < 0:
        return default

    return x


def sanitize_ei_for_principle(v, default=0.0):
    """
    EI visual semantics:
    - finite EI is shown directly.
    - +inf is preserved and mapped to the highest risk color.
    - missing / -inf is treated as default.
    """
    x = safe_float(v, np.nan)

    if np.isposinf(x):
        return np.inf

    if not np.isfinite(x):
        return default

    if x < 0:
        return default

    if x >= INF_DISPLAY_THRESHOLD:
        return np.inf

    return x


def resolve_ei_from_tem_indepth(tem_value, indepth_value, ei_value):
    """
    Core correction for already-colliding frames:

    If TEM = 0 and InDepth > 0, then EI = InDepth / TEM = +inf.
    This must override finite/default CSV values for visualization.

    If EI is missing but TEM > 0, infer EI = InDepth / TEM.
    """
    tem = sanitize_tem_for_principle(tem_value)
    indepth = sanitize_indepth_for_principle(indepth_value, default=0.0)
    ei_raw = sanitize_ei_for_principle(ei_value, default=np.nan)

    if np.isfinite(tem) and abs(tem) <= EPS_TEM_ZERO:
        if indepth > EPS_INDEPTH_POSITIVE:
            return np.inf
        if np.isfinite(ei_raw):
            return ei_raw
        return 0.0

    if np.isfinite(ei_raw):
        return ei_raw

    if np.isfinite(tem) and tem > EPS_TEM_ZERO:
        return indepth / tem

    return np.inf if indepth > EPS_INDEPTH_POSITIVE else 0.0


def principle_color_from_ei(ei_value, alpha=1.0):
    ei = sanitize_ei_for_principle(ei_value, default=0.0)

    if np.isposinf(ei):
        ei_plot = PRINCIPLE_EI_MAX
    else:
        ei_plot = clip_val(ei, 0.0, PRINCIPLE_EI_MAX)

    rgba = PRINCIPLE_CMAP(PRINCIPLE_EI_NORM(ei_plot))
    return rgba[0], rgba[1], rgba[2], alpha


def format_principle_value(v, decimals=2, inf_threshold=INF_DISPLAY_THRESHOLD):
    x = safe_float(v, np.nan)

    if np.isposinf(x):
        return "inf"

    if not np.isfinite(x):
        return "NaN"

    if x >= inf_threshold:
        return "inf"

    return f"{x:.{decimals}f}"


def derive_principle_cap(df, cols, default_value, lower, upper, treat_as_tem=False):
    values = []

    for col in cols:
        if col not in df.columns:
            continue

        arr = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)

        if treat_as_tem:
            arr = arr[np.isfinite(arr) & (arr >= 0) & (arr < INF_DISPLAY_THRESHOLD)]
        else:
            arr = arr[np.isfinite(arr) & (arr >= 0)]

        if arr.size > 0:
            values.append(arr)

    if not values:
        return default_value

    try:
        all_values = np.concatenate(values)
        q = float(np.nanquantile(all_values, 0.90 if treat_as_tem else 0.95))

        if not np.isfinite(q) or q <= 0:
            return default_value

        return clip_val(q, lower, upper)
    except Exception:
        return default_value


def format_agent_type_for_label(agent_type):
    """
    Format agent type for legend labels.

    Rule:
    - If the CSV value is already all-uppercase, keep it unchanged.
      Example: PTW -> PTW
    - If the CSV value is all-lowercase, apply the current title-case rule.
      Example: car -> Car, ptw -> Ptw
    - Otherwise, keep the original spelling as much as possible.
      Example: eBike -> eBike
    """
    s = str(agent_type).strip()

    if not s:
        return ""

    if s.isupper():
        return s

    if s.islower():
        return s.title()

    return s


# ============================================================
# Axis styling
# ============================================================

def style_scene_axis(ax, x_min, x_max, y_min, y_max):
    ax.set_facecolor("white")

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_linewidth(rcParams["axes.linewidth"])
    ax.spines["bottom"].set_linewidth(rcParams["axes.linewidth"])
    ax.spines["left"].set_color("#2A2A2A")
    ax.spines["bottom"].set_color("#2A2A2A")

    ax.set_aspect("equal")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax.tick_params(axis="both", which="major", pad=4, length=5)
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.yaxis.set_major_locator(MultipleLocator(20))

    ax.set_xlabel("Position X (m)", labelpad=6)
    ax.set_ylabel("Position Y (m)", labelpad=6)

    if GRID_ON:
        ax.grid(
            True,
            linestyle=GRID_LINESTYLE,
            linewidth=GRID_LINEWIDTH,
            color=GRID_COLOR,
            alpha=GRID_ALPHA,
        )
    else:
        ax.grid(False)


def style_curve_axis(ax, x_min, x_max):
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    ax.set_xlim(x_min, x_max)

    ax.tick_params(axis="both", which="major", pad=2, length=4)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.1f}"))

    if CURVE_GRID_ON:
        ax.grid(
            True,
            alpha=CURVE_GRID_ALPHA,
            linestyle=CURVE_GRID_LS,
            linewidth=CURVE_GRID_LW,
        )
    else:
        ax.grid(False)


# ============================================================
# Frame preparation
# ============================================================

def build_frames_from_df(df, time_col):
    df2 = df.copy()
    df2["_time_numeric_"] = pd.to_numeric(df2[time_col], errors="coerce")
    df2 = df2[np.isfinite(df2["_time_numeric_"])].copy()

    if df2.empty:
        return []

    df2["_time_key_"] = df2["_time_numeric_"].round(6)

    frames = [g.iloc[0].copy() for _, g in df2.groupby("_time_key_", sort=True)]

    for r in frames:
        r["time_actual"] = safe_float(r.get("_time_key_", np.nan), np.nan)

    return frames


def apply_frame_step(frames, frame_step):
    frame_step = max(1, int(frame_step))
    indices = list(range(0, len(frames), frame_step))
    sampled_frames = [frames[i] for i in indices]
    return sampled_frames, indices


def apply_visualize_time_range(frames, indices, visualize_time_range):
    if len(frames) == 0:
        return frames, indices, 0, 0

    start_ratio = float(visualize_time_range[0])
    end_ratio = float(visualize_time_range[1])

    start_ratio = max(0.0, min(1.0, start_ratio))
    end_ratio = max(0.0, min(1.0, end_ratio))

    if end_ratio < start_ratio:
        start_ratio, end_ratio = end_ratio, start_ratio

    n = len(frames)

    start_idx = int(math.floor(start_ratio * n))
    end_idx_exclusive = int(math.ceil(end_ratio * n))

    start_idx = max(0, min(start_idx, n - 1))
    end_idx_exclusive = max(start_idx + 1, min(end_idx_exclusive, n))

    sliced_frames = frames[start_idx:end_idx_exclusive]
    sliced_indices = indices[start_idx:end_idx_exclusive]

    return sliced_frames, sliced_indices, start_idx, end_idx_exclusive


def compute_axis_limits_from_display_frames(display_frames, posx1_col, posy1_col, posx2_col, posy2_col):
    xs = []
    ys = []

    for r in display_frames:
        x1 = safe_float(r.get(posx1_col, np.nan), np.nan)
        y1 = safe_float(r.get(posy1_col, np.nan), np.nan)
        x2 = safe_float(r.get(posx2_col, np.nan), np.nan)
        y2 = safe_float(r.get(posy2_col, np.nan), np.nan)

        if np.isfinite(x1):
            xs.append(x1)
        if np.isfinite(x2):
            xs.append(x2)
        if np.isfinite(y1):
            ys.append(y1)
        if np.isfinite(y2):
            ys.append(y2)

    if len(xs) == 0 or len(ys) == 0:
        return -50, 50, -50, 50

    x_min = float(np.min(xs)) - X_MARGIN_LEFT
    x_max = float(np.max(xs)) + X_MARGIN_RIGHT
    y_min = float(np.min(ys)) - Y_MARGIN_BOTTOM
    y_max = float(np.max(ys)) + Y_MARGIN_TOP

    if x_max - x_min < 1e-6:
        x_min -= 10.0
        x_max += 10.0

    if y_max - y_min < 1e-6:
        y_min -= 10.0
        y_max += 10.0

    return x_min, x_max, y_min, y_max


def infer_frame_durations_ms(display_frames):
    if len(display_frames) <= 1:
        return [FALLBACK_DURATION_MS]

    times = [safe_float(r.get("time_actual", np.nan), np.nan) for r in display_frames]

    valid_dts = []
    for i in range(len(times) - 1):
        t0 = times[i]
        t1 = times[i + 1]

        if np.isfinite(t0) and np.isfinite(t1) and t1 > t0:
            valid_dts.append(t1 - t0)

    if len(valid_dts) == 0:
        return [FALLBACK_DURATION_MS] * len(display_frames)

    median_dt = float(np.median(valid_dts))
    durations = []

    for i in range(len(times)):
        if i < len(times) - 1:
            t0 = times[i]
            t1 = times[i + 1]
            dt = (t1 - t0) if (np.isfinite(t0) and np.isfinite(t1) and t1 > t0) else median_dt
        else:
            dt = median_dt

        ms = int(round(dt * 1000.0))
        ms = max(MIN_GIF_DURATION_MS, ms)
        durations.append(ms)

    return durations


# ============================================================
# Metric curve configuration
# ============================================================

def get_metric_configs(df_cols):
    return [
        {
            "key": "bbox",
            "ylabel": "BBox distance (m)",
            "type": "single_pos",
            "col": choose_col(df_cols, ["BBox distance (m)"]),
            "color": COL_SOFTRED,
        },
        {
            "key": "vel",
            "ylabel": "Velocity (m/s)",
            "type": "double_pos",
            "col1": choose_col(df_cols, ["Velocity (m/s)"]),
            "col2": choose_col(df_cols, ["2_Velocity (m/s)"]),
            "color1": COL_RED,
            "color2": COL_GREEN,
        },
        {
            "key": "gei",
            "ylabel": "GEI (m/s)",
            "type": "single_pos",
            "col": choose_col(df_cols, ["GEI"]),
            "color": CURVE_MAIN_COLOR,
        },
        {
            "key": "mei",
            "ylabel": "MEI (m/s)",
            "type": "single_pos",
            "col": choose_col(df_cols, ["MEI"]),
            "color": CURVE_MAIN_COLOR,
        },
        {
            "key": "drac",
            "ylabel": "DRAC (m/s^2)",
            "type": "single_pos",
            "col": choose_col(df_cols, ["DRAC"]),
            "color": CURVE_MAIN_COLOR,
        },
        {
            "key": "alon",
            "ylabel": "Lon. accel. (m/s^2)",
            "type": "double_sym",
            "col1": choose_col(df_cols, ["A_lon", "Lon_acc", "Lon accel", "Longitudinal Acceleration"]),
            "col2": choose_col(df_cols, ["2_A_lon", "2_Lon_acc", "2_Lon accel", "2_Longitudinal Acceleration"]),
            "color1": COL_RED,
            "color2": COL_GREEN,
        },
        {
            "key": "alat",
            "ylabel": "Lat. accel. (m/s^2)",
            "type": "double_sym",
            "col1": choose_col(df_cols, ["A_lat", "Lat_acc", "Lat accel", "Lateral Acceleration"]),
            "col2": choose_col(df_cols, ["2_A_lat", "2_Lat_acc", "2_Lat accel", "2_Lateral Acceleration"]),
            "color1": COL_RED,
            "color2": COL_GREEN,
        },
        {
            "key": "ttc",
            "ylabel": "TTC (s)",
            "type": "single_tem",
            "col": choose_col(df_cols, ["TTC"]),
            "color": CURVE_MAIN_COLOR,
        },
        {
            "key": "ttc2d",
            "ylabel": "TTC2D (s)",
            "type": "single_tem",
            "col": choose_col(df_cols, ["TTC2D"]),
            "color": CURVE_MAIN_COLOR,
        },
        {
            "key": "act",
            "ylabel": "ACT (s)",
            "type": "single_tem",
            "col": choose_col(df_cols, ["ACT"]),
            "color": CURVE_MAIN_COLOR,
        },
    ]


# ============================================================
# Renderer
# ============================================================

class FastGifRenderer:
    def __init__(
        self,
        stem,
        metric_name,
        display_frames,
        x_min,
        x_max,
        y_min,
        y_max,
        value_col,
        colorbar_norm,
        colorbar_cmap,
        colorbar_label,
        colorbar_ticks,
        posx1_col,
        posy1_col,
        posx2_col,
        posy2_col,
        heading1_col,
        heading2_col,
        len1_col,
        wid1_col,
        len2_col,
        wid2_col,
        vel1_col,
        vel2_col,
        show_speed_text,
        curve_df,
        label1,
        label2,
        use_map_background=False,
        map_data=None,
        map_offset=(0.0, 0.0),
    ):
        self.stem = stem
        self.metric_name = metric_name
        self.display_frames = display_frames
        self.value_col = value_col

        self.use_map_background = bool(use_map_background)
        self.map_data = map_data
        self.map_offset = map_offset

        self.posx1_col = posx1_col
        self.posy1_col = posy1_col
        self.posx2_col = posx2_col
        self.posy2_col = posy2_col
        self.heading1_col = heading1_col
        self.heading2_col = heading2_col
        self.len1_col = len1_col
        self.wid1_col = wid1_col
        self.len2_col = len2_col
        self.wid2_col = wid2_col
        self.vel1_col = vel1_col
        self.vel2_col = vel2_col
        self.show_speed_text = show_speed_text

        self.label1 = label1
        self.label2 = label2

        self.curve_df = curve_df.copy()
        self.curve_time = pd.to_numeric(self.curve_df["time_actual"], errors="coerce").to_numpy(dtype=float)

        self.principle_tem_cap = derive_principle_cap(
            self.curve_df,
            [m[1] for m in MODE_PRINCIPLE_SPECS],
            default_value=PRINCIPLE_TEM_CAP_DEFAULT,
            lower=PRINCIPLE_TEM_CAP_RANGE[0],
            upper=PRINCIPLE_TEM_CAP_RANGE[1],
            treat_as_tem=True,
        )

        self.principle_indepth_cap = derive_principle_cap(
            self.curve_df,
            [m[2] for m in MODE_PRINCIPLE_SPECS],
            default_value=PRINCIPLE_INDEPTH_CAP_DEFAULT,
            lower=PRINCIPLE_INDEPTH_CAP_RANGE[0],
            upper=PRINCIPLE_INDEPTH_CAP_RANGE[1],
            treat_as_tem=False,
        )

        self.fig = plt.figure(figsize=FIGSIZE, dpi=FIG_DPI, facecolor="white")

        self.gs = GridSpec(
            nrows=3,
            ncols=12,
            figure=self.fig,
            height_ratios=[1.18, 0.82, 0.82],
            width_ratios=[1, 1, 1, 1, 1, 1, 1, 1, 0.92, 0.92, 0.92, 0.18],
            hspace=0.34,
            wspace=0.32,
        )

        self.ax = self.fig.add_subplot(self.gs[0, 0:8])
        self.ax_principle = self.fig.add_subplot(self.gs[0, 8:11])
        self.cax = self.fig.add_subplot(self.gs[0, 11])

        sm = ScalarMappable(norm=colorbar_norm, cmap=colorbar_cmap)
        sm.set_array([])

        self.cbar = self.fig.colorbar(sm, cax=self.cax)
        self.cbar.set_label(colorbar_label, fontsize=11.5)
        self.cbar.ax.tick_params(labelsize=9.5, width=0.9, length=3.5)
        self.cbar.outline.set_linewidth(0.9)
        self.cbar.set_ticks(colorbar_ticks)

        style_scene_axis(self.ax, x_min, x_max, y_min, y_max)

        # Draw static SIND map background only when the current CSV is a SIND file.
        # For all other ei_*.csv files, this block is skipped and the original white background is preserved.
        if self.use_map_background and self.map_data is not None:
            draw_map_background(self.ax, self.map_data, offset=self.map_offset)

        self.hist_line1, = self.ax.plot(
            [],
            [],
            color=HISTORY_TRAJ_COLOR,
            linewidth=HISTORY_TRAJ_LINEWIDTH,
            alpha=HISTORY_TRAJ_ALPHA,
            zorder=6,
            solid_capstyle="round",
        )

        self.hist_line2, = self.ax.plot(
            [],
            [],
            color=HISTORY_TRAJ_COLOR,
            linewidth=HISTORY_TRAJ_LINEWIDTH,
            alpha=HISTORY_TRAJ_ALPHA,
            zorder=6,
            solid_capstyle="round",
        )

        self.vehicle1 = Polygon(
            np.zeros((4, 2)),
            closed=True,
            facecolor=(1, 1, 1, 1),
            edgecolor=EDGE_COLOR,
            linewidth=EDGE_WIDTH,
            alpha=1.0,
            zorder=20,
            visible=False,
            joinstyle="round",
        )

        self.vehicle2 = Polygon(
            np.zeros((4, 2)),
            closed=True,
            facecolor=(1, 1, 1, 1),
            edgecolor=EDGE_COLOR,
            linewidth=EDGE_WIDTH,
            alpha=1.0,
            zorder=20,
            visible=False,
            joinstyle="round",
        )

        self.ax.add_patch(self.vehicle1)
        self.ax.add_patch(self.vehicle2)

        self.dynamic_artists = []
        self.principle_dynamic_artists = []

        self.ax_title = self.ax.set_title("", pad=8, fontsize=13)
        self.sup_title = self.fig.suptitle("", y=0.988, fontsize=12.5)

        self._init_gei_principle_panel()

        self.curve_axes = []
        self.curve_artists = []

        subgs = self.gs[1:, :].subgridspec(2, 5, hspace=0.34, wspace=0.28)
        self.metric_configs = get_metric_configs(self.curve_df.columns)

        for i, cfg in enumerate(self.metric_configs):
            r, c = divmod(i, 5)
            axc = self.fig.add_subplot(subgs[r, c])
            self.curve_axes.append(axc)
            self.curve_artists.append(self._init_curve_panel(axc, cfg))

        self.canvas = FigureCanvas(self.fig)

    def _clear_dynamic_artists(self):
        for artist in self.dynamic_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self.dynamic_artists = []

    def _clear_principle_dynamic_artists(self):
        for artist in self.principle_dynamic_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self.principle_dynamic_artists = []

    def _init_gei_principle_panel(self):
        axp = self.ax_principle

        axp.set_xlim(0, 1)
        axp.set_ylim(0, 1)
        axp.axis("off")
        axp.set_facecolor("white")

        bg = FancyBboxPatch(
            (0.01, 0.01),
            0.98,
            0.98,
            boxstyle="round,pad=0.014,rounding_size=0.03",
            facecolor=PRINCIPLE_PANEL_FACE,
            edgecolor=PRINCIPLE_PANEL_EDGE,
            linewidth=1.0,
            alpha=0.985,
            zorder=0,
        )
        axp.add_patch(bg)

        axp.text(
            0.06,
            0.955,
            "GEI Principle View",
            ha="left",
            va="top",
            fontsize=10.2,
            fontweight="bold",
            color=PRINCIPLE_TEXT_COLOR,
            zorder=2,
        )

        axp.text(
            0.06,
            0.905,
            r"$EI=\dfrac{InDepth}{TEM}$",
            ha="left",
            va="top",
            fontsize=8.8,
            color=PRINCIPLE_SUBTEXT_COLOR,
            zorder=2,
        )

        avg_x, avg_y, avg_w, avg_h = 0.055, 0.54, 0.89, 0.255

        avg_bg = FancyBboxPatch(
            (avg_x, avg_y),
            avg_w,
            avg_h,
            boxstyle="round,pad=0.010,rounding_size=0.022",
            facecolor=PRINCIPLE_CELL_FACE,
            edgecolor=PRINCIPLE_CELL_EDGE,
            linewidth=0.95,
            alpha=1.0,
            zorder=1,
        )
        axp.add_patch(avg_bg)

        axp.text(
            avg_x + 0.02 * avg_w,
            avg_y + 0.90 * avg_h,
            "GEI",
            ha="left",
            va="top",
            fontsize=8.3,
            fontweight="bold",
            color=PRINCIPLE_TEXT_COLOR,
            zorder=3,
        )

        self.avg_panel = {
            "x": avg_x,
            "y": avg_y,
            "w": avg_w,
            "h": avg_h,
            "x0": avg_x + 0.13 * avg_w,
            "x1": avg_x + 0.93 * avg_w,
            "y0": avg_y + 0.18 * avg_h,
            "y1": avg_y + 0.70 * avg_h,
        }

        self._draw_panel_axes(self.avg_panel, show_axis_labels=True, label_scale=1.0)

        positions = [
            (0.055, 0.315, 0.415, 0.145),
            (0.53, 0.315, 0.415, 0.145),
            (0.055, 0.12, 0.415, 0.145),
            (0.53, 0.12, 0.415, 0.145),
        ]

        self.principle_cells = []

        for (mode_name, tem_col, indepth_col, ei_col), (x, y, w, h) in zip(MODE_PRINCIPLE_SPECS, positions):
            cell_bg = FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.008,rounding_size=0.018",
                facecolor=PRINCIPLE_CELL_FACE,
                edgecolor=PRINCIPLE_CELL_EDGE,
                linewidth=0.85,
                alpha=1.0,
                zorder=1,
            )
            axp.add_patch(cell_bg)

            cell = {
                "mode_name": mode_name,
                "tem_col": tem_col,
                "indepth_col": indepth_col,
                "ei_col": ei_col,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "x0": x + 0.12 * w,
                "x1": x + 0.89 * w,
                "y0": y + 0.19 * h,
                "y1": y + 0.63 * h,
            }

            self.principle_cells.append(cell)
            self._draw_panel_axes(cell, show_axis_labels=False, label_scale=0.8)

            axp.text(
                x + 0.05 * w,
                y + 0.86 * h,
                mode_name,
                ha="left",
                va="top",
                fontsize=7.7,
                fontweight="bold",
                color=PRINCIPLE_TEXT_COLOR,
                zorder=3,
            )

    def _draw_panel_axes(self, panel, show_axis_labels=False, label_scale=1.0):
        axp = self.ax_principle

        x0, x1, y0, y1 = panel["x0"], panel["x1"], panel["y0"], panel["y1"]

        axp.plot(
            [x0, x1],
            [y0, y0],
            color=PRINCIPLE_AXIS_COLOR,
            linewidth=0.85,
            alpha=0.92,
            zorder=2,
        )

        axp.plot(
            [x0, x0],
            [y0, y1],
            color=PRINCIPLE_AXIS_COLOR,
            linewidth=0.85,
            alpha=0.92,
            zorder=2,
        )

        axp.plot(
            [x0, x1],
            [y0 + 0.5 * (y1 - y0), y0 + 0.5 * (y1 - y0)],
            color=PRINCIPLE_GRID_COLOR,
            linewidth=0.65,
            linestyle="--",
            alpha=0.9,
            zorder=2,
        )

        axp.plot(
            [x0 + 0.5 * (x1 - x0), x0 + 0.5 * (x1 - x0)],
            [y0, y1],
            color=PRINCIPLE_GRID_COLOR,
            linewidth=0.65,
            linestyle="--",
            alpha=0.9,
            zorder=2,
        )

        if show_axis_labels:
            axp.text(
                (x0 + x1) * 0.5,
                y0 - 0.040,
                "TEM",
                ha="center",
                va="top",
                fontsize=7.0 * label_scale,
                color=PRINCIPLE_SUBTEXT_COLOR,
                zorder=2,
            )

            axp.text(
                x0 - 0.055,
                (y0 + y1) * 0.5,
                "InDepth",
                ha="center",
                va="center",
                rotation=90,
                fontsize=7.0 * label_scale,
                color=PRINCIPLE_SUBTEXT_COLOR,
                zorder=2,
            )

    def _draw_triangle_in_panel(self, panel, tem_value, indepth_value, ei_value, draw_summary=False):
        axp = self.ax_principle

        tem = sanitize_tem_for_principle(tem_value)
        indepth = sanitize_indepth_for_principle(indepth_value, default=0.0)
        ei = resolve_ei_from_tem_indepth(tem, indepth, ei_value)

        x0, x1, y0, y1 = panel["x0"], panel["x1"], panel["y0"], panel["y1"]

        avail_w = x1 - x0
        avail_h = y1 - y0

        if np.isinf(tem):
            tem_norm = 1.0
        else:
            tem_norm = clip_val(tem / self.principle_tem_cap, 0.0, 1.0)

        indepth_norm = clip_val(indepth / self.principle_indepth_cap, 0.0, 1.0)

        xt = x0 + avail_w * tem_norm
        yt = y0 + avail_h * indepth_norm

        face_rgba = principle_color_from_ei(ei, alpha=0.84 if draw_summary else 0.80)
        edge_rgba = principle_color_from_ei(ei, alpha=0.98)

        tri = Polygon(
            np.array([[x0, y0], [xt, y0], [xt, yt]], dtype=float),
            closed=True,
            facecolor=face_rgba,
            edgecolor=edge_rgba,
            linewidth=1.0 if draw_summary else 0.85,
            zorder=5,
            joinstyle="round",
        )
        axp.add_patch(tri)
        self.principle_dynamic_artists.append(tri)

        diag, = axp.plot(
            [x0, xt],
            [y0, yt],
            color="#4E4E4E",
            linewidth=0.95 if draw_summary else 0.72,
            alpha=0.86,
            zorder=6,
        )
        self.principle_dynamic_artists.append(diag)

        if np.isfinite(tem) and abs(tem) <= EPS_TEM_ZERO and indepth > EPS_INDEPTH_POSITIVE:
            vline, = axp.plot(
                [x0, x0],
                [y0, yt],
                color=edge_rgba,
                linewidth=1.35 if draw_summary else 1.05,
                alpha=0.98,
                zorder=7,
                solid_capstyle="round",
            )
            self.principle_dynamic_artists.append(vline)

        pt = axp.scatter(
            [xt],
            [yt],
            s=22 if draw_summary else 15,
            color=[edge_rgba],
            edgecolors="white",
            linewidths=0.5,
            zorder=8,
        )
        self.principle_dynamic_artists.append(pt)

        if draw_summary:
            txt_main = axp.text(
                panel["x"] + 0.74 * panel["w"],
                panel["y"] + 0.78 * panel["h"],
                f"GEI = {format_principle_value(ei, decimals=3, inf_threshold=INF_DISPLAY_THRESHOLD)}",
                ha="center",
                va="center",
                fontsize=8.8,
                fontweight="bold",
                color=PRINCIPLE_TEXT_COLOR,
                bbox=dict(
                    boxstyle="round,pad=0.22,rounding_size=0.12",
                    facecolor=principle_color_from_ei(ei, alpha=0.88),
                    edgecolor=principle_color_from_ei(ei, alpha=1.0),
                    linewidth=0.7,
                ),
                zorder=9,
            )
            self.principle_dynamic_artists.append(txt_main)
        else:
            x, y, w, h = panel["x"], panel["y"], panel["w"], panel["h"]

            txt_ei = axp.text(
                x + 0.95 * w,
                y + 0.85 * h,
                f"EI={format_principle_value(ei)}",
                ha="right",
                va="top",
                fontsize=6.6,
                color=PRINCIPLE_SUBTEXT_COLOR,
                zorder=9,
            )
            self.principle_dynamic_artists.append(txt_ei)

    def _update_gei_principle_panel(self, row):
        self._clear_principle_dynamic_artists()

        mode_indepth_vals = []
        mode_tem_vals = []
        mode_ei_vals = []

        for cell in self.principle_cells:
            tem_raw = row.get(cell["tem_col"], np.nan)
            indepth_raw = row.get(cell["indepth_col"], np.nan)
            ei_raw = row.get(cell["ei_col"], np.nan)

            tem_clean = sanitize_tem_for_principle(tem_raw)
            indepth_clean = sanitize_indepth_for_principle(indepth_raw, default=0.0)
            ei_clean = resolve_ei_from_tem_indepth(tem_clean, indepth_clean, ei_raw)

            mode_tem_vals.append(tem_clean)
            mode_indepth_vals.append(indepth_clean)
            mode_ei_vals.append(ei_clean)

            self._draw_triangle_in_panel(
                panel=cell,
                tem_value=tem_clean,
                indepth_value=indepth_clean,
                ei_value=ei_clean,
                draw_summary=False,
            )

        indepth_eff = row.get("InDepth_eff", np.nan)
        tem_eff = row.get("TEM_eff", np.nan)
        gei = row.get("GEI", np.nan)

        if not np.isfinite(safe_float(indepth_eff, np.nan)):
            indepth_eff = sum(mode_indepth_vals) / 4.0

        tem_eff_clean = sanitize_tem_for_principle(tem_eff)
        indepth_eff_clean = sanitize_indepth_for_principle(indepth_eff, default=0.0)

        gei_clean = sanitize_ei_for_principle(gei, default=np.nan)

        if np.isinf(tem_eff_clean):
            finite_mode_tems = [v for v in mode_tem_vals if np.isfinite(v)]
            if finite_mode_tems:
                tem_eff_clean = sum(finite_mode_tems) / len(finite_mode_tems)
            else:
                tem_eff_clean = np.inf

        finite_mode_tems = [v for v in mode_tem_vals if np.isfinite(v)]
        all_valid_tems_zero = (
            len(finite_mode_tems) > 0
            and all(abs(v) <= EPS_TEM_ZERO for v in finite_mode_tems)
        )

        if all_valid_tems_zero and indepth_eff_clean > EPS_INDEPTH_POSITIVE:
            tem_eff_clean = 0.0
            gei_clean = np.inf
        else:
            gei_clean = resolve_ei_from_tem_indepth(tem_eff_clean, indepth_eff_clean, gei_clean)

        self._draw_triangle_in_panel(
            panel=self.avg_panel,
            tem_value=tem_eff_clean,
            indepth_value=indepth_eff_clean,
            ei_value=gei_clean,
            draw_summary=True,
        )

    def _update_history(self, i):
        if (not DRAW_CENTER_TRAJECTORY) or i < 1:
            self.hist_line1.set_data([], [])
            self.hist_line2.set_data([], [])
            return

        frames_history = self.display_frames[: i + 1]

        xs1 = np.array(
            [safe_float(r.get(self.posx1_col, np.nan), np.nan) for r in frames_history],
            dtype=float,
        )
        ys1 = np.array(
            [safe_float(r.get(self.posy1_col, np.nan), np.nan) for r in frames_history],
            dtype=float,
        )
        xs2 = np.array(
            [safe_float(r.get(self.posx2_col, np.nan), np.nan) for r in frames_history],
            dtype=float,
        )
        ys2 = np.array(
            [safe_float(r.get(self.posy2_col, np.nan), np.nan) for r in frames_history],
            dtype=float,
        )

        valid1 = np.isfinite(xs1) & np.isfinite(ys1)
        valid2 = np.isfinite(xs2) & np.isfinite(ys2)

        if np.sum(valid1) >= 2:
            self.hist_line1.set_data(xs1[valid1], ys1[valid1])
        else:
            self.hist_line1.set_data([], [])

        if np.sum(valid2) >= 2:
            self.hist_line2.set_data(xs2[valid2], ys2[valid2])
        else:
            self.hist_line2.set_data([], [])

    def _update_vehicle_patch(self, patch, x, y, heading, length, width, face_rgba):
        if np.isfinite(x) and np.isfinite(y) and np.isfinite(heading) and length > 0 and width > 0:
            poly = build_vehicle_polygon(x, y, heading, length, width)
            patch.set_xy(poly)
            patch.set_facecolor(face_rgba)
            patch.set_alpha(face_rgba[3])
            patch.set_visible(True)
        else:
            patch.set_visible(False)

    def _draw_speed_annotation(self, row):
        if not DRAW_SPEED_ARROW:
            return

        x1 = safe_float(row.get(self.posx1_col, np.nan), np.nan)
        y1 = safe_float(row.get(self.posy1_col, np.nan), np.nan)
        a1 = safe_float(row.get(self.heading1_col, np.nan), np.nan)
        v1 = safe_float(row.get(self.vel1_col, np.nan), np.nan)

        x2 = safe_float(row.get(self.posx2_col, np.nan), np.nan)
        y2 = safe_float(row.get(self.posy2_col, np.nan), np.nan)
        a2 = safe_float(row.get(self.heading2_col, np.nan), np.nan)
        v2 = safe_float(row.get(self.vel2_col, np.nan), np.nan)

        for x, y, heading, speed in [(x1, y1, a1, v1), (x2, y2, a2, v2)]:
            if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(heading) and np.isfinite(speed)):
                continue

            arrow_len = max(0.8, speed * ARROW_LENGTH_SCALE)
            dx = arrow_len * math.cos(heading)
            dy = arrow_len * math.sin(heading)

            arr = self.ax.arrow(
                x,
                y,
                dx,
                dy,
                width=0.0,
                head_width=ARROW_HEAD_WIDTH,
                head_length=ARROW_HEAD_LENGTH,
                length_includes_head=True,
                fc=ARROW_COLOR,
                ec=ARROW_COLOR,
                linewidth=ARROW_LINEWIDTH,
                alpha=ARROW_ALPHA,
                zorder=30,
            )
            self.dynamic_artists.append(arr)

            if not self.show_speed_text:
                continue

            norm = math.hypot(dx, dy)
            ux, uy = (dx / norm, dy / norm) if norm > 1e-8 else (1.0, 0.0)

            tx = x + dx + ux * SPEED_TEXT_OFFSET
            ty = y + dy + uy * SPEED_TEXT_OFFSET

            txt = self.ax.text(
                tx,
                ty,
                f"v={speed:.1f} m/s",
                fontsize=10.0,
                color="#555555",
                ha="left" if ux >= 0 else "right",
                va="bottom" if uy >= 0 else "top",
                bbox=dict(
                    boxstyle="round,pad=0.22,rounding_size=0.10",
                    facecolor="white",
                    edgecolor="none",
                    alpha=SPEED_TEXT_BBOX_ALPHA,
                ),
                zorder=31,
            )
            self.dynamic_artists.append(txt)

    def _compute_curve_ylim(self, cfg):
        if cfg["type"] == "single_pos":
            col = cfg["col"]
            if col is None:
                return 0, 1

            y = finite_series(self.curve_df[col]).to_numpy(dtype=float)
            y = y[np.isfinite(y)]

            if y.size == 0:
                return 0, 1

            ymax = max(1e-9, float(np.max(y)) * 1.2)

            if cfg.get("key") in ("gei", "mei", "drac") and ymax > 10.0:
                ymax = 10.0

            return 0, ymax

        if cfg["type"] == "single_tem":
            col = cfg["col"]
            if col is None:
                return 0, 1

            y = finite_series_keep_posinf(self.curve_df[col]).replace([np.inf], np.nan).to_numpy(dtype=float)
            y = y[np.isfinite(y)]
            y = y[(y >= 0) & (y < INF_DISPLAY_THRESHOLD)]

            if y.size == 0:
                return 0, 1

            return 0, min(5.0, max(1e-9, float(np.max(y)) * 1.5))

        if cfg["type"] == "double_pos":
            vals = []

            for col in [cfg["col1"], cfg["col2"]]:
                if col is None:
                    continue

                y = finite_series(self.curve_df[col]).to_numpy(dtype=float)
                y = y[np.isfinite(y)]

                if y.size > 0:
                    vals.append(y)

            if not vals:
                return 0, 1

            vmax = max(float(np.max(v)) for v in vals)
            return 0, max(1e-9, vmax * 1.5)

        if cfg["type"] == "double_sym":
            vals = []

            for col in [cfg["col1"], cfg["col2"]]:
                if col is None:
                    continue

                y = finite_series(self.curve_df[col]).to_numpy(dtype=float)
                y = y[np.isfinite(y)]

                if y.size > 0:
                    vals.append(np.abs(y))

            if not vals:
                return -1, 1

            m = max(float(np.max(v)) for v in vals)
            m = max(m * 1.35, 1e-6)

            return -m, m

        return 0, 1

    def _init_curve_panel(self, ax, cfg):
        t = self.curve_time

        xmin = float(np.nanmin(t)) if np.isfinite(t).any() else 0.0
        xmax = float(np.nanmax(t)) if np.isfinite(t).any() else 1.0

        if abs(xmax - xmin) < 1e-9:
            xmax = xmin + 1.0

        style_curve_axis(ax, xmin, xmax)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel(cfg["ylabel"])
        ax.set_ylim(*self._compute_curve_ylim(cfg))

        if cfg["type"] == "double_sym":
            ax.axhline(
                0,
                color="#C7C7C7",
                linestyle="-.",
                linewidth=1.0,
                alpha=0.9,
                zorder=0,
            )

        artists = {
            "vline": ax.axvline(
                x=np.nan,
                color=REF_VLINE_COLOR,
                linestyle=REF_VLINE_LS,
                linewidth=REF_VLINE_LW,
                alpha=REF_VLINE_ALPHA,
                zorder=2,
            )
        }

        if cfg["type"] in ("single_pos", "single_tem"):
            l_pre, = ax.plot(
                [],
                [],
                color=cfg["color"],
                linewidth=CURVE_PRE_LW,
                alpha=CURVE_PRE_ALPHA,
                linestyle="-",
                zorder=3,
            )

            l_post, = ax.plot(
                [],
                [],
                color=cfg["color"],
                linewidth=CURVE_POST_LW,
                alpha=CURVE_POST_ALPHA,
                linestyle=CURVE_POST_LS,
                zorder=3,
            )

            artists["pre"] = l_pre
            artists["post"] = l_post
            artists["nodata"] = ax.text(
                0.5,
                0.5,
                "",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=8.2,
                alpha=0.72,
            )

        else:
            l1_pre, = ax.plot(
                [],
                [],
                color=cfg["color1"],
                linewidth=LEFT_LINEWIDTH,
                alpha=CURVE_PRE_ALPHA,
                linestyle="-",
                zorder=3,
                label=self.label1,
            )

            l1_post, = ax.plot(
                [],
                [],
                color=cfg["color1"],
                linewidth=CURVE_POST_LW,
                alpha=CURVE_POST_ALPHA,
                linestyle=CURVE_POST_LS,
                zorder=3,
            )

            l2_pre, = ax.plot(
                [],
                [],
                color=cfg["color2"],
                linewidth=LEFT_LINEWIDTH,
                alpha=CURVE_PRE_ALPHA,
                linestyle="-",
                zorder=3,
                label=self.label2,
            )

            l2_post, = ax.plot(
                [],
                [],
                color=cfg["color2"],
                linewidth=CURVE_POST_LW,
                alpha=CURVE_POST_ALPHA,
                linestyle=CURVE_POST_LS,
                zorder=3,
            )

            artists["pre1"] = l1_pre
            artists["post1"] = l1_post
            artists["pre2"] = l2_pre
            artists["post2"] = l2_post
            artists["nodata"] = ax.text(
                0.5,
                0.5,
                "",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=8.2,
                alpha=0.72,
            )

            ax.legend(loc="upper right", fontsize=8.2)

        return artists

    @staticmethod
    def _set_split_data_preserve_time(pre_artist, post_artist, t_full, y_full, i):
        if len(t_full) == 0 or len(y_full) == 0:
            pre_artist.set_data([], [])
            post_artist.set_data([], [])
            return False

        t_full = np.asarray(t_full, dtype=float)
        y_full = np.asarray(y_full, dtype=float)

        n = min(len(t_full), len(y_full))
        t_full = t_full[:n]
        y_full = y_full[:n]

        if n == 0:
            pre_artist.set_data([], [])
            post_artist.set_data([], [])
            return False

        end_pre = max(0, min(i + 1, n))
        start_post = max(0, min(i, n - 1))

        t_pre = t_full[:end_pre]
        y_pre = y_full[:end_pre]

        t_post = t_full[start_post:]
        y_post = y_full[start_post:]

        pre_artist.set_data(t_pre, y_pre)
        post_artist.set_data(t_post, y_post)

        return np.isfinite(y_full).any()

    def _update_curve_panel(self, cfg, artists, i, cur_time):
        artists["vline"].set_xdata([cur_time, cur_time])

        if cfg["type"] in ("single_pos", "single_tem"):
            col = cfg["col"]

            if col is None:
                artists["pre"].set_data([], [])
                artists["post"].set_data([], [])
                artists["nodata"].set_text("Missing")
                return

            if cfg["type"] == "single_tem":
                y = finite_series_keep_posinf(self.curve_df[col]).replace([np.inf], np.nan).to_numpy(dtype=float)
                y[y >= INF_DISPLAY_THRESHOLD] = np.nan
            else:
                y = finite_series(self.curve_df[col]).to_numpy(dtype=float)

            ok = self._set_split_data_preserve_time(
                artists["pre"],
                artists["post"],
                self.curve_time,
                y,
                i,
            )

            artists["nodata"].set_text("" if ok else "No finite data")

        elif cfg["type"] in ("double_pos", "double_sym"):
            c1 = cfg["col1"]
            c2 = cfg["col2"]

            ok1 = False
            ok2 = False

            if c1 is not None:
                y1 = finite_series(self.curve_df[c1]).to_numpy(dtype=float)
                ok1 = self._set_split_data_preserve_time(
                    artists["pre1"],
                    artists["post1"],
                    self.curve_time,
                    y1,
                    i,
                )
            else:
                artists["pre1"].set_data([], [])
                artists["post1"].set_data([], [])

            if c2 is not None:
                y2 = finite_series(self.curve_df[c2]).to_numpy(dtype=float)
                ok2 = self._set_split_data_preserve_time(
                    artists["pre2"],
                    artists["post2"],
                    self.curve_time,
                    y2,
                    i,
                )
            else:
                artists["pre2"].set_data([], [])
                artists["post2"].set_data([], [])

            artists["nodata"].set_text("" if (ok1 or ok2) else "No finite data")

    def render_frame(self, i):
        row = self.display_frames[i]

        self._clear_dynamic_artists()
        self._update_history(i)

        raw_value = row.get(self.value_col, np.nan)
        veh_color = gei_to_rgba(raw_value, alpha=0.98)

        x1 = safe_float(row.get(self.posx1_col, np.nan), np.nan)
        y1 = safe_float(row.get(self.posy1_col, np.nan), np.nan)
        a1 = safe_float(row.get(self.heading1_col, np.nan), np.nan)
        l1 = safe_float(row.get(self.len1_col, 0.0), 0.0)
        w1 = safe_float(row.get(self.wid1_col, 0.0), 0.0)

        x2 = safe_float(row.get(self.posx2_col, np.nan), np.nan)
        y2 = safe_float(row.get(self.posy2_col, np.nan), np.nan)
        a2 = safe_float(row.get(self.heading2_col, np.nan), np.nan)
        l2 = safe_float(row.get(self.len2_col, 0.0), 0.0)
        w2 = safe_float(row.get(self.wid2_col, 0.0), 0.0)

        self._update_vehicle_patch(self.vehicle1, x1, y1, a1, l1, w1, veh_color)
        self._update_vehicle_patch(self.vehicle2, x2, y2, a2, l2, w2, veh_color)

        self._draw_speed_annotation(row)
        self._update_gei_principle_panel(row)

        cur_time = safe_float(row.get("time_actual", np.nan), np.nan)
        cur_value = safe_float(raw_value, np.nan)

        if np.isposinf(cur_value) or (np.isfinite(cur_value) and cur_value >= INF_DISPLAY_THRESHOLD):
            self.ax_title.set_text("GEI = inf")
        elif np.isfinite(cur_value):
            self.ax_title.set_text(f"GEI = {cur_value:.3f}")
        else:
            self.ax_title.set_text("GEI = NaN")

        map_text = " | SIND map" if self.use_map_background and self.map_data is not None else ""

        self.sup_title.set_text(
            f"{self.stem}\n"
            f"Colored by {self.metric_name}{map_text} | "
            f"Current time = {cur_time:.3f} s | "
            f"Frame {i + 1}/{len(self.display_frames)}"
        )

        for cfg, artists in zip(self.metric_configs, self.curve_artists):
            self._update_curve_panel(cfg, artists, i, cur_time)

        self.canvas.draw()
        buf = np.asarray(self.canvas.buffer_rgba())[:, :, :3].copy()

        return Image.fromarray(buf)

    def close(self):
        plt.close(self.fig)


# ============================================================
# GIF generation
# ============================================================

def save_gif_fast(
    stem,
    metric_name,
    display_frames,
    durations_ms,
    x_min,
    x_max,
    y_min,
    y_max,
    value_col,
    colorbar_norm,
    colorbar_cmap,
    colorbar_label,
    colorbar_ticks,
    posx1_col,
    posy1_col,
    posx2_col,
    posy2_col,
    heading1_col,
    heading2_col,
    len1_col,
    wid1_col,
    len2_col,
    wid2_col,
    vel1_col,
    vel2_col,
    show_speed_text,
    out_dir,
    curve_df,
    label1,
    label2,
    use_map_background=False,
    map_data=None,
    map_offset=(0.0, 0.0),
):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_gif = out_dir / f"{stem}_{metric_name}_with_curves.gif"

    renderer = FastGifRenderer(
        stem=stem,
        metric_name=metric_name,
        display_frames=display_frames,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        value_col=value_col,
        colorbar_norm=colorbar_norm,
        colorbar_cmap=colorbar_cmap,
        colorbar_label=colorbar_label,
        colorbar_ticks=colorbar_ticks,
        posx1_col=posx1_col,
        posy1_col=posy1_col,
        posx2_col=posx2_col,
        posy2_col=posy2_col,
        heading1_col=heading1_col,
        heading2_col=heading2_col,
        len1_col=len1_col,
        wid1_col=wid1_col,
        len2_col=len2_col,
        wid2_col=wid2_col,
        vel1_col=vel1_col,
        vel2_col=vel2_col,
        show_speed_text=show_speed_text,
        curve_df=curve_df,
        label1=label1,
        label2=label2,
        use_map_background=use_map_background,
        map_data=map_data,
        map_offset=map_offset,
    )

    try:
        pil_frames = [renderer.render_frame(i) for i in range(len(display_frames))]
    finally:
        renderer.close()

    if len(pil_frames) == 0:
        return None

    first = pil_frames[0]
    rest = pil_frames[1:] if len(pil_frames) > 1 else []

    first.save(
        out_gif,
        save_all=True,
        append_images=rest,
        duration=durations_ms,
        loop=GIF_LOOP,
        optimize=GIF_OPTIMIZE,
        disposal=GIF_DISPOSAL,
    )

    return out_gif


# ============================================================
# CSV validation and visualization
# ============================================================

def validate_required_columns(df, csv_path):
    missing = [c for c in REQUIRED_VIS_COLS if c not in df.columns]

    if missing:
        raise KeyError(
            f"{csv_path} is missing required visualization columns: {missing}"
        )


def generate_gif_from_csv_dataframe(csv_path, df):
    cols = df.columns
    stem = Path(csv_path).stem

    validate_required_columns(df, csv_path)

    time_col = choose_col(cols, ["Time (s)", "Time", "time", "timestamp", "Timestamp"])
    if time_col is None:
        time_col = df.columns[0]

    frames = build_frames_from_df(df, time_col)

    if len(frames) == 0:
        return None, "No valid frames for visualization."

    sampled_frames, sampled_indices = apply_frame_step(frames, FRAME_STEP)

    sampled_frames, sampled_indices, range_start_idx, range_end_idx_exclusive = apply_visualize_time_range(
        sampled_frames,
        sampled_indices,
        VISUALIZE_TIME_RANGE,
    )

    if len(sampled_frames) == 0:
        return None, "No frames remain after VISUALIZE_TIME_RANGE."

    durations_ms = infer_frame_durations_ms(sampled_frames)

    gei_col = choose_col(cols, ["GEI"])
    if gei_col is None:
        return None, "GEI column not found."

    posx1_col = choose_col(cols, ["Position X (m)"])
    posy1_col = choose_col(cols, ["Position Y (m)"])
    posx2_col = choose_col(cols, ["2_Position X (m)"])
    posy2_col = choose_col(cols, ["2_Position Y (m)"])

    heading1_col = choose_col(cols, ["Heading"])
    heading2_col = choose_col(cols, ["2_Heading"])

    len1_col = choose_col(cols, ["Length (m)"])
    wid1_col = choose_col(cols, ["Width (m)"])
    len2_col = choose_col(cols, ["2_Length (m)"])
    wid2_col = choose_col(cols, ["2_Width (m)"])

    vel1_col = choose_col(cols, ["Velocity (m/s)"])
    vel2_col = choose_col(cols, ["2_Velocity (m/s)"])

    required_for_scene = [
        posx1_col,
        posy1_col,
        posx2_col,
        posy2_col,
        heading1_col,
        heading2_col,
        len1_col,
        wid1_col,
        len2_col,
        wid2_col,
        vel1_col,
        vel2_col,
    ]

    if not all(required_for_scene):
        return None, "Some required scene-visualization columns are missing."

    x_min, x_max, y_min, y_max = compute_axis_limits_from_display_frames(
        sampled_frames,
        posx1_col,
        posy1_col,
        posx2_col,
        posy2_col,
    )

    curve_rows = [pd.Series(r) for r in sampled_frames]
    curve_df = pd.DataFrame(curve_rows).reset_index(drop=True)

    row0 = curve_df.iloc[0]

    veh1_num = row0.get("Vehicle Number", "?")
    veh1_type_raw = row0.get("Agent_Type", "")
    veh1_type = format_agent_type_for_label(veh1_type_raw)

    veh2_num = row0.get("2_Vehicle Number", "?")
    veh2_type_raw = row0.get("2_Agent_Type", "")
    veh2_type = format_agent_type_for_label(veh2_type_raw)

    label1 = f"{veh1_type} #{veh1_num}" if veh1_type else f"ID #{veh1_num}"
    label2 = f"{veh2_type} #{veh2_num}" if veh2_type else f"ID #{veh2_num}"

    use_map_background = is_SIND_csv_by_name(stem)
    map_data = None

    if use_map_background:
        map_data = load_global_map_if_needed()

    out_gif = save_gif_fast(
        stem=stem,
        metric_name="GEI",
        display_frames=sampled_frames,
        durations_ms=durations_ms,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        value_col=gei_col,
        colorbar_norm=GEI_NORM,
        colorbar_cmap=RISK_CMAP,
        colorbar_label="GEI",
        colorbar_ticks=[0.0, 0.5, 1.0, 1.5],
        posx1_col=posx1_col,
        posy1_col=posy1_col,
        posx2_col=posx2_col,
        posy2_col=posy2_col,
        heading1_col=heading1_col,
        heading2_col=heading2_col,
        len1_col=len1_col,
        wid1_col=wid1_col,
        len2_col=len2_col,
        wid2_col=wid2_col,
        vel1_col=vel1_col,
        vel2_col=vel2_col,
        show_speed_text=SHOW_SPEED_TEXT,
        out_dir=OUTPUT_GIF_DIR,
        curve_df=curve_df,
        label1=label1,
        label2=label2,
        use_map_background=use_map_background and map_data is not None,
        map_data=map_data,
        map_offset=MAP_OFFSET,
    )

    log = []
    log.append(f"  Total raw frames: {len(frames)}")
    log.append(f"  Total displayed frames: {len(sampled_frames)}")
    log.append(f"  Range slice on sampled frames: [{range_start_idx}:{range_end_idx_exclusive}]")
    log.append(f"  GEI column used: {gei_col}")
    log.append(f"  SIND filename detected: {use_map_background}")
    log.append(f"  Map background enabled: {use_map_background and map_data is not None}")
    if use_map_background:
        log.append(f"  Map file: {MAP_OSM_PATH}")
        log.append(f"  Map offset: {MAP_OFFSET}")
    log.append(f"  Saved GIF: {out_gif}")

    return out_gif, "\n".join(log)


def process_one_csv(csv_path, save_gif=True):
    file_t0 = time.perf_counter()

    csv_path = str(csv_path)
    df = pd.read_csv(csv_path)

    gif_path = None
    gif_log = "GIF generation skipped."
    gif_elapsed = 0.0

    if save_gif:
        gif_t0 = time.perf_counter()
        gif_path, gif_log = generate_gif_from_csv_dataframe(csv_path, df)
        gif_elapsed = time.perf_counter() - gif_t0

    file_t1 = time.perf_counter()
    total_ms = (file_t1 - file_t0) * 1000.0

    print(f"[OK] Processed: {csv_path}")

    if gif_path is not None:
        print(f"     GIF: {gif_path}")

    print(f"     CSV rows: {len(df)}")
    print(f"     Total file time (read + visualize): {total_ms:.3f} ms")

    if save_gif:
        print(f"     GIF time: {gif_elapsed:.3f} s")
        print(gif_log)

    return {
        "file": csv_path,
        "gif_path": str(gif_path) if gif_path is not None else None,
        "total_frames": len(df),
        "total_ms": total_ms,
    }


# ============================================================
# CLI
# ============================================================

def collect_input_files(input_path=None, pattern="GEI_*.csv"):
    """
    Collect CSV files generated by main.py.
    """
    if input_path is not None:
        p = Path(input_path)

        if not p.exists():
            raise FileNotFoundError(f"Input path does not exist: {input_path}")

        if p.is_file():
            if p.name.startswith("GEI_") and p.suffix.lower() == ".csv":
                return [str(p)]
            return []

        return sorted(
            str(x) for x in p.glob(pattern)
            if x.name.startswith("GEI_") and x.suffix.lower() == ".csv"
        )

    return sorted(
        f for f in glob.glob(pattern)
        if Path(f).name.startswith("GEI_") and Path(f).suffix.lower() == ".csv"
    )


def build_argparser():
    parser = argparse.ArgumentParser(
        description=(
            "Generate GIF visualizations from CSV files that already contain "
            "GEI/TEM/InDepth/SSM metrics. This script does not recompute metrics. "
            "CSV files whose names start with GEI_SIND automatically use the SIND map."
        )
    )

    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Single CSV file or directory. Defaults to GEI_*.csv in the current directory.",
    )

    parser.add_argument(
        "--pattern",
        type=str,
        default="GEI_*.csv",
        help="Glob pattern used when --input is a directory or omitted.",
    )

    parser.add_argument(
        "--skip-gif",
        action="store_true",
        help="Skip GIF generation.",
    )

    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()

    batch_t0 = time.perf_counter()

    csv_files = collect_input_files(args.input, args.pattern)

    if not csv_files:
        print("No CSV files were found.")
        return

    print("CSV files:")
    for f in csv_files:
        print(" -", f)

    print("\nSIND map rule: file name starts with 'GEI_SIND'")
    print(f"SIND map file: {MAP_OSM_PATH}")
    print(f"SIND map offset: {MAP_OFFSET}")
    print(f"DRAW_VIRTUAL_LINES = {DRAW_VIRTUAL_LINES}")

    batch_total_frames = 0
    batch_total_ms = 0.0
    success_count = 0
    failed_count = 0

    for csv_path in csv_files:
        try:
            stat = process_one_csv(
                csv_path=csv_path,
                save_gif=not args.skip_gif,
            )

            batch_total_frames += stat["total_frames"]
            batch_total_ms += stat["total_ms"]
            success_count += 1

        except Exception as e:
            failed_count += 1
            print(f"[ERROR] Failed to process {csv_path}: {e}")

    batch_t1 = time.perf_counter()
    elapsed_ms = (batch_t1 - batch_t0) * 1000.0

    avg_ms_per_frame = elapsed_ms / batch_total_frames if batch_total_frames > 0 else np.nan

    print("\n========== Batch Summary ==========")
    print(f"Files: {len(csv_files)}")
    print(f"Successful files: {success_count}")
    print(f"Failed files: {failed_count}")
    print(f"Total frames: {batch_total_frames}")
    print(f"Batch elapsed time: {elapsed_ms:.3f} ms")
    print(f"Average time per frame: {avg_ms_per_frame:.3f} ms/frame")


if __name__ == "__main__":
    main()
