#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Command-line and Python API entry points for GEI computation.

The module supports two primary workflows:
1. Direct single-frame GEI computation from 14 numeric vehicle-state parameters.
2. CSV enrichment, where GEI and related metrics are computed for every frame.
"""

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

# Limit numerical-library worker threads before importing numpy or pandas.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from .core import (
    compute_4mode_gei_core_metrics,
    compute_extra_ssm_metrics,
)


DEFAULT_CTRV_DT = 0.05
DEFAULT_CTRV_HORIZON = 10.0
DEFAULT_D_SAFE = 0.0
DEFAULT_ANCHOR_BACKOFF = 1e-3
DEFAULT_ROUND_DECIMALS = 4

FRAME_VALUE_NAMES = [
    "xA",
    "yA",
    "vA",
    "hA",
    "lA",
    "wA",
    "yawA",
    "xB",
    "yB",
    "vB",
    "hB",
    "lB",
    "wB",
    "yawB",
]

REQUIRED_COLS = [
    "Position X (m)",
    "Position Y (m)",
    "Velocity (m/s)",
    "Heading",
    "Length (m)",
    "Width (m)",
    "Yawrate",
    "2_Position X (m)",
    "2_Position Y (m)",
    "2_Velocity (m/s)",
    "2_Heading",
    "2_Length (m)",
    "2_Width (m)",
    "2_Yawrate",
]

OUTPUT_COLUMNS = [
    "TEM_CVCV",
    "TEM_CVCT",
    "TEM_CTCV",
    "TEM_CTCT",
    "InDepth_CVCV",
    "InDepth_CVCT_CA",
    "InDepth_CTCV_CA",
    "InDepth_CTCT_CA",
    "MEI",
    "EI_CVCT_CA",
    "EI_CTCV_CA",
    "EI_CTCT_CA",
    "GEI",
    "InDepth_eff",
    "TEM_eff",
    "DRAC",
    "DRAC2D",
    "TTC",
    "2D-TTC",
    "TAdv",
    "ACT",
    "EI",
    "TTC2D",
    "BBox distance (m)",
]

TEM_COLUMNS = [
    "TEM_CVCV",
    "TEM_CVCT",
    "TEM_CTCV",
    "TEM_CTCT",
]

INDEPTH_COLUMNS = [
    "InDepth_CVCV",
    "InDepth_CVCT_CA",
    "InDepth_CTCV_CA",
    "InDepth_CTCT_CA",
]

DEFAULT_RESULT_DICT = {
    "TEM_CVCV": np.inf,
    "TEM_CVCT": np.inf,
    "TEM_CTCV": np.inf,
    "TEM_CTCT": np.inf,
    "InDepth_CVCV": 0.0,
    "InDepth_CVCT_CA": 0.0,
    "InDepth_CTCV_CA": 0.0,
    "InDepth_CTCT_CA": 0.0,
    "MEI": 0.0,
    "EI_CVCT_CA": 0.0,
    "EI_CTCV_CA": 0.0,
    "EI_CTCT_CA": 0.0,
    "GEI": 0.0,
    "InDepth_eff": 0.0,
    "TEM_eff": np.inf,
    "DRAC": 0.0,
    "DRAC2D": 0.0,
    "TTC": np.inf,
    "2D-TTC": np.inf,
    "TAdv": np.nan,
    "ACT": np.inf,
    "EI": 0.0,
    "TTC2D": np.inf,
    "BBox distance (m)": np.nan,
}

LEGACY_OUTPUT_COLUMNS = [
    "TTC_CVCV",
    "TTC_CVCT",
    "TTC_CTCV",
    "TTC_CTCT",
    "EI_CVCV_CA",
    "InDepth",
    "InDepth_CVCT_IR",
    "InDepth_CTCV_IR",
    "InDepth_CTCT_IR",
    "D2TTC",
]


def round_value(value, decimals=DEFAULT_ROUND_DECIMALS):
    """Round finite numeric values while preserving NaN and infinity."""
    if pd.isna(value):
        return np.nan

    value = float(value)

    if np.isposinf(value):
        return np.inf
    if np.isneginf(value):
        return -np.inf

    if decimals is None:
        return value

    return round(value, decimals)


def normalize_metric_result(metric_res, decimals=DEFAULT_ROUND_DECIMALS):
    """Normalize a raw metric dictionary to the public output schema."""
    res = dict(DEFAULT_RESULT_DICT)

    if metric_res is not None:
        metric_res = dict(metric_res)
        metric_res.pop("_precomputed_ttc2d_result", None)

        if "D2TTC" in metric_res:
            metric_res["2D-TTC"] = metric_res.pop("D2TTC")

        res.update(metric_res)

    for col in TEM_COLUMNS:
        try:
            value = float(res.get(col, np.inf))
        except Exception:
            value = np.inf

        if np.isnan(value) or value < 0:
            value = np.inf

        res[col] = value

    for col in INDEPTH_COLUMNS:
        try:
            value = float(res.get(col, 0.0))
        except Exception:
            value = 0.0

        if np.isnan(value) or value < 0:
            value = 0.0

        res[col] = value

    indepth_eff = float(np.mean([res[col] for col in INDEPTH_COLUMNS]))
    res["InDepth_eff"] = indepth_eff

    try:
        gei = float(res.get("GEI", 0.0))
    except Exception:
        gei = np.nan

    if np.isnan(gei):
        tem_eff = np.nan
    elif gei == 0.0:
        tem_eff = np.inf
    elif np.isposinf(gei):
        tem_eff = 0.0
    elif gei > 0.0 and np.isfinite(gei):
        tem_eff = indepth_eff / gei
    else:
        tem_eff = np.inf

    res["TEM_eff"] = tem_eff

    return {
        col: round_value(res.get(col, DEFAULT_RESULT_DICT[col]), decimals=decimals)
        for col in OUTPUT_COLUMNS
    }


def validate_frame_values(values):
    """Return a list of human-readable input issues for one frame."""
    issues = []

    if len(values) != len(FRAME_VALUE_NAMES):
        return [f"expected {len(FRAME_VALUE_NAMES)} values, got {len(values)}"]

    for name, value in zip(FRAME_VALUE_NAMES, values):
        if not np.isfinite(value):
            issues.append(f"{name} is not finite")

    for name in ("vA", "vB"):
        value = values[FRAME_VALUE_NAMES.index(name)]
        if np.isfinite(value) and value < 0:
            issues.append(f"{name} is negative")

    for name in ("lA", "wA", "lB", "wB"):
        value = values[FRAME_VALUE_NAMES.index(name)]
        if np.isfinite(value) and value <= 0:
            issues.append(f"{name} must be positive")

    return issues


def compute_single_frame(
    xA,
    yA,
    vA,
    hA,
    lA,
    wA,
    yawA,
    xB,
    yB,
    vB,
    hB,
    lB,
    wB,
    yawB,
    compute_extra_metrics=True,
    dt=DEFAULT_CTRV_DT,
    horizon=DEFAULT_CTRV_HORIZON,
    d_safe=DEFAULT_D_SAFE,
    anchor_backoff=DEFAULT_ANCHOR_BACKOFF,
    decimals=DEFAULT_ROUND_DECIMALS,
):
    """Compute metrics for one frame.

    Public argument order for each road user is:
    x, y, speed, heading, length, width, yaw_rate.
    """
    core_raw = compute_4mode_gei_core_metrics(
        xA=xA,
        yA=yA,
        vA=vA,
        hA=hA,
        yawA=yawA,
        lA=lA,
        wA=wA,
        xB=xB,
        yB=yB,
        vB=vB,
        hB=hB,
        yawB=yawB,
        lB=lB,
        wB=wB,
        T_total=horizon,
        dt=dt,
        d_safe=d_safe,
        anchor_backoff=anchor_backoff,
    )

    precomputed_ttc2d_result = core_raw.get("_precomputed_ttc2d_result")
    metric_raw = dict(core_raw)
    metric_raw.pop("_precomputed_ttc2d_result", None)

    if compute_extra_metrics:
        extra_raw = compute_extra_ssm_metrics(
            xA=xA,
            yA=yA,
            vA=vA,
            hA=hA,
            lA=lA,
            wA=wA,
            xB=xB,
            yB=yB,
            vB=vB,
            hB=hB,
            lB=lB,
            wB=wB,
            precomputed_ttc2d_result=precomputed_ttc2d_result,
        )
        metric_raw.update(extra_raw)

    return normalize_metric_result(metric_raw, decimals=decimals)


def compute_one_frame(*args, **kwargs):
    """Backward-compatible alias for compute_single_frame."""
    return compute_single_frame(*args, **kwargs)


def compute_frame_from_values(
    values,
    compute_extra_metrics=True,
    dt=DEFAULT_CTRV_DT,
    horizon=DEFAULT_CTRV_HORIZON,
    d_safe=DEFAULT_D_SAFE,
    anchor_backoff=DEFAULT_ANCHOR_BACKOFF,
    decimals=DEFAULT_ROUND_DECIMALS,
):
    """Compute one frame from values ordered according to FRAME_VALUE_NAMES."""
    if len(values) != len(FRAME_VALUE_NAMES):
        raise ValueError(
            f"Expected {len(FRAME_VALUE_NAMES)} values, got {len(values)}."
        )

    values = [float(v) for v in values]
    return compute_single_frame(
        *values,
        compute_extra_metrics=compute_extra_metrics,
        dt=dt,
        horizon=horizon,
        d_safe=d_safe,
        anchor_backoff=anchor_backoff,
        decimals=decimals,
    )


def collect_csv_files(input_dir=".", pattern="*.csv"):
    """Collect raw CSV files and skip generated outputs."""
    input_dir = Path(input_dir)
    csv_files = []

    for path in sorted(input_dir.glob(pattern)):
        if not path.is_file() or path.suffix.lower() != ".csv":
            continue

        name = path.name
        if name.startswith("GEI_") or name.startswith("ei_"):
            continue
        if name.startswith("runtime_"):
            continue
        if name in {"all_frame_gei_runtime.csv", "gei_runtime_summary.csv"}:
            continue

        csv_files.append(str(path))

    return csv_files


def build_output_path(csv_path, output_path=None, output_dir=None):
    """Return the target path for an enriched CSV output."""
    csv_path = Path(csv_path)

    if output_path is not None:
        return Path(output_path)

    if output_dir is not None:
        return Path(output_dir) / f"GEI_{csv_path.name}"

    return csv_path.with_name(f"GEI_{csv_path.name}")


def row_to_frame_values(row_values):
    """Convert one CSV row into the public single-frame parameter order."""
    (
        xA,
        yA,
        vA,
        hA,
        lA,
        wA,
        yawA,
        xB,
        yB,
        vB,
        hB,
        lB,
        wB,
        yawB,
    ) = row_values

    return [
        xA,
        yA,
        vA,
        hA,
        lA,
        wA,
        yawA,
        xB,
        yB,
        vB,
        hB,
        lB,
        wB,
        yawB,
    ]


def process_one_csv(
    csv_path,
    output_path=None,
    output_dir=None,
    compute_extra_metrics=True,
    skip_if_output_exists=False,
    dt=DEFAULT_CTRV_DT,
    horizon=DEFAULT_CTRV_HORIZON,
    d_safe=DEFAULT_D_SAFE,
    anchor_backoff=DEFAULT_ANCHOR_BACKOFF,
    decimals=DEFAULT_ROUND_DECIMALS,
    write_output=True,
):
    """Compute metrics for every valid row in one CSV file."""
    csv_path = Path(csv_path)
    output_path = build_output_path(csv_path, output_path, output_dir)

    if skip_if_output_exists and output_path.exists():
        return {
            "file": str(csv_path),
            "output_file": str(output_path),
            "total_frames": 0,
            "input_valid_frames": 0,
            "success_frames": 0,
            "failed_frames": 0,
            "skipped": True,
            "message": f"[SKIP] Output already exists: {output_path}",
        }

    if write_output:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    missing = [col for col in REQUIRED_COLS if col not in df.columns]
    if missing:
        raise KeyError(f"{csv_path.name} is missing required columns: {missing}")

    output_like_cols = list(dict.fromkeys(OUTPUT_COLUMNS + LEGACY_OUTPUT_COLUMNS))
    existing_output_cols = [col for col in output_like_cols if col in df.columns]

    if existing_output_cols:
        df.drop(columns=existing_output_cols, inplace=True)

    input_arr = (
        df[REQUIRED_COLS]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
    )

    result_lists = {col: [] for col in OUTPUT_COLUMNS}
    total_frames = len(df)
    input_valid_frames = 0
    failed_frames = 0
    nan_input_frames = 0
    invalid_value_frames = 0

    for row_pos, row_values in enumerate(input_arr):
        original_index = df.index[row_pos]

        if np.isnan(row_values).any():
            nan_input_frames += 1
            metric_res = normalize_metric_result(
                DEFAULT_RESULT_DICT,
                decimals=decimals,
            )
        else:
            validation_issues = validate_frame_values(row_values)

            if validation_issues:
                invalid_value_frames += 1
                print(
                    f"[WARN] {csv_path.name} row {original_index} invalid input: "
                    f"{'; '.join(validation_issues)}",
                    file=sys.stderr,
                )
                metric_res = normalize_metric_result(
                    DEFAULT_RESULT_DICT,
                    decimals=decimals,
                )
            else:
                input_valid_frames += 1

                try:
                    frame_values = row_to_frame_values(row_values)
                    metric_res = compute_frame_from_values(
                        frame_values,
                        compute_extra_metrics=compute_extra_metrics,
                        dt=dt,
                        horizon=horizon,
                        d_safe=d_safe,
                        anchor_backoff=anchor_backoff,
                        decimals=decimals,
                    )
                except Exception as exc:
                    failed_frames += 1
                    print(
                        f"[WARN] {csv_path.name} row {original_index} failed: {exc}",
                        file=sys.stderr,
                    )
                    metric_res = normalize_metric_result(
                        DEFAULT_RESULT_DICT,
                        decimals=decimals,
                    )

        for col in OUTPUT_COLUMNS:
            result_lists[col].append(metric_res[col])

    if write_output:
        for col in OUTPUT_COLUMNS:
            df[col] = result_lists[col]

        df.to_csv(output_path, index=False, encoding="utf-8-sig")

    success_frames = input_valid_frames - failed_frames
    message = (
        f"[OK] Processed: {csv_path}\n"
        f"     Output: {output_path if write_output else '(not written)'}\n"
        f"     Total frames: {total_frames}, "
        f"valid input frames: {input_valid_frames}, "
        f"successful frames: {success_frames}, "
        f"failed frames: {failed_frames}, "
        f"NaN input frames: {nan_input_frames}, "
        f"invalid input frames: {invalid_value_frames}"
    )

    return {
        "file": str(csv_path),
        "output_file": str(output_path),
        "total_frames": total_frames,
        "input_valid_frames": input_valid_frames,
        "success_frames": success_frames,
        "failed_frames": failed_frames,
        "nan_input_frames": nan_input_frames,
        "invalid_value_frames": invalid_value_frames,
        "skipped": False,
        "message": message,
    }


def json_safe(value):
    """Convert numpy scalars and non-finite values to JSON-friendly values."""
    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (float, np.floating)):
        value = float(value)
        if np.isnan(value):
            return "nan"
        if np.isposinf(value):
            return "inf"
        if np.isneginf(value):
            return "-inf"
        return value

    return value


def print_frame_result(result, as_json=False):
    """Print a single-frame result as JSON or a compact table."""
    if as_json:
        payload = {key: json_safe(value) for key, value in result.items()}
        print(json.dumps(payload, indent=2, ensure_ascii=True))
        return

    for key in OUTPUT_COLUMNS:
        print(f"{key}: {json_safe(result[key])}")


def resolve_decimals(args):
    """Resolve rounding options from argparse args."""
    if getattr(args, "no_round", False):
        return None

    decimals = getattr(args, "decimals", DEFAULT_ROUND_DECIMALS)
    if decimals < 0:
        raise ValueError("--decimals must be non-negative.")

    return decimals


def validate_common_args(args):
    """Validate shared numeric CLI options."""
    if args.dt <= 0:
        raise ValueError("--dt must be positive.")
    if args.horizon <= 0:
        raise ValueError("--horizon must be positive.")
    if args.anchor_backoff < 0:
        raise ValueError("--anchor-backoff must be non-negative.")


def run_frame(args):
    validate_common_args(args)
    result = compute_frame_from_values(
        args.values,
        compute_extra_metrics=not args.core_only,
        dt=args.dt,
        horizon=args.horizon,
        d_safe=args.d_safe,
        anchor_backoff=args.anchor_backoff,
        decimals=resolve_decimals(args),
    )
    print_frame_result(result, as_json=args.json)


def run_csv(args):
    validate_common_args(args)
    stat = process_one_csv(
        csv_path=args.input,
        output_path=args.output,
        compute_extra_metrics=not args.core_only,
        skip_if_output_exists=args.skip_existing,
        dt=args.dt,
        horizon=args.horizon,
        d_safe=args.d_safe,
        anchor_backoff=args.anchor_backoff,
        decimals=resolve_decimals(args),
    )
    print(stat["message"])


def run_batch(args):
    validate_common_args(args)
    csv_files = collect_csv_files(args.input_dir, args.pattern)

    if not csv_files:
        print("No raw CSV files were found.")
        return

    print("Raw CSV files:")
    for path in csv_files:
        print(f" - {path}")

    print()
    print("Mode: serial per-frame computation")
    print(f"Extra SSM metrics: {not args.core_only}")
    print(f"Prediction horizon: {args.horizon} s")
    print(f"CTRV scan step: {args.dt} s")
    print()

    batch_t0 = time.perf_counter()
    batch_total_frames = 0
    batch_input_valid_frames = 0
    batch_success_frames = 0
    batch_failed_frames = 0
    batch_processed_files = 0
    batch_skipped_files = 0

    for csv_path in csv_files:
        try:
            stat = process_one_csv(
                csv_path=csv_path,
                output_dir=args.output_dir,
                compute_extra_metrics=not args.core_only,
                skip_if_output_exists=args.skip_existing,
                dt=args.dt,
                horizon=args.horizon,
                d_safe=args.d_safe,
                anchor_backoff=args.anchor_backoff,
                decimals=resolve_decimals(args),
            )
            print(stat["message"])

            if stat.get("skipped", False):
                batch_skipped_files += 1
                continue

            batch_processed_files += 1
            batch_total_frames += stat["total_frames"]
            batch_input_valid_frames += stat["input_valid_frames"]
            batch_success_frames += stat["success_frames"]
            batch_failed_frames += stat["failed_frames"]

        except Exception as exc:
            print(f"[ERROR] Failed to process {csv_path}: {exc}", file=sys.stderr)

    elapsed_ms = (time.perf_counter() - batch_t0) * 1000.0

    print("\n========== Batch Summary ==========")
    print(f"Raw files found: {len(csv_files)}")
    print(f"Processed files: {batch_processed_files}")
    print(f"Skipped files: {batch_skipped_files}")
    print(
        f"Total frames: {batch_total_frames}, "
        f"valid input frames: {batch_input_valid_frames}, "
        f"successful frames: {batch_success_frames}, "
        f"failed frames: {batch_failed_frames}"
    )
    print(f"Elapsed time: {elapsed_ms:.3f} ms")
    if batch_success_frames > 0:
        print(f"Average time per successful frame: {elapsed_ms / batch_success_frames:.3f} ms/frame")


def percentile(values, q):
    """Return a percentile from a non-empty numeric list."""
    return float(np.percentile(np.asarray(values, dtype=float), q))


def run_benchmark(args):
    validate_common_args(args)
    if args.repeat < 1:
        raise ValueError("--repeat must be at least 1.")

    csv_files = []

    if args.input:
        csv_files = [args.input]
    else:
        csv_files = collect_csv_files(args.input_dir, args.pattern)

    if not csv_files:
        print("No raw CSV files were found.")
        return

    decimals = resolve_decimals(args)
    repeat_elapsed = []
    total_frames = 0
    total_success_frames = 0
    total_failed_frames = 0

    print("Benchmark files:")
    for path in csv_files:
        print(f" - {path}")

    print()
    print("Mode: serial per-frame benchmark")
    print(f"Extra SSM metrics: {not args.core_only}")
    print(f"Prediction horizon: {args.horizon} s")
    print(f"CTRV scan step: {args.dt} s")
    print(f"Repeats: {args.repeat}")
    print("Output writing: disabled")
    print()

    for repeat_idx in range(args.repeat):
        repeat_t0 = time.perf_counter()
        repeat_frames = 0
        repeat_success = 0
        repeat_failed = 0

        for csv_path in csv_files:
            stat = process_one_csv(
                csv_path=csv_path,
                output_dir=args.output_dir,
                compute_extra_metrics=not args.core_only,
                skip_if_output_exists=False,
                dt=args.dt,
                horizon=args.horizon,
                d_safe=args.d_safe,
                anchor_backoff=args.anchor_backoff,
                decimals=decimals,
                write_output=False,
            )
            repeat_frames += stat["total_frames"]
            repeat_success += stat["success_frames"]
            repeat_failed += stat["failed_frames"]

        elapsed_ms = (time.perf_counter() - repeat_t0) * 1000.0
        repeat_elapsed.append(elapsed_ms)

        if repeat_idx == 0:
            total_frames = repeat_frames
            total_success_frames = repeat_success
            total_failed_frames = repeat_failed

        print(
            f"Repeat {repeat_idx + 1}/{args.repeat}: "
            f"{elapsed_ms:.3f} ms, "
            f"{elapsed_ms / repeat_success:.3f} ms/successful-frame"
            if repeat_success > 0
            else f"Repeat {repeat_idx + 1}/{args.repeat}: {elapsed_ms:.3f} ms"
        )

    per_frame = [
        elapsed / total_success_frames
        for elapsed in repeat_elapsed
        if total_success_frames > 0
    ]

    print("\n========== Benchmark Summary ==========")
    print(f"Files: {len(csv_files)}")
    print(f"Frames per repeat: {total_frames}")
    print(f"Successful frames per repeat: {total_success_frames}")
    print(f"Failed frames per repeat: {total_failed_frames}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"NumPy: {np.__version__}")
    print(f"Pandas: {pd.__version__}")

    if per_frame:
        print(f"Mean: {float(np.mean(per_frame)):.3f} ms/frame")
        print(f"Median: {float(np.median(per_frame)):.3f} ms/frame")
        print(f"P90: {percentile(per_frame, 90):.3f} ms/frame")
        print(f"P95: {percentile(per_frame, 95):.3f} ms/frame")
        print(f"P99: {percentile(per_frame, 99):.3f} ms/frame")


def add_common_compute_args(parser):
    parser.add_argument(
        "--core-only",
        action="store_true",
        help="Compute only GEI core metrics and fill traditional SSM columns with defaults.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=DEFAULT_CTRV_DT,
        help=f"CTRV collision-scan time step in seconds. Default: {DEFAULT_CTRV_DT}.",
    )
    parser.add_argument(
        "--horizon",
        type=float,
        default=DEFAULT_CTRV_HORIZON,
        help=f"Prediction horizon in seconds. Default: {DEFAULT_CTRV_HORIZON}.",
    )
    parser.add_argument(
        "--d-safe",
        type=float,
        default=DEFAULT_D_SAFE,
        help=f"Safety distance offset used in InDepth. Default: {DEFAULT_D_SAFE}.",
    )
    parser.add_argument(
        "--anchor-backoff",
        type=float,
        default=DEFAULT_ANCHOR_BACKOFF,
        help=(
            "Time backoff before the first predicted collision for CA-InDepth "
            f"anchor direction. Default: {DEFAULT_ANCHOR_BACKOFF}."
        ),
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=DEFAULT_ROUND_DECIMALS,
        help=f"Decimal places for numeric outputs. Default: {DEFAULT_ROUND_DECIMALS}.",
    )
    parser.add_argument(
        "--no-round",
        action="store_true",
        help="Preserve full floating-point precision in CLI/API-normalized outputs.",
    )


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Compute GEI from one frame or enrich CSV files frame by frame.",
    )
    subparsers = parser.add_subparsers(dest="command")

    frame_parser = subparsers.add_parser(
        "frame",
        help="Compute GEI for one frame from 14 numeric state values.",
    )
    frame_parser.add_argument(
        "--values",
        type=float,
        nargs=14,
        metavar="V",
        required=True,
        help=(
            "Values in order: "
            "xA yA vA hA lA wA yawA xB yB vB hB lB wB yawB."
        ),
    )
    frame_parser.add_argument(
        "--json",
        action="store_true",
        help="Print the single-frame result as JSON.",
    )
    add_common_compute_args(frame_parser)
    frame_parser.set_defaults(func=run_frame)

    csv_parser = subparsers.add_parser(
        "csv",
        help="Compute GEI for every frame in one CSV file.",
    )
    csv_parser.add_argument("--input", required=True, help="Input raw CSV file.")
    csv_parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path. Defaults to GEI_<input-name>.csv beside the input.",
    )
    csv_parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip processing if the output file already exists.",
    )
    add_common_compute_args(csv_parser)
    csv_parser.set_defaults(func=run_csv)

    batch_parser = subparsers.add_parser(
        "batch",
        help="Compute GEI for raw CSV files in a directory.",
    )
    batch_parser.add_argument(
        "--input-dir",
        default=".",
        help="Directory containing raw CSV files.",
    )
    batch_parser.add_argument(
        "--pattern",
        default="*.csv",
        help="Glob pattern used inside --input-dir.",
    )
    batch_parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional directory for enriched CSV outputs.",
    )
    batch_parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip processing when an output file already exists.",
    )
    add_common_compute_args(batch_parser)
    batch_parser.set_defaults(func=run_batch)

    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Benchmark CSV GEI computation without writing output files.",
    )
    benchmark_parser.add_argument(
        "--input",
        default=None,
        help="Single raw CSV file to benchmark. If omitted, --input-dir is used.",
    )
    benchmark_parser.add_argument(
        "--input-dir",
        default=".",
        help="Directory containing raw CSV files.",
    )
    benchmark_parser.add_argument(
        "--pattern",
        default="*.csv",
        help="Glob pattern used inside --input-dir.",
    )
    benchmark_parser.add_argument(
        "--output-dir",
        default=None,
        help="Unused for writing; retained for output-path compatibility.",
    )
    benchmark_parser.add_argument(
        "--repeat",
        type=int,
        default=5,
        help="Number of benchmark repeats. Default: 5.",
    )
    add_common_compute_args(benchmark_parser)
    benchmark_parser.set_defaults(func=run_benchmark)

    return parser


def main(argv=None):
    parser = build_argparser()
    args = parser.parse_args(argv)

    if args.command is None:
        args = parser.parse_args(["batch"])

    try:
        args.func(args)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
