# GEI

GEI is a Python toolkit for computing the Generalized Emergency Index (GEI) for pairs of interacting road users. It supports single-frame computation from 14 state parameters and frame-by-frame processing for trajectory CSVs.

An optional GIF visualization utility is provided for inspecting the temporal evolution of GEI in traffic-conflict cases.

## Table of Contents

- [Visual Examples](#visual-examples)
- [Why GEI?](#why-gei)
- [Method at a Glance](#method-at-a-glance)
- [Research Highlights](#research-highlights)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Input Definition](#input-definition)
- [Workflow 1: Single-Frame GEI](#workflow-1-single-frame-gei)
- [Workflow 2: CSV Frame-by-Frame GEI](#workflow-2-csv-frame-by-frame-gei)
- [Workflow 3: Optional GIF Visualization](#workflow-3-optional-gif-visualization)
- [Workflow 4: Runtime Benchmarking](#workflow-4-runtime-benchmarking)
- [Required CSV Columns](#required-csv-columns)
- [Output Columns](#output-columns)
- [Python API](#python-api)
- [Project Layout](#project-layout)
- [Development Workflow](#development-workflow)
- [License](#license)

## Visual Examples

The GIFs below show GEI-based visualizations for two vehicle--powered two-wheeler (PTW) interactions. In each example, the scene view is paired with GEI-related curves, allowing the spatial conflict and metric evolution to be inspected together.

**SIND Tianjin Intersection: High-Risk Vehicle--PTW Interaction**

![High-risk vehicle--PTW interaction in the SIND Tianjin dataset](assets/demos/sind-tianjin-intersection-vehicle-ptw-strong-interaction.gif)

This case comes from the SIND dataset and captures a high-risk vehicle--PTW interaction at an intersection in Tianjin, China.

**CIMSS-TA Hunan: PTW Cut-In Collision**

![CIMSS-TA Hunan powered-two-wheeler cut-in collision](assets/demos/cimss-ta-hunan-ptw-cut-in-collision.gif)

This case comes from the CIMSS-TA database and shows a PTW cut-in collision in Hunan, China.

## Why GEI?

Powered two-wheelers (PTWs), including motorcycles, scooters, and mopeds, are heavily involved in severe road crashes because they are highly exposed, physically vulnerable, and often interact with vehicles in complex mixed-traffic environments.

Vehicle--PTW interactions cannot be reduced to conventional vehicle--vehicle interactions with smaller body dimensions. PTWs are less lane-constrained and more maneuverable, often exhibiting pronounced two-dimensional motion patterns: they can filter, weave, cut in, turn, and make rapid lateral movements. Purely time-based surrogate safety measures may therefore miss an important component of risk: two situations may have similar temporal urgency but require substantially different evasive maneuvers.

GEI is built on a simple idea:

```text
risk = required evasive maneuver demand / remaining available evasive time
```

This makes GEI a risk measure that jointly reflects evasive maneuver demand and remaining evasive time, rather than a purely temporal proximity measure.

## Method at a Glance

GEI combines two interpretable quantities:

- `InDepth`: a geometric proxy for evasive maneuver demand, defined by the projected intrusion depth between the two road users.
- `TEM`: Time for Evasive Maneuver, the remaining time before extrapolated oriented bodies first overlap.

Instead of relying on a single short-term motion extrapolation, GEI evaluates four motion hypotheses:

```text
CV-CV, CV-CTRV, CTRV-CV, CTRV-CTRV
```

where `CV` denotes constant velocity and `CTRV` denotes constant turn rate and velocity. The four mode-specific emergency indices are aggregated into the final `GEI`, reducing dependence on any single deterministic motion assumption and improving the representation of PTW turning and lateral maneuverability.

## Research Highlights

Empirical evaluation on naturalistic vehicle--PTW conflicts and reconstructed crashes shows that GEI:

- Captures both risk escalation and risk resolution during vehicle--PTW interactions.
- Distinguishes fine-grained risk when temporal proximity is similar but evasive demand differs.
- Provides stronger crash-precursor separability in early pre-crash windows.
- Achieves the earliest sustained warnings under percentile-aligned false-alarm constraints.
- Retains the most crash-outcome-relevant information on average across the pre-crash horizon.
- Runs at low frame-level computational cost in a serial Python implementation: mean `4.27 ms/frame`, median `3.99 ms/frame` over `175,053` valid frames in the reported evaluation.

Based on the reported datasets, a preliminary calibration suggests that GEI values around `0.68-0.94 m/s` may indicate a data-dependent high-risk transition range for vehicle--PTW interactions. This threshold range is not universal and should be recalibrated for new datasets, road-user types, and deployment contexts.

## Quick Start

From the repository root, run the following commands to install the package, compute GEI for one frame, enrich an example CSV, and optionally generate a GIF.

```bash
python -m pip install -e .

gei frame --values 504.0451 -271.9787 22.9184 2.5530 17.0237 2.5907 0.0 501.8724 -278.5692 24.9702 2.4877 16.3289 2.5973 0.0 --json

gei csv --input examples/data/SIND_Tianjin_8_6_1_180_181.csv --output outputs/GEI_SIND_Tianjin_8_6_1_180_181.csv

gei-gif --input outputs/GEI_SIND_Tianjin_8_6_1_180_181.csv
```

Expected single-frame core result:

```text
GEI = 3.5840
TEM_eff = 2.2775 s
InDepth_eff = 8.1623 m
```

If `python` is not available on Windows, replace it with `py`. If the `gei` or `gei-gif` console commands are not on `PATH`, use the compatibility commands:

```bash
py main.py frame --values 504.0451 -271.9787 22.9184 2.5530 17.0237 2.5907 0.0 501.8724 -278.5692 24.9702 2.4877 16.3289 2.5973 0.0 --json
py main.py csv --input examples/data/SIND_Tianjin_8_6_1_180_181.csv --output outputs/GEI_SIND_Tianjin_8_6_1_180_181.csv
py gif_maker.py --input outputs/GEI_SIND_Tianjin_8_6_1_180_181.csv
```

## Installation

For normal use on Windows CMD:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e .
```

For PowerShell, activate the virtual environment with:

```powershell
.venv\Scripts\Activate.ps1
```

If `python` points to the Windows Store placeholder, use `py`:

```bash
py -m venv .venv
.venv\Scripts\activate
py -m pip install -e .
```

After installation, two commands are available:

```bash
gei --help
gei-gif --help
```

On some Windows installations, `pip` may warn that the Python `Scripts` directory is not on `PATH`. In that case, either add that directory to `PATH`, or use the module/script entry points:

```bash
py -m gei.cli --help
py -m gei.visualization --help
py main.py --help
py gif_maker.py --help
```

The root-level scripts are kept for compatibility:

```bash
python main.py --help
python gif_maker.py --help
```

## Input Definition

GEI is computed from the instantaneous states of two interacting road users. Each road user is represented by seven parameters.

For each road user `i` in `{A, B}`, the input state is:

```text
(x_i, y_i, v_i, h_i, L_i, W_i, omega_i)
```

where:

- `x_i`: global X position `[m]`
- `y_i`: global Y position `[m]`
- `v_i`: speed magnitude `[m/s]`
- `h_i`: heading angle `[rad]`
- `L_i`: body length `[m]`
- `W_i`: body width `[m]`
- `omega_i`: yaw rate (`yaw_rate`) `[rad/s]`

One interaction frame therefore consists of 14 values in total.

Command-line order for each road user:

```text
x y speed heading length width yaw_rate
```

Full command-line order:

```text
xA yA vA hA LA WA yawA xB yB vB hB LB WB yawB
```

### Notes on Yaw Rate Input

Some trajectory datasets do not provide yaw rate directly. In that case, yaw rate can be estimated from the historical heading sequence using finite differences, ideally with mild smoothing, such as low-pass filtering, to suppress numerical jitter.

- If a road user does not exhibit noticeable turning behavior, `yaw_rate = 0` is acceptable.
- If turning is evident, a more accurate yaw-rate estimate is strongly recommended.

Input values should be finite. Speeds should be non-negative, and body length and width must be positive. Heading and yaw rate are expected in radians and radians per second, respectively; convert degree-based datasets before calling GEI.

### Applicability Beyond Vehicle--PTW Interactions

GEI was motivated by vehicle--PTW interaction risk, but its input definition is road-user agnostic. The same format can be used for vehicle--vehicle interactions and other road-user pairs, such as vehicle--pedestrian or vehicle--cyclist interactions, as long as each participant can be represented by position, speed, heading, yaw rate, length, and width.

## Workflow 1: Single-Frame GEI

Use this workflow when you already have one frame with two road users.

The 14 input parameters follow the order defined in [Input Definition](#input-definition):

```text
xA yA vA hA LA WA yawA xB yB vB hB LB WB yawB
```

Example:

```bash
gei frame --values 504.0451 -271.9787 22.9184 2.5530 17.0237 2.5907 0.0 501.8724 -278.5692 24.9702 2.4877 16.3289 2.5973 0.0 --json
```

Equivalent compatibility command:

```bash
python main.py frame --values 504.0451 -271.9787 22.9184 2.5530 17.0237 2.5907 0.0 501.8724 -278.5692 24.9702 2.4877 16.3289 2.5973 0.0 --json
```

By default, the result includes GEI core metrics plus traditional SSM metrics. To compute only the GEI core metrics:

```bash
gei frame --values 504.0451 -271.9787 22.9184 2.5530 17.0237 2.5907 0.0 501.8724 -278.5692 24.9702 2.4877 16.3289 2.5973 0.0 --core-only --json
```

The default prediction settings are `--dt 0.05` seconds and `--horizon 10.0` seconds. These can be changed for sensitivity or runtime studies:

```bash
gei frame --values 504.0451 -271.9787 22.9184 2.5530 17.0237 2.5907 0.0 501.8724 -278.5692 24.9702 2.4877 16.3289 2.5973 0.0 --dt 0.1 --horizon 8.0 --json
```

## Workflow 2: CSV Frame-by-Frame GEI

Use this workflow when each row in a CSV is one frame and you want GEI appended to every row.

Run on one example CSV:

```bash
gei csv --input examples/data/SIND_Tianjin_8_6_1_180_181.csv
```

The default output is written beside the input with a `GEI_` prefix:

```text
examples/data/GEI_SIND_Tianjin_8_6_1_180_181.csv
```

For a cleaner workflow, write generated files to an output folder:

```bash
gei csv --input examples/data/SIND_Tianjin_8_6_1_180_181.csv --output outputs/GEI_SIND_Tianjin_8_6_1_180_181.csv
```

To process all raw CSV files in a directory:

```bash
gei batch --input-dir examples/data --pattern "*.csv" --output-dir outputs
```

Generated files whose names start with `GEI_`, `ei_`, or `runtime_` are skipped automatically.

Use `--skip-existing` to avoid overwriting existing generated CSVs. Use `--decimals N` to control output rounding, or `--no-round` to keep full floating-point precision.

## Workflow 3: Optional GIF Visualization

The visualization tool reads enriched CSV files. It does not recompute GEI.

```bash
gei-gif --input outputs/GEI_SIND_Tianjin_8_6_1_180_181.csv
```

GIF files are written to:

```text
gif_visualizations/
```

If the CSV filename starts with `GEI_SIND`, `gei-gif` uses the optional map asset:

```text
assets/maps/map_relink_law_save.osm
```

If the same SIND data are saved as a generic name such as `GEI_example.csv`, the map background is not enabled. Visualization options can be adjusted from the command line:

```bash
gei-gif --input outputs/GEI_SIND_Tianjin_8_6_1_180_181.csv --time-range 0 1 --frame-step 1 --gei-max 2.0 --output-dir gif_visualizations
```

Use `--skip-gif` to validate that an enriched CSV can be read without spending time rendering the GIF.

## Workflow 4: Runtime Benchmarking

Use `benchmark` when measuring computational cost. This command reads raw CSV files, computes the metrics repeatedly, and does not write output CSVs.

```bash
gei benchmark --input-dir examples/data --pattern "*.csv" --repeat 5
gei benchmark --input examples/data/SIND_Tianjin_8_6_1_180_181.csv --repeat 10 --core-only
```

The summary reports Python, NumPy, and pandas versions plus mean, median, p90, p95, and p99 milliseconds per successful frame. Benchmark results depend on hardware, Python version, `--dt`, `--horizon`, and whether traditional SSM metrics are included.

## Required CSV Columns

Each row must contain these columns:

```text
Position X (m)
Position Y (m)
Velocity (m/s)
Heading
Length (m)
Width (m)
Yawrate
2_Position X (m)
2_Position Y (m)
2_Velocity (m/s)
2_Heading
2_Length (m)
2_Width (m)
2_Yawrate
```

The first group is road user A. The `2_` group is road user B.

## Output Columns

CSV batch processing appends:

```text
TEM_CVCV, TEM_CVCT, TEM_CTCV, TEM_CTCT
InDepth_CVCV, InDepth_CVCT_CA, InDepth_CTCV_CA, InDepth_CTCT_CA
MEI, EI_CVCT_CA, EI_CTCV_CA, EI_CTCT_CA, GEI
InDepth_eff, TEM_eff
DRAC, DRAC2D, TTC, 2D-TTC, TAdv, ACT, EI, TTC2D, BBox distance (m)
```

Use `--core-only` when only GEI-related computations are needed. For schema compatibility, traditional SSM columns are still included in the output and filled with default values.

Important output notes:

- In `--core-only` mode, traditional SSM columns such as `DRAC`, `TTC`, `ACT`, and `BBox distance (m)` are placeholders, not computed metrics.
- JSON output represents non-finite values as strings: `"inf"`, `"-inf"`, and `"nan"`.
- `TTC2D` follows the two-dimensional TTC implementation from Yiru Jiao's `Two-Dimensional-Time-To-Collision` repository.
- `2D-TTC` refers to the method proposed in *Modeling driver's evasive behavior during safety-critical lane changes: Two-dimensional time-to-collision and deep reinforcement learning*. In the code, this column is normalized from the internal `D2TTC` key for output-schema readability.

## Python API

```python
from gei import compute_single_frame, process_one_csv

result = compute_single_frame(
    504.0451, -271.9787, 22.9184, 2.5530, 17.0237, 2.5907, 0.0,
    501.8724, -278.5692, 24.9702, 2.4877, 16.3289, 2.5973, 0.0,
)
print(result["GEI"])

process_one_csv(
    "examples/data/SIND_Tianjin_8_6_1_180_181.csv",
    output_path="outputs/GEI_SIND_Tianjin_8_6_1_180_181.csv",
)
```

## Project Layout

```text
.
|-- src/gei/
|   |-- __init__.py
|   |-- cli.py              # Public CLI and CSV workflow
|   |-- core.py             # GEI, CTRV, geometry, and SSM kernels
|   `-- visualization.py    # Optional GIF visualization
|-- examples/
|   `-- data/               # Example raw CSV inputs
|-- assets/
|   |-- demos/              # README GIF demonstrations
|   `-- maps/               # Optional map assets for visualization
|-- tests/                  # Smoke and regression tests
|-- main.py                 # Compatibility wrapper for gei CLI
|-- gif_maker.py            # Compatibility wrapper for gei-gif CLI
|-- pyproject.toml          # Package metadata and console commands
|-- requirements.txt
`-- README.md
```

This is the standard `src/` package layout used by many Python open-source projects. Algorithmic code is implemented in `src/gei/core.py`, user-facing command-line workflows are implemented in `src/gei/cli.py`, static visual resources are stored in `assets/`, and reproducible examples are stored in `examples/`.

## Development Workflow

Install in editable mode:

```bash
python -m pip install -e ".[dev]"
```

Run smoke checks:

```bash
python -m py_compile src/gei/cli.py src/gei/core.py src/gei/visualization.py main.py gif_maker.py
python -m pytest
gei frame --values 504.0451 -271.9787 22.9184 2.5530 17.0237 2.5907 0.0 501.8724 -278.5692 24.9702 2.4877 16.3289 2.5973 0.0 --json
gei csv --input examples/data/SIND_Tianjin_8_6_1_180_181.csv --output outputs/GEI_SIND_Tianjin_8_6_1_180_181.csv
gei benchmark --input-dir examples/data --pattern "*.csv" --repeat 3 --core-only
```

Open-source conventions used here:

- `src/` contains importable package code.
- `assets/` contains static resources such as demo GIFs and maps.
- `examples/` contains small reproducible input data.
- `tests/` is reserved for smoke tests and regression tests.
- Generated files go to `outputs/` or `gif_visualizations/` and are ignored by Git.

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
