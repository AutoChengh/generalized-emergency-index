# GEI

GEI is a Python toolkit for computing the Generalized Encroachment Index between two road users. It supports both single-frame calculation from 14 numeric state parameters and frame-by-frame CSV enrichment.

GIF visualization is included as a bonus tool for inspecting GEI dynamics in traffic-conflict cases.

## Table of Contents

- [Visual Examples](#visual-examples)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Workflow 1: Single-Frame GEI](#workflow-1-single-frame-gei)
- [Workflow 2: CSV Frame-by-Frame GEI](#workflow-2-csv-frame-by-frame-gei)
- [Workflow 3: GIF Visualization Bonus](#workflow-3-gif-visualization-bonus)
- [Required CSV Columns](#required-csv-columns)
- [Output Columns](#output-columns)
- [Python API](#python-api)
- [Project Layout](#project-layout)
- [Development Workflow](#development-workflow)
- [License](#license)

## Visual Examples

The examples below show GEI-based visualizations for two vehicle and powered-two-wheeler interactions. In each example, the scene view is paired with GEI-related curves so the spatial conflict and metric evolution can be inspected together.

The README uses lightweight static previews. Click a preview to open the animated GIF.

**SIND Tianjin Intersection: Strong Interaction**

[![SIND Tianjin vehicle and powered-two-wheeler strong interaction](assets/demos/sind-tianjin-intersection-vehicle-ptw-strong-interaction-poster.jpg)](assets/demos/sind-tianjin-intersection-vehicle-ptw-strong-interaction.gif)

This case comes from the SIND dataset and captures a strong vehicle and powered-two-wheeler interaction at an intersection in Tianjin, China.

**CIMSS-TA Hunan: Powered-Two-Wheeler Cut-In Collision**

[![CIMSS-TA Hunan powered-two-wheeler cut-in collision](assets/demos/cimss-ta-hunan-ptw-cut-in-collision-poster.jpg)](assets/demos/cimss-ta-hunan-ptw-cut-in-collision.gif)

This case comes from the CIMSS-TA database and shows a powered-two-wheeler cut-in event in Hunan, China that leads to a collision.

## Quick Start

Run these commands from the repository root to install the package, compute one frame, enrich an example CSV, and optionally generate a GIF.

```bash
python -m pip install -e .

gei frame --values 504.0451 -271.9787 22.9184 2.5530 0.0 17.0237 2.5907 501.8724 -278.5692 24.9702 2.4877 0.0 16.3289 2.5973 --json

gei csv --input examples/data/SIND_Tianjin_8_6_1_180_181.csv --output outputs/GEI_example.csv

gei-gif --input outputs/GEI_example.csv
```

If `python` is not available on Windows, replace it with `py` in the installation command.

## Installation

For normal use:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e .
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

The root-level scripts are kept for compatibility:

```bash
python main.py --help
python gif_maker.py --help
```

## Workflow 1: Single-Frame GEI

Use this workflow when you already have one frame with two road users.

The 14 input parameters must be ordered as:

```text
xA yA vA hA yawA lA wA xB yB vB hB yawB lB wB
```

where:

```text
x, y    position in meters
v       speed in meters per second
h       heading in radians
yaw     yaw rate in radians per second
l, w    object length and width in meters
A, B    the two road users
```

Example:

```bash
gei frame --values 504.0451 -271.9787 22.9184 2.5530 0.0 17.0237 2.5907 501.8724 -278.5692 24.9702 2.4877 0.0 16.3289 2.5973 --json
```

Equivalent compatibility command:

```bash
python main.py frame --values 504.0451 -271.9787 22.9184 2.5530 0.0 17.0237 2.5907 501.8724 -278.5692 24.9702 2.4877 0.0 16.3289 2.5973 --json
```

By default, the result includes GEI core metrics plus traditional SSM metrics. To compute only the GEI core metrics:

```bash
gei frame --values 504.0451 -271.9787 22.9184 2.5530 0.0 17.0237 2.5907 501.8724 -278.5692 24.9702 2.4877 0.0 16.3289 2.5973 --core-only --json
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
gei csv --input examples/data/SIND_Tianjin_8_6_1_180_181.csv --output outputs/GEI_example.csv
```

To process all raw CSV files in a directory:

```bash
gei batch --input-dir examples/data --pattern "*.csv" --output-dir outputs
```

Generated files whose names start with `GEI_`, `ei_`, or `runtime_` are skipped automatically.

## Workflow 3: GIF Visualization Bonus

The visualization tool reads enriched CSV files. It does not recompute GEI.

```bash
gei-gif --input outputs/GEI_example.csv
```

GIF files are written to:

```text
gif_visualizations/
```

If the CSV name starts with `GEI_SIND`, `gei-gif` uses the optional map asset:

```text
assets/maps/map_relink_law_save.osm
```

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

CSV enrichment appends:

```text
TEM_CVCV, TEM_CVCT, TEM_CTCV, TEM_CTCT
InDepth_CVCV, InDepth_CVCT_CA, InDepth_CTCV_CA, InDepth_CTCT_CA
MEI, EI_CVCT_CA, EI_CTCV_CA, EI_CTCT_CA, GEI
InDepth_eff, TEM_eff
DRAC, DRAC2D, TTC, 2D-TTC, TAdv, ACT, EI, TTC2D, BBox distance (m)
```

Use `--core-only` when only GEI-related metrics are needed. The traditional SSM columns remain in the output schema and are filled with default values.

## Python API

```python
from gei import compute_single_frame, process_one_csv

result = compute_single_frame(
    504.0451, -271.9787, 22.9184, 2.5530, 0.0, 17.0237, 2.5907,
    501.8724, -278.5692, 24.9702, 2.4877, 0.0, 16.3289, 2.5973,
)
print(result["GEI"])

process_one_csv(
    "examples/data/SIND_Tianjin_8_6_1_180_181.csv",
    output_path="outputs/GEI_example.csv",
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

This is the standard `src/` package layout used by many Python open-source projects. Algorithm code lives in `src/gei/core.py`, user-facing computation lives in `src/gei/cli.py`, static visual resources live in `assets/`, and reproducible examples live in `examples/`.

## Development Workflow

Install in editable mode:

```bash
python -m pip install -e .
```

Run smoke checks:

```bash
python -m py_compile src/gei/cli.py src/gei/core.py src/gei/visualization.py main.py gif_maker.py
gei frame --values 504.0451 -271.9787 22.9184 2.5530 0.0 17.0237 2.5907 501.8724 -278.5692 24.9702 2.4877 0.0 16.3289 2.5973 --json
gei csv --input examples/data/SIND_Tianjin_8_6_1_180_181.csv --output outputs/GEI_example.csv
```

Open-source conventions used here:

- `src/` contains importable package code.
- `assets/` contains static resources such as demo GIFs and maps.
- `examples/` contains small reproducible input data.
- `tests/` is reserved for smoke tests and regression tests.
- Generated files go to `outputs/` or `gif_visualizations/` and are ignored by git.

## License

Add a `LICENSE` file before publishing. MIT, BSD-3-Clause, and Apache-2.0 are common choices for research software.
