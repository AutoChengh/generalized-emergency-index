import csv
import warnings

import pytest

from gei import compute_single_frame, process_one_csv
from gei.cli import main


def test_compute_single_frame_smoke():
    result = compute_single_frame(
        504.0451, -271.9787, 22.9184, 2.5530, 17.0237, 2.5907, 0.0,
        501.8724, -278.5692, 24.9702, 2.4877, 16.3289, 2.5973, 0.0,
    )

    assert "GEI" in result
    assert result["GEI"] == pytest.approx(3.5840)
    assert result["TEM_eff"] == pytest.approx(2.2775)
    assert result["InDepth_eff"] == pytest.approx(8.1623)


def test_compute_single_frame_core_only_keeps_schema_defaults():
    result = compute_single_frame(
        504.0451, -271.9787, 22.9184, 2.5530, 17.0237, 2.5907, 0.0,
        501.8724, -278.5692, 24.9702, 2.4877, 16.3289, 2.5973, 0.0,
        compute_extra_metrics=False,
    )

    assert result["GEI"] == pytest.approx(3.5840)
    assert result["DRAC2D"] == 0.0
    assert result["BBox distance (m)"] != result["BBox distance (m)"]


def test_process_one_csv_writes_expected_columns(tmp_path):
    output_path = tmp_path / "GEI_example.csv"
    stat = process_one_csv(
        "examples/data/SIND_Tianjin_8_6_1_180_181.csv",
        output_path=output_path,
    )

    assert stat["total_frames"] == 90
    assert stat["success_frames"] == 90
    assert output_path.exists()

    with output_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        first_row = next(reader)

    assert "GEI" in first_row
    assert "TEM_eff" in first_row
    assert "2D-TTC" in first_row


def test_benchmark_does_not_write_outputs(tmp_path):
    output_dir = tmp_path / "bench_outputs"

    main([
        "benchmark",
        "--input",
        "examples/data/SIND_Tianjin_8_6_1_180_181.csv",
        "--repeat",
        "1",
        "--output-dir",
        str(output_dir),
        "--core-only",
    ])

    assert not output_dir.exists() or not list(output_dir.iterdir())


def test_invalid_cli_dt_exits_cleanly():
    with pytest.raises(SystemExit):
        main([
            "frame",
            "--values",
            "504.0451", "-271.9787", "22.9184", "2.5530", "17.0237", "2.5907", "0.0",
            "501.8724", "-278.5692", "24.9702", "2.4877", "16.3289", "2.5973", "0.0",
            "--dt",
            "0",
        ])


def test_core_no_numpy_cross_deprecation_warning():
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        compute_single_frame(
            504.0451, -271.9787, 22.9184, 2.5530, 17.0237, 2.5907, 0.0,
            501.8724, -278.5692, 24.9702, 2.4877, 16.3289, 2.5973, 0.0,
        )

    messages = [str(record.message) for record in records]
    assert not any("Arrays of 2-dimensional vectors are deprecated" in msg for msg in messages)
