"""Regression coverage for canonical wheel import provenance checks."""

import ctypes
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "gei_check_release", Path(__file__).resolve().parents[1] / "tools" / "check_release.py"
)
RELEASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RELEASE)


def make_package(tmp_path, source=""):
    installed = tmp_path / "wheel installation directory"
    package = installed / "gei"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(source, encoding="utf-8")
    return installed


def run_check(installed):
    return subprocess.run(
        [sys.executable, "-I", "-c", RELEASE.wheel_import_preamble(installed)],
        capture_output=True,
        text=True,
    )


def test_wheel_provenance_accepts_normalized_path(tmp_path):
    installed = make_package(tmp_path)
    result = run_check(installed)
    assert result.returncode == 0, result.stderr


def test_wheel_provenance_resolves_both_sides(tmp_path):
    installed = make_package(tmp_path)
    alias = installed / ".." / installed.name
    result = run_check(alias)
    assert result.returncode == 0, result.stderr


def test_wheel_provenance_still_rejects_outside_import(tmp_path):
    outside = tmp_path / "editable checkout" / "gei" / "__init__.py"
    installed = make_package(tmp_path, f"__file__ = {str(outside)!r}\n")
    result = run_check(installed)
    assert result.returncode != 0
    assert "GEI imported from" in result.stderr
    assert "expected under" in result.stderr


@pytest.mark.skipif(sys.platform != "win32", reason="Windows 8.3 path aliases only")
def test_wheel_provenance_accepts_windows_short_path(tmp_path):
    installed = make_package(tmp_path)
    get_short_path = ctypes.windll.kernel32.GetShortPathNameW
    get_short_path.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32]
    get_short_path.restype = ctypes.c_uint32
    size = get_short_path(str(installed), None, 0)
    if not size:
        pytest.skip("short-path lookup unavailable")
    buffer = ctypes.create_unicode_buffer(size)
    assert get_short_path(str(installed), buffer, size)
    alias = Path(buffer.value)
    if alias == installed.resolve():
        pytest.skip("8.3 aliases are disabled on this volume")
    assert alias.resolve() == installed.resolve()
    result = run_check(alias)
    assert result.returncode == 0, result.stderr
