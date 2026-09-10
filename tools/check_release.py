"""Test extracted source-distribution assets and the installed wheel, not an editable tree."""

import argparse
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath


def extract_source(archive_path, destination):
    """Extract only ordinary files/directories confined to this temporary directory."""
    destination = Path(destination).resolve()
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            name = PurePosixPath(member.name)
            target = (destination / member.name).resolve()
            if (
                name.is_absolute()
                or ".." in name.parts
                or "\\" in member.name
                or not target.is_relative_to(destination)
                or not (member.isdir() or member.isfile())
            ):
                raise ValueError(f"unsafe source archive member: {member.name!r}")
        for member in members:
            target = destination / member.name
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
    roots = list(destination.iterdir())
    if len(roots) != 1 or not roots[0].is_dir():
        raise ValueError("expected exactly one source-distribution root")
    return roots[0]


def check_release(dist):
    dist = Path(dist).resolve()
    wheels, sources = list(dist.glob("gei-*.whl")), list(dist.glob("gei-*.tar.gz"))
    if len(wheels) != 1 or len(sources) != 1:
        raise ValueError("dist must contain exactly one GEI wheel and one GEI source archive")
    with tempfile.TemporaryDirectory(prefix="gei-release-check-") as temporary:
        scratch = Path(temporary)
        source_root = extract_source(sources[0], scratch / "source")
        installed = scratch / "wheel"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--no-index",
                "--target",
                str(installed),
                str(wheels[0]),
            ],
            check=True,
            cwd=scratch,
        )
        # -I ignores caller PYTHONPATH/current-directory imports. Dependencies
        # remain available from this interpreter, but GEI must come from the wheel.
        preamble = (
            "import sys; from pathlib import Path; "
            f"sys.path.insert(0, {str(installed)!r}); "
            "import gei; "
            f"assert Path(gei.__file__).resolve().is_relative_to(Path({str(installed)!r})); "
        )
        commands = [
            "import pytest; raise SystemExit(pytest.main(['tests', '-q']))",
            "import runpy; sys.argv=['gei', 'examples/default.json']; "
            "runpy.run_module('gei.cli', run_name='__main__')",
            "import runpy; sys.argv=['gei', 'examples/weighted.json']; "
            "runpy.run_module('gei.cli', run_name='__main__')",
            "import runpy; runpy.run_path('examples/predictor_bridge.py', run_name='__main__')",
        ]
        for command in commands:
            subprocess.run(
                [sys.executable, "-I", "-X", "utf8", "-c", preamble + command],
                check=True,
                cwd=source_root,
            )
    print("Release check passed: source assets, wheel import, tests and examples.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, default=Path("dist"))
    check_release(parser.parse_args().dist)


if __name__ == "__main__":
    main()
