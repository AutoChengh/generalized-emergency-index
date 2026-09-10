"""Evaluate GEI and contact probability on exactly the same immutable futures."""

import argparse
import csv
import hashlib
import json
import platform
from dataclasses import asdict
from pathlib import Path

import numpy as np
import scipy

import gei
from gei import JointFuture, SolverOptions, Trajectory, compute_gei_from_futures


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate(input_path, output_dir, options=SolverOptions()):
    """Finalize scores only after all records pass; preserve partial failures."""
    input_path, output_dir = Path(input_path), Path(output_dir)
    input_hash = sha256(input_path)
    package_dir = Path(gei.__file__).parent
    source_hashes = {p.name: sha256(p) for p in sorted(package_dir.glob("*.py"))}
    with np.load(input_path, allow_pickle=False) as data:
        required = {
            "a",
            "b",
            "times",
            "weights",
            "size_a",
            "size_b",
            "frame_key",
            "event_id",
            "label",
            "lead_time",
        }
        if set(data.files) != required:
            raise ValueError(f"NPZ fields must be exactly {sorted(required)}")
        a, b, weights = data["a"], data["b"], data["weights"]
        if a.ndim != 4 or a.shape != b.shape or a.shape[-1] != 3 or a.shape[0] == 0:
            raise ValueError("a/b must have identical shape (frames,joint_futures,times,3)")
        count, modes, steps, _ = a.shape
        if weights.shape != (count, modes) or data["times"].shape != (steps,):
            raise ValueError("weights or times has the wrong shape")
        if data["size_a"].shape != (count, 2) or data["size_b"].shape != (count, 2):
            raise ValueError("sizes must have shape (frames,2)")
        for field in ("frame_key", "event_id", "label", "lead_time"):
            if data[field].shape != (count,):
                raise ValueError(f"{field} must have shape (frames,)")
        keys, events, labels = data["frame_key"], data["event_id"], data["label"]
        if keys.dtype.kind not in "US" or events.dtype.kind not in "US":
            raise ValueError("frame/event keys must be string arrays (not pickled objects)")
        keys, events = keys.astype(str), events.astype(str)
        if len(set(keys)) != count or np.any(keys == "") or np.any(events == ""):
            raise ValueError("frame keys must be unique; frame/event keys must be nonempty")
        if not np.all(np.isin(labels, [0, 1])):
            raise ValueError("labels must be 0 or 1")
        if not np.all(np.isfinite(data["lead_time"])) or np.any(
            data["lead_time"][labels == 0] != 0
        ):
            raise ValueError("lead_time must be finite (use zero for negative events)")
        for event in set(events):
            if len(set(labels[events == event])) != 1:
                raise ValueError("an event cannot have conflicting labels")
        output_dir.mkdir(parents=True, exist_ok=False)
        temporary = output_dir / "frame_scores.partial.csv"
        index = None
        try:
            with temporary.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(
                    ["frame_key", "event_id", "label", "lead_time", "pc", "gei", "status"]
                )
                for index in range(count):
                    futures = [
                        JointFuture(
                            Trajectory(data["times"], a[index, j]),
                            Trajectory(data["times"], b[index, j]),
                            float(weights[index, j]),
                        )
                        for j in range(modes)
                    ]
                    result = compute_gei_from_futures(
                        futures, data["size_a"][index], data["size_b"][index], options=options
                    )
                    writer.writerow(
                        [
                            keys[index],
                            events[index],
                            int(labels[index]),
                            float(data["lead_time"][index]),
                            result.contact_probability,
                            result.gei,
                            result.status,
                        ]
                    )
            if sha256(input_path) != input_hash:
                raise ValueError("input future bank changed during scoring")
            if source_hashes != {p.name: sha256(p) for p in sorted(package_dir.glob("*.py"))}:
                raise ValueError("GEI source files changed during scoring")
            manifest = {
                "completed": True,
                "gei_version": gei.__version__,
                "input_sha256": input_hash,
                "score_sha256": sha256(temporary),
                "frames": count,
                "events": len(set(events)),
                "joint_futures_per_frame": modes,
                "solver": asdict(options),
                "source_sha256": source_hashes,
                "environment": {
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                    "scipy": scipy.__version__,
                },
            }
            manifest_temporary = output_dir / "manifest.partial.json"
            manifest_temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            temporary.rename(output_dir / "frame_scores.csv")
            # Publishing the manifest is the final success marker.
            manifest_temporary.rename(output_dir / "manifest.json")
        except Exception as exc:
            (output_dir / "failure.json").write_text(
                json.dumps(
                    {
                        "error": str(exc),
                        "completed": False,
                        "frame_index": index,
                        "frame_key": str(keys[index]) if index is not None else None,
                    }
                ),
                encoding="utf-8",
            )
            raise
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path, help="new output directory")
    args = parser.parse_args()
    evaluate(args.input, args.output)


if __name__ == "__main__":
    main()
