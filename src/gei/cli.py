"""Small JSON command-line interface; errors never become safe-risk records."""

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

from . import ActorState, JointFuture, SolverOptions, Trajectory
from .api import compute_gei, compute_gei_from_futures


def _json_safe(value):
    # JSON has no numeric infinity. Preserve its meaning explicitly.
    if isinstance(value, float) and not math.isfinite(value):
        return None if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def evaluate_document(document):
    """Evaluate either a current-state JSON document or a joint-future document."""
    options = SolverOptions(**document.get("solver", {}))
    mode = document["mode"]
    if mode == "states":
        return compute_gei(
            ActorState(**document["actor_a"]),
            ActorState(**document["actor_b"]),
            horizon=document["horizon"],
            options=options,
        )
    if mode == "futures":
        times = document["times"]
        futures = [
            JointFuture(
                Trajectory(times, f["a"]),
                Trajectory(times, f["b"]),
                f["weight"],
                f.get("label", ""),
            )
            for f in document["futures"]
        ]
        return compute_gei_from_futures(
            futures, document["size_a"], document["size_b"], options=options
        )
    raise ValueError("mode must be 'states' or 'futures'")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="GEI from current states or weighted future poses."
    )
    parser.add_argument("input", type=Path, help="input JSON; see examples/")
    parser.add_argument("--output", type=Path, help="new result JSON (refuses to overwrite)")
    args = parser.parse_args(argv)
    try:
        result = evaluate_document(json.loads(args.input.read_text(encoding="utf-8")))
        text = json.dumps(_json_safe(asdict(result)), indent=2, allow_nan=False)
        if args.output:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(text + "\n")
        else:
            print(text)
    except (ValueError, TypeError, KeyError, OSError, RuntimeError) as exc:
        parser.exit(2, f"gei: {exc}\n")


if __name__ == "__main__":
    main()
