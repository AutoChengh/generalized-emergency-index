"""Paper Table III protocol: negative event maxima and pooled positive frames."""

import argparse
import csv
import hashlib
import io
import json
import platform
import re
from pathlib import Path

import numpy as np
import sklearn
from sklearn.metrics import average_precision_score


def _load_scores(path, *, allow_unverified=False):
    path = Path(path)
    if "partial" in path.name.lower().split(".") or (path.parent / "failure.json").exists():
        raise ValueError("failed or partial scoring run; do not analyze incomplete results")
    # Hash and parse the same bytes, even if another process replaces the file.
    raw = path.read_bytes()
    score_hash = hashlib.sha256(raw).hexdigest()
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline=""))
    required = {"frame_key", "event_id", "label", "lead_time", "pc", "gei", "status"}
    fields = reader.fieldnames or []
    if len(fields) != len(set(fields)) or not required.issubset(fields):
        raise ValueError("score CSV requires unique headers including pc, gei and status")
    rows = list(reader)
    if not rows:
        raise ValueError("no frame scores")
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError("malformed score CSV row")
    keys = [r["frame_key"] for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate frame keys")
    events = {}
    for row in rows:
        if not row["frame_key"].strip() or not row["event_id"].strip():
            raise ValueError("event and frame identifiers must be nonempty")
        label = int(row["label"])
        if label not in (0, 1):
            raise ValueError("label must be 0 or 1")
        value = np.array([float(row["pc"]), float(row["gei"])])
        if not np.all(np.isfinite(value)) or not 0 <= value[0] <= 1 + 1e-9 or value[1] < 0:
            raise ValueError("nonfinite or invalid score; do not silently exclude this event")
        status = row["status"]
        if status not in ("no_contact", "finite_contact"):
            raise ValueError(f"unsupported classification status: {status!r}; inspect the run")
        if (status == "no_contact" and np.any(value != 0)) or (
            status == "finite_contact" and value[0] <= 0
        ):
            raise ValueError("inconsistent status, contact probability and GEI")
        # Contact with zero InDepth can legitimately have positive PC but zero GEI.
        event = events.setdefault(row["event_id"], {"label": label, "rows": []})
        if event["label"] != label:
            raise ValueError("inconsistent event label")
        lead = float(row["lead_time"])
        if not np.isfinite(lead) or (label == 0 and lead != 0):
            raise ValueError("lead_time must be finite; use zero for negative events")
        event["rows"].append((lead, value))
    negative, positive = [], []
    for key in sorted(events):
        event = events[key]
        if event["label"] == 0:
            negative.append(np.max([v for _, v in event["rows"]], axis=0))
        else:
            slots = {}
            for lead, value in event["rows"]:
                index = int(round(lead * 10))
                if abs(lead - index / 10) > 1e-7 or not 1 <= index <= 20 or index in slots:
                    raise ValueError("positive events require unique 0.1,...,2.0 s lead times")
                slots[index] = value
            if set(slots) != set(range(1, 21)):
                raise ValueError("every positive event must contain all 20 prescribed frames")
            positive.append([slots[i] for i in range(1, 21)])
    if not negative or not positive:
        raise ValueError("both positive and negative events are required")
    verification = {
        "mode": "unverified_csv_import",
        "score_sha256": score_hash,
        "source_registry_completeness": "not_checked_by_this_analyzer",
    }
    manifest_path = path.parent / "manifest.json"
    if manifest_path.exists():
        manifest_raw = manifest_path.read_bytes()
        try:
            manifest = json.loads(manifest_raw)
        except (ValueError, UnicodeDecodeError) as exc:
            raise ValueError("invalid scoring manifest") from exc
        if not isinstance(manifest, dict) or manifest.get("completed") is not True:
            raise ValueError("scoring manifest must certify completed=true")
        if manifest.get("score_sha256") != score_hash:
            raise ValueError("score hash does not match the completed scoring manifest")
        for field, expected in (("frames", len(rows)), ("events", len(events))):
            if type(manifest.get(field)) is not int or manifest[field] != expected:
                raise ValueError(f"scoring manifest {field} count does not match the CSV")
        input_hash = manifest.get("input_sha256")
        if not isinstance(input_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", input_hash):
            raise ValueError("scoring manifest requires a valid input bank SHA-256")
        verification.update(
            mode="completed_scoring_manifest",
            manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(),
            input_bank_sha256=input_hash,
        )
    elif not allow_unverified:
        raise ValueError(
            "completed scoring manifest.json required; use --allow-unverified only for "
            "independently checked external CSV imports"
        )
    return np.array(negative), np.array(positive), verification


def load_scores(path, *, allow_unverified=False):
    """Validate a completed scoring run and preserve the paper aggregation protocol."""
    negative, positive, _ = _load_scores(path, allow_unverified=allow_unverified)
    return negative, positive


def estimands(negative, positive):
    """Four pooled-frame window APs, then their unweighted arithmetic mean.

    Preserve all five scores from every selected crash. Pooling is deliberately
    different from taking a within-crash maximum or averaging per-instant APs.
    """
    values = []
    for start in range(0, 20, 5):
        frames = positive[:, start : start + 5].reshape(-1, 2)
        scores = np.vstack([negative, frames])
        labels = np.r_[np.zeros(len(negative)), np.ones(len(frames))]
        values.append([average_precision_score(labels, scores[:, m]) for m in range(2)])
    values.append(np.mean(values, axis=0))
    return np.array(values)


def analyze(input_path, output_dir, bootstrap_reps=10000, seed=20260909, *, allow_unverified=False):
    if type(bootstrap_reps) is not int or bootstrap_reps < 2:
        raise ValueError("at least two bootstrap replicates are required")
    input_path, output_dir = Path(input_path), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    negative, positive, verification = _load_scores(input_path, allow_unverified=allow_unverified)
    point = estimands(negative, positive)
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(bootstrap_reps):
        # One event draw shared by all time slices and both metrics.
        n = rng.integers(len(negative), size=len(negative))
        p = rng.integers(len(positive), size=len(positive))
        sampled = estimands(negative[n], positive[p])
        differences.append(sampled[:, 1] - sampled[:, 0])
    intervals = np.quantile(differences, [0.025, 0.975], axis=0).T
    names = [f"window_{i / 10:.1f}_{(i + 4) / 10:.1f}s" for i in (1, 6, 11, 16)] + [
        "mean_four_windows"
    ]
    output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "input_sha256": verification["score_sha256"],
        "input_verification": verification,
        "negative_events": len(negative),
        "positive_events": len(positive),
        "positive_frames_per_window": len(positive) * 5,
        "negative_aggregation": "complete_event_maximum",
        "positive_aggregation": "five_individual_frames_per_crash_pooled_within_each_window",
        "primary_estimand": "arithmetic_mean_of_four_pooled_window_auprcs",
        "auprc_definition": "noninterpolated_average_precision",
        "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "bootstrap": {
            "unit": "source_event_cluster_retaining_all_five_positive_frames",
            "paired": True,
            "stratified": True,
            "replicates": bootstrap_reps,
            "seed": seed,
            "interval": "percentile_95",
            "multiplicity_adjusted": False,
        },
        "results": [
            {
                "comparison": name,
                "pc_auprc": float(v[0]),
                "gei_auprc": float(v[1]),
                "difference": float(v[1] - v[0]),
                "difference_ci": ci.tolist(),
            }
            for name, v, ci in zip(names, point, intervals)
        ],
    }
    (output_dir / "discrimination.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path, help="new output directory")
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument(
        "--allow-unverified",
        action="store_true",
        help="allow external CSV without a manifest; report is explicitly marked unverified",
    )
    args = parser.parse_args()
    report = analyze(
        args.input,
        args.output,
        args.bootstrap_reps,
        args.seed,
        allow_unverified=args.allow_unverified,
    )
    if report["input_verification"]["mode"] == "unverified_csv_import":
        print("WARNING: unverified external CSV import; source completeness was not established.")


if __name__ == "__main__":
    main()
