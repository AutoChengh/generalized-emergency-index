import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def module(name):
    path = Path(__file__).parents[1] / "experiments" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_same_future_evaluation_and_manifest(tmp_path):
    run = module("evaluate_future_bank")
    bank = tmp_path / "bank.npz"
    np.savez(
        bank,
        a=np.array([[[[0, 0, 0], [3, 0, 0]]]], dtype=float),
        b=np.array([[[[5, 0, 0], [5, 0, 0]]]], dtype=float),
        times=np.array([0, 3]),
        weights=np.ones((1, 1)),
        size_a=np.array([[2, 2]]),
        size_b=np.array([[2, 2]]),
        frame_key=np.array(["e0:0"]),
        event_id=np.array(["e0"]),
        label=np.array([0]),
        lead_time=np.array([0.0]),
    )
    manifest = run.evaluate(bank, tmp_path / "out")
    assert manifest["completed"] and manifest["frames"] == manifest["events"] == 1
    with (tmp_path / "out/frame_scores.csv").open() as stream:
        row = list(csv.DictReader(stream))[0]
    assert float(row["pc"]) == 1 and float(row["gei"]) == pytest.approx(2 / 3)
    assert manifest == json.loads((tmp_path / "out/manifest.json").read_text())


def test_event_protocol_and_paired_bootstrap(tmp_path):
    run = module("analyze_discrimination")
    path = tmp_path / "scores.csv"
    with path.open("w", newline="") as stream:
        w = csv.writer(stream)
        w.writerow(["frame_key", "event_id", "label", "lead_time", "pc", "gei", "status"])
        # Negative full-event maxima deliberately include an outlying frame.
        w.writerows(
            [
                ["n0", "n", 0, 0, 0.1, 0.1, "finite_contact"],
                ["n1", "n", 0, 0, 0.9, 0.3, "finite_contact"],
            ]
        )
        for i in range(1, 21):
            w.writerow([f"p{i}", "p", 1, i / 10, 0.8, 0.7, "finite_contact"])
    write_manifest(path)
    negative, positive = run.load_scores(path)
    assert negative[0].tolist() == [0.9, 0.3]
    assert positive.shape == (1, 20, 2)
    report = run.analyze(path, tmp_path / "report", bootstrap_reps=2)
    assert len(report["results"]) == 5
    assert report["positive_frames_per_window"] == 5
    for result in report["results"]:
        assert result["pc_auprc"] == pytest.approx(5 / 6)
        assert result["gei_auprc"] == 1
    with path.open("a", newline="") as stream:
        csv.writer(stream).writerow(["p21", "p", 1, 2.0, 0.8, 0.7, "finite_contact"])
    with pytest.raises(ValueError, match="unique"):
        run.load_scores(path)


def test_pooled_frames_are_not_event_maxima_or_mean_instant_ap():
    run = module("analyze_discrimination")
    from sklearn.metrics import average_precision_score

    negative = np.array([[0.5, 0.5], [0.8, 0.8]])
    frames = np.array([[0.1, 0.9], [0.2, 0.8], [0.6, 0.7], [0.7, 0.6], [0.9, 0.1]])
    positive = np.tile(frames, (4, 1))[None, ...]
    result = run.estimands(negative, positive)
    labels = [0, 0, 1, 1, 1, 1, 1]
    expected = [
        average_precision_score(labels, np.r_[negative[:, m], frames[:, m]]) for m in range(2)
    ]
    assert result == pytest.approx(np.tile(expected, (5, 1)))
    event_max_ap = average_precision_score([0, 0, 1], [0.5, 0.8, 0.9])
    assert result[0, 0] != pytest.approx(event_max_ap)
    mean_instant_ap = np.mean(
        [average_precision_score([0, 0, 1], [0.5, 0.8, x]) for x in frames[:, 0]]
    )
    assert result[0, 0] != pytest.approx(mean_instant_ap)


def write_manifest(path):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    manifest = {
        "completed": True,
        "score_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "input_sha256": "a" * 64,
        "frames": len(rows),
        "events": len({row["event_id"] for row in rows}),
    }
    (path.parent / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def score_fixture(path, *, pc=0, gei=0, status="no_contact"):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame_key", "event_id", "label", "lead_time", "pc", "gei", "status"])
        writer.writerow(["n0", "n", 0, 0, pc, gei, status])
        for i in range(1, 21):
            writer.writerow([f"p{i}", "p", 1, i / 10, 1, 0.7, "finite_contact"])
    return path


def bank_fixture(path, *, invalid_last=False):
    count = 22 if invalid_last else 21
    sizes = np.full((count, 2), 2.0)
    if invalid_last:
        sizes[-1, 0] = 0
    np.savez(
        path,
        a=np.tile([[[[0.0, 0, 0], [3, 0, 0]]]], (count, 1, 1, 1)),
        b=np.tile([[[[5.0, 0, 0], [5, 0, 0]]]], (count, 1, 1, 1)),
        times=np.array([0.0, 3]),
        weights=np.ones((count, 1)),
        size_a=sizes,
        size_b=np.full((count, 2), 2.0),
        frame_key=np.array([f"p{i}" for i in range(1, 21)] + [f"n{i}" for i in range(count - 20)]),
        event_id=np.array(["p"] * 20 + [f"n{i}" for i in range(count - 20)]),
        label=np.array([1] * 20 + [0] * (count - 20)),
        lead_time=np.r_[np.arange(1, 21) / 10, np.zeros(count - 20)],
    )
    return path


def test_completed_bank_to_verified_analysis(tmp_path):
    bank = bank_fixture(tmp_path / "bank.npz")
    manifest = module("evaluate_future_bank").evaluate(bank, tmp_path / "scores")
    report = module("analyze_discrimination").analyze(
        tmp_path / "scores/frame_scores.csv", tmp_path / "report", bootstrap_reps=2
    )
    assert report["input_verification"]["mode"] == "completed_scoring_manifest"
    assert report["input_verification"]["input_bank_sha256"] == manifest["input_sha256"]
    assert report["negative_events"] == report["positive_events"] == 1


def test_failed_bank_cannot_be_analyzed_or_renamed_to_success(tmp_path):
    bank = bank_fixture(tmp_path / "bank.npz", invalid_last=True)
    with pytest.raises(ValueError):
        module("evaluate_future_bank").evaluate(bank, tmp_path / "scores")
    partial = tmp_path / "scores/frame_scores.partial.csv"
    failure = json.loads((partial.parent / "failure.json").read_text())
    assert failure["completed"] is False and failure["frame_index"] == 21
    assert not (partial.parent / "manifest.json").exists()
    run = module("analyze_discrimination")
    for unverified in (False, True):
        with pytest.raises(ValueError, match="failed or partial"):
            run.analyze(partial, tmp_path / "report", 2, allow_unverified=unverified)
    copied = partial.with_name("renamed.csv")
    copied.write_bytes(partial.read_bytes())
    with pytest.raises(ValueError, match="failed or partial"):
        run.load_scores(copied, allow_unverified=True)
    assert not (tmp_path / "report").exists()


def test_external_csv_requires_explicit_unverified_import(tmp_path):
    path = score_fixture(tmp_path / "scores.csv")
    run = module("analyze_discrimination")
    with pytest.raises(ValueError, match="manifest.json required"):
        run.load_scores(path)
    report = run.analyze(path, tmp_path / "report", 2, allow_unverified=True)
    assert report["input_verification"]["mode"] == "unverified_csv_import"
    assert report["input_verification"]["source_registry_completeness"].startswith("not_checked")


@pytest.mark.parametrize(
    "field,value",
    [
        ("completed", False),
        ("completed", 1),
        ("score_sha256", "b" * 64),
        ("input_sha256", "missing"),
        ("frames", 20),
        ("frames", True),
        ("events", 3),
        ("events", True),
    ],
)
def test_invalid_manifest_cannot_be_bypassed(tmp_path, field, value):
    path = score_fixture(tmp_path / "scores.csv")
    manifest = write_manifest(path)
    manifest[field] = value
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    for unverified in (False, True):
        with pytest.raises(ValueError):
            module("analyze_discrimination").load_scores(path, allow_unverified=unverified)


@pytest.mark.parametrize("content", ["{", "[]", "null"])
def test_malformed_manifest_rejected(tmp_path, content):
    path = score_fixture(tmp_path / "scores.csv")
    (tmp_path / "manifest.json").write_text(content)
    with pytest.raises(ValueError, match="manifest"):
        module("analyze_discrimination").load_scores(path)


@pytest.mark.parametrize(
    "pc,gei,status",
    [
        (0, 1, "no_contact"),
        (0.2, 0, "no_contact"),
        (0, 0, "finite_contact"),
        (0, 0, "numerical_failure"),
        (1, float("inf"), "current_overlap"),
        (1, 0.3, "current_overlap"),
    ],
)
def test_inconsistent_or_failed_scores_rejected(tmp_path, pc, gei, status):
    path = score_fixture(tmp_path / "scores.csv", pc=pc, gei=gei, status=status)
    write_manifest(path)
    with pytest.raises(ValueError):
        module("analyze_discrimination").load_scores(path)


def test_positive_contact_probability_with_zero_gei_is_valid(tmp_path):
    path = score_fixture(tmp_path / "scores.csv", pc=0.5, gei=0, status="finite_contact")
    write_manifest(path)
    negative, _ = module("analyze_discrimination").load_scores(path)
    assert negative.tolist() == [[0.5, 0]]


def test_missing_status_rejected(tmp_path):
    path = score_fixture(tmp_path / "scores.csv")
    lines = path.read_text().splitlines()
    path.write_text("\n".join(line.rsplit(",", 1)[0] for line in lines))
    with pytest.raises(ValueError, match="headers"):
        module("analyze_discrimination").load_scores(path, allow_unverified=True)


def test_bank_changed_during_run_does_not_publish_success(tmp_path, monkeypatch):
    bank = bank_fixture(tmp_path / "bank.npz")
    run = module("evaluate_future_bank")
    original = run.sha256
    calls = 0

    def changed_hash(path):
        nonlocal calls
        if path == bank:
            calls += 1
            if calls > 1:
                return "changed"
        return original(path)

    monkeypatch.setattr(run, "sha256", changed_hash)
    with pytest.raises(ValueError, match="bank changed"):
        run.evaluate(bank, tmp_path / "scores")
    assert (tmp_path / "scores/failure.json").exists()
    assert not (tmp_path / "scores/manifest.json").exists()
