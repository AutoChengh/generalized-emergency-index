import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

from gei import (
    BodySize,
    JointFuture,
    Trajectory,
    compute_gei_from_futures,
    independent_joint_futures,
)
from gei.cli import evaluate_document, main


def future_bank():
    a = Trajectory([0, 3], [[0, 0, 0], [3, 0, 0]])
    b = Trajectory([0, 3], [[5, 0, 0]] * 2)
    stop = Trajectory([0, 3], [[0, 0, 0]] * 2)
    return [JointFuture(a, b, 0.25, "continue"), JointFuture(stop, b, 0.75, "stop")]


def test_weighted_decomposition():
    r = compute_gei_from_futures(future_bank(), (2, 2), BodySize(2, 2))
    assert r.gei == pytest.approx(1 / 6)
    assert r.contact_probability == 0.25
    assert r.conditional_ei == pytest.approx(2 / 3)
    assert r.gei == r.contact_probability * r.conditional_ei


def test_zero_contact_conditional_mean_undefined():
    bank = [replace(future_bank()[1], weight=1)]
    r = compute_gei_from_futures(bank, (2, 2), (2, 2))
    assert r.gei == 0 and r.contact_probability == 0 and r.conditional_ei is None


def test_zero_weight_not_solved():
    bank = future_bank()
    bank = [replace(bank[0], weight=0), replace(bank[1], weight=1)]
    r = compute_gei_from_futures(bank, (2, 2), (2, 2))
    assert r.gei == 0 and len(r.evaluations) == 1


def test_current_overlap_and_zero_weight_are_not_nan():
    p = Trajectory([0, 1], [[0, 0, 0]] * 2)
    r = compute_gei_from_futures([JointFuture(p, p, 0), JointFuture(p, p, 1)], (2, 2), (2, 2))
    assert math.isinf(r.gei) and r.contact_probability == 1


def test_weights_need_explicit_normalization():
    bank = [replace(f, weight=f.weight * 4) for f in future_bank()]
    with pytest.raises(ValueError, match="sum to 1"):
        compute_gei_from_futures(bank, (2, 2), (2, 2))
    assert compute_gei_from_futures(
        bank, (2, 2), (2, 2), normalize_weights=True
    ).gei == pytest.approx(1 / 6)


@pytest.mark.parametrize("weight", [-1, float("nan"), float("inf")])
def test_invalid_weights(weight):
    with pytest.raises(ValueError):
        replace(future_bank()[0], weight=weight)


def test_inconsistent_current_pose_rejected():
    bank = future_bank()
    bank[1] = replace(bank[1], trajectory_a=Trajectory([0, 3], [[1, 0, 0]] * 2))
    with pytest.raises(ValueError, match="current pose"):
        compute_gei_from_futures(bank, (2, 2), (2, 2))


def test_independent_modes():
    bank = future_bank()
    joint = independent_joint_futures(
        [f.trajectory_a for f in bank], [0.25, 0.75], [bank[0].trajectory_b], [1]
    )
    assert compute_gei_from_futures(joint, (2, 2), (2, 2)).gei == pytest.approx(1 / 6)
    with pytest.raises(ValueError):
        independent_joint_futures([bank[0].trajectory_a], [0.5], [bank[0].trajectory_b], [2])


@pytest.mark.parametrize("name,expected", [("default", 2 / 3), ("weighted", 1 / 6)])
def test_documented_examples(name, expected):
    doc = json.loads((Path(__file__).parents[1] / "examples" / f"{name}.json").read_text())
    assert evaluate_document(doc).gei == pytest.approx(expected)


def test_cli_roundtrip_and_no_overwrite(tmp_path):
    example = Path(__file__).parents[1] / "examples" / "weighted.json"
    dest = tmp_path / "result.json"
    main([str(example), "--output", str(dest)])
    assert json.loads(dest.read_text())["gei"] == pytest.approx(1 / 6)
    before = dest.read_bytes()
    with pytest.raises(SystemExit):
        main([str(example), "--output", str(dest)])
    assert dest.read_bytes() == before


def test_arbitrary_mode_count():
    f = future_bank()[0]
    for n in (1, 5, 6, 8):
        bank = [replace(f, weight=1 / n) for _ in range(n)]
        r = compute_gei_from_futures(bank, (2, 2), (2, 2))
        assert r.gei == pytest.approx(2 / 3)
        assert len(r.evaluations) == n
