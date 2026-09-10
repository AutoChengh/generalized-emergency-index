import math

import numpy as np
import pytest

from gei.adapters import empd_futures, qcnet_futures, trajectories_from_positions


def test_stationary_heading_and_yaw_bound():
    modes = np.array([[[0, 0], [0, 0], [-1, 0], [-1, 0]]])
    path = trajectories_from_positions(modes, [0.1, 0.2, 0.3, 0.4], (0, 0, 0.4), max_yaw_rate=0.5)[
        0
    ]
    assert path.headings[0] == 0.4 and path.headings[1] == 0.4 and path.headings[2] == 0.4
    assert np.max(np.abs(path.heading_slopes)) <= 0.5 + 1e-12


def test_coordinate_transform():
    path = trajectories_from_positions(
        [[[1, 0], [2, 0]]],
        [1, 2],
        (10, 20, math.pi / 2),
        origin=(10, 20),
        frame_heading=math.pi / 2,
        max_yaw_rate=1,
    )[0]
    assert path.centers[-1] == pytest.approx([10, 22])
    assert path.headings[-1] == pytest.approx(math.pi / 2)


@pytest.mark.parametrize(
    "adapter,key", [(empd_futures, "y_hat"), (qcnet_futures, "loc_refine_pos")]
)
def test_predictor_schema_and_all_modes(adapter, key):
    xy = np.zeros((2, 6, 30, 2))
    logits = np.array([[1000, 999, 998, 997, 996, 995]] * 2)
    paths, p = adapter({key: xy, "pi": logits}, 1, np.arange(1, 31) / 10, (0, 0, 0), max_yaw_rate=1)
    assert len(paths) == len(p) == 6 and np.all(p > 0)
    assert p.sum() == pytest.approx(1)


def test_adapter_rejects_wrong_time_axis():
    with pytest.raises(ValueError):
        trajectories_from_positions([[[1, 0]]], [0], (0, 0, 0), max_yaw_rate=1)
