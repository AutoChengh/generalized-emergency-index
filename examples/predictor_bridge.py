"""Runnable synthetic output-shape example; does NOT run a pretrained model."""

import numpy as np

from gei import compute_gei_from_futures, independent_joint_futures
from gei.adapters import empd_futures, qcnet_futures


def main():
    times = np.arange(1, 31) / 10
    # Two actors, six modes, thirty future samples, x/y coordinates.
    xy = np.zeros((2, 6, 30, 2))
    xy[0, :, :, 0] = times[None, :]
    xy[1, :, :, 0] = 5.0
    logits = np.zeros((2, 6))
    for name, adapter, key in (
        ("EMP-D", empd_futures, "y_hat"),
        ("QCNet", qcnet_futures, "loc_refine_pos"),
    ):
        output = {key: xy, "pi": logits}
        a, qa = adapter(output, 0, times, (0, 0, 0), max_yaw_rate=1.0)
        b, pb = adapter(output, 1, times, (5, 0, 0), max_yaw_rate=1.0)
        result = compute_gei_from_futures(independent_joint_futures(a, qa, b, pb), (2, 2), (2, 2))
        print(f"GEI ({name}): {result.gei:.6f} m/s; {len(result.evaluations)} joint futures")


if __name__ == "__main__":
    main()
