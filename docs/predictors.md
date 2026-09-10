# Connecting future generators

## Preferred contract

Supply world/scene-frame geometric-centre poses (x,y,heading), including the
measured current pose at t=0, plus body sizes and joint weights. Both actors
must share coordinates and times. No map or neural architecture is assumed
inside the GEI solver.

Direct joint futures are preferred when available. If a model instead outputs
per-actor modes, `independent_joint_futures` constructs their Cartesian product
under the stated independence approximation. The helper validates each marginal
probability vector separately. It neither selects top-k modes nor inserts CV
or CTRV modes.

## EMP-D and QCNet output adapters

The adapters correspond to output keys observed in the archived AV2 pipeline:

| Adapter | Positions | Scores | Selected dimension |
|---|---|---|---|
| empd_futures | y_hat: (B,K,T,C), first 2 channels x/y | pi: (B,K), logits | batch_index |
| qcnet_futures | loc_refine_pos: (N,K,T,C), first 2 channels x/y | pi: (N,K), logits | actor_index |

They accept NumPy arrays or tensors exposing detach/cpu/numpy. Softmax is applied
once to logits. Do not pass already normalized probabilities as logits.
All modes are retained. T must match `future_times`; horizon truncation must be
explicit upstream, not inferred from array shape.

`origin` and `frame_heading` specify the prediction coordinate frame:

```text
world_position = R(frame_heading) × predicted_position + origin
```

For a target-centred frame, use that target's current world position and heading,
not automatically the ego pose. For world-frame predictions use the default
zero origin and heading. `current_pose` is always a world-frame body pose.

## Heading policy

When body-heading predictions are unavailable, the adapter:

1. Preserves the measured current heading exactly.
2. Follows predicted displacement direction under a forward-motion assumption.
3. Holds the previous heading below a speed threshold.
4. Limits heading change to an explicitly supplied maximum yaw rate.

Choose `max_yaw_rate` for the road user and record it; the core does not prescribe
a universal value. The example's 1 rad/s is illustrative, not a validated
vehicle/PTW parameter. The default low-speed threshold is 0.1 m/s.
For reverse motion, sideslip, or reliable model-predicted body orientation,
construct `Trajectory` directly instead.

## What is and is not included

`examples/predictor_bridge.py` is a synthetic interface test with two actors and
six modes each. It is **not** model inference or an AV2 experimental result.
Model installation, checkpoints, scene encoding, map extraction and historical
observation masks are separate upstream responsibilities.

In the archived Sensor-log protocol, up to 50 history slots were right-aligned
and masked; 50 actual observations were not required. The final QCNet registry
did not impose a 150 m actor cutoff. A future full-pipeline export must preserve
these settings rather than importing an earlier adapter copy.

The relevant model papers are:

- A. Prutsch, H. Bischof, H. Possegger, [*Efficient Motion Prediction: A Lightweight
  & Accurate Trajectory Prediction Model With Fast Training and Inference
  Speed*](https://arxiv.org/abs/2409.16154), IROS 2024.
  [Official EMP code](https://github.com/a-pru/emp). EMP-D denotes the
  DETR-decoder configuration of EMP, not a separate paper.
- Z. Zhou, J. Wang, Y.-H. Li, Y.-K. Huang, [*Query-Centric Trajectory Prediction*](https://openaccess.thecvf.com/content/CVPR2023/html/Zhou_Query-Centric_Trajectory_Prediction_CVPR_2023_paper.html),
  CVPR 2023. [Official QCNet code](https://github.com/ZikangZhou/QCNet).

These output adapters do not redistribute either implementation or its weights.
