# AV2-derived GIF media notice

This notice accompanies:

- [AV2S-PTW-0334_three_method.gif](AV2S-PTW-0334_three_method.gif)
- [AV2S-PTW-0275_three_method.gif](AV2S-PTW-0275_three_method.gif)

## Source and attribution

Sensor imagery, LiDAR and annotations: **Argoverse 2 Sensor Dataset**,
© 2021 Argo AI, LLC.

- [Official dataset source](https://www.argoverse.org/av2.html)
- [Dataset terms](https://www.argoverse.org/about.html)
- License: **Creative Commons Attribution-NonCommercial-ShareAlike 4.0
  International ([CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/))**.

Dataset citation: Benjamin Wilson et al., *Argoverse 2: Next Generation Datasets
for Self-Driving Perception and Forecasting*, NeurIPS Datasets and Benchmarks,
2021. [Paper](https://arxiv.org/abs/2301.00493).

## Adaptation and licensing

The GEI project adapted these data through timestamp-based clip selection, PTW
cuboid overlays, a target-detail crop, trajectory and risk-history overlays,
multi-panel composition, resizing and GIF encoding. A subsequent revision
displays CV/CTRV absolute futures and removes redundant labels and in-image
footer text. Risk scores remain the unchanged archived paper results.

The adapted media are shared under **CC BY-NC-SA 4.0**, separately from the
software license. The repository's MIT code license does not relicense these
AV2-derived media. No endorsement by the dataset creators is implied.

Retain this attribution, source and license links, and indication of changes when
redistributing the GIFs. When embedding them in a README or web page, place an
attribution and license link next to the GIFs and retain this notice with the
downloadable files.

## Replay conventions

- Each GIF is 1600 × 970 pixels. Case 0334 contains 26 frames; case 0275 contains
  40 frames. Original approximately 10 Hz samples play at half speed, with a
  1.1 s final-frame hold. No generated intermediate frames are inserted.
- Times are seconds after the displayed integer origin `t0`, preserving the AV2
  dataset timestamp. They are not claimed UTC calendar dates.
- The faint curve ahead of the cursor is offline display context, not an input
  to the risk computation. Saturated curves show history through the current
  frame. These are open-loop research replays, not evidence of warning
  effectiveness or closed-loop safety.
- All three BEVs use the same frozen current ego coordinate frame. Blue denotes
  ego and red denotes PTW. The default GEI panel shows absolute body-center CV
  and CTRV futures for both actors over 10 s: dashed lines denote CV, solid lines
  denote CTRV, and faint dotted lines denote observed history. Nearly identical
  modes can visually overlap. Paths outside the displayed bounds are clipped
  by the axes, not truncated in the motion model.
- GEI (EMP-D) and GEI (QCNet) use the archived 3 s predictions with six modes per
  actor, giving 36 weighted combinations. Line opacity reflects mode weight.
  Different horizons and future banks mean these examples are not a controlled
  comparison of predictor accuracy.
- The GIFs preserve archived paper experiment scores; they do not recompute
  risk with the refactored public package. EMP-D/QCNet output adapters in this
  repository do not include pretrained neural inference or raw sensor data.

The selected media were copied without re-encoding from the validated clean-label
research previews. Raw sensor caches, per-frame research result tables and
internal validation manifests are not included in this repository.
