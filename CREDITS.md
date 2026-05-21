# Attribution

This repository contains code adapted from the following sources:

## TAPIR / TAP-Vid
Doersch et al. (2023). TAPIR: Tracking Any Point with per-frame Initialization and temporal Refinement. ICCV 2023.
Doersch et al. (2022). TAP-Vid: A Benchmark for Tracking Any Point in a Video. NeurIPS 2022.
- Repository: https://github.com/google-deepmind/tapnet
- License: Apache 2.0
- Used in: `scripts/metrics/run_metrics_real.py`, `scripts/metrics/run_metrics_stylized.py`, `scripts/eigencam/run_eigencam_real_final.py`, `scripts/eigencam/run_eigencam_stylized.py`, `scripts/feature_similarity/run_feature_similarity.py`

## EigenCAM
Muhammad & Yeasin (2020). EigenCAM: Class Activation Map using Principal Components. arXiv:2008.00299.
- Implementation adapted from: https://github.com/shyhyawJou/EigenCAM-Pytorch
- License: GPL-3.0
- Used in: `scripts/eigencam/run_eigencam_real_final.py`, `scripts/eigencam/run_eigencam_stylized.py`, `scripts/feature_similarity/run_feature_similarity.py`

## StyleMaster
Ye et al. (2025). StyleMaster: Stylize Your Video with Artistic Generation and Translation. CVPR 2025.
- Repository: https://github.com/KwaiVGI/StyleMaster
- Adaptation: padding added to handle videos shorter than 81 frames by repeating the last frame.
- Used in: `scripts/stylemaster/inference_stylemaster_v2v.py`
