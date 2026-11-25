# BI-GW and RLV for infrared–visible image fusion

This repository contains the six supervision variants used to evaluate
batch-independent gradient-weighted structural supervision (BI-GW)
and relative local variance matching (RLV) on the SwinFusion backbone.

- [中文实验过程与结果](EXPERIMENTS_ZH.md) summarizes the experiments and links to the corresponding scores and downloads.
- [Reproduction guide](REPRODUCE.md) covers the datasets, training runs,
  checkpoint-based inference, evaluations, and manuscript tables and figures.
- [Paper weights and figure materials](https://github.com/DQEQEQ/bigw-rlv-fusion-public/releases/tag/paper-v1)
  contains the ten final EMA checkpoints, their checksums, and the scripts and
  data used to render Figures 3–5.
- [Reference scores and input manifests](reference) contains the per-image
  archives used to verify the results and the test-set identifiers.

Use `python -m pip install -r requirements.txt` and
`python -m unittest discover -s tests` for the CPU supervision checks.
For training and inference, follow the environments in the reproduction guide.
