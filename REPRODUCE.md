# Reproducing the paper

The experiments use the [SwinFusion backbone](https://github.com/Linfeng-Tang/SwinFusion/tree/ab8c9ee46a4111527b9b55b6d0e77715f2593ada).
The pinned upstream revision was checked against the archived source for
the network, dataset loader, trainer, and original VIF loss. Use a separate
checkout of that revision. The experiment scripts do not modify it.

The [paper-v1 release](https://github.com/DQEQEQ/bigw-rlv-fusion-public/releases/tag/paper-v1)
provides ten final EMA checkpoints, their SHA-256 checksums, and the
reproduction attachment for manuscript Figures 3–5. This repository adds the
training and data preparation commands, per-image reference scores, and the
Figure 6 layout.

## Prepare the datasets

Download the source archives from [MSRS](https://github.com/Linfeng-Tang/MSRS),
[RoadScene](https://github.com/hanna-xu/RoadScene), [LLVIP](https://github.com/bupt-ai-cz/LLVIP),
and [TNO](https://figshare.com/articles/dataset/TNO_Image_Fusion_Dataset/1008029).

| Split | ZIP used in the local checks | Pairs | Role |
| --- | --- | ---: | --- |
| `msrs-train` | `MSRS-main.zip` | 1,083 | Training |
| `msrs-test` | `MSRS-main.zip` | 361 | Ablations, mechanism, seed stability, comparison |
| `roadscene` | `RoadScene-master.zip` | 221 | External test |
| `llvip` | `LLVIP_test3463_pairs.zip` | 3,463 | External test |
| `tno49` | `TNO_Image_Fusion_Dataset.zip` | 49 | External test and comparison |

Install the inference dependencies, then run one command per split.

~~~sh
python -m pip install -r requirements-inference.txt
python -m scripts.prepare_pairs --archive MSRS-main.zip --split msrs-train --output-dir data/msrs-train
python -m scripts.prepare_pairs --archive MSRS-main.zip --split msrs-test --output-dir data/msrs-test
python -m scripts.prepare_pairs --archive RoadScene-master.zip --split roadscene --output-dir data/roadscene
python -m scripts.prepare_pairs --archive LLVIP_test3463_pairs.zip --split llvip --output-dir data/llvip
python -m scripts.prepare_pairs --archive TNO_Image_Fusion_Dataset.zip --split tno49 --output-dir data/tno49
~~~

The LLVIP command also accepts the full dataset ZIP if it includes the
`infrared/test` and `visible/test` directories. Training and inference consume
`ir` and `vi_y`. The six fusion metrics consume `ir` and `vi`.
Use the original experimental `VI_Y` folders if available. The preparation
script generates grayscale inputs by OpenCV decoding, and the original
MSRS luminance files were unavailable for a byte-level comparison.
RoadScene uses the aligned `cropinfrared` and `crop_LR_visible` images.

The TNO aliases and image dimensions come from the original experiment
audit. Of the 49 raw pairs, 29 match an archived exact duplicate or
zero-distance perceptual match. The other 20 were reconstructed from the
scene names, modality filenames, and archived dimensions. Each row records
the evidence in `reference/splits/tno49-origin-matches.csv`. Treat these
reconstructed inputs as candidates until compared with the curated input
pixels or the archived inference scores.

## Reproduce the training runs

The archived seed-123 joint training run used Python 3.10, PyTorch 2.1.2,
torchvision 0.16.2, CUDA 11.8, and one NVIDIA RTX A5000. The separate training
requirements preserve those PyTorch and torchvision versions. Install them
from the CUDA 11.8 index, then install the remaining dependencies.

~~~sh
python -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements-training.txt
python -m scripts.train --repo-root path/to/SwinFusion --ir-dir data/msrs-train/ir --vi-dir data/msrs-train/vi_y --variant bigw_rlv --seed 123 --output-dir runs/seed123-bigw-rlv --dry-run
~~~

Remove `--dry-run` to train on one CUDA GPU. The full run defaults to 10,000
optimizer steps. The launcher uses batches of 12, split into three strided
micro-batches of four, with 128 × 128 crops and EMA decay 0.999.
It trains from initialization and writes `10000_E.pth` for inference.

Run `baseline`, `bi_eq`, `bils`, `bigw`, `rlv_only`, and `bigw_rlv` at seed 123.
Use distinct output folders for each run. For seeds 456 and 789 run
`baseline` and `bigw_rlv` with matched seeds. The protocol can be checked on
CPU with `--dry-run` and on CUDA with a short `--steps` run before starting the
full training schedule.

## Inference and evaluation

Use the trained checkpoints from the release or the matching fresh
checkpoint from the preceding step. Here is the seed-123 joint inference
command for MSRS.

~~~sh
python -m scripts.infer --repo_root path/to/SwinFusion --ir_dir data/msrs-test/ir --vi_dir data/msrs-test/vi_y --model_path weights/seed123-bigw-rlv-step10000-ema.pth --output_dir results/msrs-test/seed123/bigw-rlv --device auto --expected_count 361
~~~

Repeat over the six seed-123 MSRS variants and two matched pairs for seeds
456 and 789. For external evaluation run the seed-123 baseline, BI-GW and joint
models on each prepared test split without training on those splits. Change
the expected count to the value in the dataset table.

~~~sh
python -m scripts.evaluate_ivif --ir_dir data/msrs-test/ir --vi_dir data/msrs-test/vi --methods baseline=results/msrs-test/seed123/baseline bigw_rlv=results/msrs-test/seed123/bigw-rlv --output_dir results/msrs-test/seed123/metrics --expected_count 361
python -m scripts.evaluate_rlv --ir-dir data/msrs-test/ir --vi-dir data/msrs-test/vi --baseline-dir results/msrs-test/seed123/baseline --bigw-dir results/msrs-test/seed123/bigw --rlv-only-dir results/msrs-test/seed123/rlv-only --bigw-rlv-dir results/msrs-test/seed123/bigw-rlv --output-dir results/msrs-test/seed123/rlv --expected-count 361
~~~

The RLV evaluator reports error differences as candidate minus reference
and relative reductions as positive when the error falls. Its Wilcoxon
tests are two-sided, with Holm adjustment across five comparisons. The
published anchor-gradient deviations are in
`reference/scores/anchor-context.csv`. The original fixed-anchor
tensors and exact context selection were unavailable in the archive.

## Rebuild the tables and figures

Regenerate all five manuscript tables from the archived reference scores.

~~~sh
python -m scripts.build_tables --output-dir results/manuscript-tables
~~~

Table 3 uses full-precision per-image data and the sample standard deviation
across the three seed-matched runs. Table 4 uses archived 256 × 256 complexity
measurements for the common backbone. Table 5 uses the repaired per-image
scores for the representative methods and the official SwinFusion checkpoint.
Recomputing those scores from images requires the respective methods'
original public code, weights, and test configurations.

Extract `reproduction-materials.zip` from the `paper-v1` release, then follow
its README to render Figures 3–5. For Figure 6, use the paired source
images and four sets of seed-123 fused outputs.

~~~sh
python -m scripts.plot_fig6 --ir data/msrs-test/ir --vi data/msrs-test/vi --baseline results/msrs-test/seed123/baseline --bigw results/msrs-test/seed123/bigw --rlv-only results/msrs-test/seed123/rlv-only --bigw-rlv results/msrs-test/seed123/bigw-rlv --output results/figure6.svg
~~~

The layout uses the manuscript's three scene identifiers, fixed crop bounds,
and panel geometry. Figures 1 and 2 are method schematics.

## Verification and provenance

The local checks covered input-pair counts and dimensions for all listed
splits, MSRS training data loading and patch dimensions, loss and evaluation
checks, table reconstruction, and the 18 full views and 18 inset crops in
Figure 6. The existing release weights were checked against the archived
SHA-256 checksums. The training launcher has not yet completed a fresh
10,000-step GPU run on this machine.
The CPU rerun of the seed-123 joint checkpoint on MSRS scene 00959N matched
97.93% of the archived output pixels exactly. The remaining pixels differed
by one 8-bit gray level.
