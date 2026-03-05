# CholecSeg8k Benchmark

Semantic segmentation experiment for laparoscopic cholecystectomy using the CholecSeg8k dataset.
Trains segmentation models and performs stitch inference and evaluation via StitchInferencer.

## Table of Contents

- [Environment Setup](#environment-setup)
- [Datasets](#datasets)
- [Directory Structure](#directory-structure)
- [Reproducing the Experiment](#reproducing-the-experiment)
- [Main Scripts](#main-scripts)

---

## Environment Setup

From the project root (`stitch_seg_dev/`), run:

```bash
cd /path/to/stitch_seg_dev
uv venv
uv sync
```

Dependencies are managed in `pyproject.toml`.

---

## Datasets

This experiment requires the following two datasets. Please refer to `stitch_seg_dev/data/DATASET.md` for detail.

### Cholec80

Contains laparoscopic videos (`video01.mp4` through `video80.mp4`).
Used for stitch inference with StitchInferencer.

### CholecSeg8k

Contains segmentation annotations (mask images).
Used as ground truth for training and evaluation.

---

## Directory Structure

To reproduce the experiment, the following directory layout is required:

```
data/
├── Cholec80/
│   └── videos/
│       ├── video01.mp4
│       ├── video02.mp4
│       └── ... (videoXX.mp4)
│
└── CholecSeg8k/
    ├── video01/
    │   ├── video01_16585/
    │   │   ├── frame_16585_endo_watershed_mask.png
    │   │   ├── frame_16585_endo.png
    │   │   └── ...
    │   └── ...
    ├── video09/
    └── ...
```

- **Cholec80**: Video files in `videos/` as `video{XX}.mp4`
- **CholecSeg8k**: Masks under `video{XX}/video{XX}_{start_frame}/` as `frame_{frame_id}_endo_watershed_mask.png`

To use different paths, pass `--videos_dir` to `infer_unetpp.py` and update the data root in `split.py` accordingly.

---

## Reproducing the Experiment

All scripts are run from the **repository root** (`stitch_seg_dev/`).

### 1. Generate Data Splits

```bash
python exp/cholecseg8k_benchmark/split.py
```

Output: `exp/cholecseg8k_benchmark/splits/cholecseg8k_train_{0..4}.csv`, `...test_{0..4}.csv`

### 2. Training

Train Unet++ for each of the 5 folds:

```bash
# For fold 0 (set CFG.fold = 0 in train_unetpp.py)
python exp/cholecseg8k_benchmark/train_unetpp.py
```

- Model output: `exp/cholecseg8k_benchmark/models/unetpp/fold{N}.pth`
- Training config: see the `CFG` class in `train_unetpp.py`

### 3. Inference and Evaluation

Run stitch inference with StitchInferencer and Dice evaluation:

```bash
python exp/cholecseg8k_benchmark/infer_unetpp.py \
    --fold 0 \
    --videos_dir data/Cholec80/videos
```

Main options:

| Option | Description | Default |
|--------|-------------|---------|
| `--fold` | Fold to use (0–4) | 0 |
| `--csv_path` | Evaluation CSV path | `splits/cholecseg8k_test_{fold}.csv` |
| `--weights_path` | Model weights path | `models/unetpp/fold{fold}.pth` |
| `--videos_dir` | Video directory | `../../data/Cholec80/videos` |
| `--out_dir` | Output directory for predictions | `infer_outputs/stitch_unetpp_fold` |
| `--debug` | Save input/GT/prediction masks | false |

### 4. Post-hoc Evaluation (from saved predictions)

```bash
python exp/cholecseg8k_benchmark/evaluate_fast.py \
    --root infer_outputs/stitch_unetpp_internal_fixed_0
```

---

## Main Scripts

| File | Description |
|------|-------------|
| `split.py` | Generate train/test splits from CholecSeg8k (5-fold) |
| `train_unetpp.py` | Train Unet++ |
| `infer_unetpp.py` | Stitch inference and Dice evaluation with StitchInferencer |
| `evaluate_fast.py` | Post-hoc evaluation from saved prediction images |
| `model.py` | UnetPlusPlus model definition (ConvNeXt backbone) |
| `metrics.py` | Dice, size/distance-wise Dice, Temporal Consistency metrics |
| `copy_gt_orig.py` | Copy evaluation GT to `infer_outputs/GT_orig_{fold}/` |

Additional inference scripts: `infer_deeplab.py`, `infer_hrnet.py`, `infer_segformer.py`, `infer_upernet.py`, etc.
