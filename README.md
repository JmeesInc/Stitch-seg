## stitch_seg

An experimental project for **stitching a temporal sequence of frames into a canvas** and running **segmentation inference on the stitched canvas**, then **warping predictions back to the original frame coordinates**.

`stitch_seg.StitchInferencer` keeps the stitching/state (homographies, canvas, masks, etc.) and calls a user-provided segmentation model (`torch.nn.Module`) on cropped canvas regions.

---

## Features

- **Canvas stitching**: sequentially paste frames onto a canvas while tracking cumulative transforms
- **Tool masking**: generate a tool mask using a pre-trained tool detector to suppress invalid regions
- **(Optional) depth masking**: when `enable_depth_mask=True`, build a depth-based mask (Depth Anything–style)
- **Segmentation on canvas**: crop the canvas region relevant to the current frame and run the segmentation model
- **Warp back to frame**: return the prediction warped to the current frame as `(C, H, W)`

---

## Requirements

- **Python**: `>= 3.9`
- **PyTorch / CUDA**: GPU is recommended (this repo also includes a `uv` index for CUDA wheels in `pyproject.toml`)
- **Key deps**: `kornia`, `ptlflow`, `segmentation-models-pytorch`, `opencv-python`, etc. (see `pyproject.toml`)

---

## Installation

### uv (recommended)

This repository includes `uv.lock` for reproducible environments.

```bash
cd /path/to/stitch_seg
uv sync
```

### pip (development / editable)

```bash
cd /path/to/stitch_seg
python3 -m pip install -e .
```

Notes:
- Installing a CUDA-enabled PyTorch build is environment-specific. Follow your platform/team guidance.

---

## Weights

By default, the code expects these relative paths (from the repository root):

- **ALIKED**: `weights/aliked-n16.pth` (`cfg.aliked_weights`)
- **LightGlue**: `weights/aliked_lightglue_v0-1_arxiv.pth` (`cfg.lightglue_weights`)
- **Tool detector (U-Net)**: `weights/convnext-unet-best.pth` (`cfg.tool_detector_weights`, default if unset)

Notes:
- If you run from a different working directory, relative paths may break. For production/use in notebooks, prefer **absolute paths** in `cfg.*_weights`.
- For depth masking, set `cfg.enable_depth_mask=True` and provide a valid checkpoint path via `cfg.depth_anything_v2_model` (the default in `stitch_seg/config.py` points outside this repo).

---

## Usage

### 1) Run the example script (`main.py`)

`main.py` is an example runner. Edit the following fields in `CFG` to match your environment:

- `CFG.video_path`
- `CFG.start_frame`, `CFG.end_frame`
- `CFG.segmentation_weights` (your segmentation model checkpoint)
- `CFG.output_dir`

Run:

```bash
python3 main.py
```

### 2) Use as a library (minimal example)

`StitchInferencer` handles stitching + coordinate transforms and expects a segmentation model callable as `model(crop)` where `crop` is `(B, 3, H, W)`.

```python
import numpy as np
import torch

from stitch_seg import StitchInferencer


class MySeg(torch.nn.Module):
    def forward(self, x):  # x: (B, 3, H, W)
        # Normalize + infer here, return (B, C, H, W)
        return torch.zeros((x.shape[0], 1, x.shape[2], x.shape[3]), device=x.device)


seg_model = MySeg().cuda().eval()

# cfg can be any object with attributes (if None, defaults are applied)
infer = StitchInferencer(model=seg_model, start_frame=0, cfg=None)

# frame_rgb is assumed to be (H, W, 3) uint8 in RGB
frame_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
frame_u = (
    torch.from_numpy(frame_rgb)
    .permute(2, 0, 1)
    .unsqueeze(0)
    .to(infer.device)
    .float()
)

infer.step_canvas(frame_u)
pred = infer.model_inference(frame_shape=frame_rgb.shape[:2])  # (C, H, W)
```

---

## Configuration (`cfg`)

Default values live in `stitch_seg/config.py` (`DEFAULT_CFG_VALUES`). Common knobs:

- **stitch / features**: `method`, `canvas_scale_x/y`, `feature_*`, `feature_ransac_thresh`
- **masking**: `seg_min_score`, `seg_max_coverage`, `tool_mask_dilate_px`
- **depth**: `enable_depth_mask`, `depth_anything_v2_model`
- **device**: `device`

---

## Repository layout (high level)

- `stitch_seg/`: library code
  - `inferencer.py`: `StitchInferencer` (stitch + inference + inverse warp)
  - `models.py`: loaders for tool detector / depth / feature pipeline
  - `stitch_utils_torch.py`: geometry/warping/mask utilities
- `weights/`: default weight locations (may vary by your environment)
- `main.py`: example runner

---

## Troubleshooting

- **`import stitch_seg` works but `from stitch_seg import StitchInferencer` fails**
  - `StitchInferencer` pulls in heavy runtime deps (e.g. `ptlflow`). Ensure **Python >= 3.9**, and that your `ptlflow`/`torch` stack is compatible.
- **Missing weights / relative path errors**
  - Use **absolute paths** for `cfg.aliked_weights`, `cfg.lightglue_weights`, and `cfg.tool_detector_weights`.
- **Depth masking crashes**
  - Provide a valid checkpoint at `cfg.depth_anything_v2_model` (the default points outside this repo).

---

## Development

- Editable install:

```bash
python3 -m pip install -e .
```

- Reproducible deps:
  - If you use `uv`, `uv.lock` is the reference.

---

## Contributing

Issues and PRs are welcome. Please include repro details (video conditions, your `CFG`/`cfg`, and logs).

---

## License

TBD (add a `LICENSE` file if you plan to distribute).

---

## Acknowledgements

This project depends on / is inspired by:

- LightGlue / ALIKED
- ptlflow
- Video Depth Anything–style models


