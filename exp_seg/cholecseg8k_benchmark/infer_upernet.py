"""
Script that stitches the corresponding videoXX.mp4 sequentially with StitchInferencer
for masks listed in cholecseg8k_test_{fold}.csv (validation), runs UPerNet inference
from the canvas crop at matching frames, and evaluates Dice.

Assumptions:
- The `file` column in the CSV is in the format:
  /data4/shared/Cholecystostomy/CholecSeg8k/video01/video01_16585/frame_16611_endo_watershed_mask.png
  where 16585 in `video01_16585` is the start_frame and `frame_16611_...` is the frame_id.
- Videos exist as `video{XX}.mp4` at:
  /data4/shared/Cholecystostomy/Cholec80/videos/video01.mp4

Example usage:
 nohup python infer_upernet.py --fold 1 --debug > internal1.log &

Important:
- `step_canvas` is run on **all frames from frame 0 to the end of the video** (not just the clip range).
- Inference (`model_inference`) and evaluation are run on frames listed in the CSV.
- Inference results are saved under `infer_outputs/stitch_upernet/`. In debug mode, input images and masks are also saved.
"""

import argparse
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import segmentation_models_pytorch as smp

from stitch_seg import StitchInferencer

from metrics import (  # noqa: E402
    DEFAULT_IGNORE_GT_LABELS,
    RAFTFlowEstimator,
    binned_dice_by_size_and_distance,
    dice_from_stats as dice_from_stats_np,
    dice_stats_per_class_numpy,
    iou_per_class_numpy,
    mean_dice as mean_dice_np,
    warp_label_with_backward_flow,
)

# Make `model.py` under `exp/02_CholecSeg8k` importable
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)


def fix_key(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remove prefixes introduced by DDP / torch.compile to make the state dict loadable."""
    out: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]
        elif k.startswith("_orig_mod."):
            k = k[10:]
        out[k] = v
    return out


class CFG:
    # UPerNet settings (must match training config)
    backbone = "tu-convnext_base"
    mask_num = 13
    num_classes = 13
    image_size = 512
    bbox_mode = "internal"
    dynamic_shape = False

    # StitchInferencer settings (minimal required; remaining defaults are filled in by stitch_seg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_static_roi = True
    apply_ellipse_mask = True
    equalize_hist_rgb = False
    enable_depth_mask = False
    tool_class_ch = 10
    startmid = False

    # label mapping (must match training config)
    LABEL2CH = {
        0: 0,# background
        50: 0,
        255: 0,
        5: 1, # Liver ligament
        11: 2, # Abdominal wall
        12: 3, # Fat
        13: 4,# Gastrointestinal Tract
        21: 5, # Liver
        22: 6, # Gallbladder
        23: 7, # Connective tissue
        24: 8, # Blood
        25: 9, # Cystic Duct
        31: 10, # Grasper (Tool)
        32: 11, # L-hook Electrocautery
        33: 12, # Hepatic Veins
    }
    unknown_label_value = 0
    aliked_model = "aliked-n16"
    aliked_weights = "../../weights/aliked-n16.pth"
    lightglue_weights = "../../weights/aliked_lightglue_v0-1_arxiv.pth"
    tool_detector_weights = "../../weights/convnext-unet-best.pth"
    port_detector_weights = "../../weights/convnext_base-unet-cholec80_port.pt"
    depth_anything_v2_model = "../../weights/video_depth_anything_vitl.pth"



def build_lut(label2ch: Dict[int, int], unknown: int = 0) -> np.ndarray:
    lut = np.full(256, int(unknown), dtype=np.uint8)
    for k, v in label2ch.items():
        kk = int(k)
        if 0 <= kk <= 255:
            lut[kk] = np.uint8(int(v))
    return lut


_LUT = build_lut(CFG.LABEL2CH, CFG.unknown_label_value)


def mask_raw_to_index(mask_raw: np.ndarray) -> np.ndarray:
    if mask_raw.dtype != np.uint8:
        mask_raw = mask_raw.astype(np.uint8, copy=False)
    return _LUT[mask_raw]


#
# NOTE:
# Dice/meanDice/size/dist/TC definitions are consolidated in exp/02_CholecSeg8k/metrics.py.
# The inference pipeline (logits_to_pred_idx etc.) is kept here; evaluation metrics are delegated to metrics.
#


def logits_to_pred_idx(x: torch.Tensor) -> torch.Tensor:
    """
    Normalizes the input to an (H,W) class index regardless of whether it is logits, one-hot, or an index.
    Expected shapes:
    - (C,H,W): return from stitched model_inference (logits)
    - (1,C,H,W): return from direct inference (logits)
    - (H,W): already an index
    - (1,H,W): squeezed to an index
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"expected torch.Tensor, got {type(x)}")
    if x.dim() == 4:
        # (1,C,H,W) -> (H,W)
        return x.argmax(dim=1).squeeze(0).to(torch.long)
    if x.dim() == 3:
        if x.shape[0] == 1:
            return x.squeeze(0).to(torch.long)
        # (C,H,W) -> (H,W)
        return x.argmax(dim=0).to(torch.long)
    if x.dim() == 2:
        return x.to(torch.long)
    raise ValueError(f"unexpected tensor dim={x.dim()} shape={tuple(x.shape)}")


class CanvasSegModel(nn.Module):
    """
    Wrapper that converts the canvas crop (RGB) passed by StitchInferencer
    to UPerNet input format (512, normalized), runs inference, and resizes back to the crop resolution.
    """

    def __init__(self, model: nn.Module, device: torch.device, input_size: int = 512, dynamic_shape: bool = False):
        super().__init__()
        self.model = model
        self.device = device
        self.input_size = int(input_size)
        self.dynamic_shape = dynamic_shape
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def forward(self, canvas_rgb: torch.Tensor) -> torch.Tensor:
        # canvas_rgb: (1, 3, H, W), RGB, uint8 or float
        x = canvas_rgb
        if x.dtype != torch.float32 and x.dtype != torch.float16:
            x = x.to(torch.float32)
        if x.max() > 1.0:
            x = x / 255.0
        # normalize
        x = (x - self.mean) / self.std
        # resize -> model -> resize back
        target_hw = x.shape[-2:]
        if self.dynamic_shape:
            #dynamic shape
            b, c, h, w = x.shape
            new_h = int(np.ceil(h / 32) * 32)
            new_w = int(np.ceil(w / 32) * 32)
            x_in = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)
        else:
            x_in = F.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"):
            logits = self.model(x_in)  # (1, C, 512, 512)
        if logits.shape[-2:] != target_hw:
            logits = F.interpolate(logits, size=target_hw, mode="bilinear", align_corners=False)
        return logits


@dataclass(frozen=True)
class SampleKey:
    video_id: int


_RE_VIDEO = re.compile(r"/video(?P<vid>\d{2})/")
_RE_START = re.compile(r"/video\d{2}_(?P<start>\d+)/")


def parse_video_id_and_start(mask_path: str) -> Tuple[int, int]:
    m1 = _RE_VIDEO.search(mask_path)
    m2 = _RE_START.search(mask_path)
    if not m1 or not m2:
        raise ValueError(f"Cannot parse video_id/start_frame from: {mask_path}")
    return int(m1.group("vid")), int(m2.group("start"))

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def build_ellipse_mask_from_gt(frame_to_masks: Dict[int, List[str]], num_votes: int = 5) -> Optional[np.ndarray]:
    """
    Creates an ellipse_mask by majority vote using multiple GT masks within a video.
    """
    mask_paths: List[str] = []
    for fid in sorted(frame_to_masks.keys()):
        mask_paths.extend(frame_to_masks[fid])
    if not mask_paths:
        return None
    sel = mask_paths[: max(1, int(num_votes))]
    bin_masks: List[np.ndarray] = []
    for p in sel:
        m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise FileNotFoundError(f"Mask not found: {p}")
        bin_masks.append(((m == 255) | (m == 50)).astype(np.uint8))
    stack = np.stack(bin_masks, axis=0)  # (N,H,W)
    votes = stack.sum(axis=0)
    thresh = (len(bin_masks) + 1) // 2  # >= ceiling(N/2)
    return (votes >= thresh).astype(np.uint8) * 255


def _to_uint8_rgb_hwc(frame_u: torch.Tensor) -> np.ndarray:
    """
    frame_u: (1,3,H,W) or (3,H,W) RGB on GPU/CPU, uint8 or float
    returns: (H,W,3) uint8 RGB
    """
    if not isinstance(frame_u, torch.Tensor):
        raise TypeError(f"expected torch.Tensor, got {type(frame_u)}")
    t = frame_u
    if t.dim() == 4:
        t = t[0]
    t = t.detach().to("cpu")
    if t.dtype != torch.uint8:
        tt = t.to(torch.float32)
        if tt.max() <= 1.0:
            tt = tt * 255.0
        t = tt.clamp(0, 255).to(torch.uint8)
    return t.permute(1, 2, 0).contiguous().numpy()

def _to_uint8_bgr_hwc(img: torch.Tensor | np.ndarray) -> Optional[np.ndarray]:
    if img is None:
        return None
    if isinstance(img, torch.Tensor):
        t = img.detach().cpu()
        if t.dim() == 4:
            t = t[0]
        if t.dim() == 3 and t.shape[0] in (1, 3):
            t = t.permute(1, 2, 0)
        t = t.to(torch.float32)
        if t.max() <= 1.0:
            t = t * 255.0
        arr = t.clamp(0, 255).to(torch.uint8).numpy()
    else:
        arr = img
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

    # 1ch -> BGR
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if arr.ndim == 3 and arr.shape[2] == 1:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)

    # 3ch: assume RGB (from torch) or already BGR (unknown). Treat as RGB and convert to BGR uniformly.
    if arr.ndim == 3 and arr.shape[2] == 3:
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return None


def build_palette(num_classes: int) -> np.ndarray:
    rng = np.random.default_rng(42)
    palette = rng.integers(0, 255, size=(max(num_classes, 2), 3), dtype=np.uint8)
    palette[0] = np.array([0, 0, 0], dtype=np.uint8)
    return palette


_PALETTE_BGR = build_palette(CFG.mask_num)  # (C,3) BGR


def colorize_label_map_bgr(label_hw: np.ndarray) -> np.ndarray:
    lab = label_hw.astype(np.int32, copy=False)
    lab = np.clip(lab, 0, CFG.mask_num - 1)
    return _PALETTE_BGR[lab]


def overlay_label_on_bgr(frame_bgr: np.ndarray, label_hw: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    if frame_bgr is None:
        return None
    h, w = frame_bgr.shape[:2]
    lab = label_hw
    if lab.shape[:2] != (h, w):
        lab = cv2.resize(lab, (w, h), interpolation=cv2.INTER_NEAREST)
    color = colorize_label_map_bgr(lab)
    blended = cv2.addWeighted(frame_bgr, 1.0 - alpha, color, alpha, 0.0)
    return blended


def _draw_label(img_bgr: np.ndarray, text: str) -> np.ndarray:
    if not text:
        return img_bgr
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness_bg = 6
    thickness_fg = 2
    org = (16, 42)
    cv2.putText(img_bgr, text, org, font, font_scale, (0, 0, 0), thickness_bg, cv2.LINE_AA)
    cv2.putText(img_bgr, text, org, font, font_scale, (255, 255, 255), thickness_fg, cv2.LINE_AA)
    return img_bgr


def build_debug_panel(labeled_bgr_images: list[tuple[str, Optional[np.ndarray]]], frame_shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = frame_shape_hw
    panels: list[np.ndarray] = []
    for label, img in labeled_bgr_images:
        if img is None:
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
        else:
            canvas = img.copy()
            if canvas.ndim == 2:
                canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
            if canvas.shape[:2] != (h, w):
                canvas = cv2.resize(canvas, (w, h), interpolation=cv2.INTER_LINEAR)
        canvas = _draw_label(canvas, label)
        panels.append(canvas)

    while len(panels) < 6:
        panels.append(np.zeros((h, w, 3), dtype=np.uint8))

    row1 = np.hstack(panels[:3])
    row2 = np.hstack(panels[3:6])
    stacked = np.vstack([row1, row2])
    # Downscale because the panel is too large
    resized = cv2.resize(stacked, (int(stacked.shape[1] * 0.33), int(stacked.shape[0] * 0.33)), interpolation=cv2.INTER_LINEAR)
    return resized


def save_pred_and_debug(
    out_dir_video: str,
    out_dir_original_video: Optional[str],
    frame_idx: int,
    pred_idx: torch.Tensor,
    frame_u: torch.Tensor,
    gt_idx: Optional[torch.Tensor],
    debug: bool,
    inferencer: Optional[StitchInferencer] = None,
    direct_pred_idx: Optional[torch.Tensor] = None,
) -> None:
    """
    - Always saves the predicted mask (class index).
    - When debug=True, also saves the input image (after ROI) and the GT/prediction masks.
    """
    ensure_dir(out_dir_video)
    pred_np = pred_idx.detach().to("cpu").to(torch.uint8).numpy()
    cv2.imwrite(os.path.join(out_dir_video, f"frame_{frame_idx:06d}_pred_idx.png"), pred_np)

    if out_dir_original_video is not None and direct_pred_idx is not None:
        ensure_dir(out_dir_original_video)
        direct_np = direct_pred_idx.detach().to("cpu").to(torch.uint8).numpy()
        cv2.imwrite(
            os.path.join(out_dir_original_video, f"frame_{frame_idx:06d}_pred_idx.png"),
            direct_np,
        )

    if not debug:
        return

    rgb = _to_uint8_rgb_hwc(frame_u)
    frame_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(out_dir_video, f"frame_{frame_idx:06d}_input.png"), frame_bgr)
    if gt_idx is not None:
        gt_np = gt_idx.detach().to("cpu").to(torch.uint8).numpy()
        cv2.imwrite(os.path.join(out_dir_video, f"frame_{frame_idx:06d}_gt_idx.png"), gt_np)
    if direct_pred_idx is not None:
        direct_np = direct_pred_idx.detach().to("cpu").to(torch.uint8).numpy()
        cv2.imwrite(
            os.path.join(out_dir_video, f"frame_{frame_idx:06d}_pred_original_idx.png"),
            direct_np,
        )

    # --- panel: current frame / canvas / model input(+pred) / seg-stitched / seg-original / GT ---
    # current frame
    cur = frame_bgr

    # canvas (inferencer.canvas is assumed to be RGB)
    canvas_bgr = None
    if inferencer is not None and getattr(inferencer, "canvas", None) is not None:
        canvas_bgr = _to_uint8_bgr_hwc(inferencer.canvas)

    # model input with inference
    model_in_bgr = None
    if inferencer is not None and getattr(inferencer, "last_model_input", None) is not None:
        model_in_bgr = _to_uint8_bgr_hwc(inferencer.last_model_input)
        # last_model_output: (1,C,Hc,Wc)
        if getattr(inferencer, "last_model_output", None) is not None:
            out = inferencer.last_model_output
            if isinstance(out, torch.Tensor):
                out = out.detach()
                if out.dim() == 4:
                    out = out[0]
                # (C,H,W) -> idx
                idx = out.argmax(dim=0).to(torch.uint8).cpu().numpy()
                if model_in_bgr is not None:
                    model_in_bgr = overlay_label_on_bgr(model_in_bgr, idx, alpha=0.45)

    # seg - stitched (overlay on current frame)
    overlay_stitched = overlay_label_on_bgr(frame_bgr, pred_np, alpha=0.45)

    # seg - original (direct prediction on current frame)
    overlay_direct = None
    if direct_pred_idx is not None:
        direct_np = direct_pred_idx.detach().to("cpu").to(torch.uint8).numpy()
        overlay_direct = overlay_label_on_bgr(frame_bgr, direct_np, alpha=0.45)

    # ground truth (colorize)
    gt_bgr = None
    if gt_idx is not None:
        gt_np = gt_idx.detach().to("cpu").to(torch.uint8).numpy()
        gt_bgr = colorize_label_map_bgr(gt_np)

    panel = build_debug_panel(
        [
            ("current frame", cur),
            ("canvas", canvas_bgr),
            ("model input (+pred)", model_in_bgr),
            ("seg - stitched", overlay_stitched),
            ("seg - original", overlay_direct),
            ("ground truth", gt_bgr),
        ],
        frame_bgr.shape[:2],
    )
    cv2.imwrite(os.path.join(out_dir_video, f"frame_{frame_idx:06d}_panel.png"), panel)


def crop_gt_with_inferencer_roi(gt_raw: np.ndarray, inferencer: StitchInferencer) -> np.ndarray:
    if gt_raw is None:
        return gt_raw
    if getattr(inferencer, "use_roi", False) and getattr(inferencer, "roi", None) is not None:
        x, y, w, h = inferencer.roi
        return gt_raw[y : y + h, x : x + w]
    return gt_raw


def load_model(weights_path: str, device: torch.device) -> nn.Module:
    model = smp.UPerNet(
        encoder_name=CFG.backbone, encoder_weights="imagenet", classes=13, activation="softmax"
    )
    state = torch.load(weights_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(fix_key(state), strict=True)
    model.eval()
    return model


def _resolve_path_maybe_relative_to_this_dir(path: str) -> str:
    """
    Since stitch_seg passes the path directly to torch.load, relative paths would be resolved
    against the runtime cwd. Here we convert to an absolute path relative to `exp/02_CholecSeg8k` (=_THIS_DIR).
    """
    if not path:
        return path
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(_THIS_DIR, path))


def evaluate_group(
    key: SampleKey,
    rows: pd.DataFrame,
    videos_dir: str,
    inferencer_cfg: CFG,
    seg_model: nn.Module,
    device: torch.device,
    pbar: Optional[tqdm] = None,
    out_dir: Optional[str] = None,
    debug: bool = False,
    start_frame_exclusive: Optional[int] = 0,
    stop_frame_exclusive: Optional[int] = None,
    compute_tc: bool = False,
    raft_variant: str = "small",
    tc_every: int = 1,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[float],
]:
    """
    rows: samples belonging to the video, expected in ascending frame_id order.
    step_canvas is run from frame 0 to the end of the video; inference and evaluation are performed on frames listed in the CSV.
    Returns: (inter_sum, pred_sum, gt_sum), each of shape (C,)
    """
    video_path = os.path.join(videos_dir, f"video{key.video_id:02d}.mp4")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")

    # StitchInferencer is stateful, so create a new one per video (start_frame=0 covers the full video)
    inferencer = StitchInferencer(seg_model, start_frame=0, cfg=inferencer_cfg)
    inferencer = inferencer.to(device)
    inferencer.eval()

    # target frame_id -> mask_path list (handles multiple masks per frame)
    frame_to_masks: Dict[int, List[str]] = {}
    for _, r in rows.iterrows():
        fid = int(r["frame_id"])
        frame_to_masks.setdefault(fid, []).append(str(r["file"]))
    target_set = set(frame_to_masks.keys())

    out_dir_video = None
    out_dir_orig_video = None
    if out_dir is not None:
        out_dir_video = os.path.join(out_dir, f"video{key.video_id:02d}")
        ensure_dir(out_dir_video)
    if getattr(inferencer_cfg, "out_dir_original", None) is not None:
        out_dir_orig_video = os.path.join(str(inferencer_cfg.out_dir_original), f"video{key.video_id:02d}")

    ellipse_mask_np = None
    if inferencer_cfg.apply_ellipse_mask and target_set:
        ellipse_mask_np = build_ellipse_mask_from_gt(frame_to_masks, num_votes=5)

    # Start reading from start_frame_exclusive
    start_f = int(start_frame_exclusive or 0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

    # stitched (class-wise)
    inter_sum = np.zeros((CFG.mask_num,), dtype=np.float64)
    pred_sum = np.zeros((CFG.mask_num,), dtype=np.float64)
    gt_sum = np.zeros((CFG.mask_num,), dtype=np.float64)
    # original (direct) (class-wise)
    inter_sum_org = np.zeros((CFG.mask_num,), dtype=np.float64)
    pred_sum_org = np.zeros((CFG.mask_num,), dtype=np.float64)
    gt_sum_org = np.zeros((CFG.mask_num,), dtype=np.float64)

    # binned dice (size/dist): bins are fixed by metrics.py defaults
    size_inter = np.zeros((3,), dtype=np.float64)
    size_pred = np.zeros((3,), dtype=np.float64)
    size_gt = np.zeros((3,), dtype=np.float64)
    dist_inter = np.zeros((3,), dtype=np.float64)
    dist_pred = np.zeros((3,), dtype=np.float64)
    dist_gt = np.zeros((3,), dtype=np.float64)

    size_inter_org = np.zeros((3,), dtype=np.float64)
    size_pred_org = np.zeros((3,), dtype=np.float64)
    size_gt_org = np.zeros((3,), dtype=np.float64)
    dist_inter_org = np.zeros((3,), dtype=np.float64)
    dist_pred_org = np.zeros((3,), dtype=np.float64)
    dist_gt_org = np.zeros((3,), dtype=np.float64)

    # temporal consistency (TC): store mean IoU per evaluated pair
    tc_mious: List[float] = []
    raft: Optional[RAFTFlowEstimator] = None
    prev_tc_frame: Optional[torch.Tensor] = None
    prev_tc_pred_idx: Optional[torch.Tensor] = None
    prev_tc_ok = False
    tc_every = max(1, int(tc_every or 1))
    if compute_tc:
        raft = RAFTFlowEstimator(device=device, variant=raft_variant)

    ellipse_applied = False
    use_gt_ellipse = inferencer_cfg.apply_ellipse_mask and ellipse_mask_np is not None
    if use_gt_ellipse:
        inferencer.apply_ellipse_mask = False
        inferencer.ellipse_mask = None

    # curr_frame_idx is kept in sync with the in-video frame number (consistent with cap seek)
    curr_frame_idx = start_f
    while True:
        if stop_frame_exclusive is not None and curr_frame_idx >= int(stop_frame_exclusive):
            break
        ok, frame_bgr = cap.read()
        if not ok:
            break
        #
        #inferencer.ellipse_mask = None
        frame_u = inferencer.preprocess_frame(frame_bgr)  # (1,3,H,W) RGB

        if use_gt_ellipse and not ellipse_applied:
            mask_np = ellipse_mask_np
            if getattr(inferencer, "use_roi", False) and getattr(inferencer, "roi", None) is not None:
                x, y, w, h = inferencer.roi
                mask_np = mask_np[y : y + h, x : x + w]
            ellipse_t = torch.from_numpy(mask_np).to(device=device, dtype=torch.long).unsqueeze(0).unsqueeze(0)
            inferencer.ellipse_mask = ellipse_t
            inferencer.apply_ellipse_mask = True
            ellipse_applied = True

        inferencer.step_canvas(frame_u)

        # --- TC: stitched prediction every frame (optional, expensive) ---
        if compute_tc and raft is not None and ((curr_frame_idx - start_f) % tc_every == 0):
            logits_tc = inferencer.model_inference(tuple(frame_u.shape[-2:]))  # (C,H,W)
            if logits_tc is not None:
                pred_tc = logits_to_pred_idx(logits_tc)  # (H,W) (ROI size if ROI is enabled)
                frame_tc = frame_u
                if getattr(inferencer, "use_roi", False) and getattr(inferencer, "roi", None) is not None:
                    x, y, w, h = inferencer.roi
                    frame_tc = frame_u[:, :, y : y + h, x : x + w]
                if prev_tc_frame is not None and prev_tc_pred_idx is not None and prev_tc_ok:
                    try:
                        # backward flow: t -> t-1
                        flow = raft.flow_t_to_tm1(frame_tc, prev_tc_frame)
                        warped_prev = warp_label_with_backward_flow(
                            prev_tc_pred_idx.to(device=device),
                            flow.to(device=device),
                            num_classes=CFG.mask_num,
                        )
                        _, miou = iou_per_class_numpy(
                            warped_prev.detach().to("cpu").numpy(),
                            pred_tc.detach().to("cpu").numpy(),
                            num_classes=CFG.mask_num,
                            exclude_classes=(0,) + tuple(DEFAULT_IGNORE_GT_LABELS),
                        )
                        if not np.isnan(miou):
                            tc_mious.append(float(miou))
                    except Exception:
                        # TC is best-effort; failures do not interrupt the main evaluation
                        pass
                prev_tc_frame = frame_tc.detach()
                prev_tc_pred_idx = pred_tc.detach()
                prev_tc_ok = True
            else:
                prev_tc_ok = False

        if curr_frame_idx in target_set:
            logits = inferencer.model_inference(tuple(frame_u.shape[-2:]))  # (C,H,W)
            if logits is None:
                curr_frame_idx += 1
                if pbar is not None:
                    pbar.update(1)
                continue

            pred_idx = logits_to_pred_idx(logits)  # (H,W)
            debug_gt_to_save: Optional[torch.Tensor] = None
            direct_pred_idx: Optional[torch.Tensor] = None

            # Run direct inference on the current frame (seg - original): computed regardless of debug for Dice comparison
            with torch.no_grad():
                direct_logits = seg_model(frame_u)  # (1,C,H,W)
            direct_pred_idx = logits_to_pred_idx(direct_logits)  # (H,W)

            for j, mask_path in enumerate(frame_to_masks.get(curr_frame_idx, [])):
                gt_raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if gt_raw is None:
                    raise FileNotFoundError(f"Mask not found: {mask_path}")
                gt_raw = crop_gt_with_inferencer_roi(gt_raw, inferencer)
                gt_idx_np = mask_raw_to_index(gt_raw)  # (H,W) uint8
                gt_idx = torch.from_numpy(gt_idx_np).to(device=device, dtype=torch.long)

                if gt_idx.shape != pred_idx.shape:
                    raise ValueError(
                        f"GT/pred shape mismatch after ROI crop: gt={tuple(gt_idx.shape)} pred={tuple(pred_idx.shape)} "
                        f"mask={mask_path}"
                    )

                # ---- Metrics (GT tool regions excluded) ----
                pred_np = pred_idx.detach().to("cpu").numpy()
                direct_np = direct_pred_idx.detach().to("cpu").numpy()

                inter, ps, gs = dice_stats_per_class_numpy(
                    pred_np, gt_idx_np, num_classes=CFG.mask_num, ignore_gt_labels=DEFAULT_IGNORE_GT_LABELS
                )
                inter_sum += inter
                pred_sum += ps
                gt_sum += gs

                inter_o, ps_o, gs_o = dice_stats_per_class_numpy(
                    direct_np, gt_idx_np, num_classes=CFG.mask_num, ignore_gt_labels=DEFAULT_IGNORE_GT_LABELS
                )
                inter_sum_org += inter_o
                pred_sum_org += ps_o
                gt_sum_org += gs_o

                # size/dist bins (instance-based)
                s_stat, d_stat = binned_dice_by_size_and_distance(
                    pred_np,
                    gt_idx_np,
                    num_classes=CFG.mask_num,
                    ignore_gt_labels=DEFAULT_IGNORE_GT_LABELS,
                    exclude_classes=(0,) + tuple(DEFAULT_IGNORE_GT_LABELS),
                )
                size_inter += s_stat.inter
                size_pred += s_stat.pred_sum
                size_gt += s_stat.gt_sum
                dist_inter += d_stat.inter
                dist_pred += d_stat.pred_sum
                dist_gt += d_stat.gt_sum

                s_stat_o, d_stat_o = binned_dice_by_size_and_distance(
                    direct_np,
                    gt_idx_np,
                    num_classes=CFG.mask_num,
                    ignore_gt_labels=DEFAULT_IGNORE_GT_LABELS,
                    exclude_classes=(0,) + tuple(DEFAULT_IGNORE_GT_LABELS),
                )
                size_inter_org += s_stat_o.inter
                size_pred_org += s_stat_o.pred_sum
                size_gt_org += s_stat_o.gt_sum
                dist_inter_org += d_stat_o.inter
                dist_pred_org += d_stat_o.pred_sum
                dist_gt_org += d_stat_o.gt_sum

                if debug and j == 0:
                    debug_gt_to_save = gt_idx

            if out_dir_video is not None:
                save_pred_and_debug(
                    out_dir_video=out_dir_video,
                    out_dir_original_video=out_dir_orig_video,
                    frame_idx=curr_frame_idx,
                    pred_idx=pred_idx,
                    frame_u=frame_u,
                    gt_idx=debug_gt_to_save,
                    debug=debug,
                    inferencer=inferencer,
                    direct_pred_idx=direct_pred_idx,
                )

        curr_frame_idx += 1
        if pbar is not None:
            pbar.update(1)

    cap.release()
    return (
        inter_sum,
        pred_sum,
        gt_sum,
        inter_sum_org,
        pred_sum_org,
        gt_sum_org,
        size_inter,
        size_pred,
        size_gt,
        dist_inter,
        dist_pred,
        dist_gt,
        size_inter_org,
        size_pred_org,
        size_gt_org,
        dist_inter_org,
        dist_pred_org,
        dist_gt_org,
        tc_mious,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument(
        "--csv_path",
        type=str,
        default=None,
        help="validation split csv. default: exp/02_CholecSeg8k/splits/cholecseg8k_test_{fold}.csv",
    )
    parser.add_argument(
        "--weights_path",
        type=str,
        default=None,
        help="model weights. default: exp/02_CholecSeg8k/models/upernet/fold{fold}.pth",
    )
    parser.add_argument(
        "--videos_dir",
        type=str,
        default="../../data/Cholec80/videos",
        help="directory containing videoXX.mp4",
    )
    parser.add_argument("--apply_ellipse_mask", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--use_static_roi", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--equalize_hist_rgb", type=str, default="false", choices=["true", "false"])
    parser.add_argument("--limit", type=int, default=0, help="0 evaluates all samples. >0 evaluates only the first N samples.")
    parser.add_argument(
        "--trust_frame_id",
        type=str,
        default="false",
        choices=["true", "false"],
        help="true: use frame_id from the CSV as-is / false: prefer actual_frame_id if present",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join("infer_outputs", f"stitch_upernet_fold"),
        help="Output directory for inference results (infer_outputs/stitch_upernet_{bbox_mode}_{'dyn' if dynamic_shape else 'fixed'}_{fold}/)",
    )
    parser.add_argument(
        "--out_dir_original",
        type=str,
        default=os.path.join("infer_outputs", "upernet_{fold}"),
        help="Output directory for predictions without stitching (original model). Empty string disables saving. upernet_{fold}/",
    )
    parser.add_argument("--debug", action="store_true", help="Also save input images, GT, and predicted masks")
    parser.add_argument(
        "--compute_tc",
        type=str,
        default="false",
        choices=["true", "false"],
        help="Compute Temporal Consistency (RAFT); set to true only when needed as it is expensive",
    )
    parser.add_argument(
        "--raft_variant",
        type=str,
        default="small",
        choices=["small", "large"],
        help="torchvision RAFT model variant",
    )
    parser.add_argument(
        "--tc_every",
        type=int,
        default=1,
        help="Interval in frames for TC computation (1=every frame, 2=every 2 frames, ...)",
    )
    args = parser.parse_args()

    args.out_dir = args.out_dir.replace("fold"  , str(args.fold))
    args.out_dir_original = args.out_dir_original.replace("fold", str(args.fold))
    device = CFG.device
    fold = int(args.fold)
    
    csv_path = args.csv_path or os.path.join(_THIS_DIR, "splits", f"cholecseg8k_test_{fold}.csv")
    weights_path = args.weights_path or os.path.join(_THIS_DIR, "models", "upernet", f"fold{fold}.pth")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    # Build cfg for StitchInferencer (missing keys will be filled with defaults by stitch_seg)
    infer_cfg = CFG()
    infer_cfg.apply_ellipse_mask = args.apply_ellipse_mask == "true"
    infer_cfg.use_static_roi = args.use_static_roi == "true"
    infer_cfg.equalize_hist_rgb = args.equalize_hist_rgb == "true"
    infer_cfg.device = device
    infer_cfg.out_dir_original = args.out_dir_original if args.out_dir_original else None
    # Convert weight paths referenced by stitch_seg to absolute paths (to avoid cwd dependency)
    infer_cfg.aliked_weights = _resolve_path_maybe_relative_to_this_dir(str(infer_cfg.aliked_weights))
    infer_cfg.lightglue_weights = _resolve_path_maybe_relative_to_this_dir(str(infer_cfg.lightglue_weights))
    infer_cfg.tool_detector_weights = _resolve_path_maybe_relative_to_this_dir(str(infer_cfg.tool_detector_weights))
    if hasattr(infer_cfg, "depth_anything_v2_model"):
        infer_cfg.depth_anything_v2_model = _resolve_path_maybe_relative_to_this_dir(
            str(getattr(infer_cfg, "depth_anything_v2_model"))
        )

    print(f"[INFO] device={device}")
    print(f"[INFO] csv_path={csv_path}")
    print(f"[INFO] weights_path={weights_path}")
    print(f"[INFO] videos_dir={args.videos_dir}")
    print(f"[INFO] out_dir={args.out_dir} out_dir_original={infer_cfg.out_dir_original} debug={args.debug}")
    print(
        f"[INFO] stitch cfg: use_static_roi={infer_cfg.use_static_roi} "
        f"apply_ellipse_mask={infer_cfg.apply_ellipse_mask} equalize_hist_rgb={infer_cfg.equalize_hist_rgb}"
    )
    print(f"[INFO] trust_frame_id={args.trust_frame_id}")

    df = pd.read_csv(csv_path)
    ####
    if args.limit and int(args.limit) > 0:
        df = df.head(int(args.limit)).copy()

    use_offset = args.trust_frame_id == "false"
    if use_offset and "actual_frame_id" in df.columns:
        df["frame_id"] = df["actual_frame_id"].fillna(df["frame_id"]).astype(int)

    # assign group key
    vids: List[int] = []
    starts: List[int] = []
    for p in df["file"].astype(str).tolist():
        vid, st = parse_video_id_and_start(p)
        vids.append(vid)
        starts.append(st)
    df["video_id"] = vids
    df["start_frame"] = starts
    df["frame_id"] = df["frame_id"].astype(int)

    # model load
    model = load_model(weights_path, device=device)
    seg_model = CanvasSegModel(model, device=device, input_size=CFG.image_size, dynamic_shape=CFG.dynamic_shape).to(device)
    seg_model.eval()

    # group by video_id (to run step_canvas over the entire video)
    groups = list(df.groupby(["video_id"], sort=True))
    print(f"[INFO] num_samples={len(df)} num_videos={len(groups)}")

    ensure_dir(args.out_dir)
    if infer_cfg.out_dir_original is not None:
        ensure_dir(str(infer_cfg.out_dir_original))

    # A progress bar over the number of processed frames is hard to estimate, so it is shown separately by sample count
    inter_all = np.zeros((CFG.mask_num,), dtype=np.float64)
    pred_all = np.zeros((CFG.mask_num,), dtype=np.float64)
    gt_all = np.zeros((CFG.mask_num,), dtype=np.float64)
    inter_all_org = np.zeros((CFG.mask_num,), dtype=np.float64)
    pred_all_org = np.zeros((CFG.mask_num,), dtype=np.float64)
    gt_all_org = np.zeros((CFG.mask_num,), dtype=np.float64)

    size_inter_all = np.zeros((3,), dtype=np.float64)
    size_pred_all = np.zeros((3,), dtype=np.float64)
    size_gt_all = np.zeros((3,), dtype=np.float64)
    dist_inter_all = np.zeros((3,), dtype=np.float64)
    dist_pred_all = np.zeros((3,), dtype=np.float64)
    dist_gt_all = np.zeros((3,), dtype=np.float64)

    size_inter_all_org = np.zeros((3,), dtype=np.float64)
    size_pred_all_org = np.zeros((3,), dtype=np.float64)
    size_gt_all_org = np.zeros((3,), dtype=np.float64)
    dist_inter_all_org = np.zeros((3,), dtype=np.float64)
    dist_pred_all_org = np.zeros((3,), dtype=np.float64)
    dist_gt_all_org = np.zeros((3,), dtype=np.float64)

    tc_all: List[float] = []

    for (video_id,), gdf in groups:
        gdf = gdf.sort_values("frame_id").reset_index(drop=True)
        key = SampleKey(video_id=int(video_id))
        video_path = os.path.join(args.videos_dir, f"video{key.video_id:02d}.mp4")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open: {video_path}")
        # Cut off at gdf['frame_id'].max()+1 for stop_frame_exclusive
        # (pbar is aligned with the number of frames processed from start_frame_exclusive)
        stop_frame_exclusive = int(gdf["frame_id"].max()) + 1
        if infer_cfg.startmid:
            start_frame_exclusive = int(gdf["frame_id"].min())
        else:
            start_frame_exclusive = 0
        total_frames = max(0, stop_frame_exclusive - start_frame_exclusive)
        cap.release()
        with tqdm(total=total_frames, desc=f"video{key.video_id:02d}", leave=False) as pbar:
            (
                inter,
                ps,
                gs,
                inter_o,
                ps_o,
                gs_o,
                s_i,
                s_p,
                s_g,
                d_i,
                d_p,
                d_g,
                s_io,
                s_po,
                s_go,
                d_io,
                d_po,
                d_go,
                tc_mious,
            ) = evaluate_group(
                key,
                gdf,
                videos_dir=args.videos_dir,
                inferencer_cfg=infer_cfg,
                seg_model=seg_model,
                device=device,
                pbar=pbar,
                out_dir=args.out_dir,
                debug=bool(args.debug),
                start_frame_exclusive=start_frame_exclusive,
                stop_frame_exclusive=stop_frame_exclusive,
                compute_tc=(args.compute_tc == "true"),
                raft_variant=str(args.raft_variant),
                tc_every=int(args.tc_every),
            )
        inter_all += inter
        pred_all += ps
        gt_all += gs
        inter_all_org += inter_o
        pred_all_org += ps_o
        gt_all_org += gs_o

        size_inter_all += s_i
        size_pred_all += s_p
        size_gt_all += s_g
        dist_inter_all += d_i
        dist_pred_all += d_p
        dist_gt_all += d_g

        size_inter_all_org += s_io
        size_pred_all_org += s_po
        size_gt_all_org += s_go
        dist_inter_all_org += d_io
        dist_pred_all_org += d_po
        dist_gt_all_org += d_go

        if tc_mious:
            tc_all.extend(tc_mious)

    dice_st_cpu = dice_from_stats_np(inter_all, pred_all, gt_all)  # (C,)
    dice_org_cpu = dice_from_stats_np(inter_all_org, pred_all_org, gt_all_org)  # (C,)

    # mean dice (ignoring nan)
    # mean dice: tools (10,11) are always excluded. fg also excludes background (0).
    mean_all_st = float(np.nanmean(dice_st_cpu[[c for c in range(CFG.mask_num) if c not in DEFAULT_IGNORE_GT_LABELS]]))
    mean_fg_st = mean_dice_np(dice_st_cpu, exclude_classes=(0,) + tuple(DEFAULT_IGNORE_GT_LABELS))
    mean_all_org = float(np.nanmean(dice_org_cpu[[c for c in range(CFG.mask_num) if c not in DEFAULT_IGNORE_GT_LABELS]]))
    mean_fg_org = mean_dice_np(dice_org_cpu, exclude_classes=(0,) + tuple(DEFAULT_IGNORE_GT_LABELS))

    print("\n=== Dice (per class) [stitched vs original] ===")
    for c in range(CFG.mask_num):
        vs = dice_st_cpu[c]
        vo = dice_org_cpu[c]
        ss = "nan" if np.isnan(vs) else f"{vs:.4f}"
        so = "nan" if np.isnan(vo) else f"{vo:.4f}"
        print(f"class {c:02d}: stitched={ss} | original={so}")
    print("========================")
    print(f"mean_dice_all (incl bg): stitched={mean_all_st:.4f} | original={mean_all_org:.4f}")
    print(f"mean_dice_fg  (excl bg): stitched={mean_fg_st:.4f} | original={mean_fg_org:.4f}")

    # --- size / distance bins ---
    size_d = dice_from_stats_np(size_inter_all, size_pred_all, size_gt_all)
    dist_d = dice_from_stats_np(dist_inter_all, dist_pred_all, dist_gt_all)
    size_d_org = dice_from_stats_np(size_inter_all_org, size_pred_all_org, size_gt_all_org)
    dist_d_org = dice_from_stats_np(dist_inter_all_org, dist_pred_all_org, dist_gt_all_org)

    print("\n=== Dice by target size (instance-bbox weighted) [stitched vs original] ===")
    for name, vs, vo in zip(["small(<32^2)", "medium(32^2-96^2)", "large(>=96^2)"], size_d, size_d_org):
        ss = "nan" if np.isnan(vs) else f"{vs:.4f}"
        so = "nan" if np.isnan(vo) else f"{vo:.4f}"
        print(f"{name}: stitched={ss} | original={so}")
    print("\n=== Dice by distance from center (instance-bbox weighted) [stitched vs original] ===")
    for name, vs, vo in zip(["center", "mid", "periphery"], dist_d, dist_d_org):
        ss = "nan" if np.isnan(vs) else f"{vs:.4f}"
        so = "nan" if np.isnan(vo) else f"{vo:.4f}"
        print(f"{name}: stitched={ss} | original={so}")

    # --- Temporal Consistency (TC) ---
    if args.compute_tc == "true":
        tc_mean = float(np.mean(tc_all)) if tc_all else float("nan")
        print("\n=== Temporal Consistency (TC, mean IoU of warped prev vs current, stitched) ===")
        print(f"TC mean IoU: {tc_mean:.4f}  (num_pairs={len(tc_all)})")

    # Optional: comparison for a specific class (if CFG.tool_class_ch is defined)
    if hasattr(CFG, "tool_class_ch"):
        tc = int(getattr(CFG, "tool_class_ch"))
        if 0 <= tc < CFG.mask_num:
            vs = dice_st_cpu[tc]
            vo = dice_org_cpu[tc]
            ss = "nan" if np.isnan(vs) else f"{vs:.4f}"
            so = "nan" if np.isnan(vo) else f"{vo:.4f}"
            print(f"tool_class_ch={tc}: stitched={ss} | original={so}")


if __name__ == "__main__":
    main()
