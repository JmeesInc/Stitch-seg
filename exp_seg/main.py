"""Run segmentation on a stitched canvas using StitchInferencer."""
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
from typing import Optional

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from tqdm import tqdm
import matplotlib.pyplot as plt

from stitch_seg import StitchInferencerDev as StitchInferencer
import time
import cProfile
import pstats
from .exp.model import UnetPlusPlus


class CFG:
    #video_path = "/mnt/devices/dl2/ex-data-2/data11/share/TLH/standardized_videos/001510725.mp4"
    video_path = "video01.mp4"
    start_frame = 29000
    end_frame = 30000
    enable_depth_mask = False
    output_dir = "1217_test"
    method = "pyramid"
    bbox_mode = "internal"
    apply_ellipse_mask = True
    output_mode = "logits"
    laplacian_var_min = 60
    alpha_overlap = 0.7
    segmentation_weights = "models/unetpp/fold0.pth"
    num_classes = 13
    backbone = "tu-convnext_base"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Set this only if a dedicated tool class channel exists in the segmentation model.
    tool_class_ch = None

    debug = True
    debug_dir = "0114_check"
    debug_video_filename = "unetpp_stitch.mp4"

class CanvasSegModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        # モデル初期化
        self.model = UnetPlusPlus(
            encoder_name=getattr(cfg, "backbone", "tu-convnext_base"),
            encoder_weights=None,
            in_channels=3,
            classes=int(getattr(cfg, "num_classes", 1)),
            activation=None,
        ).to(cfg.device)

        # 重みのロード
        state = torch.load(cfg.segmentation_weights, map_location=cfg.device)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        
        self.device = cfg.device
        #self.input_size = getattr(cfg, "segm_input_size", 512)
        self.last_model_input_image = None
        self.last_model_output = None

    def _prepare_input(self, img_tensor: torch.Tensor):
        img_tensor = img_tensor.float() / 255.0
        # reshape H, W to nearest values divisible by 32
        b, c, h, w = img_tensor.shape
        new_h = int(np.ceil(h / 32) * 32)
        new_w = int(np.ceil(w / 32) * 32)
        if new_h != h or new_w != w:
            img_tensor = F.interpolate(img_tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)
        return img_tensor

    def _predict_to_shape(self, tensor: torch.Tensor, target_hw) -> torch.Tensor:
        pred = self.model(tensor)
        pred_resized = F.interpolate(
            pred, 
            size=target_hw, 
            mode='bilinear', 
            align_corners=False
        )
        return pred_resized

    def forward(self, canvas_bgr: torch.Tensor) -> torch.Tensor:
        input_tensor = self._prepare_input(canvas_bgr)
        self.last_model_input_image = input_tensor
        target_hw = canvas_bgr.shape[-2:]
        output = self._predict_to_shape(input_tensor, target_hw)
        self.last_model_output = output
            
        return output

    def predict_on_frame(self, frame_bgr: torch.Tensor):
        """
        Helper for single frame prediction.
        Similar to forward but strictly keeps batch dim logic for clarity.
        """
        output = self.forward(frame_bgr)
        return output, self.last_model_input_image


def _resolve_output_mode(output_mode: Optional[str] = None) -> str:
    mode = str(output_mode or getattr(CFG, "output_mode", "logits")).lower()
    if mode not in {"logits", "sigmoid", "softmax"}:
        raise ValueError(f"Unsupported output_mode={mode!r}")
    return mode


def prediction_to_prob(mask, output_mode: Optional[str] = None):
    if mask is None:
        return None
    mode = _resolve_output_mode(output_mode)
    if isinstance(mask, torch.Tensor):
        pred = mask.detach().float()
        if mode == "logits":
            if pred.dim() >= 3 and pred.shape[-3] > 1:
                return pred.softmax(dim=-3)
            return pred.sigmoid()
        return pred
    arr = np.asarray(mask)
    if mode != "logits":
        return arr
    if arr.ndim >= 3 and arr.shape[-1] > 1:
        shifted = arr - arr.max(axis=-1, keepdims=True)
        exp = np.exp(shifted)
        denom = np.clip(exp.sum(axis=-1, keepdims=True), a_min=1e-8, a_max=None)
        return exp / denom
    return 1.0 / (1.0 + np.exp(-arr))


def ensure_dir(path: str):
    if not path:
        return
    os.makedirs(path, exist_ok=True)


def to_numpy_image(img):
    if img is None:
        return None
    if isinstance(img, torch.Tensor):
        t = img.detach().cpu()
        if t.dim() == 4:
            t = t[0]
        if t.dim() == 3 and t.shape[0] in (1, 3):
            t = t.permute(1, 2, 0)
        t = t.float()
        if t.max() <= 1.0:
            t = t * 255.0
        t = t.clamp(0, 255).byte().numpy()
        return t
    if isinstance(img, np.ndarray) and img.dtype != np.uint8:
        arr = img.astype(np.float32)
        if arr.max() <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr
    return img


def to_numpy_mask(mask):
    if mask is None:
        return None
    if isinstance(mask, torch.Tensor):
        m = mask.detach().cpu()
        if m.dim() == 4 and m.shape[0] == 1:
            m = m.squeeze(0)
        if m.dim() == 3:
            if m.shape[0] == 1:
                m = m.squeeze(0)
            elif m.shape[0] > 1:
                m = m.permute(1, 2, 0)
        return m.numpy()
    return mask


def prediction_to_label(mask, output_mode: Optional[str] = None):
    prob = prediction_to_prob(mask, output_mode=output_mode)
    if prob is None:
        return None
    prob_np = to_numpy_mask(prob)
    if prob_np is None:
        return None
    prob_np = np.asarray(prob_np)
    if prob_np.ndim == 3 and prob_np.shape[2] > 1:
        return np.argmax(prob_np, axis=2).astype(np.uint8)
    prob_single = np.squeeze(prob_np)
    return (prob_single >= 0.5).astype(np.uint8)


def save_segmentation(mask: np.ndarray, frame_idx: int, output_mode: Optional[str] = None):
    if mask is None:
        return
    ensure_dir(CFG.output_dir)
    label_map = prediction_to_label(mask, output_mode=output_mode)
    if label_map is None:
        return
    mask_vis = label_map.astype(np.uint8)
    out_path = os.path.join(CFG.output_dir, f"frame_{frame_idx:06d}.png")
    cv2.imwrite(out_path, mask_vis)


def build_palette(num_classes: int) -> np.ndarray:
    np.random.seed(42)
    palette = np.random.randint(0, 255, size=(max(num_classes, 2), 3), dtype=np.uint8)
    palette[0] = np.array([0, 0, 0], dtype=np.uint8)
    return palette


def render_segmentation(mask: np.ndarray, output_mode: Optional[str] = None):
    if mask is None:
        return None
    labels = prediction_to_label(mask, output_mode=output_mode)
    if labels is None:
        return None
    if labels.ndim == 2 and CFG.num_classes > 1:
        palette = build_palette(CFG.num_classes)
        return palette[labels]
    mask_bin = labels.astype(np.uint8) * 255
    return cv2.applyColorMap(mask_bin, cv2.COLORMAP_TURBO)


def overlay_mask(frame: np.ndarray, mask: np.ndarray, alpha: float = 0.5, output_mode: Optional[str] = None) -> np.ndarray:
    frame = to_numpy_image(frame)
    if frame is None:
        return None
    if mask is None:
        return frame
    label_map = prediction_to_label(mask, output_mode=output_mode)
    if label_map is None:
        return frame
    label_map = cv2.resize(label_map.astype(np.uint8), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    if CFG.num_classes > 1:
        color = build_palette(CFG.num_classes)[label_map]
    else:
        green = np.zeros_like(frame)
        green[..., 1] = 255
        color = frame * (1 - label_map[..., None]) + green * label_map[..., None]
    blended = cv2.addWeighted(color.astype(np.uint8), alpha, frame, 1.0 - alpha, 0)
    return blended


def build_debug_panel(labeled_images, frame_shape):
    h, w = frame_shape
    panels = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 3  # Larger font size
    thickness_bg = 10  # Thicker background
    thickness_fg = 4  # Thicker foreground
    y_offset = 42     # Move label down a bit for bigger font
    for label, img in labeled_images:
        img_np = to_numpy_image(img)
        if img_np is None:
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
        else:
            canvas = img_np.copy()
            if canvas.ndim == 2:
                canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
            if canvas.shape[:2] != (h, w):
                canvas = cv2.resize(canvas, (w, h), interpolation=cv2.INTER_LINEAR)
        if label:
            cv2.putText(canvas, label, (16, y_offset), font, font_scale, (0, 0, 0), thickness_bg, cv2.LINE_AA)
            cv2.putText(canvas, label, (16, y_offset), font, font_scale, (255, 255, 255), thickness_fg, cv2.LINE_AA)
        panels.append(canvas)
    while len(panels) < 6:
        panels.append(np.zeros((h, w, 3), dtype=np.uint8))
    row1 = np.hstack(panels[:3])
    row2 = np.hstack(panels[3:6])
    stacked = np.vstack([row1, row2])
    new_h = int(stacked.shape[0] * 0.33)
    new_w = int(stacked.shape[1] * 0.33)
    resized = cv2.resize(stacked, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return resized


# --- Debug video writer (for saving panels as a video) ---
DEBUG_VIDEO_WRITER = None
DEBUG_VIDEO_PATH = None

def _init_debug_video_writer(frame_size_wh, fps: float):
    ensure_dir(CFG.debug_dir)
    global DEBUG_VIDEO_WRITER, DEBUG_VIDEO_PATH
    # Build output path (e.g., debug_044300_045300.mp4)
    end_name = getattr(CFG, "end_frame", None)
    end_name = f"{end_name:06d}" if isinstance(end_name, int) else "end"
    filename = CFG.debug_video_filename
    path = os.path.join(CFG.debug_dir, filename)
    # Choose codec
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    safe_fps = float(fps if fps and fps > 0 else 30.0)
    DEBUG_VIDEO_WRITER = cv2.VideoWriter(path, fourcc, safe_fps, frame_size_wh, True)
    DEBUG_VIDEO_PATH = path

def close_debug_video():
    global DEBUG_VIDEO_WRITER
    if DEBUG_VIDEO_WRITER is not None:
        DEBUG_VIDEO_WRITER.release()
        DEBUG_VIDEO_WRITER = None

def save_debug_panel(panel: np.ndarray, frame_idx: int):
    # Save debug panel into a video instead of per-frame images
    if panel is None:
        return
    global DEBUG_VIDEO_WRITER
    if DEBUG_VIDEO_WRITER is None:
        # Initialize writer on first use with current panel size and video FPS
        fps = float(getattr(CFG, "debug_video_fps", 30.0))
        h, w = panel.shape[:2]
        _init_debug_video_writer((w, h), fps)
    # Ensure panel is 3-channel uint8
    #panel = cv2.cvtColor(panel, cv2.COLOR_BGR2RGB)
    DEBUG_VIDEO_WRITER.write(panel)


def main():
    ensure_dir(CFG.output_dir)
    if CFG.debug:
        ensure_dir(CFG.debug_dir)
    seg_model = CanvasSegModel(CFG)
    inferencer = StitchInferencer(seg_model, cfg=CFG, start_frame=CFG.start_frame)

    cap = cv2.VideoCapture(CFG.video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {CFG.video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, CFG.start_frame)

    # Store FPS for debug video writer
    setattr(CFG, "debug_video_fps", cap.get(cv2.CAP_PROP_FPS))

    frame_idx = CFG.start_frame
    stride = max(1, int(getattr(CFG, "segm_stride", 1)))
    end_frame = getattr(CFG, "end_frame", None)

    # Prepare progress bar over the portion of the video being stitched.
    total_frames = None
    if end_frame is not None:
        total_frames = end_frame - CFG.start_frame + 1
    else:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_frames = max(0, total - CFG.start_frame)
    pbar = tqdm(total=total_frames, desc="Stitch+Seg")
    profiler = cProfile.Profile()

    while True:
        if end_frame is not None and frame_idx > end_frame:
            break
        t0 = time.time()
        ok, frame = cap.read()
        if not ok:
            break
        frame_u = inferencer.preprocess_frame(frame)
        profiler.enable()
        inferencer.step_canvas(frame_u) # 4-6s
        if (frame_idx - CFG.start_frame) % stride == 0:
            seg_map = inferencer.model_inference() # 0.02s
            profiler.disable()
            if seg_map is not None:
                if CFG.debug:
                    direct_seg, direct_input = seg_model.predict_on_frame(frame_u)
                    frame_vis = to_numpy_image(frame_u)
                    overlay_stitched = overlay_mask(frame_vis, seg_map, output_mode=CFG.output_mode)
                    overlay_direct = overlay_mask(frame_vis, direct_seg, output_mode="logits")
                    canvas_vis = inferencer.canvas
                    canvas_vis = to_numpy_image(canvas_vis)
                    if canvas_vis is not None and frame_vis is not None and canvas_vis.shape[:2] != frame_vis.shape[:2]:
                        canvas_vis = cv2.resize(canvas_vis, (frame_vis.shape[1], frame_vis.shape[0]), interpolation=cv2.INTER_LINEAR)
                    model_input = inferencer.last_model_input if inferencer.last_model_input is not None else seg_model.last_model_input_image
                    model_output = inferencer.last_model_output if inferencer.last_model_output is not None else seg_model.last_model_output
                    model_input = to_numpy_image(model_input)
                    model_input = overlay_mask(model_input, model_output, alpha=0.2, output_mode="logits")
                    # Build a 3x2 grid for visual sanity checks.
                    panel = build_debug_panel(
                        [
                            ("current frame", frame_vis),
                            ("canvas", canvas_vis),
                            ("model_input", model_input),
                            ("seg - stitched", overlay_stitched),
                            ("seg - original", overlay_direct),
                            ('merge_mask',  inferencer.combined_mask_prev_raw.squeeze(0).squeeze(0).cpu().detach().numpy()),
                        ],
                        frame_vis.shape[:2] if frame_vis is not None else frame.shape[:2],
                    )
                    panel = cv2.cvtColor(panel, cv2.COLOR_BGR2RGB)
                    save_debug_panel(panel, frame_idx)
                else:
                    #save_segmentation(seg_map, frame_idx)
                    pass
        frame_idx += 1
        pbar.update(1)

    pbar.close()
    
    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats('cumtime')
    stats.print_stats(20)

    cap.release()
    # Close debug video if used
    close_debug_video()


if __name__ == "__main__":
    main()
