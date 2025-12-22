"""Run segmentation on a stitched canvas using StitchInferencer."""
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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
#from stitch_seg import StitchInferencer
import time
import cProfile
import pstats


class CFG:
    video_path = "/mnt/data/data4/shared/Cholecystostomy/Cholec80/videos/video09.mp4"
    start_frame = 0
    end_frame = 1000
    #segmentation_weights =  "checkpoint/best.pth"
    output_dir = "1217_test"
    method = "feature"
    apply_ellipse_mask = True
    laplacian_var_min = 60
    segmentation_weights = "/mnt/devices/dl1/in-data/data3/result/Hysterectomy/Ureter/v10.0/cv1/last.pth"
    num_classes = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tool_class_ch = 0

    debug = True
    debug_dir = "video"
    method = "dev"
    debug_video_filename = "video09.mp4"

class CanvasSegModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # 活性化関数の決定
        activation = "sigmoid" if cfg.num_classes == 1 else "softmax"
        
        # モデル初期化
        self.model = smp.FPN(
            encoder_name="efficientnet-b7",
            encoder_weights="imagenet",
            activation=activation,
            in_channels=3,
            classes=cfg.num_classes,
        ).to(cfg.device)

        # 重みのロード
        state = torch.load(cfg.segmentation_weights, map_location=cfg.device)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        
        self.device = cfg.device
        self.input_size = getattr(cfg, "segm_input_size", 512)
        self.last_model_input_image = None

    def _prepare_input(self, img_tensor: torch.Tensor):
        img_tensor = img_tensor.float() / 255.0
        return F.interpolate(img_tensor, size=(self.input_size, self.input_size), mode='bilinear', align_corners=False)

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
            
        return output

    def predict_on_frame(self, frame_bgr: torch.Tensor):
        """
        Helper for single frame prediction.
        Similar to forward but strictly keeps batch dim logic for clarity.
        """
        output = self.forward(frame_bgr)
        return output, self.last_model_input_image


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


def save_segmentation(mask: np.ndarray, frame_idx: int):
    if mask is None:
        return
    mask = to_numpy_mask(mask)
    ensure_dir(CFG.output_dir)
    mask_single = np.squeeze(mask)
    mask_bin = (mask_single >= 0.5).astype(np.uint8) * 100
    mask_vis = mask_bin.astype(np.uint8)
    out_path = os.path.join(CFG.output_dir, f"frame_{frame_idx:06d}.png")
    cv2.imwrite(out_path, mask_vis)


def build_palette(num_classes: int) -> np.ndarray:
    np.random.seed(42)
    palette = np.random.randint(0, 255, size=(max(num_classes, 2), 3), dtype=np.uint8)
    palette[0] = np.array([0, 0, 0], dtype=np.uint8)
    return palette


def render_segmentation(mask: np.ndarray):
    if mask is None:
        return None
    mask = to_numpy_mask(mask)
    if mask.ndim == 3 and mask.shape[2] > 1:
        labels = np.argmax(mask, axis=2).astype(np.uint8)
        palette = build_palette(CFG.num_classes)
        return palette[labels]
    mask_single = np.squeeze(mask)
    mask_bin = (mask_single >= 0.5).astype(np.uint8) * 255
    return cv2.applyColorMap(mask_bin, cv2.COLORMAP_TURBO)


def overlay_mask(frame: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    frame = to_numpy_image(frame)
    if frame is None:
        return None
    if mask is None:
        return frame
    mask = to_numpy_mask(mask)
    mask_bin = np.squeeze(mask)
    mask_bin = (mask_bin >= 0.5).astype(np.uint8)
    if mask_bin.ndim == 3 and mask_bin.shape[2] > 1:
        mask_bin = np.argmax(mask_bin, axis=2).astype(np.uint8)
    mask_bin = cv2.resize(mask_bin, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    green = np.zeros_like(frame)
    green[..., 1] = 255
    color = frame * (1 - mask_bin[..., None]) + green * mask_bin[..., None]
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
    inferencer = StitchInferencer(seg_model, start_frame=CFG.start_frame, cfg=CFG)

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
            seg_map = inferencer.model_inference(frame_u.shape[-2:]) # 0.02s
            profiler.disable()
            if seg_map is not None:
                if CFG.debug:
                    direct_seg, direct_input = seg_model.predict_on_frame(frame_u)
                    frame_vis = to_numpy_image(frame_u)
                    overlay_stitched = overlay_mask(frame_vis, seg_map)
                    overlay_direct = overlay_mask(frame_vis, direct_seg)
                    canvas_vis = inferencer.canvas
                    canvas_vis = to_numpy_image(canvas_vis)
                    if canvas_vis is not None and frame_vis is not None and canvas_vis.shape[:2] != frame_vis.shape[:2]:
                        canvas_vis = cv2.resize(canvas_vis, (frame_vis.shape[1], frame_vis.shape[0]), interpolation=cv2.INTER_LINEAR)
                    model_input = inferencer.last_model_input if inferencer.last_model_input is not None else seg_model.last_model_input_image
                    model_output = inferencer.last_model_output if inferencer.last_model_output is not None else seg_model.last_model_output
                    model_input = to_numpy_image(model_input)
                    model_output = to_numpy_image(model_output)
                    
                    model_input = overlay_mask(model_input, model_output)
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
