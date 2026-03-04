"""Run segmentation on a stitched canvas using StitchInferencer."""
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
from contextlib import contextmanager
from functools import wraps
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

#from stitch_seg import StitchInferencerDev as StitchInferencer
from stitch_seg import Stitcher_ONNX as Stitcher
from stitch_seg import compute_static_roi


@contextmanager
def nvtx_range(name: str):
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()
    else:
        yield


def nvtx_annotate(name: str):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with nvtx_range(name):
                return func(*args, **kwargs)

        return wrapper

    return decorator

class CFG:
    #video_path = "/mnt/devices/dl2/ex-data-2/data11/share/TLH/standardized_videos/001510725.mp4"
    video_path = "1003_1127.mp4"
    start_frame = 45198
    end_frame = 46000
    method = "pyramid"
    bbox_mode = "internal"
    apply_ellipse_mask = True
    laplacian_var_min = 60
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    debug = True
    debug_dir = "0114_check"
    debug_video_filename = "eso_1003_1127.avi"


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
    filename = CFG.debug_video_filename
    path = os.path.join(CFG.debug_dir, filename)
    safe_fps = float(fps if fps and fps > 0 else 30.0)

    # Try codecs in order: XVID (.avi), mp4v (.mp4)
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".avi":
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
    else:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    DEBUG_VIDEO_WRITER = cv2.VideoWriter(path, fourcc, safe_fps, frame_size_wh, True)
    if not DEBUG_VIDEO_WRITER.isOpened():
        # Fallback to XVID + .avi
        path = os.path.splitext(path)[0] + ".avi"
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        DEBUG_VIDEO_WRITER = cv2.VideoWriter(path, fourcc, safe_fps, frame_size_wh, True)
        print(f"[warn] Fallback to XVID: {path}")
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
    ok = DEBUG_VIDEO_WRITER.write(panel)
    if ok is False:
        print(f"[warn] VideoWriter.write() returned False at frame {frame_idx}")


def main():
    inferencer = Stitcher(cfg=CFG, input_size=(1080, 1920))
    roi = None

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

    # Fixed panel size for debug video (set on first frame, kept constant)
    panel_size_wh = None

    while True:
        with nvtx_range(f"frame_{frame_idx}"):
            if end_frame is not None and frame_idx > end_frame:
                break
            ok, frame = cap.read()
            if not ok:
                break

            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
                result = preprocess_frame(frame, roi)
                if result is None:
                    frame_idx += 1
                    pbar.update(1)
                    continue
                frame_u, roi, ellipse_mask = result

                # Resize to match Stitcher's expected input_size
                exp_h, exp_w = 1080, 1920
                if frame_u.shape[-2] != exp_h or frame_u.shape[-1] != exp_w:
                    frame_u = F.interpolate(frame_u.float(), size=(exp_h, exp_w), mode='bilinear', align_corners=False).to(frame_u.dtype)
                    ellipse_mask = F.interpolate(ellipse_mask.float(), size=(exp_h, exp_w), mode='nearest').to(ellipse_mask.dtype)

                if frame_idx == CFG.start_frame:
                    canvas, canvas_mask, H_cum_curr, prev_keypoints, prev_descriptors = inferencer.first_frame(frame_u, ellipse_mask)

                if (frame_idx - CFG.start_frame) % stride == 0:
                    canvas, canvas_mask, H_cum_curr, prev_keypoints, prev_descriptors, needs_reset = inferencer(
                        frame_u,
                        ellipse_mask,
                        H_cum_curr,
                        prev_keypoints,
                        prev_descriptors,
                        canvas,
                        canvas_mask,
                    )
                    if needs_reset:
                        canvas, canvas_mask, H_cum_curr, prev_keypoints, prev_descriptors = inferencer.first_frame(frame_u, ellipse_mask)

                    if CFG.debug:
                        # --- Debug: raw tensor stats ---
                        frame_vis = to_numpy_image(frame_u)
                        canvas_vis = to_numpy_image(canvas.clone())

                        if frame_vis is not None:
                            # Save first frame as PNG for inspection
                            if frame_idx == CFG.start_frame:
                                ensure_dir(CFG.debug_dir)
                                # frame_vis is RGB from to_numpy_image -> save as BGR for cv2
                                cv2.imwrite(os.path.join(CFG.debug_dir, "dbg_frame_rgb.png"), cv2.cvtColor(frame_vis, cv2.COLOR_RGB2BGR))
                                if canvas_vis is not None:
                                    cv2.imwrite(os.path.join(CFG.debug_dir, "dbg_canvas_rgb.png"), cv2.cvtColor(canvas_vis, cv2.COLOR_RGB2BGR))

                            # Convert RGB -> BGR for VideoWriter
                            frame_bgr = cv2.cvtColor(frame_vis, cv2.COLOR_RGB2BGR)
                            canvas_bgr = cv2.cvtColor(canvas_vis, cv2.COLOR_RGB2BGR) if canvas_vis is not None else None

                            fh, fw = frame_bgr.shape[:2]
                            # Resize canvas to match frame height, keep aspect ratio
                            if canvas_bgr is not None:
                                ch, cw = canvas_bgr.shape[:2]
                                new_cw = max(1, int(cw * fh / ch))
                                canvas_bgr = cv2.resize(canvas_bgr, (new_cw, fh), interpolation=cv2.INTER_LINEAR)

                            # Side-by-side: frame | canvas
                            if canvas_bgr is not None:
                                combined = np.hstack([frame_bgr, canvas_bgr])
                            else:
                                combined = frame_bgr

                            # Determine fixed panel size on first debug frame
                            if panel_size_wh is None:
                                # Round down to multiple of 16 for codec compatibility
                                pw = (combined.shape[1] // 16) * 16
                                ph = (combined.shape[0] // 16) * 16
                                panel_size_wh = (pw, ph)

                            # Resize to fixed panel size and ensure contiguous uint8
                            panel = cv2.resize(combined, panel_size_wh, interpolation=cv2.INTER_LINEAR)
                            panel = np.ascontiguousarray(panel, dtype=np.uint8)
                            save_debug_panel(panel, frame_idx)

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    close_debug_video()

def preprocess_frame(frame: np.ndarray, roi: tuple=None) -> torch.Tensor:
    """
    frame: (H, W, 3) BGR uint8
    Return: (1, 3, H, W) RGB float (without scale/normalize)
    """
    if frame is None:
        return None

    # Convert BGR (cv2) -> RGB torch tensor on device
    frame_rgb_full = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    torch_frame_full = torch.from_numpy(frame_rgb_full).permute(2, 0, 1).unsqueeze(0).to(CFG.device)

    if roi is None:
        roi = compute_static_roi(torch_frame_full.float())
    x, y, w, h = roi

    # Crop both image and ellipse mask to the same ROI.
    # NOTE: Previously ellipse_mask was computed on the full frame and then resized to ROI size,
    # which makes the ellipse scale/position incorrect relative to the cropped frame.
    frame_roi = frame[y : y + h, x : x + w]

    frame_rgb = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2RGB)
    torch_frame = torch.from_numpy(frame_rgb).permute(2, 0, 1).unsqueeze(0).to(CFG.device)

    gray = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(gray)
    # 1. 二値化 (閾値は環境に合わせて調整。10-30あたりが一般的)
    _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    # 2. モルフォロジー演算（ノイズ除去と穴埋め）
    kernel = np.ones((5,5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    # 3. 輪郭抽出
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    max_contour = max(contours, key=cv2.contourArea)
    (x, y), radius = cv2.minEnclosingCircle(max_contour)
    radius -= 4
    mask = cv2.circle(mask, (int(x), int(y)), int(radius), 1, -1)
    mask = np.ones_like(mask) - mask
    ellipse_mask = torch.from_numpy(mask).to(CFG.device).unsqueeze(0).unsqueeze(0)

    return torch_frame, roi, ellipse_mask


if __name__ == "__main__":
    main()
