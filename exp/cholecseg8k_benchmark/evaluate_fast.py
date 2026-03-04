#!/usr/bin/env python3
"""
Optimized Evaluation Script

Features:
1. Class-wise Dice (Micro-average aggregated over all images)
2. Size-wise Dice (Small/Medium/Large)
3. Distance-wise Dice (Center/Mid/Periphery, 3-bin & 12-bin split)
   For each distance zone, pixel statistics are aggregated across all images,
   class-wise Dice is computed, then macro-averaged.
4. Temporal Consistency (RAFT Optical Flow, GPU-based)

Usage:
  python evaluate_fast.py --root /path/to/pred_dir --compute-tc
"""

import argparse
import os
import re
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Sequence

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from torchvision.models.optical_flow import (
    raft_large, raft_small, Raft_Large_Weights, Raft_Small_Weights
)

# --- Constants ---

# Mask value to class ID mapping
LABEL2CH = {
    0: 0, 50: 0, 255: 0,  # Background / Unlabeled
    5: 1,                 # Liver ligament
    11: 2,                # Abdominal wall
    12: 3,                # Fat
    13: 4,                # Gastrointestinal Tract
    21: 5,                # Liver
    22: 6,                # Gallbladder
    23: 7,                # Connective tissue
    24: 8,                # Blood
    25: 9,                # Cystic Duct
    31: 10,               # Grasper (Tool)
    32: 11,               # L-hook Electrocautery
    33: 12,               # Hepatic Veins
}

NUM_CLASSES = 13
DEFAULT_IGNORE_GT_LABELS = (10, 11)  # GT label values to exclude from evaluation (e.g. Grasper, L-hook)


# --- Helper Functions ---

def build_label_lut(label2ch: Dict[int, int], unknown: int = 0) -> np.ndarray:
    lut = np.full(256, unknown, dtype=np.uint8)
    for k, v in label2ch.items():
        if 0 <= int(k) <= 255:
            lut[int(k)] = np.uint8(v)
    return lut

_LABEL_LUT = build_label_lut(LABEL2CH)

def load_idx_image(path: str) -> np.ndarray:
    """Load a grayscale index image."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Failed to load image: {path}")
    return img.astype(np.int64)

def extract_frame_id_from_filename(filename: str) -> Optional[int]:
    """Extract frame ID from filename."""
    match = re.search(r"frame_(\d+)_", filename)
    if match:
        return int(match.group(1))
    return None

def dice_from_stats(inter: np.ndarray, pred_sum: np.ndarray, gt_sum: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Compute Dice from (inter, pred, gt) statistics."""
    den = pred_sum + gt_sum
    with np.errstate(divide='ignore', invalid='ignore'):
        dice_val = (2.0 * inter) / (den + eps)
        dice_val[den == 0] = np.nan
    return dice_val

# --- Core Logic (Vectorized Bincount) ---

def fast_binned_stats(
    pred: np.ndarray,
    gt: np.ndarray,
    bin_map: np.ndarray,
    num_classes: int,
    num_bins: int,
    valid_mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fast function that computes per-(class, bin) statistics in batch using np.bincount.
    Return shape: (num_classes, num_bins) for inter, pred_sum, gt_sum.
    """
    # 1. Extract valid pixels only (flatten)
    p_flat = pred[valid_mask]
    g_flat = gt[valid_mask]
    b_flat = bin_map[valid_mask]

    # Clamp class IDs to valid range
    p_flat = np.clip(p_flat, 0, num_classes - 1)
    g_flat = np.clip(g_flat, 0, num_classes - 1)

    # 2. Create unique IDs: id = class * num_bins + bin
    # This represents each (class, bin) combination as a single integer.

    # GT sum
    gt_ids = g_flat * num_bins + b_flat
    gt_sum_flat = np.bincount(gt_ids, minlength=num_classes * num_bins)

    # Pred sum
    pred_ids = p_flat * num_bins + b_flat
    pred_sum_flat = np.bincount(pred_ids, minlength=num_classes * num_bins)

    # Intersection (only where pred == gt)
    match_mask = (p_flat == g_flat)
    inter_ids = gt_ids[match_mask]
    inter_flat = np.bincount(inter_ids, minlength=num_classes * num_bins)

    # 3. Reshape and return (class, bin)
    return (
        inter_flat.reshape(num_classes, num_bins),
        pred_sum_flat.reshape(num_classes, num_bins),
        gt_sum_flat.reshape(num_classes, num_bins)
    )

# --- Worker Function (for parallel processing) ---

def process_single_image(args_tuple: Tuple) -> Dict:
    """
    Worker that processes a single image pair and returns statistics.
    """
    (pred_path, gt_path, num_classes, ignore_labels, exclude_classes,
     dist_edges_12, ellipse_edges_12, gt_is_raw) = args_tuple

    # Result container
    # Distance/Ellipse stats are stored in (Class, Bin) shape
    result = {
        "status": "ok",
        "inter": np.zeros(num_classes),
        "pred_sum": np.zeros(num_classes),
        "gt_sum": np.zeros(num_classes),
        # Size stats (3 bins: S/M/L) - aggregated across all classes per instance
        "size_inter": np.zeros(3), "size_pred": np.zeros(3), "size_gt": np.zeros(3),
        # Ellipse stats (num_classes, 12 bins)
        "ell_inter": np.zeros((num_classes, 12)),
        "ell_pred": np.zeros((num_classes, 12)),
        "ell_gt": np.zeros((num_classes, 12)),
    }

    try:
        if not os.path.exists(gt_path):
            return {"status": "missing_gt"}

        pred = load_idx_image(pred_path)
        gt = load_idx_image(gt_path)
        # If GT is the original watershed mask (0-255), convert to class IDs (0-12)
        if gt_is_raw:
            gt = _LABEL_LUT[gt.astype(np.uint8)].astype(np.int64)

        # Resize if shapes don't match (nearest neighbor)
        if pred.shape != gt.shape:
            h, w = pred.shape[:2]
            gt = cv2.resize(gt.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(np.int64)

        h, w = gt.shape

        # Valid mask (exclude pixels with ignore labels)
        valid = np.ones_like(gt, dtype=bool)
        for lab in ignore_labels:
            valid &= (gt != int(lab))

        # --- 1. Basic Class-wise Stats ---
        p_flat = pred[valid]
        g_flat = gt[valid]
        # Fast aggregation via bincount
        result["pred_sum"] = np.bincount(p_flat, minlength=num_classes)
        result["gt_sum"] = np.bincount(g_flat, minlength=num_classes)
        match = (p_flat == g_flat)
        # Intersection aggregated by GT class
        result["inter"] = np.bincount(g_flat[match], minlength=num_classes)

        # Classes excluding the exclude list
        target_classes = [c for c in range(num_classes) if c not in exclude_classes]

        # --- 2. Size-wise Stats (Target Area Based) ---
        small_thr, medium_thr = 32*32, 96*96

        for c in target_classes:
            gt_mask_c = (gt == c) & valid
            if not gt_mask_c.any(): continue

            area = gt_mask_c.sum()
            bin_idx = 0 if area < small_thr else (1 if area < medium_thr else 2)

            pred_mask_c = (pred == c) & valid
            inter_val = (pred_mask_c & gt_mask_c).sum()
            pred_val = pred_mask_c.sum()

            # Accumulate size-wise stats across all classes
            result["size_inter"][bin_idx] += inter_val
            result["size_pred"][bin_idx] += pred_val
            result["size_gt"][bin_idx] += area

        # --- 3. Distance-wise Stats (12 bins) ---
        # Compute normalized distance map from center
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        max_r = np.hypot(cx, cy) + 1e-12
        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        dist_map = np.hypot(xx - cx, yy - cy) / max_r

        # Build 12-bin map
        dist_bin_map = np.full((h, w), -1, dtype=np.int32)
        for i in range(len(dist_edges_12) - 1):
            mask_bin = (dist_edges_12[i] <= dist_map) & (dist_map < dist_edges_12[i+1])
            dist_bin_map[mask_bin] = i

        # --- 4. Ellipse Distance-wise Stats (12 bins) ---
        ellipse_mask = (gt != 0)  # Treat non-background pixels as the ellipse region
        if ellipse_mask.any():
            # Distance transform
            ellipse_uint8 = ellipse_mask.astype(np.uint8)
            ell_dist_map = cv2.distanceTransform(ellipse_uint8, cv2.DIST_L2, 5).astype(np.float64)
            max_edist = ell_dist_map.max() + 1e-12
            ell_dist_norm = ell_dist_map / max_edist

            # Bin map (inside ellipse only)
            ell_bin_map = np.full((h, w), -1, dtype=np.int32)
            for i in range(len(ellipse_edges_12) - 1):
                mask_bin = (ellipse_edges_12[i] <= ell_dist_norm) & \
                           (ell_dist_norm < ellipse_edges_12[i+1]) & ellipse_mask
                ell_bin_map[mask_bin] = i

            valid_ellipse = valid & ellipse_mask

            e_inter, e_pred, e_gt = fast_binned_stats(
                pred, gt, ell_bin_map, num_classes, 12, valid_ellipse
            )
            result["ell_inter"] = e_inter
            result["ell_pred"] = e_pred
            result["ell_gt"] = e_gt

    except Exception as e:
        # print(f"Error processing {pred_path}: {e}")  # Enable for debugging
        return {"status": "error", "msg": str(e)}

    return result


# --- RAFT / Temporal Consistency ---

class RAFTFlowEstimator:
    def __init__(self, device: torch.device, variant: str = "small"):
        self.device = device
        self.model, self.transforms = self._build_model(variant)
        self.model.to(self.device).eval()

    def _build_model(self, variant: str):
        if variant == "large":
            weights = Raft_Large_Weights.DEFAULT
            model = raft_large(weights=weights, progress=False)
        else:
            weights = Raft_Small_Weights.DEFAULT
            model = raft_small(weights=weights, progress=False)
        transforms = weights.transforms()
        return model, transforms

    @torch.no_grad()
    def flow_t_to_tm1(self, frame_t: torch.Tensor, frame_tm1: torch.Tensor) -> torch.Tensor:
        # frame: (1, 3, H, W) float [0, 1]
        im1, im2 = frame_t.to(self.device), frame_tm1.to(self.device)
        if self.transforms:
            im1, im2 = self.transforms(im1, im2)

        # Pad to multiple of 8
        _, _, h, w = im1.shape
        ph = (8 - (h % 8)) % 8
        pw = (8 - (w % 8)) % 8
        if ph or pw:
            im1 = F.pad(im1, (0, pw, 0, ph), mode='replicate')
            im2 = F.pad(im2, (0, pw, 0, ph), mode='replicate')

        flows = self.model(im1, im2)
        flow = flows[-1]  # List[flow] -> last flow

        if ph or pw:
            flow = flow[..., :h, :w]
        return flow

def warp_label_with_backward_flow(prev_idx: torch.Tensor, flow: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Warp t-1 labels to frame t using backward optical flow."""
    h, w = prev_idx.shape
    # One-hot
    oh = F.one_hot(prev_idx.clamp(0, num_classes-1), num_classes).permute(2, 0, 1).unsqueeze(0).float()
    oh = oh.to(flow.device)

    # Grid generation
    yy, xx = torch.meshgrid(torch.arange(h, device=flow.device), torch.arange(w, device=flow.device), indexing="ij")
    grid = torch.stack([xx, yy], dim=0).unsqueeze(0).float()  # (1, 2, H, W)
    vgrid = grid + flow  # Apply flow

    # Normalize to [-1, 1]
    vgrid[:, 0] = 2.0 * vgrid[:, 0] / max(w - 1, 1) - 1.0
    vgrid[:, 1] = 2.0 * vgrid[:, 1] / max(h - 1, 1) - 1.0
    vgrid = vgrid.permute(0, 2, 3, 1)  # (1, H, W, 2)

    warped_oh = F.grid_sample(oh, vgrid, mode='bilinear', padding_mode='zeros', align_corners=True)
    return warped_oh.argmax(dim=1).squeeze(0)  # (H, W)


def compute_temporal_consistency(pred_files: List[Path], num_classes: int, device_str: str = "cuda") -> float:
    """
    Compute Temporal Consistency (IoU between warped t-1 and t predictions).
    Runs sequentially in the main process as it uses GPU memory.
    """
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"Initializing RAFT on {device}...")
    try:
        raft = RAFTFlowEstimator(device, variant="small")
    except Exception as e:
        print(f"Failed to init RAFT: {e}")
        return np.nan

    # Sort by video and frame order
    # Filename convention: .../video18/frame_000123_pred_idx.png
    def parse_key(p: Path):
        # Extract video name (e.g. video01, video12) and distinguish from other dirs
        vid = p.parent.name if re.match(r"^video\d+$", p.parent.name) else "unknown"
        fid = extract_frame_id_from_filename(p.name) or -1
        return vid, fid

    sorted_files = sorted(pred_files, key=parse_key)

    ious = []
    prev_vid = None
    prev_fid = -1
    prev_img_t = None
    prev_pred_t = None

    # Resolve corresponding RGB image paths from prediction file directory
    # Adjust path resolution logic to match actual dataset structure if needed

    print("Computing Temporal Consistency...")
    for pred_path in tqdm(sorted_files):
        vid, fid = parse_key(pred_path)

        # Check frame continuity with previous frame
        is_continuous = (vid == prev_vid) and (fid == prev_fid + 1 if prev_fid != -1 else False)

        # Resolve RGB image path from prediction filename
        # Assumes: pred_idx.png -> frame_xxx.png (RGB)
        img_name = pred_path.name.replace("_pred_idx.png", ".png").replace("_pred_idx", "")
        img_path = pred_path.parent / img_name

        if not img_path.exists():
            # Fallback pattern: frame_xxxxxx.png
            img_path = pred_path.parent / f"frame_{fid:06d}.png"

        if not img_path.exists():
            # Skip and reset if image not found
            prev_vid, prev_fid = vid, fid
            prev_img_t, prev_pred_t = None, None
            continue

        try:
            curr_bgr = cv2.imread(str(img_path))
            if curr_bgr is None: continue
            curr_rgb = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2RGB)
            curr_pred = load_idx_image(str(pred_path))
        except:
            continue

        h, w = curr_pred.shape
        curr_img_t = torch.from_numpy(curr_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0  # (1,3,H,W)
        curr_pred_t = torch.from_numpy(curr_pred).long().to(device)

        if is_continuous and prev_img_t is not None and prev_pred_t is not None:
            # Resize if needed
            if prev_img_t.shape[-2:] != (h, w):
                prev_img_resized = F.interpolate(prev_img_t, size=(h, w), mode='bilinear')
                prev_pred_resized = F.interpolate(prev_pred_t.float().unsqueeze(0).unsqueeze(0), size=(h, w), mode='nearest').long().squeeze()
            else:
                prev_img_resized = prev_img_t
                prev_pred_resized = prev_pred_t

            # Compute flow from t to t-1
            flow = raft.flow_t_to_tm1(curr_img_t, prev_img_resized)

            # Warp t-1 labels to t
            warped_prev = warp_label_with_backward_flow(prev_pred_resized, flow, num_classes)

            # Compute IoU (exclude background 0 and ignore labels)
            wp = warped_prev.cpu().numpy()
            cp = curr_pred

            c_ious = []
            for c in range(1, num_classes):  # Exclude background (0)
                if c in DEFAULT_IGNORE_GT_LABELS: continue
                m1 = (wp == c)
                m2 = (cp == c)
                i = (m1 & m2).sum()
                u = (m1 | m2).sum()
                if u > 0:
                    c_ious.append(i / u)

            if c_ious:
                ious.append(np.mean(c_ious))

        # Update state
        prev_vid, prev_fid = vid, fid
        prev_img_t = curr_img_t
        prev_pred_t = curr_pred_t

    return np.mean(ious) if ious else np.nan


# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="Medical Segmentation Evaluation Script (Optimized)")
    parser.add_argument("--root", type=str, required=True, help="Root directory containing *_pred_idx.png")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES)
    parser.add_argument("--exclude-classes", type=int, nargs="+", default=[0, 10, 11], help="Classes to exclude from Mean Dice")
    parser.add_argument("--compute-tc", action="store_true", help="Compute Temporal Consistency (Slow, uses GPU)")
    parser.add_argument("--gt-dir", type=str, default=None, help="GT dir name under infer_outputs (e.g. ground_truth_0 or GT_orig_0). Default: ground_truth_{fold}")
    parser.add_argument("--gt-format", choices=["idx", "raw"], default="idx",
                        help="idx: GT is class index 0-12. raw: GT is watershed mask 0-255 (use for GT_orig_*)")
    args = parser.parse_args()

    root_path = Path(args.root)
    if not root_path.exists():
        print(f"Error: Directory {root_path} not found.")
        sys.exit(1)

    # 1. Scan prediction files
    print("Scanning files...")
    pred_files = sorted(list(root_path.glob("**/frame_*_pred_idx.png")))
    if not pred_files:
        print("No prediction files found.")
        sys.exit(0)

    # Infer fold number from directory name suffix
    fold = 0
    match = re.search(r"(\d+)$", str(root_path))
    if match:
        fold = int(match.group(1))

    # GT directory base path (sibling of root_path's parent, e.g. ground_truth_0)
    gt_dir_name = args.gt_dir if args.gt_dir is not None else f"ground_truth_{fold}"
    # When root_path is infer_outputs/xxx_0, parent is infer_outputs -> ground_truth_0 is a sibling
    gt_base_dir = root_path.parent / gt_dir_name

    # Build task list
    tasks = []
    dist_edges_12 = np.linspace(0.0, 1.01, 13)  # 12 bins

    print(f"Matching GT for {len(pred_files)} predictions...")
    for pred_path in pred_files:
        # Resolve GT path
        frame_id = extract_frame_id_from_filename(pred_path.name)
        # Video ID: parent directory of frame (e.g. video01, video12)
        video = pred_path.parent.name if re.match(r"^video\d+$", pred_path.parent.name) else None

        gt_path_str = None
        # Priority 1: Global GT dir
        if video and frame_id is not None:
            cand = gt_base_dir / video / f"frame_{frame_id:06d}_gt_idx.png"
            if cand.exists():
                gt_path_str = str(cand)

        # Priority 2: Local dir (same directory as prediction)
        if gt_path_str is None:
            cand = pred_path.parent / pred_path.name.replace("_pred_idx.png", "_gt_idx.png")
            if cand.exists():
                gt_path_str = str(cand)

        if gt_path_str:
            gt_is_raw = args.gt_format == "raw"
            tasks.append((
                str(pred_path),
                gt_path_str,
                args.num_classes,
                DEFAULT_IGNORE_GT_LABELS,
                tuple(args.exclude_classes),
                dist_edges_12,
                dist_edges_12,
                gt_is_raw,
            ))

    if not tasks:
        print("Error: No matching GT files found.")
        sys.exit(1)

    print(f"Starting evaluation on {len(tasks)} pairs using {cpu_count()} cores...")

    # --- Initialize aggregation variables ---
    # Class-wise: (num_classes,)
    total_inter = np.zeros(args.num_classes)
    total_pred = np.zeros(args.num_classes)
    total_gt = np.zeros(args.num_classes)

    # Size-wise: (3 bins,) - S, M, L
    size_inter = np.zeros(3)
    size_pred = np.zeros(3)
    size_gt = np.zeros(3)

    # Ellipse-wise: (num_classes, 12 bins)
    ell_inter = np.zeros((args.num_classes, 12))
    ell_pred = np.zeros((args.num_classes, 12))
    ell_gt = np.zeros((args.num_classes, 12))

    # --- Parallel execution ---
    with Pool(processes=cpu_count()) as pool:
        for res in tqdm(pool.imap_unordered(process_single_image, tasks, chunksize=16), total=len(tasks)):
            if res["status"] != "ok":
                continue

            # Accumulate statistics
            total_inter += res["inter"]
            total_pred += res["pred_sum"]
            total_gt += res["gt_sum"]

            size_inter += res["size_inter"]
            size_pred += res["size_pred"]
            size_gt += res["size_gt"]

            ell_inter += res["ell_inter"]
            ell_pred += res["ell_pred"]
            ell_gt += res["ell_gt"]

    # --- Compute and display results ---
    print("\n" + "="*80)
    print(f"Evaluation Results (Micro-average over images)")
    print("="*80)

    valid_classes = [c for c in range(args.num_classes) if c not in args.exclude_classes]

    # 1. Class-wise Dice & Mean Dice
    dice_c = dice_from_stats(total_inter, total_pred, total_gt)
    mean_dice = np.nanmean(dice_c[valid_classes])

    print(f"\n[1] Class-wise Dice:")
    for c in range(args.num_classes):
        if c not in args.exclude_classes:
            print(f"  Class {c:2d}: {dice_c[c]:.4f}")
    print(f"  --> Mean Dice: {mean_dice:.4f}")

    # 2. Size-wise Dice
    dice_size = dice_from_stats(size_inter, size_pred, size_gt)
    print(f"\n[2] Size-wise Dice (aggregated across target classes):")
    for i, name in enumerate(["Small", "Medium", "Large"]):
        print(f"  {name:8s}: {dice_size[i]:.4f}")

    # 3. Distance-wise Dice (Special Logic: Mean of Class-wise Dices)
    def calculate_and_print_dist_stats(inter_cb, pred_cb, gt_cb, title):
        # inter_cb shape: (num_classes, 12)
        den = pred_cb + gt_cb
        with np.errstate(divide='ignore', invalid='ignore'):
            dice_cb = (2.0 * inter_cb) / den
            dice_cb[den == 0] = np.nan

        # Exclude specified classes
        dice_valid = dice_cb[valid_classes, :]  # (num_valid, 12)

        # (A) 12-bin result (average over classes)
        mean_dice_12 = np.nanmean(dice_valid, axis=0)

        # (B) 3-bin result (Center/Mid/Periphery)
        # Aggregate to 3 bins per class, compute Dice, then average across classes
        # Center: 0-3, Mid: 4-7, Periphery: 8-11
        dice_3bins_per_class = []
        for start, end in [(0, 4), (4, 8), (8, 12)]:
            i_sum = inter_cb[valid_classes, start:end].sum(axis=1)
            p_sum = pred_cb[valid_classes, start:end].sum(axis=1)
            g_sum = gt_cb[valid_classes, start:end].sum(axis=1)

            d = (2.0 * i_sum) / (p_sum + g_sum + 1e-6)
            d[(p_sum + g_sum) == 0] = np.nan
            dice_3bins_per_class.append(np.nanmean(d))

        print(f"\n{title} (Mean of Class-wise Dice):")
        print("  [3 Bins Summary]")
        for i, name in enumerate(["Center", "Mid", "Periphery"]):
            print(f"    {name:10s}: {dice_3bins_per_class[i]:.4f}")

        print("  [12 Bins Detail]")
        for i in range(12):
            print(f"    Bin {i:2d}: {mean_dice_12[i]:.4f}")

        return dice_3bins_per_class, mean_dice_12

    ellipse_3bins, ellipse_12bins = calculate_and_print_dist_stats(
        ell_inter, ell_pred, ell_gt, "[4] Ellipse Distance-wise Dice"
    )

    # 4. Temporal Consistency
    if args.compute_tc:
        tc_score = compute_temporal_consistency(pred_files, args.num_classes)
        print(f"\n[5] Temporal Consistency (mIoU): {tc_score:.4f}")

    # --- CSV Output (separate files) ---
    def build_distance_bin_names(num_bins: int) -> List[str]:
        names: List[str] = []
        for i in range(num_bins):
            if i == 0:
                names.append("center")
            elif i == 1:
                names.append("mid")
            else:
                names.append(f"ring{i + 1}")
        if num_bins == 3:
            names = ["center", "mid", "periphery"]
        return names

    # [1] Class-wise Dice: mean only
    class_dice_path = root_path / "class_dice.csv"
    pd.DataFrame([{"class": "mean", "dice": float(mean_dice)}]).to_csv(class_dice_path, index=False)
    print(f"\nSaved: {class_dice_path}")

    # [2] Size-wise Dice
    size_dice_path = root_path / "size_dice.csv"
    size_rows = []
    for i, name in enumerate(["small", "medium", "large"]):
        v = dice_size[i]
        if not np.isnan(v):
            size_rows.append({"size": name, "dice": float(v)})
    pd.DataFrame(size_rows).to_csv(size_dice_path, index=False)
    print(f"Saved: {size_dice_path}")

    # [4] Ellipse Distance-wise Dice
    # 3-bin summary
    ellipse_3bin_path = root_path / "ellipse_distance_dice_3bin.csv"
    ellipse_3bin_names = build_distance_bin_names(3)
    ellipse_3bin_rows = [
        {"distance": name, "dice": float(val)}
        for name, val in zip(ellipse_3bin_names, ellipse_3bins)
        if not np.isnan(val)
    ]
    pd.DataFrame(ellipse_3bin_rows).to_csv(ellipse_3bin_path, index=False)
    print(f"Saved: {ellipse_3bin_path}")

    # 12-bin detail
    ellipse_12bin_path = root_path / "ellipse_distance_dice_12bin.csv"
    ellipse_12bin_names = build_distance_bin_names(12)
    ellipse_12bin_rows = [
        {"distance": name, "dice": float(val)}
        for name, val in zip(ellipse_12bin_names, ellipse_12bins)
        if not np.isnan(val)
    ]
    pd.DataFrame(ellipse_12bin_rows).to_csv(ellipse_12bin_path, index=False)
    print(f"Saved: {ellipse_12bin_path}")

if __name__ == "__main__":
    main()
