"""
Evaluation metrics for exp/cholecseg8k_benchmark.

Requirements:
- Regions in the ground truth with labels Grasper(10) or L-hook Electrocautery(11)
  are excluded from all subsequent metric evaluation (= pixels where GT is 10/11 are ignored).
- Metrics:
  1. Per-class Dice and mean Dice
  2. Dice by target size (small/medium/large)
  3. Dice by distance from image center
  4. Temporal consistency (TC): warp the t-1 prediction to t using RAFT optical flow,
     then compute IoU between the warped prediction and the prediction at t.

Notes:
- Size-based / distance-based Dice is aggregated at the "GT connected-component (instance)" level.
  Predictions and GT are compared within each instance's bounding box, and
  (inter, pred_sum, gt_sum) inside the bbox are accumulated
  (= yields a Dice weighted by instance area).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models.optical_flow import (  # noqa: WPS433
    raft_large,
    raft_small,
    Raft_Large_Weights,
    Raft_Small_Weights,
)

DEFAULT_IGNORE_GT_LABELS: Tuple[int, int] = (0, 10, 11)


def _as_numpy_int_hw(x: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().to("cpu").to(torch.int64).numpy()
    if not isinstance(x, np.ndarray):
        raise TypeError(f"expected np.ndarray or torch.Tensor, got {type(x)}")
    if x.dtype.kind not in ("i", "u"):
        x = x.astype(np.int64, copy=False)
    return x


def valid_mask_excluding_gt_labels(gt_idx_hw: np.ndarray, ignore_gt_labels: Sequence[int]) -> np.ndarray:
    """
    Returns: (H,W) bool. Sets pixels where GT is in ignore_gt_labels to False.
    """
    if not ignore_gt_labels:
        return np.ones_like(gt_idx_hw, dtype=bool)
    m = np.ones_like(gt_idx_hw, dtype=bool)
    for lab in ignore_gt_labels:
        m &= gt_idx_hw != int(lab)
    return m


def dice_from_stats(inter: np.ndarray, pred_sum: np.ndarray, gt_sum: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    den = pred_sum + gt_sum
    out = (2.0 * inter) / (den + eps)
    out = out.astype(np.float64, copy=False)
    out[den <= 0] = np.nan
    return out


def mean_dice(dice_c: np.ndarray, exclude_classes: Sequence[int] = (0, 10, 11)) -> float:
    if dice_c.size == 0:
        return float("nan")
    mask = np.ones_like(dice_c, dtype=bool)
    for c in exclude_classes:
        if 0 <= int(c) < dice_c.size:
            mask[int(c)] = False
    vals = dice_c[mask]
    return float(np.nanmean(vals)) if np.any(~np.isnan(vals)) else float("nan")


def dice_stats_per_class_numpy(
    pred_idx_hw: np.ndarray,
    gt_idx_hw: np.ndarray,
    num_classes: int,
    ignore_gt_labels: Sequence[int] = DEFAULT_IGNORE_GT_LABELS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (inter, pred_sum, gt_sum) per class as numpy arrays.
    Pixels where GT is in ignore_gt_labels are ignored.
    """
    pred = _as_numpy_int_hw(pred_idx_hw)
    gt = _as_numpy_int_hw(gt_idx_hw)
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: pred={pred.shape} gt={gt.shape}")
    valid = valid_mask_excluding_gt_labels(gt, ignore_gt_labels)
    pf = pred[valid].reshape(-1)
    gf = gt[valid].reshape(-1)
    pf = np.clip(pf, 0, num_classes - 1)
    gf = np.clip(gf, 0, num_classes - 1)
    pred_sum = np.bincount(pf, minlength=num_classes).astype(np.float64)
    gt_sum = np.bincount(gf, minlength=num_classes).astype(np.float64)
    eq = pf == gf
    inter = np.bincount(gf[eq], minlength=num_classes).astype(np.float64)
    return inter, pred_sum, gt_sum


@dataclass
class BinnedDiceStats:
    bin_names: List[str]
    inter: np.ndarray  # (B,)
    pred_sum: np.ndarray  # (B,)
    gt_sum: np.ndarray  # (B,)

    @classmethod
    def create(cls, bin_names: Sequence[str]) -> "BinnedDiceStats":
        b = int(len(bin_names))
        return cls(
            bin_names=list(bin_names),
            inter=np.zeros((b,), dtype=np.float64),
            pred_sum=np.zeros((b,), dtype=np.float64),
            gt_sum=np.zeros((b,), dtype=np.float64),
        )

    def add(self, bin_idx: int, inter: float, pred_sum: float, gt_sum: float) -> None:
        self.inter[bin_idx] += float(inter)
        self.pred_sum[bin_idx] += float(pred_sum)
        self.gt_sum[bin_idx] += float(gt_sum)

    def dice(self) -> Dict[str, float]:
        d = dice_from_stats(self.inter, self.pred_sum, self.gt_sum)
        return {name: float(d[i]) for i, name in enumerate(self.bin_names)}


@dataclass
class BinnedMicroDiceStats:
    bin_names: List[str]
    sum_dice: np.ndarray  # (B,)
    count: np.ndarray  # (B,)

    @classmethod
    def create(cls, bin_names: Sequence[str]) -> "BinnedMicroDiceStats":
        b = int(len(bin_names))
        return cls(
            bin_names=list(bin_names),
            sum_dice=np.zeros((b,), dtype=np.float64),
            count=np.zeros((b,), dtype=np.float64),
        )

    def add(self, bin_idx: int, dice_val: float) -> None:
        self.sum_dice[bin_idx] += float(dice_val)
        self.count[bin_idx] += 1.0

    def dice(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for i, name in enumerate(self.bin_names):
            if self.count[i] > 0:
                out[name] = float(self.sum_dice[i] / self.count[i])
            else:
                out[name] = float("nan")
        return out

def _size_bin_index(area_px: int, small_thr: int, medium_thr: int) -> int:
    if area_px < small_thr:
        return 0
    if area_px < medium_thr:
        return 1
    return 2


def _dist_bin_index(norm_dist: float, edges: Sequence[float]) -> int:
    # edges: [0, e1, e2, ..., 1.01]
    x = float(norm_dist)
    for i in range(len(edges) - 1):
        if edges[i] <= x < edges[i + 1]:
            return i
    return len(edges) - 2


def binned_dice_by_size_and_distance(
    pred_idx_hw: np.ndarray,
    gt_idx_hw: np.ndarray,
    num_classes: int,
    ignore_gt_labels: Sequence[int] = DEFAULT_IGNORE_GT_LABELS,
    exclude_classes: Sequence[int] = (0, 10, 11),
    # size thresholds in pixels (area):
    small_area_thr: int = 32 * 32,
    medium_area_thr: int = 96 * 96,
    # distance bins (normalized by max radius):
    dist_edges: Sequence[float] = (0.0, 0.33, 0.66, 1.01),
) -> Tuple[BinnedDiceStats, BinnedDiceStats]:
    """
    Per class:
    - Dice stats by size (small/medium/large), determined by the total area of each class
    - Dice stats by distance from center (center/mid/periphery), binned per pixel distance

    Returns:
    - size_stats: bins=["small","medium","large"]
    - dist_stats: bins=["center","mid","periphery"] (depends on the number of intervals in dist_edges)
    """
    pred = _as_numpy_int_hw(pred_idx_hw)
    gt = _as_numpy_int_hw(gt_idx_hw)
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: pred={pred.shape} gt={gt.shape}")

    h, w = gt.shape[:2]
    valid = valid_mask_excluding_gt_labels(gt, ignore_gt_labels)

    size_bins = ["small", "medium", "large"]
    dist_bins = []
    for i in range(len(dist_edges) - 1):
        if i == 0:
            dist_bins.append("center")
        elif i == 1:
            dist_bins.append("mid")
        else:
            dist_bins.append(f"ring{i+1}")
    if len(dist_bins) == 3:
        dist_bins = ["center", "mid", "periphery"]

    size_stats = BinnedDiceStats.create(size_bins)
    dist_stats = BinnedDiceStats.create(dist_bins)

    # image center / max distance
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    max_r = float(np.hypot(cx, cy) + 1e-12)

    # Pre-compute the distance mask (compute the distance bin for each pixel)
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    dist_map = np.hypot(xx - cx, yy - cy) / max_r  # (H, W) normalized distance
    dist_bin_map = np.zeros_like(dist_map, dtype=np.int32)
    for i in range(len(dist_edges) - 1):
        mask_bin = (dist_edges[i] <= dist_map) & (dist_map < dist_edges[i + 1])
        dist_bin_map[mask_bin] = i

    exclude = set(int(c) for c in exclude_classes)
    classes: Iterable[int] = (c for c in range(num_classes) if c not in exclude)

    for c in classes:
        # GT for the target class (excluding ignored regions)
        gt_mask_c = (gt == c) & valid
        if not gt_mask_c.any():
            continue

        # Compute the total area of the class
        class_area = int(gt_mask_c.sum())

        # Determine the size bin
        sb = _size_bin_index(class_area, small_area_thr, medium_area_thr)

        # Compute Dice stats for the entire class
        pred_mask_c = (pred == c) & valid
        inter_all = np.logical_and(pred_mask_c, gt_mask_c).sum()
        pred_sum_all = pred_mask_c.sum()
        gt_sum_all = gt_mask_c.sum()

        if pred_sum_all + gt_sum_all > 0:
            size_stats.add(sb, float(inter_all), float(pred_sum_all), float(gt_sum_all))

        # Compute Dice stats by distance (add each pixel to its distance bin)
        for db in range(len(dist_bins)):
            dist_mask = dist_bin_map == db
            # Consider only pixels within this distance bin
            pred_in_bin = pred_mask_c & dist_mask
            gt_in_bin = gt_mask_c & dist_mask
            inter_in_bin = np.logical_and(pred_in_bin, gt_in_bin).sum()
            pred_sum_in_bin = pred_in_bin.sum()
            gt_sum_in_bin = gt_in_bin.sum()

            if pred_sum_in_bin + gt_sum_in_bin > 0:
                dist_stats.add(db, float(inter_in_bin), float(pred_sum_in_bin), float(gt_sum_in_bin))

    return size_stats, dist_stats


def micro_dice_by_distance( # renamed: macro -> micro
    pred_idx_hw: np.ndarray,
    gt_idx_hw: np.ndarray,
    num_classes: int,
    ignore_gt_labels: Sequence[int] = DEFAULT_IGNORE_GT_LABELS,
    exclude_classes: Sequence[int] = (0, 10, 11),
    dist_edges: Sequence[float] = (0.0, 0.33, 0.66, 1.01),
) -> BinnedDiceStats: # return type changed: BinnedMacroDiceStats -> BinnedDiceStats
    """
    Accumulates (inter, pred_sum, gt_sum) per distance bin (for micro-averaging).
    """
    pred = _as_numpy_int_hw(pred_idx_hw)
    gt = _as_numpy_int_hw(gt_idx_hw)
    # ... (validation and preprocessing unchanged) ...
    h, w = gt.shape[:2]
    valid = valid_mask_excluding_gt_labels(gt, ignore_gt_labels)

    # Bin definition (unchanged)
    dist_bins = []
    for i in range(len(dist_edges) - 1):
        if i == 0:
            dist_bins.append("center")
        elif i == 1:
            dist_bins.append("mid")
        else:
            dist_bins.append(f"ring{i+1}")
    if len(dist_bins) == 3:
        dist_bins = ["center", "mid", "periphery"]

    # Change: use BinnedDiceStats
    dist_stats = BinnedDiceStats.create(dist_bins)

    # ... (distance map computation unchanged) ...
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    max_r = float(np.hypot(cx, cy) + 1e-12)
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    dist_map = np.hypot(xx - cx, yy - cy) / max_r
    dist_bin_map = np.zeros_like(dist_map, dtype=np.int32)
    for i in range(len(dist_edges) - 1):
        mask_bin = (dist_edges[i] <= dist_map) & (dist_map < dist_edges[i + 1])
        dist_bin_map[mask_bin] = i

    exclude = set(int(c) for c in exclude_classes)
    classes: Iterable[int] = (c for c in range(num_classes) if c not in exclude)

    for c in classes:
        gt_mask_c = (gt == c) & valid
        if not gt_mask_c.any():
            continue

        pred_mask_c = (pred == c) & valid

        for db in range(len(dist_bins)):
            dist_mask = dist_bin_map == db
            pred_in_bin = pred_mask_c & dist_mask
            gt_in_bin = gt_mask_c & dist_mask
            
            # Change: accumulate raw counts without computing Dice
            inter = np.logical_and(pred_in_bin, gt_in_bin).sum()
            ps = pred_in_bin.sum()
            gs = gt_in_bin.sum()

            if ps + gs > 0:
                dist_stats.add(db, float(inter), float(ps), float(gs))

    return dist_stats


def build_ellipse_mask_from_gt(gt_idx_hw: np.ndarray, bg_class: int = 0) -> np.ndarray:
    """
    Builds an ellipse mask from the GT. Excludes the background class (0) and outer border regions.
    Returns: (H, W) bool, True = inside the ellipse (region to be evaluated)
    """
    gt = _as_numpy_int_hw(gt_idx_hw)
    # Region where any class other than background exists = inside the ellipse
    ellipse_mask = gt != bg_class
    return ellipse_mask


def micro_dice_by_ellipse_distance( # renamed
    pred_idx_hw: np.ndarray,
    gt_idx_hw: np.ndarray,
    num_classes: int,
    ignore_gt_labels: Sequence[int] = DEFAULT_IGNORE_GT_LABELS,
    exclude_classes: Sequence[int] = (0, 10, 11),
    dist_edges: Sequence[float] = (0.0, 0.33, 0.66, 1.01),
) -> BinnedDiceStats: # type changed
    import cv2

    pred = _as_numpy_int_hw(pred_idx_hw)
    gt = _as_numpy_int_hw(gt_idx_hw)
    # ... (preprocessing unchanged) ...
    
    h, w = gt.shape[:2]
    valid = valid_mask_excluding_gt_labels(gt, ignore_gt_labels)
    ellipse_mask = build_ellipse_mask_from_gt(gt, bg_class=0)

    # Bin definition (unchanged)
    dist_bins = []
    for i in range(len(dist_edges) - 1):
        if i == 0: dist_bins.append("center")
        elif i == 1: dist_bins.append("mid")
        else: dist_bins.append(f"ring{i+1}")
    if len(dist_bins) == 3:
        dist_bins = ["center", "mid", "periphery"]

    if not ellipse_mask.any():
        return BinnedDiceStats.create(dist_bins) # Change: BinnedDiceStats

    # ... (distance map computation etc. unchanged) ...
    ellipse_uint8 = ellipse_mask.astype(np.uint8)
    dist_map = cv2.distanceTransform(ellipse_uint8, cv2.DIST_L2, 5).astype(np.float64)
    max_dist = float(dist_map[ellipse_mask].max()) + 1e-12
    dist_map_norm = dist_map / max_dist
    dist_map_norm[~ellipse_mask] = -1.0
    
    # Change: use BinnedDiceStats
    dist_stats = BinnedDiceStats.create(dist_bins)

    dist_bin_map = np.full((h, w), -1, dtype=np.int32)
    for i in range(len(dist_edges) - 1):
        mask_bin = (dist_edges[i] <= dist_map_norm) & (dist_map_norm < dist_edges[i + 1])
        dist_bin_map[mask_bin] = i

    eval_mask = ellipse_mask & valid
    exclude = set(int(c) for c in exclude_classes)
    classes: Iterable[int] = (c for c in range(num_classes) if c not in exclude)

    for c in classes:
        gt_mask_c = (gt == c) & eval_mask
        if not gt_mask_c.any():
            continue
        pred_mask_c = (pred == c) & eval_mask

        for db in range(len(dist_bins)):
            dist_mask = dist_bin_map == db
            pred_in_bin = pred_mask_c & dist_mask
            gt_in_bin = gt_mask_c & dist_mask
            
            # Change: accumulate raw counts without computing Dice
            inter = np.logical_and(pred_in_bin, gt_in_bin).sum()
            ps = pred_in_bin.sum()
            gs = gt_in_bin.sum()

            if ps + gs > 0:
                dist_stats.add(db, float(inter), float(ps), float(gs))

    return dist_stats


class RAFTFlowEstimator:
    """
    Optical flow estimator using torchvision RAFT (intended for computing backward flow t -> t-1).
    """

    def __init__(self, device: torch.device, variant: str = "small"):
        self.device = device
        self.variant = str(variant)
        self.model, self.transforms = self._build_model_and_transforms(self.variant)
        self.model.to(self.device).eval()

    @staticmethod
    def _build_model_and_transforms(variant: str):
        """
        Build the torchvision RAFT model and transforms.
        variant:
          - "small": raft_small
          - "large": raft_large
        """
        v = str(variant).lower()
        if v == "large":
            weights = Raft_Large_Weights.DEFAULT
            model = raft_large(weights=weights, progress=False)
        else:
            weights = Raft_Small_Weights.DEFAULT
            model = raft_small(weights=weights, progress=False)

        # Use the transforms bundled with the torchvision weights if available (otherwise None)
        transforms = getattr(weights, "transforms", None)
        transforms = transforms() if callable(transforms) else None
        return model, transforms

    @torch.no_grad()
    def flow_t_to_tm1(self, frame_t: torch.Tensor, frame_tm1: torch.Tensor) -> torch.Tensor:
        """
        frame_*: (1,3,H,W) RGB, uint8 or float in [0,1] or [0,255]
        Returns: flow (1,2,H,W) in pixels, t -> t-1
        """
        if frame_t.shape != frame_tm1.shape:
            raise ValueError(f"frame shape mismatch: t={tuple(frame_t.shape)} tm1={tuple(frame_tm1.shape)}")

        def to_float01(x: torch.Tensor) -> torch.Tensor:
            y = x
            if y.dtype != torch.float32 and y.dtype != torch.float16:
                y = y.to(torch.float32)
            if y.max() > 1.0:
                y = y / 255.0
            return y

        im1 = to_float01(frame_t).to(self.device)
        im2 = to_float01(frame_tm1).to(self.device)

        # Apply weights.transforms if it supports torch tensors
        if self.transforms is not None:
            try:
                im1, im2 = self.transforms(im1, im2)
            except Exception:
                # Ignore and continue if transforms assumes PIL input or similar
                pass

        # RAFT recommends dimensions be multiples of 8: pad accordingly
        _, _, h, w = im1.shape
        pad_h = (8 - (h % 8)) % 8
        pad_w = (8 - (w % 8)) % 8
        if pad_h or pad_w:
            im1p = F.pad(im1, (0, pad_w, 0, pad_h), mode="replicate")
            im2p = F.pad(im2, (0, pad_w, 0, pad_h), mode="replicate")
        else:
            im1p, im2p = im1, im2

        flows = self.model(im1p, im2p)
        # torchvision RAFT is expected to return list[flow]
        flow = flows[-1] if isinstance(flows, (list, tuple)) else flows
        if pad_h or pad_w:
            flow = flow[..., :h, :w]
        return flow


def warp_label_with_backward_flow(
    prev_idx: torch.Tensor,
    flow_t_to_tm1: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """
    Warp prev_idx (t-1) to t coordinates using flow (t->t-1) (backward sampling).
    prev_idx: (H,W) long on device
    flow_t_to_tm1: (1,2,H,W) float on device, pixel displacement
    Returns: warped_prev_idx (H,W) long
    """
    if prev_idx.dim() != 2:
        raise ValueError(f"prev_idx must be (H,W), got {tuple(prev_idx.shape)}")
    if flow_t_to_tm1.dim() != 4 or flow_t_to_tm1.shape[0] != 1 or flow_t_to_tm1.shape[1] != 2:
        raise ValueError(f"flow must be (1,2,H,W), got {tuple(flow_t_to_tm1.shape)}")
    h, w = prev_idx.shape
    if tuple(flow_t_to_tm1.shape[-2:]) != (h, w):
        raise ValueError(f"shape mismatch: idx={(h,w)} flow={tuple(flow_t_to_tm1.shape[-2:])}")

    # one-hot: (1,C,H,W)
    oh = F.one_hot(prev_idx.clamp(0, num_classes - 1), num_classes=num_classes).permute(2, 0, 1).unsqueeze(0)
    oh = oh.to(flow_t_to_tm1.device, dtype=torch.float32)

    # grid in pixel coords
    yy, xx = torch.meshgrid(
        torch.arange(h, device=flow_t_to_tm1.device),
        torch.arange(w, device=flow_t_to_tm1.device),
        indexing="ij",
    )
    base = torch.stack([xx, yy], dim=0).unsqueeze(0).to(flow_t_to_tm1.dtype)  # (1,2,H,W)
    src = base + flow_t_to_tm1  # (1,2,H,W): coordinates in t-1

    # normalize to [-1,1] for grid_sample (x, y)
    x = src[:, 0]  # (1,H,W)
    y = src[:, 1]
    x = (2.0 * x / max(w - 1, 1)) - 1.0
    y = (2.0 * y / max(h - 1, 1)) - 1.0
    grid = torch.stack([x, y], dim=-1)  # (1,H,W,2)

    warped = F.grid_sample(oh, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    warped_idx = warped.argmax(dim=1).squeeze(0).to(torch.long)
    return warped_idx


def iou_per_class_numpy(
    a_idx_hw: np.ndarray,
    b_idx_hw: np.ndarray,
    num_classes: int,
    exclude_classes: Sequence[int] = (0, 10, 11),
    eps: float = 1e-6,
) -> Tuple[np.ndarray, float]:
    """
    Returns the class-wise IoU between a and b, and their mean IoU (excluding exclude_classes).
    """
    a = _as_numpy_int_hw(a_idx_hw)
    b = _as_numpy_int_hw(b_idx_hw)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: a={a.shape} b={b.shape}")
    a = np.clip(a, 0, num_classes - 1)
    b = np.clip(b, 0, num_classes - 1)
    ious = np.full((num_classes,), np.nan, dtype=np.float64)
    for c in range(num_classes):
        if c in set(int(x) for x in exclude_classes):
            continue
        aa = a == c
        bb = b == c
        inter = np.logical_and(aa, bb).sum()
        union = np.logical_or(aa, bb).sum()
        if union > 0:
            ious[c] = float(inter) / float(union + eps)
    miou = float(np.nanmean(ious)) if np.any(~np.isnan(ious)) else float("nan")
    return ious, miou


