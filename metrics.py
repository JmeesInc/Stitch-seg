import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from typing import Tuple


def _warp_image_to_curr(prev_bgr: np.ndarray, H_prev_to_curr: np.ndarray, out_shape_hw: Tuple[int, int]):
    if H_prev_to_curr is None:
        return None, None
    h, w = out_shape_hw
    warped = cv2.warpPerspective(
        prev_bgr, H_prev_to_curr, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
    )
    valid_prev = np.ones(prev_bgr.shape[:2], np.uint8) * 255
    valid = cv2.warpPerspective(
        valid_prev, H_prev_to_curr, (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    return warped, valid


def overlap_ssim(prev_bgr: np.ndarray, curr_bgr: np.ndarray, H_prev_to_curr: np.ndarray, invalid_mask: np.ndarray = None):
    """重複領域のSSIM（高いほど良い）とoverlap比率"""
    warped_prev, valid = _warp_image_to_curr(prev_bgr, H_prev_to_curr, curr_bgr.shape[:2])
    if warped_prev is None:
        return np.nan, 0.0
    g1 = cv2.cvtColor(warped_prev, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    g2 = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    mask = (valid > 0)
    if invalid_mask is not None and invalid_mask.shape[:2] == curr_bgr.shape[:2]:
        mask &= (invalid_mask == 0)
    if mask.sum() == 0:
        return np.nan, 0.0
    _, ssim_map = ssim(g1, g2, data_range=1.0, full=True)
    return float(np.mean(ssim_map[mask])), float(mask.mean())


def overlap_edge_misalignment(prev_bgr: np.ndarray, curr_bgr: np.ndarray, H_prev_to_curr: np.ndarray, invalid_mask: np.ndarray = None):
    """重複領域のエッジずれ[pixel]（低いほど良い。双方向DTの中央値の平均）"""
    warped_prev, valid = _warp_image_to_curr(prev_bgr, H_prev_to_curr, curr_bgr.shape[:2])
    if warped_prev is None:
        return np.nan
    g1 = cv2.cvtColor(warped_prev, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)
    e1 = cv2.Canny(g1, 50, 150)
    e2 = cv2.Canny(g2, 50, 150)
    mask = (valid > 0)
    if invalid_mask is not None and invalid_mask.shape[:2] == curr_bgr.shape[:2]:
        mask &= (invalid_mask == 0)
    if mask.sum() == 0:
        return np.nan
    dt1 = cv2.distanceTransform((e1 == 0).astype(np.uint8), cv2.DIST_L2, 3)
    dt2 = cv2.distanceTransform((e2 == 0).astype(np.uint8), cv2.DIST_L2, 3)
    d12 = dt2[(e1 > 0) & mask]
    d21 = dt1[(e2 > 0) & mask]
    if d12.size == 0 or d21.size == 0:
        return np.nan
    return float(0.5 * (np.median(d12) + np.median(d21)))


def residual_flow_error(prev_bgr: np.ndarray, curr_bgr: np.ndarray, H_prev_to_curr: np.ndarray, invalid_mask: np.ndarray = None):
    """ワープ後の残差フロー大きさの中央値（低いほど良い）"""
    warped_prev, valid = _warp_image_to_curr(prev_bgr, H_prev_to_curr, curr_bgr.shape[:2])
    if warped_prev is None:
        return np.nan
    p = cv2.cvtColor(warped_prev, cv2.COLOR_BGR2GRAY)
    q = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(p, q, None, 0.5, 3, 21, 3, 5, 1.2, 0)
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    mask = (valid > 0)
    if invalid_mask is not None and invalid_mask.shape[:2] == curr_bgr.shape[:2]:
        mask &= (invalid_mask == 0)
    if mask.sum() == 0:
        return np.nan
    return float(np.median(mag[mask]))


def homography_smoothness(prev_rel: np.ndarray, curr_rel: np.ndarray):
    """連続フレームの相対変換差分ノルム（低いほど良い）"""
    if prev_rel is None or curr_rel is None:
        return np.nan
    delta = np.linalg.inv(prev_rel) @ curr_rel
    return float(np.linalg.norm(delta - np.eye(3, dtype=np.float32)))


def stitched_laplacian_variance(img_bgr: np.ndarray):
    """合成ビューのシャープネス（高いほど良い）"""
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(g, cv2.CV_32F)
    return float(lap.var())


