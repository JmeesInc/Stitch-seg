import numpy as np
import torch
import segmentation_models_pytorch as smp
from pathlib import Path

from .lightglue import ALIKED, LightGlue

def init_feature_pipeline(cfg):
    extractor = ALIKED(
        weights=cfg.aliked_weights,
        model_name=cfg.aliked_model,
        detection_threshold=cfg.feature_detection_threshold,
        nms_radius=cfg.feature_nms_radius,
    ).to(cfg.device).eval()
    matcher = LightGlue(
        weights=cfg.lightglue_weights,
        features="aliked",
        depth_confidence=cfg.lightglue_depth_confidence,
        width_confidence=cfg.lightglue_width_confidence,
        filter_threshold=cfg.lightglue_filter_threshold,
    ).to(cfg.device).eval()
    return extractor, matcher


def load_masking_model(cfg):
    seg_model = smp.Unet(
        encoder_name="tu-convnext_tiny.dinov3_lvd1689m",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation="sigmoid",
    ).to(cfg.device)
    ckpt_path = getattr(cfg, "tool_detector_weights", "weights/convnext_tiny-unet-best.pt")
    state = torch.load(ckpt_path, map_location=cfg.device)
    seg_model.load_state_dict(state, strict=True)
    seg_model.eval()
    return seg_model

def load_masking_model2(cfg):
    seg_model = smp.Unet(
        encoder_name="tu-convnext_tiny",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation="sigmoid",
    ).to(cfg.device)
    ckpt_path = getattr(cfg, "port_detector_weights", "weights/convnext_tiny-unet-cholec80_port.pt")
    state = torch.load(ckpt_path, map_location=cfg.device)
    seg_model.load_state_dict(state, strict=True)
    seg_model.eval()
    return seg_model

def motion_mask(flow: torch.Tensor) -> torch.Tensor:
    """
    input: flow (1, 1, 2, H, W)
    output: motion_mask (H, W) uint8, 0 or 255
    
    optical flowから動いている部分のマスクを推定する
    """
    dx = flow[:, 0, 0, ...]
    dy = flow[:, 0, 1, ...]
    median_dx = dx.median()
    median_dy = dy.median()
    rel_dx, rel_dy = dx - median_dx, dy - median_dy
    magnitude = torch.sqrt(rel_dx**2 + rel_dy**2)
    mean_mag = torch.mean(magnitude)
    std_mag = torch.std(magnitude)
    mag_thresh = mean_mag + 3 * std_mag
    motion_mask = (magnitude > mag_thresh).to(torch.uint8) * 255
    return motion_mask