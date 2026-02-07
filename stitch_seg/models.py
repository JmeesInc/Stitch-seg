import numpy as np
import torch
import segmentation_models_pytorch as smp
from pathlib import Path

from .video_depth_anything.video_depth_stream import VideoDepthAnything


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

def load_depth_model(cfg):
    if VideoDepthAnything is None:
        raise RuntimeError("VideoDepthAnything is not available")
    if cfg.enable_depth_mask:
        depth_model = VideoDepthAnything(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024])
        depth_model.load_state_dict(torch.load(cfg.depth_anything_v2_model, map_location='cpu'), strict=True)
        depth_model = depth_model.to(cfg.device).eval()
        return depth_model
    return None