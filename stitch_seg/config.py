import torch


DEFAULT_CFG_VALUES = {
    "tool_class_ch": 0,
    "enable_depth_mask": False,
    "enable_flow_translation": False,
    "enable_motion_mask": False,
    "equalize_hist_rgb": False,
    "stride": 1,
    "method": "pyramid",#feature, pyramid, poisson
    "canvas_superres_scale": 1.0,
    "canvas_scale_x": 3,
    "canvas_scale_y": 3,
    "alpha_overlap": 0.95,
    "gradient_radius": 201,
    "canvas_border_trim_px": 12,
    "laplacian_var_min": 60, # if turn off the filtering, set to 0
    "reset_shear_angle": 15.0,
    "reset_rotate_angle": 15.0,
    "reset_scale_factor": 2.0,
    "seg_overlay_alpha": 0.45,
    "seg_min_score": 0.3,
    "seg_min_area": 0,
    "seg_max_coverage": 0.6,
    "segm_stride": 1,
    "segm_input_size": 512,
    "keypoint_color": (0, 255, 255),
    "keypoint_radius": 3,
    "inject_tool_mask": False,
    "aliked_model": "aliked-n16",
    "aliked_weights": "weights/aliked-n16.pth",
    "lightglue_weights": "weights/aliked_lightglue_v0-1_arxiv.pth",
    "port_detector_weights": "weights/convnext_tiny-unet-cholec80_port.pt",
    "checkpoint_dir": "checkpoint",
    "feature_detection_threshold": 0.2,
    "feature_nms_radius": 2,
    "lightglue_depth_confidence": 0.95,
    "lightglue_width_confidence": 0.99,
    "lightglue_filter_threshold": 0.1,
    "feature_ransac_thresh": 3.0,
    "feature_min_matches": 8,
    "feature_min_inliers": 15,
    "optflow_pyr_scale": 0.5,
    "optflow_levels": 3,
    "optflow_winsize": 21,
    "optflow_iterations": 3,
    "optflow_poly_n": 5,
    "optflow_poly_sigma": 1.2,
    "optflow_residual_factor": 50.0,###
    "optflow_magnitude_factor": 50.0,
    "optflow_min_valid_ratio": 0.05,
    "tool_label_keywords": ["tool"],
    "tool_label_ids": None,
    "tool_mask_dilate_px": 12,
    "bbox_mode": "internal", # "external" or "internal"
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}


def apply_stitch_defaults(cfg):
    for key, value in DEFAULT_CFG_VALUES.items():
        if not hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def build_default_cfg():
    class _Cfg:
        pass
    return apply_stitch_defaults(_Cfg())
