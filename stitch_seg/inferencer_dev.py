import cv2
from kornia.filters.blur_pool import blur_pool2d
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import kornia
import kornia.augmentation as K
import time
import matplotlib.pyplot as plt
import ptlflow
from ptlflow.utils import flow_utils
from ptlflow.utils.io_adapter import IOAdapter

from .models import (
    load_masking_model,
    load_depth_model,
    init_feature_pipeline,
)
from .config import apply_stitch_defaults, build_default_cfg
from .stitch_utils_torch import (
    compute_static_roi,
    equalize_hist_rgb,
    compute_depth_mask,
    merge_masks,
    translation_matrix_from_shift,
    warp_with_transform,
    filter_features_by_mask,
    paste_current_to_canvas_forward,paste_current_to_canvas_forward_poisson,paste_current_to_canvas_forward_multiband,
    invert_canvas_valid_mask,
    extract_canvas_features,
    convert_homography_to_raw_space,
    translation_matrix_from_offset,
    shear_angle_from_homography,
    rotate_angle_from_homography,
    scale_factor_from_homography,
    reset_canvas_orientation,
    get_tool_class_ids,
    laplacian_var
)


class StitchInferencerDev(nn.Module):
    """Stateful stitching engine that exposes stitched crops to a seg model.

    Flow summary:
      1. Preprocess + crop each raw frame using `compute_static_roi`.
      2. Obtain tool/depth masks + motion estimates to build clean canvases.
      4. For every frame, estimate camera shift, paste onto both canvases,
         and cache the homography so predictions can be warped back.
      5. When `model_inference` is called, crop the mask-free canvas around
         the current frame footprint, run the seg model, and warp the output
         into the current frame space.
    """

    def __init__(self, model, start_frame: int, cfg=None):
        super().__init__()
        if cfg is None:
            cfg = build_default_cfg()
        else:
            cfg = apply_stitch_defaults(cfg)
        self.model = model
        self.cfg = cfg
        self.start_frame = int(start_frame)
        self.device = cfg.device
        self.roi = None
        self.ellipse_mask = None
        self.use_roi = bool(getattr(cfg, "use_static_roi", True))
        self.roi = None
        self.apply_ellipse_mask = bool(getattr(cfg, "apply_ellipse_mask", True))
        self.equalize_hist = bool(getattr(cfg, "equalize_hist_rgb", True))

        self.masking_model = load_masking_model(cfg)
        self.seg_processor = nn.Sequential(
            K.Resize(size=(512, 512)),
            K.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        )
        self.depth_model = load_depth_model(cfg)
        self.extractor, self.matcher = init_feature_pipeline(cfg)
        
        self.scope_kernel = torch.ones(5, 5, device=self.device)
        self.last_model_input = None
        self.last_model_output = None
        self.last_tool_mask = None

        self.reset_state()

    def reset_state(self):
        self.canvas = None # for model inference
        self.canvas_memory = None # for step and update canvas
        self.canvas_mask = None # for model inference
        self.canvas_mask_memory = None # for step and update canvas
        self.canvas4model = None
        self.canvas_mask4model = None
        self.offset_xy = (0, 0)
        self.prev_rgb_raw = None
        self.prev_feats = None
        self.prev_stab_transform = torch.eye(3, device=self.device)
        self.H_cum = torch.eye(3, device=self.device)
        self.last_canvas_crop = None
        self.last_canvas_bbox = None
    
    def extract_scope_mask(self, torch_frame: torch.Tensor):
        """
        画像から内視鏡の円/楕円領域のパラメータを推定する
        Return: mask (x, y) - 1が内視鏡視野
        """
        gray = cv2.cvtColor(torch_frame[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8), cv2.COLOR_BGR2GRAY)
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
        radius -= self.cfg.canvas_border_trim_px
        mask = cv2.circle(mask, (int(x), int(y)), int(radius), 1, -1)
        mask = np.ones_like(mask) - mask
        self.ellipse_mask = torch.from_numpy(mask).to(self.device).unsqueeze(0).unsqueeze(0).to(torch.long)




    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """
        frame: (H, W, 3) BGR uint8
        Return: (1, 3, H, W) RGB float (without scale/normalize)
        """
        if frame is None:
            return None

        # Convert BGR (cv2) -> RGB torch tensor on device
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        torch_frame = torch.from_numpy(frame_rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)

        if self.equalize_hist:
            torch_frame = equalize_hist_rgb(torch_frame)

        if self.use_roi:
            if self.roi is None:
                self.roi = compute_static_roi(torch_frame.float())
            x, y, w, h = self.roi
            torch_frame = torch_frame[..., y : y + h, x : x + w].clone()
        if self.apply_ellipse_mask:
            if self.ellipse_mask is None:
                self.extract_scope_mask(torch_frame)

        return torch_frame

    def _to_lightglue_gray(self, rgb_tensor: torch.Tensor) -> torch.Tensor:
        if rgb_tensor is None:
            return None
        if rgb_tensor.dim() == 3:
            rgb_tensor = rgb_tensor.unsqueeze(0)
        if rgb_tensor.max() > 1.0:
            rgb_tensor = rgb_tensor / 255.0
        gray = kornia.color.rgb_to_grayscale(rgb_tensor.float())
        return gray

    def _predict_masks(self, frame_u: torch.Tensor):
        """
        input: (1, 3, H, W) RGB uint8/float
        output: tool_mask: (H, W) uint8 (device on self.device)
                depth_mask: (H, W) uint8 (device on self.device)
        """
        input_img = self.seg_processor(frame_u/255.0)
        input_img_dpt = torch.nn.functional.interpolate(input_img, size=(518, 518), mode='bilinear', align_corners=False).unsqueeze(0)
        depth_tensor = None
        with torch.autocast(dtype=torch.float16, device_type=self.device.type, enabled=True):
            masking = self.masking_model(input_img)
            if self.depth_model is not None:
                depth_feature = self.depth_model.forward_features(input_img_dpt)
                depth_tensor = self.depth_model.forward_depth(depth_feature, input_img_dpt.shape)[0]
        
        masking = torch.nn.functional.interpolate(masking, size=frame_u.shape[-2:], mode='bilinear', align_corners=False)
        masking = (masking > 0.5).to(torch.uint8) * 255
        
        if depth_tensor is not None:
            depth_tensor = torch.nn.functional.interpolate(depth_tensor, size=frame_u.shape[-2:], mode='bilinear', align_corners=False)
            depth_mask = compute_depth_mask(depth_tensor, self.cfg)
        else:
            depth_mask = None
        
        if self.apply_ellipse_mask:
            masking = masking | torch.nn.functional.interpolate(self.ellipse_mask, size=frame_u.shape[-2:], mode='nearest')
        
        return masking, depth_mask

    def _current_to_canvas_h(self):
        scale = getattr(self.cfg, "canvas_superres_scale", 1.0)
        S = torch.eye(3, device=self.device)
        S[0, 0] = scale
        S[1, 1] = scale
        T = translation_matrix_from_offset(self.offset_xy, device=self.device)
        # Apply scale first, then translate so the offset is not scaled again.
        H_cum = self.H_cum.clone()
        H_cum = H_cum.to(torch.float32)
        ret = T @ S @ H_cum
        return ret

    def _current_canvas_bbox(self, frame_shape):
        h, w = frame_shape # int, int
        H = self._current_to_canvas_h()
        
        corners = torch.tensor([[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]], device=self.device)
        ones = torch.ones((4, 1), device=self.device)
        pts = torch.cat([corners, ones], dim=1) # (4, 3)
        
        proj = (H @ pts.T).T
        denom = torch.clamp(proj[:, 2:3], min=1e-6)
        proj_xy = proj[:, :2] / denom

        mode = getattr(self.cfg, "bbox_mode", "external")
        # print(f"DEBUG: bbox_mode={mode}")
        
        if mode == "internal":
            # 最小矩形として「現在フレーム（投影後）の外接矩形」を必ず含む
            # その上で、canvas_mask（+ ellipse）で広げられるだけ広げる
            xs = proj_xy[:, 0]
            ys = proj_xy[:, 1]
            center = proj_xy.mean(dim=0)
            
            try:
                # 外接bbox（現在フレーム全体が必ず入る）
                base_x1 = max(0, int(torch.floor(xs.min()).item()))
                base_y1 = max(0, int(torch.floor(ys.min()).item()))
                base_x2 = min(self.canvas4model.shape[-1], int(torch.ceil(xs.max()).item()))
                base_y2 = min(self.canvas4model.shape[-2], int(torch.ceil(ys.max()).item()))
                x1, y1, x2, y2 = base_x1, base_y1, base_x2, base_y2
                
                # 以降は「拡張のみ」。縮小はしない（= 現在フレーム包含保証を壊さない）
                if self.canvas4model_mask is not None:
                    mask = self.canvas4model_mask.squeeze().clone() # (H, W) Copy to avoid modifying actual canvas mask
                    H_mask, W_mask = mask.shape
                    
                    # Explicitly mask out current frame's ellipse/scope mask from the bbox calculation
                    # (To ensure we don't include black borders even if they were somehow marked valid)
                    if self.apply_ellipse_mask and self.ellipse_mask is not None:
                        # self.ellipse_mask: 1=Invalid(Outside), 0=Valid(Inside)
                        # Warp it to canvas space
                        ch, cw = self.canvas4model.shape[-2:]
                        warped_ellipse = warp_with_transform(
                            self.ellipse_mask.float(), 
                            H, 
                            (ch, cw), 
                            interpolation='nearest', 
                            border_mode='zeros' # Padding with 0 (Valid) to avoid shrinking from canvas borders unnecessarily
                        )
                        # warped_ellipse: (1, 1, CH, CW) or (CH, CW) depending on batch
                        if warped_ellipse.dim() == 4:
                            warped_ellipse = warped_ellipse.squeeze(0).squeeze(0)
                        elif warped_ellipse.dim() == 3:
                            warped_ellipse = warped_ellipse.squeeze(0)
                            
                        # Apply to mask: Set regions where ellipse is 1 (Invalid) to 0
                        mask[warped_ellipse > 0.5] = 0

                    # Validate coords
                    x1 = max(0, min(x1, W_mask))
                    y1 = max(0, min(y1, H_mask))
                    x2 = max(0, min(x2, W_mask))
                    y2 = max(0, min(y2, H_mask))

                    # Helper for counting consecutive True values
                    def count_consecutive(tensor_1d, from_start=True):
                        if not tensor_1d.any(): return 0
                        if not from_start:
                            tensor_1d = tensor_1d.flip(0)
                        return tensor_1d.cumprod(dim=0).sum().item()

                    # Expand only: Expand edges as long as ALL pixels are valid
                    # (Valid is >128 to be robust to interpolation/noise)
                    if x2 > x1 and y2 > y1:
                        for _ in range(10): # Converges quickly
                            changed = False

                            # Left: count continuous valid columns immediately adjacent to x1
                            if x1 > 0:
                                roi_left = mask[y1:y2, 0:x1]
                                valid_cols = (roi_left > 128).all(dim=0)
                                expand = count_consecutive(valid_cols, from_start=False)
                                if expand > 0:
                                    x1 -= int(expand)
                                    changed = True

                            # Right
                            if x2 < W_mask:
                                roi_right = mask[y1:y2, x2:W_mask]
                                valid_cols = (roi_right > 128).all(dim=0)
                                expand = count_consecutive(valid_cols, from_start=True)
                                if expand > 0:
                                    x2 += int(expand)
                                    changed = True

                            # Top
                            if y1 > 0:
                                roi_top = mask[0:y1, x1:x2]
                                valid_rows = (roi_top > 128).all(dim=1)
                                expand = count_consecutive(valid_rows, from_start=False)
                                if expand > 0:
                                    y1 -= int(expand)
                                    changed = True

                            # Bottom
                            if y2 < H_mask:
                                roi_bottom = mask[y2:H_mask, x1:x2]
                                valid_rows = (roi_bottom > 128).all(dim=1)
                                expand = count_consecutive(valid_rows, from_start=True)
                                if expand > 0:
                                    y2 += int(expand)
                                    changed = True

                            if not changed:
                                break

                    # Aspect ratio constraint:
                    # Keep bbox aspect reasonably close to current frame aspect, but NEVER smaller than base bbox.
                    # Define distortion as max(bbox_ar/frame_ar, frame_ar/bbox_ar) and clamp to <= 1.33.
                    max_aspect_distortion = float(getattr(self.cfg, "bbox_aspect_max_distortion", 1.33))
                    if max_aspect_distortion < 1.0:
                        max_aspect_distortion = 1.0

                    if x2 > x1 and y2 > y1:
                        frame_ar = float(w) / float(h) if h > 0 else 1.0
                        bbox_w = float(x2 - x1)
                        bbox_h = float(y2 - y1)
                        bbox_ar = bbox_w / bbox_h if bbox_h > 0 else frame_ar

                        lower_ar = frame_ar / max_aspect_distortion
                        upper_ar = frame_ar * max_aspect_distortion

                        # Helper to clamp interval while keeping base bbox inside expanded bbox
                        def _shrink_width_keep_base(desired_w: float):
                            nonlocal x1, x2
                            desired_w_i = int(max(1, round(desired_w)))
                            base_w_i = int(max(1, base_x2 - base_x1))
                            desired_w_i = max(desired_w_i, base_w_i)
                            # Feasible x1 range
                            lo = max(x1, base_x2 - desired_w_i)
                            hi = min(x2 - desired_w_i, base_x1)
                            if lo > hi:
                                return False
                            new_x1 = min(max(x1, lo), hi)
                            x1 = int(new_x1)
                            x2 = int(new_x1 + desired_w_i)
                            return True

                        def _shrink_height_keep_base(desired_h: float):
                            nonlocal y1, y2
                            desired_h_i = int(max(1, round(desired_h)))
                            base_h_i = int(max(1, base_y2 - base_y1))
                            desired_h_i = max(desired_h_i, base_h_i)
                            lo = max(y1, base_y2 - desired_h_i)
                            hi = min(y2 - desired_h_i, base_y1)
                            if lo > hi:
                                return False
                            new_y1 = min(max(y1, lo), hi)
                            y1 = int(new_y1)
                            y2 = int(new_y1 + desired_h_i)
                            return True

                        # If too wide, shrink width; if too tall, shrink height.
                        if bbox_ar > upper_ar:
                            # Need w/h <= upper_ar  => w <= upper_ar * h
                            target_w = upper_ar * bbox_h
                            _shrink_width_keep_base(target_w)
                        elif bbox_ar < lower_ar:
                            # Need w/h >= lower_ar => h <= w / lower_ar
                            target_h = bbox_w / lower_ar if lower_ar > 1e-8 else bbox_h
                            _shrink_height_keep_base(target_h)

            except Exception as e:
                # print(f"DEBUG: Exception in internal bbox: {e}")
                return None, None, None, None
                #return 0, 0, self.canvas.shape[-1], self.canvas.shape[-2]
        else:
            xs = proj_xy[:, 0]
            ys = proj_xy[:, 1]
            try:
                x1 = max(0, int(torch.floor(xs.min()).item()))
                y1 = max(0, int(torch.floor(ys.min()).item()))
                x2 = min(self.canvas4model.shape[-1], int(torch.ceil(xs.max()).item()))
                y2 = min(self.canvas4model.shape[-2], int(torch.ceil(ys.max()).item()))
            except Exception as e:
                return None, None, None, None

        if x2 <= x1 or y2 <= y1:
            # Fallback to full canvas if calculation fails (though rare)
            return 0, 0, self.canvas4model.shape[-1], self.canvas4model.shape[-2]
        return x1, y1, x2, y2
    
    @torch.inference_mode()
    def model_inference(self, frame_shape) -> torch.Tensor:
        """
        Returns: Seg Map Tensor (Classes, H, W)
        """
        source_canvas = self.canvas4model
        if self.model is None or source_canvas is None:
            return None
            
        x1, y1, x2, y2 = self._current_canvas_bbox(frame_shape)
        if x1 is None or y1 is None or x2 is None or y2 is None:
            self.reset_state()
            return torch.zeros((1, self.cfg.num_classes, frame_shape[0], frame_shape[1]), dtype=torch.float32, device=self.device)

        crop = source_canvas[..., y1:y2, x1:x2]
        if crop.numel() == 0: return None
        
        # Model Inference (crop is (1, 3, Hc, Wc))
        # self.model expects (B, 3, H, W)
        canvas_out = self.model(crop) # (1, Classes, Hc, Wc)
        
        # Resize output to crop size (in case model output different stride)
        if canvas_out.shape[-2:] != crop.shape[-2:]:
            canvas_out = torch.nn.functional.interpolate(canvas_out, size=crop.shape[-2:], mode='bilinear', align_corners=False)
            
        # Create full canvas prediction holder
        canvas_pred = torch.zeros((1, canvas_out.shape[1], source_canvas.shape[-2], source_canvas.shape[-1]), 
                                  dtype=torch.float32, device=self.device)
        canvas_pred[..., y1:y2, x1:x2] = canvas_out
        
        self.last_canvas_crop = crop
        self.last_model_input = crop
        self.last_model_output = canvas_out
        self.last_canvas_bbox = (x1, y1, x2, y2)
        
        # Warp back to current frame
        h, w = frame_shape
        H = self._current_to_canvas_h()
        try:
            H_canvas_to_curr = torch.linalg.inv(H)
        except Exception as e:
            self.reset_state()
            return torch.zeros((1, self.cfg.num_classes, frame_shape[0], frame_shape[1]), dtype=torch.float32, device=self.device)
        # Warp (1, Classes, H_canv, W_canv) -> (1, Classes, H, W)
        warped_pred = warp_with_transform(canvas_pred, H_canvas_to_curr, (h, w), interpolation='nearest', border_mode='zeros')

        if self.cfg.apply_ellipse_mask: # ellipse maskの1の部分は0にする
            warped_pred[:, 0, :, :] = self.ellipse_mask.squeeze(0).float()*255.0
        # Optionally inject tool mask into a dedicated class channel.
        # IMPORTANT: For binary segmentation (num_classes=1), injecting would overwrite the only channel
        # and make downstream visualizations look like "tool segmentation".
        if self.cfg.tool_class_ch is not None and getattr(self.cfg, "num_classes", 0) > 1:
            if self.last_tool_mask is not None and 0 <= int(self.cfg.tool_class_ch) < warped_pred.shape[1]:
                warped_pred[:, int(self.cfg.tool_class_ch), :, :] = self.last_tool_mask.squeeze(0).float()
        return warped_pred.squeeze(0) # (Classes, H, W)
    
    def first_frame(self, frame_u, tool_mask_raw, depth_mask_raw):
        if self.canvas4model is None:
            _, _, h0, w0 = frame_u.shape
            self.canvas_h = int(self.cfg.canvas_scale_y * h0 * self.cfg.canvas_superres_scale)
            self.canvas_w = int(self.cfg.canvas_scale_x * w0 * self.cfg.canvas_superres_scale)
            self.canvas4model = torch.zeros((1, 3, self.canvas_h, self.canvas_w), dtype=torch.float32, device=self.device)
            self.canvas4model_mask = torch.zeros((1, 1, self.canvas_h, self.canvas_w), dtype=torch.uint8, device=self.device)
            self.canvas = self.canvas4model.clone()
            self.canvas_mask = self.canvas4model_mask.clone()
            # Correct offset calculation based on UN-scaled dimensions
            # canvas_w is scaled, so divide by scale first
            scale = getattr(self.cfg, "canvas_superres_scale", 1.0)
            base_cw = int(self.canvas_w / scale) if scale > 0 else self.canvas_w
            base_ch = int(self.canvas_h / scale) if scale > 0 else self.canvas_h
            self.offset_xy = (base_cw // 2 - w0 // 2, base_ch // 2 - h0 // 2)

            self.prev_rgb_raw = frame_u.clone()
            self.prev_stab_transform = torch.eye(3, device=self.device)
            combined_mask_raw = merge_masks(tool_mask_raw, depth_mask_raw)
            self.combined_mask_prev_raw = combined_mask_raw
            # Features
            tensor_lg = self._to_lightglue_gray(frame_u)
            with torch.no_grad():
                feats = self.extractor.extract(tensor_lg)
            self.prev_feats = filter_features_by_mask(feats, self.combined_mask_prev_raw) if self.combined_mask_prev_raw is not None else feats
            self.H_cum = torch.eye(3, device=self.device)
            # Paste
            if self.cfg.method == "pyramid":
                self.canvas, self.canvas_mask, self.canvas4model, self.canvas4model_mask = paste_current_to_canvas_forward_multiband(
                    self.canvas, self.canvas_mask, self.H_cum, self.offset_xy, frame_u, combined_mask_raw, self.cfg, self.cfg.alpha_overlap
                )
            elif self.cfg.method == "poisson":
                self.canvas, self.canvas_mask, self.canvas4model, self.canvas4model_mask = paste_current_to_canvas_forward_poisson(
                    self.canvas, self.canvas_mask, self.H_cum, self.offset_xy, frame_u, combined_mask_raw, self.cfg, self.cfg.alpha_overlap
                )
            else:
                self.canvas, self.canvas_mask, self.canvas4model, self.canvas4model_mask = paste_current_to_canvas_forward(
                    self.canvas, self.canvas_mask, self.H_cum, self.offset_xy, frame_u, combined_mask_raw, self.cfg, self.cfg.alpha_overlap
                )
            self.last_tool_mask = tool_mask_raw.clone()
    
    def step_canvas(self, frame_u: torch.Tensor):
        """
        Main logic step.
        frame_u: (1, 3, H, W) RGB Float
        """
        with torch.autocast(device_type=self.device.type, dtype=torch.float16):
            # === 1. Predict Masks ===
            tool_mask_raw, depth_mask_raw = self._predict_masks(frame_u)
            # === 2. Convert to Grayscale for Flow ===
            # === 3. Initialization ===
            if self.canvas4model is None:
                self.first_frame(frame_u, tool_mask_raw, depth_mask_raw)
                return

            # === 5. Mask Merge ===
            combined_mask_raw = merge_masks(tool_mask_raw, depth_mask_raw)

            # === 6. Feature Extraction ===
            tensor_lg = self._to_lightglue_gray(frame_u)
            with torch.no_grad():
                curr_feats = self.extractor.extract(tensor_lg)
            curr_feats = filter_features_by_mask(curr_feats, combined_mask_raw) if combined_mask_raw is not None else curr_feats
            
            # === 7. Global Homography ===
            H_cum_curr = self.H_cum
            if curr_feats is not None and self.prev_feats is not None:
                self.prev_feats['keypoints'] = self.prev_feats['keypoints'].to(torch.float32)
                #H_rel = self.estimate_homography_from_features(
                #    self.prev_feats, curr_feats
                #)
                H_inv = self.estimate_homography_from_features(
                    curr_feats, self.prev_feats
                )
                #if H_rel is not None:
                #    if isinstance(H_rel, np.ndarray):
                #        H_rel_t = torch.from_numpy(H_rel).to(self.device, dtype=torch.float32)
                #    else:
                #        H_rel_t = H_rel.to(torch.float32)
                    
                    # H_rel maps prev_raw -> curr_stab (p_stab = H_rel @ p_prev)
                    # We want H_{curr_raw -> prev_raw} (p_prev = H_step @ p_curr)
                    # p_stab = T_curr @ p_curr
                    # T_curr @ p_curr = H_rel @ p_prev
                    # p_prev = inv(H_rel) @ T_curr @ p_curr
                    #H_cum_curr = self.H_cum @ torch.linalg.inv(H_rel_t)
                if isinstance(H_inv, tuple):
                    self.reset_state()
                    self.first_frame(frame_u, tool_mask_raw, depth_mask_raw)
                    return
                H_cum_curr = self.H_cum @ H_inv

            # === 8. Paste Current Frame to Canvas (update canvas) ===
            # self.canvas4model/self.canvas4model_mask: current frame inference用なので、blurでも更新してよい
            # self.canvas/self.canvas_mask: 次フレームの推論に使うので、blur時は更新しない
            blur = laplacian_var(frame_u.float()) < self.cfg.laplacian_var_min

            # NOTE:
            # paste_current_to_canvas_forward*() は引数 canvas/canvas_mask をインプレース更新するため、
            # blur時に self.canvas をそのまま渡すと「代入しなくても」self.canvasが更新されてしまう。
            # blur時は clone を渡して current-frame 用 (canvas4model) だけ更新する。
            canvas_in = self.canvas if not blur else self.canvas.clone()
            canvas_mask_in = self.canvas_mask if not blur else self.canvas_mask.clone()
            update_mode = "only_new" if blur else "full"
            if self.cfg.method == "pyramid":
                new_canvas, new_canvas_mask, self.canvas4model, self.canvas4model_mask = paste_current_to_canvas_forward_multiband(
                    canvas_in,
                    canvas_mask_in,
                    H_cum_curr,
                    self.offset_xy,
                    frame_u,
                    combined_mask_raw,
                    self.cfg,
                    self.cfg.alpha_overlap,
                    update_mode=update_mode,
                )
            elif self.cfg.method == "poisson":
                new_canvas, new_canvas_mask, self.canvas4model, self.canvas4model_mask = paste_current_to_canvas_forward_poisson(
                    canvas_in,
                    canvas_mask_in,
                    H_cum_curr,
                    self.offset_xy,
                    frame_u,
                    combined_mask_raw,
                    self.cfg,
                    self.cfg.alpha_overlap,
                    update_mode=update_mode,
                )
            else:
                new_canvas, new_canvas_mask, self.canvas4model, self.canvas4model_mask = paste_current_to_canvas_forward(
                    canvas_in,
                    canvas_mask_in,
                    H_cum_curr,
                    self.offset_xy,
                    frame_u,
                    combined_mask_raw,
                    self.cfg,
                    self.cfg.alpha_overlap,
                    update_mode=update_mode,
                )
                # 次フレーム用のbase canvasは、blur時は更新しない
            if not blur:
                self.canvas, self.canvas_mask = new_canvas, new_canvas_mask
            
            # Transform current features to raw space for next iteration
            self.prev_feats = curr_feats
            if curr_feats is not None:
                kps = curr_feats["keypoints"] # (B, N, 2)
                if kps.numel() > 0:
                    try:
                        # kornia.geometry.transform.transform_points might be missing or moved
                        # Manual implementation: P_out = H @ P_in
                        B_k, N_k, _ = kps.shape
                        ones = torch.ones((B_k, N_k, 1), device=kps.device, dtype=kps.dtype)
                        kps_homo = torch.cat([kps, ones], dim=2) # (B, N, 3)
                        kps_raw = kps_homo[..., :2]
                        
                        self.prev_feats["keypoints"] = kps_raw
                    except RuntimeError:
                        # Inversion failed, keep as is (likely bad transform)
                        pass
        self.H_cum = H_cum_curr
        
        # === 9. Reset Logic === # degreeからradianにしてnumpy消したい
        shear = shear_angle_from_homography(H_cum_curr)
        rot = rotate_angle_from_homography(H_cum_curr)
        scale = scale_factor_from_homography(H_cum_curr)

        reset = (shear > getattr(self.cfg, "reset_shear_angle", 15.0) or
                 rot > getattr(self.cfg, "reset_rotate_angle", 15.0) or
                 scale > getattr(self.cfg, "reset_scale_factor", 2.0))

        if not blur:
            if reset:
                H_cum_curr = H_cum_curr.to(torch.float32)
                self.canvas, self.canvas_mask, self.offset_xy = reset_canvas_orientation(
                    self.canvas, self.canvas_mask, H_cum_curr, frame_u.shape[-2:], self.cfg, self.offset_xy
                )
                self.H_cum = torch.eye(3, device=self.device)
            #self.canvas = self.canvas4model.clone()
            #self.canvas_mask = self.canvas4model_mask.clone()
            #print(laplacian_var(frame_u.float()), self.cfg.laplacian_var_min, "canvas update")
        
        # === 13. Update State ===
        self.prev_rgb_raw = frame_u.clone()
        self.combined_mask_prev_raw = combined_mask_raw if combined_mask_raw is not None else None
        self.prev_stab_transform = H_cum_curr
        self.last_tool_mask = tool_mask_raw.clone()

    @torch.inference_mode()
    def inference_video(self, video_path: str):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open: {video_path}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        ok, first = cap.read()
        if not ok:
            raise RuntimeError("Failed to read first frame")

        first_u = self.preprocess_frame(first)
        self.step_canvas(first_u)
        first_seg = self.model_inference(first_u.shape[-2:])

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = range(self.start_frame + 1, total)
        outputs = [first_seg]

        # Each loop iteration stitches the latest frame (always), but the
        # caller can decide how often to consume the stitched segmentation
        # results (e.g., via `segm_stride` logic in the CLI).
        for fidx in tqdm(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ok, frame = cap.read()
            if not ok:
                break
            frame_u = self.preprocess_frame(frame)
            self.step_canvas(frame_u)
            seg_curr = self.model_inference(frame_u.shape[-2:])
            outputs.append(seg_curr)
        cap.release()
        return outputs
    
    def _preprocess(self, img: torch.Tensor):
        """
        入力: (C, H, W) in [0, 1] -> 出力: (1, 1, 3, H, W) in [0, 255]
        """
        # 1. 範囲調整 [0, 1] -> [0, 255]
        if img.max() > 1.0:
            img = img / 255.0
            
        return img
    
    def estimate_homography_from_features(self, prev_feats, curr_feats, min_matches: int = 4):
        """
        特徴点対応からホモグラフィ（curr→prev）を推定する (Pure PyTorch / Kornia版)。

        返り値:
        - H_rel: 推定されたホモグラフィ (3x3) Tensor または None
        """
        H_rel = None
        inliers = 0
        pts_prev = None
        pts_curr = None
        inlier_mask = None

        if prev_feats is None or curr_feats is None:
            return H_rel, inliers, pts_prev, pts_curr, inlier_mask

        # Matcher実行 (GPU)
        with torch.inference_mode():
            result = self.matcher({"image0": prev_feats, "image1": curr_feats})
        
        matches = result.get("matches") # (M, 2)
        scores = result.get("scores")

        # LightGlueはバッチ対応していますが、ここではBatch=0のみ取得する前提
        # matchesは (M, 2) のTensor
        match_idx = matches[0]

        # 最小マッチ数チェック (DLTには最低4点必要)
        if match_idx.shape[0] < max(min_matches, 4):
            return H_rel, inliers, pts_prev, pts_curr, inlier_mask

        # ポイント抽出 (GPU上のTensorのまま)
        # prev_feats["keypoints"] is typically (B, N, 2) -> take batch 0
        kp_prev = prev_feats["keypoints"][0]
        kp_curr = curr_feats["keypoints"][0]
        
        pts_prev = kp_prev[match_idx[:, 0]] # (M, 2)
        pts_curr = kp_curr[match_idx[:, 1]] # (M, 2)

        H_est = kornia.geometry.find_homography_dlt(pts_prev.unsqueeze(0), pts_curr.unsqueeze(0), weights=scores[0].unsqueeze(0))
        if H_est.isinf().any():
            ransac = kornia.geometry.RANSAC(
                model_type='homography',
                batch_size=1,
                max_iter=100,
                confidence=0.95
            )
            H_est, mask = ransac(pts_prev, pts_curr, weights=scores[0])
        H_est = H_est.squeeze(0) # (3, 3)
        return H_est
