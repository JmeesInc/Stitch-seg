import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
import kornia.augmentation as K
import ptlflow

from .config import apply_track_defaults, build_default_track_cfg

from .models import init_feature_pipeline, load_masking_model, load_masking_model2, motion_mask
from .stitch_utils_torch import (
    compute_static_roi,
    equalize_hist_rgb,
    merge_masks,
    warp_with_transform,
    filter_features_by_mask,
    paste_current_to_canvas_forward_multiband,
    translation_matrix_from_offset,
    reset_canvas_orientation,
)

class StitchTracker(nn.Module):
    """
    Stateful stitching + motion separation with simplified 3-step API.

    Usage::

        tracker = StitchTracker2(cfg=cfg)

        for frame in frames:
            crop_bbox, crop, frame_u = tracker.step(frame)
            pts_crop = external_tracker.update(crop)
            pts_current = tracker.reproject(pts_crop)
    """
    def __init__(self, cfg=None, canvas_channels=3):
        super().__init__()
        if cfg is None:
            cfg = build_default_track_cfg()
        else:
            cfg = apply_track_defaults(cfg)
        self.canvas_channels = canvas_channels
        self.cfg = cfg
        self.device = cfg.device
        self.roi = None
        self.ellipse_mask = None
        self.use_roi = bool(getattr(cfg, "use_static_roi", True))
        self.roi = None
        self.apply_ellipse_mask = bool(getattr(cfg, "apply_ellipse_mask", True))
        self.equalize_hist = bool(getattr(cfg, "equalize_hist_rgb", True))
        self.use_flow_mask = bool(getattr(cfg, "use_flow_mask", False))

        self.masking_model = load_masking_model(cfg)
        self.masking_model2 = load_masking_model2(cfg)
        self.flow_model = ptlflow.get_model('neuflow2', ckpt_path="mixed").to(self.device)
        self.seg_processor = nn.Sequential(
            K.Resize(size=(512, 512)),
            K.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        )
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
        self.canvas4model_curr = None
        self.canvas4model_curr_mask = None
        self.offset_xy = (0, 0)
        self.prev_rgb_raw = None
        self.prev_feats = None
        self.prev_stab_transform = torch.eye(3, device=self.device)
        self.H_cum = torch.eye(3, device=self.device)
        self.last_canvas_bbox = None
        self._first_canvas_corners = None
        self._last_crop_bbox = None
        self._last_frame_shape = None
    
    def extract_scope_mask_torch(self, torch_frame: torch.Tensor):
        """
        画像から内視鏡の円/楕円領域のパラメータを推定する
        Return: mask (x, y) - 1が内視鏡視野
        """
        gray = torch.mean(torch_frame, dim=1, keepdim=True)
        # 1. 二値化 (閾値は環境に合わせて調整。10-30あたりが一般的)
        thresh = (gray > (15 / 255.0)).float()
        # 2. モルフォロジー演算（ノイズ除去と穴埋め）
        thresh = kornia.morphology.opening(thresh, self.scope_kernel)
        thresh = kornia.morphology.closing(thresh, self.scope_kernel)
        # 4. 円のパラメータ推定 (モーメント法による中心と半径の推定)
            # 輪郭抽出(findContours)の代わりに、1が立っている座標の重心を求める
        # 座標グリッドの生成
        B, C, H, W = thresh.shape
        y_coords, x_coords = torch.meshgrid(
            torch.arange(H, device=thresh.device),
            torch.arange(W, device=thresh.device),
            indexing='ij'
        )
        
        # 重心 (Center of Mass) の計算
        sum_thresh = thresh.sum()
        if sum_thresh == 0:
            return None
            
        center_y = (y_coords * thresh).sum() / sum_thresh
        center_x = (x_coords * thresh).sum() / sum_thresh
        
        # 半径の推定 (面積 S = πr^2 から逆算、または重心からの最大距離)
        # ここでは面積から逆算するのがノイズに強く安定します
        radius = torch.sqrt(sum_thresh / torch.pi) - self.cfg.canvas_border_trim_px
        
        # 5. マスクの再生成 (円の外側を1にする)
        dist_sq = (x_coords - center_x)**2 + (y_coords - center_y)**2
        final_mask = (dist_sq > radius**2).long() # 円の外側を1にする
        
        self.ellipse_mask = final_mask.unsqueeze(0).unsqueeze(0) # [1, 1, H, W]
    
    def extract_scope_mask(self, torch_frame: torch.Tensor):
        """
        画像から内視鏡の円/楕円領域のパラメータを推定する
        Return: mask (x, y) - 1が内視鏡視野
        """
        if isinstance(torch_frame, torch.Tensor):
            gray = cv2.cvtColor(torch_frame[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8), cv2.COLOR_BGR2GRAY)
        else:
            gray = cv2.cvtColor(torch_frame, cv2.COLOR_BGR2GRAY)
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
        self.ellipse_mask = torch.from_numpy(mask).to(self.device).unsqueeze(0).unsqueeze(0)




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
        # Avoid expensive full-image reductions (max over all pixels) where possible.
        # - uint/int tensors are assumed to be in [0,255]
        # - float tensors are assumed to be in [0,1] unless a sparse sample suggests [0,255]
        if rgb_tensor.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            rgb_f = rgb_tensor.to(torch.float32) / 255.0
        else:
            rgb_f = rgb_tensor.to(torch.float32)
            sample = rgb_f[..., ::16, ::16]
            max_val = float(sample.max().item()) if sample.numel() else float(rgb_f.max().item())
            if max_val > 1.5:
                rgb_f = rgb_f / 255.0

        gray = torch.mean(rgb_f, dim=1, keepdim=True)
        return gray

    def _predict_masks(self, frame_u: torch.Tensor):
        """
        input: (1, 3, H, W) RGB uint8/float
        output: tool_mask: (H, W) uint8 (device on self.device)
                depth_mask: (H, W) uint8 (device on self.device)
        """
        frame_u = frame_u.to(self.device)
        input_img = self.seg_processor(frame_u / 255.0)
        # Pure inference: avoid autograd graphs to reduce VRAM usage.
        with torch.inference_mode(), torch.autocast(dtype=torch.float16, device_type=self.device.type, enabled=True):
            input_img_fp16 = input_img.to(torch.float16)
            masking = self.masking_model(input_img_fp16)
            masking2 = self.masking_model2(input_img_fp16)
        
        masking = torch.nn.functional.interpolate(masking, size=frame_u.shape[-2:], mode='bilinear', align_corners=False)
        masking2 = torch.nn.functional.interpolate(masking2, size=frame_u.shape[-2:], mode='bilinear', align_corners=False)
        masking = (masking > 0.5).to(torch.uint8) * 255
        masking2 = (masking2 > 0.5).to(torch.uint8) * 255
        masking = masking | masking2
        
        if self.apply_ellipse_mask:
            masking = masking | torch.nn.functional.interpolate(self.ellipse_mask.float(), size=frame_u.shape[-2:], mode='nearest').long()
        if self.prev_rgb_raw is not None:
            # Stack along time dimension: (1, 3, H, W) -> (1, 2, 3, H, W)
            images = torch.stack([self.prev_rgb_raw.to(self.device), frame_u.to(self.device).float()], dim=1)
            flow = self.flow_model({"images": images})["flows"] #1, 1, 2, H, W
            flow_mask = motion_mask(flow)
        else:
            flow_mask = torch.zeros_like(masking)
        if self.use_flow_mask:
            masking = masking | flow_mask
        return masking, flow_mask

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

    def _update_canvas4model_in_current(self, frame_shape):
        """
        Project the stitched canvas into the current frame coordinates.
        Stores result in self.canvas4model_curr / self.canvas4model_curr_mask.
        """
        if self.canvas4model is None or self.canvas4model_mask is None:
            self.canvas4model_curr = None
            self.canvas4model_curr_mask = None
            return

        if isinstance(frame_shape, tuple):
            out_h, out_w = frame_shape
        else:
            out_h, out_w = frame_shape[-2], frame_shape[-1]

        H_current_to_canvas = self._current_to_canvas_h()
        if H_current_to_canvas is None or not torch.isfinite(H_current_to_canvas).all():
            self.canvas4model_curr = None
            self.canvas4model_curr_mask = None
            return

        H_canvas_to_current = torch.linalg.inv(H_current_to_canvas.to(torch.float32))
        self.canvas4model_curr = warp_with_transform(
            self.canvas4model,
            H_canvas_to_current,
            output_shape=(out_h, out_w),
            interpolation="bilinear",
            border_mode="zeros",
        )
        self.canvas4model_curr_mask = warp_with_transform(
            self.canvas4model_mask.float(),
            H_canvas_to_current,
            output_shape=(out_h, out_w),
            interpolation="nearest",
            border_mode="zeros",
        ).to(torch.uint8)

    def _current_canvas_bbox(self, frame_shape):
        h, w = frame_shape # int, int
        H = self._current_to_canvas_h()
        
        corners = torch.tensor([[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]], device=self.device)
        ones = torch.ones((4, 1), device=self.device)
        pts = torch.cat([corners, ones], dim=1) # (4, 3)
        
        proj = (H @ pts.T).T # ここがnanになってる
        denom = torch.clamp(proj[:, 2:3], min=1e-6)
        proj_xy = proj[:, :2] / denom

        mode = getattr(self.cfg, "bbox_mode", "internal")
        # print(f"DEBUG: bbox_mode={mode}")

        def _original_bbox():
            xs = proj_xy[:, 0]
            ys = proj_xy[:, 1]
            try:
                ox1 = max(0, int(torch.floor(xs.min()).item()))
                oy1 = max(0, int(torch.floor(ys.min()).item()))
                ox2 = min(self.canvas4model.shape[-1], int(torch.ceil(xs.max()).item()))
                oy2 = min(self.canvas4model.shape[-2], int(torch.ceil(ys.max()).item()))
            except Exception:
                print("Error in _original_bbox")
                return None, None, None, None
            return ox1, oy1, ox2, oy2

        if mode == "internal":
            x1, y1, x2, y2 = _original_bbox()
            if x1 is None or y1 is None or x2 is None or y2 is None:
                print("Error in _internal_bbox")
                return None, None, None, None
            if self.canvas4model_mask is not None:
                mask = self.canvas4model_mask.squeeze().clone()
                H_mask, W_mask = mask.shape
                x1 = max(0, min(x1, W_mask))
                y1 = max(0, min(y1, H_mask))
                x2 = max(0, min(x2, W_mask))
                y2 = max(0, min(y2, H_mask))

                # Only expand if the original bbox is fully inside the valid mask.
                if x2 > x1 and y2 > y1 and (mask[y1:y2, x1:x2] > 0).all():
                    def count_consecutive(tensor_1d, from_start=True):
                        if not tensor_1d.any():
                            return 0
                        if not from_start:
                            tensor_1d = tensor_1d.flip(0)
                        return tensor_1d.cumprod(dim=0).sum().item()

                    for _ in range(10):
                        changed = False

                        if x1 > 0:
                            roi_left = mask[y1:y2, 0:x1]
                            valid_cols = (roi_left > 0).all(dim=0)
                            expand = count_consecutive(valid_cols, from_start=False)
                            if expand > 0:
                                x1 -= int(expand)
                                changed = True

                        if x2 < W_mask:
                            roi_right = mask[y1:y2, x2:W_mask]
                            valid_cols = (roi_right > 0).all(dim=0)
                            expand = count_consecutive(valid_cols, from_start=True)
                            if expand > 0:
                                x2 += int(expand)
                                changed = True

                        if y1 > 0:
                            roi_top = mask[0:y1, x1:x2]
                            valid_rows = (roi_top > 0).all(dim=1)
                            expand = count_consecutive(valid_rows, from_start=False)
                            if expand > 0:
                                y1 -= int(expand)
                                changed = True

                        if y2 < H_mask:
                            roi_bottom = mask[y2:H_mask, x1:x2]
                            valid_rows = (roi_bottom > 0).all(dim=1)
                            expand = count_consecutive(valid_rows, from_start=True)
                            if expand > 0:
                                y2 += int(expand)
                                changed = True

                        if not changed:
                            break
        elif mode == "first":
            if self._first_canvas_corners is not None:
                xs = self._first_canvas_corners[:, 0]
                ys = self._first_canvas_corners[:, 1]
                canvas_w = self.canvas4model.shape[-1]
                canvas_h = self.canvas4model.shape[-2]
                x1 = max(0, int(torch.floor(xs.min()).item()))
                y1 = max(0, int(torch.floor(ys.min()).item()))
                x2 = min(canvas_w, int(torch.ceil(xs.max()).item()))
                y2 = min(canvas_h, int(torch.ceil(ys.max()).item()))
                if x2 <= x1 or y2 <= y1:
                    return 0, 0, canvas_w, canvas_h
                self.last_canvas_bbox = (x1, y1, x2, y2)
                return x1, y1, x2, y2
            else:
                x1, y1, x2, y2 = _original_bbox()
        elif mode == "external":
            if self.canvas4model_mask is None:
                x1, y1, x2, y2 = _original_bbox()
            else:
                mask = self.canvas4model_mask.squeeze()
                coords = (mask > 0).nonzero(as_tuple=False)
                if coords.numel() == 0:
                    x1, y1, x2, y2 = _original_bbox()
                else:
                    ys = coords[:, 0]
                    xs = coords[:, 1]
                    x1 = int(xs.min().item())
                    y1 = int(ys.min().item())
                    x2 = int(xs.max().item()) + 1
                    y2 = int(ys.max().item()) + 1
        elif mode == "entire":
            x1, y1, x2, y2 = 0, 0, self.canvas4model.shape[-1], self.canvas4model.shape[-2]
        else:
            x1, y1, x2, y2 = _original_bbox()

        if x1 is None or y1 is None or x2 is None or y2 is None:
            print("Some of the bbox values are None")
            return None, None, None, None
        if x2 <= x1 or y2 <= y1:
            # Fallback to full canvas if calculation fails (though rare)
            return 0, 0, self.canvas4model.shape[-1], self.canvas4model.shape[-2]
        return self._stabilize_canvas_bbox(x1, y1, x2, y2)

    def _stabilize_canvas_bbox(self, x1: int, y1: int, x2: int, y2: int):
        last = self.last_canvas_bbox
        if last is None:
            self.last_canvas_bbox = (x1, y1, x2, y2)
            return x1, y1, x2, y2

        max_delta = int(getattr(self.cfg, "bbox_max_delta_px", 0) or 0)
        if max_delta > 0:
            lx1, ly1, lx2, ly2 = last
            x1 = max(lx1 - max_delta, min(lx1 + max_delta, x1))
            y1 = max(ly1 - max_delta, min(ly1 + max_delta, y1))
            x2 = max(lx2 - max_delta, min(lx2 + max_delta, x2))
            y2 = max(ly2 - max_delta, min(ly2 + max_delta, y2))

        alpha = float(getattr(self.cfg, "bbox_smooth_alpha", 0.0) or 0.0)
        if alpha > 0.0:
            lx1, ly1, lx2, ly2 = last
            x1 = int(round(alpha * x1 + (1.0 - alpha) * lx1))
            y1 = int(round(alpha * y1 + (1.0 - alpha) * ly1))
            x2 = int(round(alpha * x2 + (1.0 - alpha) * lx2))
            y2 = int(round(alpha * y2 + (1.0 - alpha) * ly2))

        min_area_ratio = float(getattr(self.cfg, "bbox_min_area_ratio", 0.0) or 0.0)
        if min_area_ratio > 0.0:
            new_area = max(0, x2 - x1) * max(0, y2 - y1)
            last_area = max(0, last[2] - last[0]) * max(0, last[3] - last[1])
            if last_area > 0 and new_area < last_area * min_area_ratio:
                x1, y1, x2, y2 = last

        if x2 <= x1 or y2 <= y1:
            x1, y1, x2, y2 = last

        self.last_canvas_bbox = (x1, y1, x2, y2)
        return x1, y1, x2, y2
    def first_frame(self, frame_u, tool_mask_raw, current_img=None):
        if self.canvas4model is None:
            # `current_img` is what gets pasted into the canvas.
            # - For normal stitching: current_img = frame_u (RGB, 3ch)
            # - For pred stitching: current_img = pred (Classes, Cch)
            if current_img is None:
                current_img = frame_u
            if current_img.dim() == 3:
                current_img = current_img.unsqueeze(0)
            if current_img.dim() != 4:
                raise ValueError(f"current_img must be 4D (B,C,H,W), got {tuple(current_img.shape)}")
            if int(current_img.shape[1]) != int(self.canvas_channels):
                raise ValueError(
                    f"canvas_channels mismatch: canvas_channels={self.canvas_channels} "
                    f"but current_img has C={int(current_img.shape[1])}. "
                    f"Pass canvas_channels matching the tensor you want to stitch."
                )
            _, _, h0, w0 = frame_u.shape
            self.canvas_h = int(self.cfg.canvas_scale_y * h0 * self.cfg.canvas_superres_scale)
            self.canvas_w = int(self.cfg.canvas_scale_x * w0 * self.cfg.canvas_superres_scale)
            self.canvas4model = torch.zeros((1, self.canvas_channels, self.canvas_h, self.canvas_w), dtype=torch.float32, device=self.device)
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
            combined_mask_raw = tool_mask_raw
            self.combined_mask_prev_raw = combined_mask_raw
            # Features
            tensor_lg = self._to_lightglue_gray(frame_u)
            with torch.no_grad():
                feats = self.extractor.extract(tensor_lg)
            self.prev_feats = filter_features_by_mask(feats, self.combined_mask_prev_raw) if self.combined_mask_prev_raw is not None else feats
            self.H_cum = torch.eye(3, device=self.device)
            # Paste
            self.canvas, self.canvas_mask, self.canvas4model, self.canvas4model_mask = paste_current_to_canvas_forward_multiband(
                self.canvas, self.canvas_mask, self.H_cum, self.offset_xy, current_img, combined_mask_raw, self.cfg, self.cfg.alpha_overlap
            )
            self.last_tool_mask = tool_mask_raw.clone()
            # Store first frame corners in canvas coordinates for bbox_mode="first"
            H_first = self._current_to_canvas_h()
            corners_frame = torch.tensor(
                [[0.0, 0.0], [w0, 0.0], [w0, h0], [0.0, h0]],
                device=self.device,
            )
            ones = torch.ones((4, 1), device=self.device)
            corners_homo = torch.cat([corners_frame, ones], dim=1)  # (4, 3)
            proj = (H_first @ corners_homo.T).T  # (4, 3)
            self._first_canvas_corners = proj[:, :2] / proj[:, 2:3].clamp(min=1e-8)
    
    @torch.inference_mode()
    def step_canvas(self, frame_u: torch.Tensor, transform: str = "homography"):
        """
        Main logic step.
        frame_u: (1, 3, H, W) RGB Float
        transform: "homography" (default) or "tps"
        
        Returns:
            crop_bbox: (x1, y1, x2, y2) bounding box of crop region in canvas coordinates, or None if failed
            crop: (1, 3, H, W) cropped canvas tensor ready for tracker input, or None if failed
        """
        with torch.autocast(device_type=self.device.type, dtype=torch.float16):
            transform_mode = str(transform).lower() if transform is not None else "homography"
            use_tps = transform_mode == "tps"
            tps_params = None
            # === 1. Predict Masks ===
            tool_mask_raw, flow_mask_raw = self._predict_masks(frame_u)
            # === 3. Initialization ===
            if self.canvas4model is None:
                self.first_frame(frame_u, tool_mask_raw)
                # Get crop region after first frame initialization
                try:
                    x1, y1, x2, y2 = self._current_canvas_bbox(frame_u.shape[-2:])
                    if x1 is None or y1 is None or x2 is None or y2 is None:
                        return None, None
                    
                    crop = self.canvas4model[..., y1:y2, x1:x2]
                    crop = torch.nn.functional.interpolate(
                        crop, size=frame_u.shape[-2:], mode='bilinear', align_corners=False
                    )
                    
                    if crop.max() <= 1.0:
                        crop = crop * 255.0
                    crop = crop.float()
                    
                    return (x1, y1, x2, y2), crop
                except Exception:
                    return None, None

            # === 5. Mask Merge ===
            combined_mask_raw = merge_masks(tool_mask_raw, flow_mask_raw)

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
                if H_inv is None or isinstance(H_inv, tuple):
                    H_cum_curr = self.H_cum
                elif torch.isfinite(H_inv).all():
                    # H_inv is valid, update H_cum
                    H_cum_curr = self.H_cum @ H_inv
                    if not torch.isfinite(H_cum_curr).all():
                        H_cum_curr = self.H_cum
                else:
                    # H_inv contains inf/nan, don't update
                    H_cum_curr = self.H_cum

            if use_tps and curr_feats is not None and self.prev_feats is not None:
                tps_params = self.estimate_tps_params(self.prev_feats, curr_feats)
                if tps_params is None:
                    use_tps = False

            # === 8. Paste Current Frame to Canvas (update canvas) ===
            # self.canvas4model/self.canvas4model_mask: current frame inference用なので、blurでも更新してよい
            # self.canvas/self.canvas_mask: 次フレームの推論に使うので、blur時は更新しない
            #blur = laplacian_var(frame_u.float()) < self.cfg.laplacian_var_min
            #print(blur)
            blur=False

            # NOTE:
            # paste_current_to_canvas_forward*() は引数 canvas/canvas_mask をインプレース更新するため、
            # blur時に self.canvas をそのまま渡すと「代入しなくても」self.canvasが更新されてしまう。
            # blur時は clone を渡して current-frame 用 (canvas4model) だけ更新する。
            canvas_in = self.canvas if not blur else self.canvas.clone()
            canvas_mask_in = self.canvas_mask if not blur else self.canvas_mask.clone()
            update_mode = "only_new" if blur else "full"
            new_canvas, new_canvas_mask, self.canvas4model, self.canvas4model_mask = paste_current_to_canvas_forward_multiband(
                canvas_in,
                canvas_mask_in,
                H_cum_curr,
                self.offset_xy,
                frame_u,
                tool_mask_raw,
                self.cfg,
                self.cfg.alpha_overlap,
                update_mode=update_mode,
                tps_params=tps_params if use_tps else None,
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

        if getattr(self.cfg, "bbox_mode", "internal") != "first":
            # Re-orient the canvas so the current frame is axis-aligned at the center.
            # This resets H_cum to identity and updates offset_xy.
            H_cum_curr = H_cum_curr.to(torch.float32)
            old_offset_xy = self.offset_xy
            self.canvas, self.canvas_mask = reset_canvas_orientation(
                self.canvas, self.canvas_mask, H_cum_curr, frame_u.shape[-2:], self.cfg, old_offset_xy
            )
            # Keep `canvas4model` in the SAME coordinate system after reset.
            if self.canvas4model is not None and self.canvas4model_mask is not None:
                self.canvas4model, self.canvas4model_mask = reset_canvas_orientation(
                    self.canvas4model, self.canvas4model_mask, H_cum_curr, frame_u.shape[-2:], self.cfg, old_offset_xy
                )
            self.H_cum = torch.eye(3, device=self.device)
        # In "first" mode: skip reset entirely.
        # The canvas stays in the first-frame coordinate system.
        # H_cum keeps accumulating (current frame → first frame space).
        # _first_canvas_corners are fixed at the position set in first_frame().

        # === 13. Update State ===
        self.prev_rgb_raw = frame_u.clone()
        self.combined_mask_prev_raw = combined_mask_raw if combined_mask_raw is not None else None
        self.prev_stab_transform = H_cum_curr
        self.last_tool_mask = tool_mask_raw.clone()
        if bool(getattr(self.cfg, "canvas4model_project_to_current", False)):
            self._update_canvas4model_in_current(frame_u.shape[-2:])
        
        # === 14. Get crop region for tracker input ===
        try:
            x1, y1, x2, y2 = self._current_canvas_bbox(frame_u.shape[-2:])
            if x1 is None or y1 is None or x2 is None or y2 is None:
                return None, None
            
            # Get crop from canvas4model
            crop = self.canvas4model[..., y1:y2, x1:x2]
            crop = torch.nn.functional.interpolate(
                crop, size=frame_u.shape[-2:], mode='bilinear', align_corners=False
            )
            
            # Normalize crop values to 0-255 range
            if crop.max() <= 1.0:
                crop = crop * 255.0
            crop = crop.float()  # (1, 3, H, W)
            
            return (x1, y1, x2, y2), crop
        except Exception:
            return None, None
    
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
    
    def pts_projection(self, pts_crop, crop_bbox: tuple, frame_shape: tuple):
        """
        Transform points from crop coordinate system to current frame coordinate system.
        
        Args:
            pts_crop: (N, 2) points in crop coordinate system (output from tracker)
                     Can be numpy.ndarray or torch.Tensor
            crop_bbox: (x1, y1, x2, y2) bounding box of crop region in canvas coordinates
            frame_shape: (H, W) shape of preprocessed frame
        
        Returns:
            pts_curr: (N, 2) points in current frame coordinate system (with ROI offset applied)
                     Returns the same type as input (numpy.ndarray or torch.Tensor)
        """
        is_tensor = isinstance(pts_crop, torch.Tensor)
        
        if pts_crop is None or (is_tensor and pts_crop.numel() == 0) or (not is_tensor and len(pts_crop) == 0):
            if is_tensor:
                return torch.empty((0, 2), dtype=torch.float32, device=pts_crop.device if pts_crop is not None else self.device)
            else:
                return np.array([], dtype=np.float32).reshape(0, 2)
        
        x1, y1, x2, y2 = crop_bbox
        frame_h, frame_w = frame_shape
        
        # 1. Crop座標系 -> Canvas座標系への変換
        crop_w = max(x2 - x1, 1)
        crop_h = max(y2 - y1, 1)
        scale_x = crop_w / max(frame_w, 1)
        scale_y = crop_h / max(frame_h, 1)
        
        if is_tensor:
            # Tensor版の処理
            device = pts_crop.device
            dtype = pts_crop.dtype
            
            # Canvas座標系への変換
            pts_canvas = pts_crop.clone()
            pts_canvas[:, 0] = pts_crop[:, 0] * scale_x + x1
            pts_canvas[:, 1] = pts_crop[:, 1] * scale_y + y1
            
            # Canvas座標系 -> Current frame（preprocess後）座標系への変換
            H_curr_to_canvas = self._current_to_canvas_h()
            H_canvas_to_curr = torch.linalg.inv(H_curr_to_canvas.to(torch.float32))
            
            # Homography変換を適用（korniaまたは手動実装）
            ones = torch.ones((pts_canvas.shape[0], 1), device=device, dtype=dtype)
            pts_canvas_homo = torch.cat([pts_canvas, ones], dim=1)  # (N, 3)
            
            # H @ pts^T -> (3, N) -> transpose -> (N, 3)
            pts_curr_homo = (H_canvas_to_curr.to(device) @ pts_canvas_homo.T).T
            pts_curr = pts_curr_homo[:, :2] / (pts_curr_homo[:, 2:3].clamp(min=1e-8))
            
            # ROI offsetを適用
            if self.roi is not None:
                roi_x, roi_y, _, _ = self.roi
                pts_curr = pts_curr + torch.tensor([[roi_x, roi_y]], device=device, dtype=dtype)
            
            return pts_curr
        else:
            # NumPy版の処理（既存の実装）
            pts_canvas = np.empty_like(pts_crop, dtype=np.float32)
            pts_canvas[:, 0] = pts_crop[:, 0] * scale_x + x1
            pts_canvas[:, 1] = pts_crop[:, 1] * scale_y + y1
            
            # 2. Canvas座標系 -> Current frame（preprocess後）座標系への変換
            H_curr_to_canvas = self._current_to_canvas_h()
            H_canvas_to_curr = torch.linalg.inv(H_curr_to_canvas.to(torch.float32))
            H_canvas_to_curr_np = H_canvas_to_curr.cpu().numpy().astype(np.float32)
            
            pts_canvas_h = pts_canvas.reshape(-1, 1, 2).astype(np.float32)
            pts_curr = cv2.perspectiveTransform(pts_canvas_h, H_canvas_to_curr_np).reshape(-1, 2)
            
            # 3. Current frame（preprocess後）座標系 -> 元フレーム座標系への変換（ROI offset）
            if self.roi is not None:
                roi_x, roi_y, _, _ = self.roi
                pts_curr[:, 0] += roi_x
                pts_curr[:, 1] += roi_y
            
            return pts_curr
    
    def bbox_projection(self, bbox_canvas: tuple, crop_bbox: tuple, frame_shape: tuple) -> tuple:
        """
        Transform bounding box from canvas coordinate system to current frame coordinate system.
        
        Args:
            bbox_canvas: (x1, y1, x2, y2) bounding box in canvas coordinate system
            crop_bbox: (x1, y1, x2, y2) bounding box of crop region in canvas coordinates
            frame_shape: (H, W) shape of preprocessed frame
        
        Returns:
            bbox_curr: (x1, y1, w, h) bounding box in current frame coordinate system (with ROI offset applied)
        """
        if bbox_canvas is None:
            return None
        
        bx1_canvas, by1_canvas, bx2_canvas, by2_canvas = bbox_canvas
        
        # 4つのコーナーを変換
        corners_canvas = np.array([
            [bx1_canvas, by1_canvas],
            [bx2_canvas, by1_canvas],
            [bx2_canvas, by2_canvas],
            [bx1_canvas, by2_canvas],
        ], dtype=np.float32)
        
        # Canvas座標系 -> Current frame（preprocess後）座標系への変換
        H_curr_to_canvas = self._current_to_canvas_h()
        H_canvas_to_curr = torch.linalg.inv(H_curr_to_canvas.to(torch.float32))
        H_canvas_to_curr_np = H_canvas_to_curr.cpu().numpy().astype(np.float32)
        
        corners_canvas_h = corners_canvas.reshape(-1, 1, 2)
        corners_curr = cv2.perspectiveTransform(corners_canvas_h, H_canvas_to_curr_np).reshape(-1, 2)
        
        # ROI offsetを適用
        if self.roi is not None:
            roi_x, roi_y, _, _ = self.roi
            corners_curr[:, 0] += roi_x
            corners_curr[:, 1] += roi_y
        
        # Bboxを再計算
        new_x1 = float(corners_curr[:, 0].min())
        new_y1 = float(corners_curr[:, 1].min())
        new_x2 = float(corners_curr[:, 0].max())
        new_y2 = float(corners_curr[:, 1].max())
        
        return (int(new_x1), int(new_y1), int(new_x2 - new_x1), int(new_y2 - new_y1))

    # ==================================================================
    # Simplified 3-step API
    # ==================================================================

    def step(self, frame, transform="homography"):
        """
        Preprocess frame, update canvas, and return crop for tracker input.

        Args:
            frame: BGR uint8 (H, W, 3) numpy array, or already-preprocessed
                   (1, 3, H, W) float tensor.
            transform: "homography" (default) or "tps".

        Returns:
            crop_bbox: (x1, y1, x2, y2) in canvas coordinates, or None
            crop: (1, C, H, W) cropped canvas resized to frame dims, or None
            frame_u: (1, 3, H, W) preprocessed frame tensor
        """
        if isinstance(frame, np.ndarray):
            frame_u = self.preprocess_frame(frame)
        else:
            frame_u = frame

        crop_bbox, crop = self.step_canvas(frame_u, transform=transform)

        self._last_crop_bbox = crop_bbox
        self._last_frame_shape = tuple(frame_u.shape[-2:])

        return crop_bbox, crop, frame_u

    def reproject(self, pts_crop, crop_bbox=None, frame_shape=None):
        """
        Transform points from crop coordinates to current frame coordinates.
        Uses stored crop_bbox/frame_shape from last step() by default.

        Args:
            pts_crop: (N, 2) points in crop coordinate system (numpy or torch)
            crop_bbox: override crop_bbox (default: last step's value)
            frame_shape: override frame_shape (default: last step's value)

        Returns:
            pts_curr: (N, 2) in current frame coordinates (same type as input)
        """
        bbox = crop_bbox if crop_bbox is not None else self._last_crop_bbox
        shape = frame_shape if frame_shape is not None else self._last_frame_shape
        if bbox is None or shape is None:
            raise RuntimeError(
                "Call step() before reproject(), or provide crop_bbox and frame_shape."
            )
        return self.pts_projection(pts_crop, bbox, shape)

    def reproject_bbox(self, bbox_canvas, crop_bbox=None, frame_shape=None):
        """
        Transform bounding box from canvas coordinates to current frame coordinates.
        Uses stored crop_bbox/frame_shape from last step() by default.

        Args:
            bbox_canvas: (x1, y1, x2, y2) in canvas coordinates
            crop_bbox: override crop_bbox (default: last step's value)
            frame_shape: override frame_shape (default: last step's value)

        Returns:
            (x1, y1, w, h) in current frame coordinates
        """
        bbox = crop_bbox if crop_bbox is not None else self._last_crop_bbox
        shape = frame_shape if frame_shape is not None else self._last_frame_shape
        if bbox is None or shape is None:
            raise RuntimeError(
                "Call step() before reproject_bbox(), or provide crop_bbox and frame_shape."
            )
        return self.bbox_projection(bbox_canvas, bbox, shape)
