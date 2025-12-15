import cv2
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
    load_tool_detector,
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
    fast_gradient_mask,
)


class StitchInferencer(nn.Module):
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
        self.apply_ellipse_mask = bool(getattr(cfg, "apply_ellipse_mask", True))
        self.equalize_hist = bool(getattr(cfg, "equalize_hist_rgb", True))

        self.seg_model = load_tool_detector(cfg)
        self.seg_processor = nn.Sequential(
            K.Resize(size=(512, 512)),
            K.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        )
        self.depth_model = load_depth_model(cfg)
        self.extractor, self.matcher = init_feature_pipeline(cfg)
        self.flow_model = ptlflow.get_model("neuflow2", "mixed").to(self.cfg.device)
        self.flow_model.eval()
        for param in self.flow_model.parameters():
            param.requires_grad_(False)

        self.reset_state()

    def reset_state(self):
        self.canvas = None
        self.canvas_mask = None
        self.offset_xy = None
        self.roi = None
        self.ellipse_mask = None
        self.prev_flow_gray_raw = None
        self.prev_flow_gray_stab = None
        self.prev_rgb_raw = None
        self.prev_rgb_stab = None
        self.prev_feats = None
        self.canvas_feats = None
        self.canvas_invalid_mask = None
        self.prev_stab_transform = torch.eye(3, device=self.device)
        self.H_cum = torch.eye(3, device=self.device)
        self.last_model_input = None
        self.last_model_output = None
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

    def _ensure_mask_4d(self, mask: torch.Tensor):
        if mask is None:
            return None
        if not isinstance(mask, torch.Tensor):
            return mask
        m = mask
        if m.dim() == 2:
            m = m.unsqueeze(0).unsqueeze(0)
        elif m.dim() == 3:
            m = m.unsqueeze(0)
        return m

    def _predict_masks(self, frame_u: torch.Tensor):
        """
        input: (1, 3, H, W) RGB uint8/float
        output: tool_mask: (H, W) uint8 (device on self.device)
                depth_mask: (H, W) uint8 (device on self.device)
        """
        input_img = self.seg_processor(frame_u/255.0)
        input_img_dpt = torch.nn.functional.interpolate(input_img, size=(518, 518), mode='bilinear', align_corners=False).unsqueeze(0)
        depth_tensor = None
        with torch.autocast(device_type=self.device.type, enabled=True):
            tool_mask = self.seg_model(input_img)
            if self.depth_model is not None:
                depth_feature = self.depth_model.forward_features(input_img_dpt)
                depth_tensor = self.depth_model.forward_depth(depth_feature, input_img_dpt.shape)[0]
        
        tool_mask = torch.nn.functional.interpolate(tool_mask, size=frame_u.shape[-2:], mode='bilinear', align_corners=False)
        tool_mask = (tool_mask > 0.5).to(torch.uint8) * 255
        
        if depth_tensor is not None:
            depth_tensor = torch.nn.functional.interpolate(depth_tensor, size=frame_u.shape[-2:], mode='bilinear', align_corners=False)
            depth_mask = compute_depth_mask(depth_tensor, self.cfg)
        else:
            depth_mask = None
        
        if self.apply_ellipse_mask:
            tool_mask = tool_mask | torch.nn.functional.interpolate(self.ellipse_mask, size=frame_u.shape[-2:], mode='nearest')
        
        return tool_mask, depth_mask

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
        
        xs = proj_xy[:, 0]
        ys = proj_xy[:, 1]
        
        try:
            x1 = max(0, int(torch.floor(xs.min()).item()))
            y1 = max(0, int(torch.floor(ys.min()).item()))
            x2 = min(self.canvas.shape[-1], int(torch.ceil(xs.max()).item()))
            y2 = min(self.canvas.shape[-2], int(torch.ceil(ys.max()).item()))
        except Exception as e:
            print(xs)
            print(ys)
            print(proj)
            print(H, pts)
            raise e
            #return 0, 0, self.canvas.shape[-1], self.canvas.shape[-2]
        x1 = max(0, int(torch.floor(xs.min()).item()))
        y1 = max(0, int(torch.floor(ys.min()).item()))
        x2 = min(self.canvas.shape[-1], int(torch.ceil(xs.max()).item()))
        y2 = min(self.canvas.shape[-2], int(torch.ceil(ys.max()).item()))
        
        if x2 <= x1 or y2 <= y1:
            return 0, 0, self.canvas.shape[-1], self.canvas.shape[-2]
        return x1, y1, x2, y2
    
    @torch.inference_mode()
    def model_inference(self, frame_shape) -> torch.Tensor:
        """
        Returns: Seg Map Tensor (Classes, H, W)
        """
        source_canvas = self.canvas
        if self.model is None or source_canvas is None:
            return None
            
        x1, y1, x2, y2 = self._current_canvas_bbox(frame_shape)#現在は外接、内接でも良さそう
        # Crop mask check
        if self.canvas_mask is not None:
             # Basic check using nonzero
             valid_indices = torch.nonzero(self.canvas_mask.squeeze())
             if valid_indices.shape[0] > 0:
                 ymin_m, xmin_m = valid_indices.min(dim=0)[0]
                 ymax_m, xmax_m = valid_indices.max(dim=0)[0]
                 x1 = max(0, min(x1, int(xmin_m.item())))
                 y1 = max(0, min(y1, int(ymin_m.item())))
                 x2 = min(self.canvas_mask.shape[-1], max(x2, int(xmax_m.item()) + 1))
                 y2 = min(self.canvas_mask.shape[-2], max(y2, int(ymax_m.item()) + 1))

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
        H_canvas_to_curr = torch.linalg.inv(self._current_to_canvas_h())
        
        # Warp (1, Classes, H_canv, W_canv) -> (1, Classes, H, W)
        warped_pred = warp_with_transform(canvas_pred, H_canvas_to_curr, (h, w), interpolation='nearest', border_mode='zeros')
        return warped_pred.squeeze(0) # (Classes, H, W)

    def step_canvas(self, frame_u: torch.Tensor):
        """
        Main logic step.
        frame_u: (1, 3, H, W) RGB Float
        """
        with torch.autocast(device_type=self.device.type, dtype=torch.float16):
            # === 1. Predict Masks ===
            tool_mask_raw, depth_mask_raw = self._predict_masks(frame_u)
            #tool_mask_raw = self._ensure_mask_4d(tool_mask_raw)
            #depth_mask_raw = self._ensure_mask_4d(depth_mask_raw)
            
            # === 2. Convert to Grayscale for Flow ===
            curr_flow_gray_raw = kornia.color.rgb_to_grayscale(frame_u) # (1, 1, H, W)

            # === 3. Initialization ===
            if self.canvas is None:
                _, _, h0, w0 = frame_u.shape
                self.canvas_h = int(self.cfg.canvas_scale_y * h0 * self.cfg.canvas_superres_scale)
                self.canvas_w = int(self.cfg.canvas_scale_x * w0 * self.cfg.canvas_superres_scale)
                self.canvas = torch.zeros((1, 3, self.canvas_h, self.canvas_w), dtype=torch.float32, device=self.device)
                self.canvas_mask = torch.zeros((1, 1, self.canvas_h, self.canvas_w), dtype=torch.uint8, device=self.device)
                
                # Correct offset calculation based on UN-scaled dimensions
                # canvas_w is scaled, so divide by scale first
                scale = getattr(self.cfg, "canvas_superres_scale", 1.0)
                base_cw = int(self.canvas_w / scale) if scale > 0 else self.canvas_w
                base_ch = int(self.canvas_h / scale) if scale > 0 else self.canvas_h
                self.offset_xy = (base_cw // 2 - w0 // 2, base_ch // 2 - h0 // 2)
                
                self.prev_flow_gray_raw = curr_flow_gray_raw.clone()
                self.prev_flow_gray_stab = curr_flow_gray_raw.clone()
                self.prev_rgb_raw = frame_u.clone()
                self.prev_rgb_stab = frame_u.clone()
                self.prev_stab_transform = torch.eye(3, device=self.device)
                combined_mask_raw = merge_masks(tool_mask_raw, depth_mask_raw)
                self.combined_mask_prev_raw = combined_mask_raw
                self.combined_mask_prev_stab = combined_mask_raw.clone() if combined_mask_raw is not None else None
                # Features
                tensor_lg = self._to_lightglue_gray(frame_u)
                with torch.no_grad():
                    feats = self.extractor.extract(tensor_lg)
                self.prev_feats = filter_features_by_mask(feats, self.combined_mask_prev_stab) if self.combined_mask_prev_stab is not None else feats
                self.H_cum = torch.eye(3, device=self.device)
                # Paste
                self.canvas, self.canvas_mask = paste_current_to_canvas_forward(
                    self.canvas, self.canvas_mask, self.H_cum, self.offset_xy, frame_u, combined_mask_raw, self.cfg, self.cfg.alpha_overlap
                )
                self.canvas_invalid_mask = invert_canvas_valid_mask(self.canvas_mask)
                self.canvas_feats = extract_canvas_features(self.extractor, self.canvas, self.cfg, self.canvas_invalid_mask)
                return

            # === 4. Stability Estimation ===
            if self.cfg.enable_flow_translation:
                shift = self.estimate_camera_shift(self.prev_rgb_raw, frame_u, self.combined_mask_prev_raw)
                curr_stab_transform = translation_matrix_from_shift(shift, device=self.device)
            else:
                curr_stab_transform = torch.eye(3, device=self.device)
            
            # === 5. Warp current to stabilized space ===
            frame_u_stab = warp_with_transform(frame_u, curr_stab_transform, interpolation='bilinear')
            curr_flow_gray_stab = warp_with_transform(curr_flow_gray_raw, curr_stab_transform, interpolation='bilinear')

            # === 6. Warp Tool/Depth Masks ===
            tool_mask_stab = None
            if tool_mask_raw is not None:
                warped_tool = warp_with_transform(tool_mask_raw.float(), curr_stab_transform, interpolation='nearest', border_mode='zeros')
                tool_mask_stab = (warped_tool > 0).to(torch.uint8) * 255

            depth_mask_stab = None
            if depth_mask_raw is not None:
                warped_depth = warp_with_transform(depth_mask_raw.float(), curr_stab_transform, interpolation='nearest', border_mode='zeros')
                depth_mask_stab = (warped_depth > 0).to(torch.uint8) * 255

            # === 7. Motion Mask ===
            if self.cfg.enable_motion_mask:
                motion_mask_stab, _, _, _ = self.compute_optical_flow_mask(
                    self.prev_flow_gray_stab, curr_flow_gray_stab, self.combined_mask_prev_stab
                )
                motion_mask_stab = self._ensure_mask_4d(motion_mask_stab)
                motion_mask_raw = None
                if motion_mask_stab is not None:
                    inv_stab = torch.linalg.inv(curr_stab_transform)
                    motion_mask_raw = warp_with_transform(motion_mask_stab.float(), inv_stab, interpolation='nearest')
                    motion_mask_raw = (motion_mask_raw > 0).to(torch.uint8) * 255
            else:
                motion_mask_stab = None
                motion_mask_raw = None

            # === 8. Mask Merge ===
            combined_mask_raw = merge_masks(tool_mask_raw, motion_mask_raw, depth_mask_raw)
            combined_mask_stab = merge_masks(tool_mask_stab, motion_mask_stab, depth_mask_stab)

            # === 9. Feature Extraction (Stabilized) ===
            tensor_lg = self._to_lightglue_gray(frame_u_stab)
            with torch.no_grad():
                curr_feats = self.extractor.extract(tensor_lg)
            curr_feats = filter_features_by_mask(curr_feats, combined_mask_stab) if combined_mask_stab is not None else curr_feats
            
            # === 10. Global Homography (Matching & DLT) ===
            H_cum_curr = self.H_cum
            if curr_feats is not None and self.prev_feats is not None:
                self.prev_feats['keypoints'] = self.prev_feats['keypoints'].to(torch.float32)
                H_rel = self.estimate_homography_from_features(
                    self.prev_feats, curr_feats
                )
                if H_rel is not None:
                    if isinstance(H_rel, np.ndarray):
                        H_rel_t = torch.from_numpy(H_rel).to(self.device, dtype=torch.float32)
                    else:
                        H_rel_t = H_rel.to(torch.float32)
                    
                    # H_rel maps prev_raw -> curr_stab (p_stab = H_rel @ p_prev)
                    # We want H_{curr_raw -> prev_raw} (p_prev = H_step @ p_curr)
                    # p_stab = T_curr @ p_curr
                    # T_curr @ p_curr = H_rel @ p_prev
                    # p_prev = inv(H_rel) @ T_curr @ p_curr
                    H_rel_inv = torch.linalg.inv(H_rel_t)
                    H_step = H_rel_inv @ curr_stab_transform
                    H_cum_curr = self.H_cum @ H_step

            # === 11. Paste Current Frame to Canvas ===
            self.canvas, self.canvas_mask = paste_current_to_canvas_forward(
                self.canvas, self.canvas_mask, H_cum_curr, self.offset_xy, frame_u, combined_mask_raw, self.cfg, self.cfg.alpha_overlap
            )
            self.canvas_invalid_mask = invert_canvas_valid_mask(self.canvas_mask)
            # self.canvas_feats = extract_canvas_features(...) # Expensive, skipped in high-speed loop usually
            
            # Transform current features to raw space for next iteration
            self.prev_feats = curr_feats
            if curr_feats is not None:
                # Shallow copy to modify keypoints
                self.prev_feats = curr_feats.copy()
                kps = curr_feats["keypoints"] # (B, N, 2)
                if kps.numel() > 0:
                    try:
                        inv_T = torch.linalg.inv(curr_stab_transform)
                        inv_T_batch = inv_T.unsqueeze(0)
                        if inv_T_batch.shape[0] != kps.shape[0]:
                            inv_T_batch = inv_T_batch.expand(kps.shape[0], -1, -1)
                            
                        # kornia.geometry.transform.transform_points might be missing or moved
                        # Manual implementation: P_out = H @ P_in
                        B_k, N_k, _ = kps.shape
                        ones = torch.ones((B_k, N_k, 1), device=kps.device, dtype=kps.dtype)
                        kps_homo = torch.cat([kps, ones], dim=2) # (B, N, 3)
                        
                        # inv_T_batch: (B, 3, 3). We want (B, N, 3) result.
                        # (B, N, 3) x (B, 3, 3)^T -> (B, N, 3)
                        kps_raw_homo = torch.matmul(kps_homo, inv_T_batch.transpose(1, 2))
                        kps_raw = kps_raw_homo[..., :2]
                        
                        self.prev_feats["keypoints"] = kps_raw
                    except RuntimeError:
                        # Inversion failed, keep as is (likely bad transform)
                        pass
        self.H_cum = H_cum_curr
        
        # === 12. Reset Logic ===
        shear = shear_angle_from_homography(H_cum_curr)
        rot = rotate_angle_from_homography(H_cum_curr)
        scale = scale_factor_from_homography(H_cum_curr)

        reset = (shear > getattr(self.cfg, "reset_shear_angle", 15.0) or
                 rot > getattr(self.cfg, "reset_rotate_angle", 15.0) or
                 scale > getattr(self.cfg, "reset_scale_factor", 2.0))

        if reset:
            H_cum_curr = H_cum_curr.to(torch.float32)
            self.canvas, self.canvas_mask, self.offset_xy = reset_canvas_orientation(
                self.canvas, self.canvas_mask, H_cum_curr, frame_u.shape[-2:], self.cfg, self.offset_xy
            )
            self.H_cum = torch.eye(3, device=self.device)
            self.canvas_invalid_mask = invert_canvas_valid_mask(self.canvas_mask)
        
        # === 13. Update State ===
        self.prev_flow_gray_raw = curr_flow_gray_raw
        self.prev_flow_gray_stab = curr_flow_gray_stab
        self.prev_rgb_raw = frame_u.clone()
        self.prev_rgb_stab = frame_u_stab.clone()
        fallback_raw = merge_masks(tool_mask_raw, depth_mask_raw)
        fallback_stab = merge_masks(tool_mask_stab, depth_mask_stab)
        self.combined_mask_prev_raw = combined_mask_raw if combined_mask_raw is not None else fallback_raw
        self.combined_mask_prev_stab = combined_mask_stab if combined_mask_stab is not None else fallback_stab
        self.prev_stab_transform = curr_stab_transform

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

    def compute_flow(self, prev_tensor: torch.Tensor, curr_tensor: torch.Tensor):
        """
        NeuFlow2推論の実行
        prev_tensor, curr_tensor: torch.Size([1, 3, 1080, 1920]), torch.Size([1, 3, 1080, 1920])
        Return: flow (H, W, 2) : (1080, 1920, 2)
        """

        # 前処理とスタック
        img1 = self._preprocess(prev_tensor)
        img2 = self._preprocess(curr_tensor)
        
        # (1, 2, 3, H, W) の形式を作成
        images = torch.stack([img1, img2], dim=1)
        
        inputs = {"images": images}
        
        with torch.no_grad():
            preds = self.flow_model(inputs)
            
        # Flow取得: (1, 1, 2, H, W) -> (H, W, 2)
        # 処理を一貫させるためGPU上のTensorのまま返す
        flow = preds["flows"].squeeze(0).squeeze(0).permute(1, 2, 0)
        return flow
    def _morphology_close_open(self, mask: torch.Tensor, kernel_size: int = 5):
        """
        korniaを使用したモルフォロジー演算 (Closing -> Opening)
        mask: (H, W) or (1, H, W) range [0, 1] (float or bool)
        """
        # Korniaは (B, C, H, W) の形状を期待するため整形
        if mask.dim() == 2:
            m = mask.view(1, 1, mask.shape[0], mask.shape[1]).float()
        elif mask.dim() == 3:
            m = mask.unsqueeze(0).float()
        else:
            m = mask.float()

        # カーネルの作成 (すべて1ならRect形状になります)
        kernel = torch.ones(kernel_size, kernel_size, device=mask.device)

        m_closed = kornia.morphology.closing(m, kernel)
        m_opened = kornia.morphology.opening(m_closed, kernel)

        # 元の (H, W) 形状に戻して bool 化
        return m_opened.squeeze() > 0.5
    def estimate_camera_shift(self, prev_img: torch.Tensor, curr_img: torch.Tensor, base_mask: torch.Tensor = None):
        """
        カメラのシフト量を推定 (Torch完結版)
        Return: (dx, dy) as float tuple
        """
        if prev_img is None or curr_img is None:
            return (0.0, 0.0)

        flow = self.compute_flow(prev_img, curr_img)
        dx = flow[..., 0]
        dy = flow[..., 1]

        # マスク処理
        h, w = prev_img.shape[-2:]
        if base_mask is not None:
             if base_mask.dim() == 4:
                 mask_2d = base_mask[0, 0]
             elif base_mask.dim() == 3:
                 mask_2d = base_mask[0]
             else:
                 mask_2d = base_mask
             valid_mask = (mask_2d == 0)
        else:
             valid_mask = torch.ones((h, w), dtype=torch.bool, device=self.device)

        valid_mask &= torch.isfinite(dx) & torch.isfinite(dy)
    
        if not valid_mask.any():
             return (0.0, 0.0)
        med_dx = dx[valid_mask].median()
        med_dy = dy[valid_mask].median()
        
        return (med_dx, med_dy)

    def compute_optical_flow_mask(self, prev_img: torch.Tensor, curr_img: torch.Tensor, base_mask: torch.Tensor = None):
        """
        オプティカルフローとモーションマスクの計算 (Torch完結版)
        """
        # stats は呼び出し元で使われていないため、item()呼び出しを含む計算をスキップ
        stats = {} 

        if prev_img is None or curr_img is None:
            h, w = (0, 0)
            if prev_img is not None: h, w = prev_img.shape[-2:]
            return torch.zeros((h, w), dtype=torch.uint8, device=self.device), (0.0, 0.0), None, stats

        # Flow推論
        flow = self.compute_flow(prev_img, curr_img)  # (H, W, 2)
        dx = flow[..., 0]
        dy = flow[..., 1]
        mag = torch.sqrt(dx**2 + dy**2)

        h, w = prev_img.shape[-2:]
        
        # マスク作成
        if base_mask is not None:
            if base_mask.dim() == 4:
                mask_2d = base_mask[0, 0]
            elif base_mask.dim() == 3:
                mask_2d = base_mask[0]
            else:
                mask_2d = base_mask
            valid_mask = (mask_2d == 0)
        else:
            valid_mask = torch.ones((h, w), dtype=torch.bool, device=self.device)

        valid_mask &= torch.isfinite(dx) & torch.isfinite(dy)
        
        # item() 回避のため比率チェックは省略、またはTensorのまま比較するが
        # ここでは後続の処理に必要なため簡略化
        if not valid_mask.any():
            zero_mask = torch.zeros((h, w), dtype=torch.uint8, device=self.device)
            return zero_mask, (0.0, 0.0), flow, stats

        dx_valid = dx[valid_mask]
        dy_valid = dy[valid_mask]
        
        med_dx = dx_valid.median()
        med_dy = dy_valid.median()

        residual = torch.sqrt((dx - med_dx) ** 2 + (dy - med_dy) ** 2)
        
        residual_valid = residual[valid_mask]
        mag_valid = mag[valid_mask]

        med_res = residual_valid.median()
        mad_res = (residual_valid - med_res).abs().median() + 1e-6
        res_thresh = med_res + self.cfg.optflow_residual_factor * mad_res

        med_mag = mag_valid.median()
        mad_mag = (mag_valid - med_mag).abs().median() + 1e-6
        mag_thresh = med_mag + self.cfg.optflow_magnitude_factor * mad_mag

        raw_mask = ((residual > res_thresh) | (mag > mag_thresh)).detach().cpu().numpy().astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel)
        raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel)
        motion_mask = torch.from_numpy(raw_mask).to(self.device)
        shift = (med_dx, med_dy)
        return motion_mask, shift, flow, stats
    
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
