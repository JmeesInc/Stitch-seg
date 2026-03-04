import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple

from .models import (
    load_masking_model,
    load_masking_model2,
)
from .lightglue_dynamo import ALIKED, LightGlue
from .config import apply_stitch_defaults, build_default_cfg

try:
    from .cuda_ops import fused_warp_perspective as _cuda_warp, fast_gradient_mask_cuda as _cuda_grad_mask, is_available as _cuda_ops_available
    _USE_CUDA_OPS = _cuda_ops_available()
except ImportError:
    _USE_CUDA_OPS = False
    print("[warn] CUDA ops not available")

class Stitcher_ONNX(nn.Module):
    """Stateful stitching engine that exposes stitched crops to a seg model.

    Flow summary:
      1. Preprocess + crop each raw frame using `compute_static_roi`.
      2. Obtain tool/port masks to build clean canvases.
      4. For every frame, estimate camera shift, paste onto both canvases,
         and cache the homography so predictions can be warped back.
      5. When `model_inference` is called, crop the mask-free canvas around
         the current frame footprint, run the seg model, and warp the output
         into the current frame space.
    """

    def __init__(self, cfg=None, input_size=(480, 854)):
        super().__init__()
        if cfg is None:
            cfg = build_default_cfg()
        else:
            cfg = apply_stitch_defaults(cfg)
        self.cfg = cfg
        self.device = cfg.device
        self.apply_ellipse_mask = bool(getattr(cfg, "apply_ellipse_mask", True))
        self.masking_model = load_masking_model(cfg)
        self.masking_model2 = load_masking_model2(cfg)
        self.masking_input_size = (512, 512)
        self.masking_input_mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.masking_input_std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        self.extractor = ALIKED(
            model_name="aliked-n16",
            device="cuda",
            top_k=2048,
            scores_th=0.2,
            n_limit=2048,
        )
        self.matcher = LightGlue(
            weights="weights/aliked_lightglue_v0-1_arxiv.pth",
            depth_confidence=0.95,
            width_confidence=0.99,
            filter_threshold=0.1,
        ).to("cuda").eval()
        h0, w0 = input_size
        self.corners = torch.tensor([[0.0, 0.0], [w0, 0.0], [w0, h0], [0.0, h0]], device=self.device)
        self.pts_scale = torch.tensor([w0, h0], device=self.device)
        # Canvas geometry:
        # - Canvas resolution is scaled by `canvas_superres_scale`
        # - We apply the scale AFTER translation: p_canvas = S @ T @ p_raw
        #   so `T` is defined in the *pre-scale (raw) coordinate system*.
        scale = torch.tensor(float(getattr(self.cfg, "canvas_superres_scale", 1.0)), dtype=torch.float32)
        sx = torch.tensor(float(getattr(self.cfg, "canvas_scale_x", 3.0)), dtype=torch.float32)
        sy = torch.tensor(float(getattr(self.cfg, "canvas_scale_y", 3.0)), dtype=torch.float32)
        if scale <= 0:
            scale = 1.0

        self.canvas_h = torch.tensor(sy * h0 * scale).to(torch.int32)
        self.canvas_w = torch.tensor(sx * w0 * scale).to(torch.int32)

        # Center the (scaled) frame footprint in the (scaled) canvas.
        # Condition (x): scale * (w0/2 + off_x) == canvas_w/2
        off_x = (self.canvas_w / (2.0 * scale)) - (w0 / 2.0)
        off_y = (self.canvas_h / (2.0 * scale)) - (h0 / 2.0)
        self.T = translation_matrix_from_offset((off_x, off_y), device=self.device)

        self.S = torch.eye(3, device=self.device, dtype=torch.float32)
        self.S[0, 0] = scale
        self.S[1, 1] = scale
        self.last_model_input = None
        self.last_model_output = None
        self.thresh_shear = torch.tensor(float(getattr(cfg, "reset_shear_angle", 15.0)), device=self.device)
        self.thresh_rot   = torch.tensor(float(getattr(cfg, "reset_rotate_angle", 15.0)), device=self.device)
        self.thresh_scale = torch.tensor(float(getattr(cfg, "reset_scale_factor", 2.0)), device=self.device)

        self.trim_px = 12 #max(0, int(getattr(cfg, "canvas_border_trim_px", 0)))
        self.paste_alpha_radius = int(getattr(cfg, "gradient_radius", 201))

        self.pyr_kernel = get_gaussian_kernel_5x5(self.device)
    
    def _predict_masks(self, frame_u: torch.Tensor, ellipse_mask: torch.Tensor):
        """
        input: (1, 3, H, W) RGB uint8/float
        output: tool_mask: (H, W) uint8 (device on self.device)
                depth_mask: (H, W) uint8 (device on self.device)
        """
        input_img = F.interpolate(
            frame_u / 255.0, 
            size=self.masking_input_size, 
            mode='bilinear', 
            align_corners=False
        )
        # 2. 正規化 (ブロードキャストが正しく機能する)
        input_img = (input_img - self.masking_input_mean) / self.masking_input_std
        masking = self.masking_model(input_img)
        masking2 = self.masking_model2(input_img)
        
        masking = F.interpolate(masking, size=frame_u.shape[-2:], mode='bilinear', align_corners=False)
        masking2 = F.interpolate(masking2, size=frame_u.shape[-2:], mode='bilinear', align_corners=False)
        masking_bool = (masking > 0.5) | (masking2 > 0.5)

        ellipse_bool = F.interpolate(
            ellipse_mask.float(), size=frame_u.shape[-2:], mode='nearest'
        ) > 0.5
        masking_bool = masking_bool | ellipse_bool

        return masking_bool.to(torch.uint8) * 255
    
    def first_frame(self, frame_u, ellipse_mask):
        print("canvas is initialized")
        combined_mask_raw = self._predict_masks(frame_u, ellipse_mask)
        # Features
        h0, w0 = frame_u.shape[-2:]
        canvas = torch.zeros((1, 3, self.canvas_h.item(), self.canvas_w.item()), dtype=torch.float32, device=self.device)
        canvas_mask = torch.zeros((1, 1, self.canvas_h.item(), self.canvas_w.item()), dtype=torch.uint8, device=self.device)
        curr_kps, curr_desc, _ = self.extractor(frame_u / 255.0) # 想定: forwardでextract)
        curr_kps = torch.stack(curr_kps, dim=0)
        curr_desc = torch.stack(curr_desc, dim=0)
        curr_kps, curr_desc = filter_features_by_mask(curr_kps, curr_desc, combined_mask_raw, h0, w0)
        curr_kps = F.pad(curr_kps, (0, 0, 0, 2048 - curr_kps.shape[1]), mode='constant', value=-2.0)
        curr_desc = F.pad(curr_desc, (0, 0, 0, 2048 - curr_desc.shape[1]), mode='constant', value=0.0)
        H_cum = torch.eye(3, device=self.device)
        # Paste
        canvas, canvas_mask = self.paste_current_to_canvas_forward_multiband(
            canvas, canvas_mask, H_cum, self.T, self.S, frame_u, combined_mask_raw
        )
        return canvas, canvas_mask, H_cum, curr_kps, curr_desc
    
    def forward(self, frame_u: torch.Tensor, ellipse_mask: torch.Tensor, H_cum: torch.Tensor, prev_keypoints: torch.Tensor, prev_descriptors: torch.Tensor, canvas: torch.Tensor, canvas_mask: torch.Tensor):
        """
        Main logic step.
        frame_u: (1, 3, H, W) RGB Float
        ellipse_mask: (1, 1, H, W) Float
        """
        # === 1. Initialization ===
        # === 2. Predict Masks ===
        combined_mask_raw = self._predict_masks(frame_u, ellipse_mask)
        self.combined_mask_prev_raw = combined_mask_raw
        # === 3. Feature Extraction ===
        curr_kps, curr_desc, _ = self.extractor(frame_u / 255.0)
        curr_kps = torch.stack(curr_kps, dim=0).detach()
        curr_desc = torch.stack(curr_desc, dim=0).detach()
        curr_kps, curr_desc = filter_features_by_mask(curr_kps, curr_desc, combined_mask_raw, frame_u.shape[-2], frame_u.shape[-1])
            
        # === 4. Global Homography ===
        kpts = pad_and_cat(prev_keypoints, curr_kps, padding_value=-2.0)# (2B, N, 2), normalized
        descs = pad_and_cat(prev_descriptors, curr_desc)# (2B, N, D)
        matches, scores = self.matcher(kpts, descs) # (N, 2), (N,) fixed-size

        # scores > 0 のエントリのみが有効なマッチ
        n_valid = (scores > 0).sum()
        needs_reset = (n_valid < 4)

        # ポイント抽出 (GPU上のTensorのまま)
        kp_prev = prev_keypoints[0]
        kp_curr = curr_kps[0]

        # pad_and_cat で埋めた領域へのマッチが混ざることがあるため除外する
        # NonZero を使わず weight=0 でマスクする
        n_prev = kp_prev.shape[0]
        n_curr = kp_curr.shape[0]
        valid = (
            (matches[:, 0] >= 0) & (matches[:, 0] < n_prev) &
            (matches[:, 1] >= 0) & (matches[:, 1] < n_curr)
        )
        scores = scores * valid.float()  # 無効エントリの重みを0に

        # 安全なインデックス（無効エントリはclampで0に、weight=0で結果に影響なし）
        safe_idx0 = matches[:, 0].clamp(0, n_prev - 1)
        safe_idx1 = matches[:, 1].clamp(0, n_curr - 1)

        pts_prev = kp_prev[safe_idx0]  # (N, 2)
        pts_curr = kp_curr[safe_idx1]  # (N, 2)

        # unormalize pts_prev and pts_curr from [-1, 1] to pixel coordinates
        pts_prev = (pts_prev + 1) / 2 * self.pts_scale
        pts_curr = (pts_curr + 1) / 2 * self.pts_scale

        H_est = find_homography_dlt_onnx(pts_curr, pts_prev, w=scores).squeeze(0)
        needs_reset = needs_reset | H_est.isinf().any()
        # needs_reset 時は単位行列で代用し、分岐なしで処理を続行
        eye3 = torch.eye(3, device=self.device, dtype=H_cum.dtype)
        H_est_safe = torch.where(needs_reset, eye3, H_est)
        H_cum = torch.where(needs_reset, eye3, H_cum @ H_est_safe)

        # === 8. Paste Current Frame to Canvas (update canvas) ===
        blur = laplacian_var(frame_u.float()) < self.cfg.laplacian_var_min
        blur_bool = blur.to(torch.bool)

        # Keep inputs immutable for branch selection (blur/non-blur) using tensor ops.
        canvas_in = canvas.clone()
        canvas_mask_in = canvas_mask.clone()
        new_canvas, new_canvas_mask = self.paste_current_to_canvas_forward_multiband(
            canvas_in,
            canvas_mask_in,
            H_cum,
            self.T,
            self.S,
            frame_u,
            combined_mask_raw,
            blur=blur_bool,
        )
        canvas = torch.where(blur_bool, canvas, new_canvas)
        canvas_mask = torch.where(blur_bool, canvas_mask, new_canvas_mask)
        
        # Transform current features to raw space for next iteration
        prev_keypoints = kpts[1:]
        prev_descriptors = descs[1:]
        # === 9. Reset Logic === # degreeからradianにしてnumpy消したい
        H_cum = H_cum.to(torch.float32)
        reset_canvas, reset_canvas_mask = reset_canvas_orientation(
            canvas, canvas_mask, H_cum, self.T, self.S
        )

        reset = self.get_reset_condition(H_cum).to(torch.bool)
        apply_reset = reset & (~blur_bool)
        eye3 = torch.eye(3, device=self.device, dtype=H_cum.dtype)
        canvas = torch.where(apply_reset, reset_canvas, canvas)
        canvas_mask = torch.where(apply_reset, reset_canvas_mask, canvas_mask)
        H_cum = torch.where(apply_reset, eye3, H_cum)

        return canvas, canvas_mask, H_cum, prev_keypoints, prev_descriptors, needs_reset

    def get_reset_condition(self, H: torch.Tensor) -> torch.Tensor:
        """
        ホモグラフィ行列からリセット判定を Tensor (bool) で一括計算する。
        QR分解/SVDを閉形式の計算に置き換え、ONNXエクスポートに対応。
        Input: H [3, 3] or [B, 3, 3]
        Output: bool Tensor [1] or [B]
        """
        device = H.device
        dtype = torch.float32
        H = H.to(dtype)
        H = H.unsqueeze(0) # [1, 3, 3]

        # 1. 準備：ゼロ除算を避けるための safe_denom
        # H[..., 2, 2] でバッチ対応
        denom = H[..., 2, 2]
        is_denom_zero = denom.abs() < 1e-8
        safe_denom = torch.where(is_denom_zero, torch.ones_like(denom), denom)
        
        # 2x2 行列の抽出と正規化 [B, 2, 2]
        # ブロードキャストのために safe_denom を [B, 1, 1] に変形
        A = H[..., :2, :2] / safe_denom.view(-1, 1, 1)

        # --- Rotation Angle (degrees) ---
        # rot = atan2(H[1,0], H[0,0])
        rot_rad = atan2_onnx(H[..., 1, 0], H[..., 0, 0]).abs()
        rot_deg = rad2deg_onnx(rot_rad)

        # --- Shear Angle (degrees) ---
        # QR分解(Gram-Schmidt)の解析解:
        # A = [a0, a1], R01 = (a0 . a1) / ||a0||, R11 = sqrt(||a1||^2 - R01^2)
        # Shear k = R01 / R11
        
        a0 = A[..., :, 0] # [B, 2] col0
        a1 = A[..., :, 1] # [B, 2] col1
        
        norm_a0_sq = (a0 ** 2).sum(dim=-1) # ||a0||^2
        norm_a1_sq = (a1 ** 2).sum(dim=-1) # ||a1||^2
        dot_a0_a1  = (a0 * a1).sum(dim=-1) # a0 . a1
        
        eps = 1e-8
        r00 = torch.sqrt(norm_a0_sq + eps)
        r01 = dot_a0_a1 / r00
        
        # R11^2 = ||a1||^2 - R01^2
        r11_sq = norm_a1_sq - r01**2
        r11 = torch.sqrt(torch.clamp(r11_sq, min=torch.tensor(eps, device=self.device)))
        
        k = r01 / (r11 + eps)
        shear_deg = rad2deg_onnx(torch.atan(k).abs())
        
        # H[2, 2] が 0 の場合は 0.0 とする
        shear_deg = torch.where(is_denom_zero, torch.tensor(0.0, device=device), shear_deg)

        # --- Scale Factor ---
        # SVD特異値の解析解:
        # 特異値^2 は A^T A の固有値。2x2固有値は解の公式で計算可能。
        # Trace = sum(A^2), Det = (det(A))^2
        
        trace_M = norm_a0_sq + norm_a1_sq
        det_A = a0[..., 0] * a1[..., 1] - a0[..., 1] * a1[..., 0]
        det_M = det_A ** 2
        
        # 判別式 D = sqrt(Tr^2 - 4*Det)
        discriminant = torch.sqrt(torch.clamp(trace_M**2 - 4*det_M, min=torch.tensor(0.0, device=self.device)))
        
        # 固有値 lambda
        eig_max = (trace_M + discriminant) / 2.0
        eig_min = (trace_M - discriminant) / 2.0
        
        # 特異値 sigma = sqrt(lambda)
        scale_major = torch.sqrt(torch.clamp(eig_max, min=torch.tensor(eps, device=self.device)))
        scale_minor = torch.sqrt(torch.clamp(eig_min, min=torch.tensor(eps, device=self.device)))
        
        inv_minor = 1.0 / torch.clamp(scale_minor, min=torch.tensor(eps, device=self.device))
        scale_res = torch.max(scale_major, inv_minor)
        
        # scale_minor が小さすぎる場合は scale_major を採用
        scale_factor = torch.where(scale_minor <= eps, scale_major, scale_res)
        scale_factor = torch.where(is_denom_zero, torch.tensor(1.0, device=device), scale_factor)

        # --- Final Logic (Bitwise OR) ---
        reset_bool = (shear_deg > self.thresh_shear) | \
                     (rot_deg > self.thresh_rot)     | \
                     (scale_factor > self.thresh_scale)

        return reset_bool
    
    def paste_current_to_canvas_forward_multiband(self,
        canvas, canvas_mask, H_to_canvas, T, S, current_img, tool_mask,
        blur=torch.tensor(False),
    ):
        """
        Multiband Blending with Mask Softening (Erosion + Gaussian Blur).
        """
        device = current_img.device
        
        # --- 1. Preparation & Warping ---
        H_total = S @ T @ H_to_canvas.to(device)
        
        img_to_warp = current_img.clone()
        
        # Mask out tool in input image to prevent bleeding: tool_mask -> (H, W) uint8
        resized_mask = tool_mask[0, 0].to(torch.uint8)
        img_to_warp.masked_fill_(tool_mask > 0, 0)

        ch, cw = canvas.shape[-2:]

        # Warp Masks (geometry mask + tool mask)
        mask = torch.ones(current_img.shape[-2:], dtype=torch.float32, device=device)      # stitch用
        mask4model = torch.ones(current_img.shape[-2:], dtype=torch.float32, device=device) # 推論用
        edge_trim_mask = torch.ones_like(mask)
        mask[resized_mask > 0] = 0
        mask4model[resized_mask > 0] = 0
        edge_trim_mask[:self.trim_px, :] = 0; edge_trim_mask[-self.trim_px:, :] = 0; edge_trim_mask[:, :self.trim_px] = 0; edge_trim_mask[:, -self.trim_px:] = 0
        
        mask = mask * edge_trim_mask
        mask_b = mask.unsqueeze(0).unsqueeze(0)
        mask4model_b = mask4model.unsqueeze(0).unsqueeze(0)

        # ---- Batch warp (image + geometry mask + tool mask) to reduce overhead ----
        # All share H_total, dsize and nearest mode -> pack into channels and warp once.
        rm_b = resized_mask.float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

        c_img = img_to_warp.shape[1]
        packed = torch.cat([img_to_warp.float(), mask_b, mask4model_b, rm_b], dim=1)  # (1, C+3, H, W)
        warped_packed = warp_perspective_onnx(
            packed, (H_total.unsqueeze(0) + torch.eye(3, device=device)*1e-6), dsize=(ch, cw), mode='nearest', padding_mode='zeros'
        )
        warped = warped_packed[:, :c_img].to(canvas.dtype)
        warped_mask = warped_packed[:, c_img:c_img + 1] 
        warped_tool_mask = warped_packed[:, c_img + 2:c_img + 3]

        # Logic Masks (Binary)
        wm_bool = warped_mask > 0.5
        cm_bool = canvas_mask > 0
        wtm_bool = warped_tool_mask > 0.5
        
        valid_new_region = wm_bool & (~wtm_bool)
        
        overlap = valid_new_region & cm_bool
        only_new = valid_new_region & (~cm_bool)
        blur_bool = blur.to(torch.bool)
        blur_bool = blur_bool.to(torch.bool)
        overlap_case = overlap.any().to(torch.bool)

        # Base update (non-overlap/only-new) is always safe to compute.
        canvas_base = torch.where(only_new, warped, canvas)
        canvas_mask_base = torch.where(valid_new_region, torch.full_like(canvas_mask, 255), canvas_mask)

        # --- 2. Blend overlap region on full canvas (no dynamic Slice/ScatterND) ---
        # Compute gradient mask on the full valid_new_region.
        # inv_mask is naturally 1.0 outside the valid region, so border handling is implicit.
        roi_mask_full = valid_new_region.float()
        inv_mask_full = 1.0 - roi_mask_full

        if _USE_CUDA_OPS and not torch.onnx.is_in_onnx_export():
            dist_out = _cuda_grad_mask(inv_mask_full, self.paste_alpha_radius, 4)
        else:
            dist_out = self.fast_gradient_mask(inv_mask_full)
        grad_in = 1.0 - dist_out
        update_weight = torch.clamp(grad_in * roi_mask_full, torch.tensor(0.0, device=self.device), torch.tensor(1.0, device=self.device))

        # Apply blending on the full canvas (weight is 0 outside valid region → no change there)
        canvas_nb = canvas_base * (1.0 - update_weight) + warped.float() * update_weight
        canvas_nb = canvas_nb.to(canvas_base.dtype)
        canvas_nb = torch.where(overlap_case, canvas_nb, canvas_base)

        # Branch selection (blur / non-blur) as tensor ops.
        canvas_out = torch.where(blur_bool, canvas_base, canvas_nb)
        canvas_mask_out = canvas_mask_base
        return canvas_out, canvas_mask_out
    
    def fast_gradient_mask(self, mask: torch.Tensor) -> torch.Tensor:
        h, w = mask.shape[-2:]
        mask = (mask > 0.5).float()

        S = 4  # downscale factor
        R_small = self.paste_alpha_radius // S  # 201 // 4 = 50

        # Downscale: AvgPool gives area-based downscale (CUDA-supported)
        mask_small = F.avg_pool2d(mask, kernel_size=S, stride=S)
        mask_small = (mask_small > 0.3).float()

        current_mask = mask_small
        accum = torch.zeros_like(mask_small)

        for _ in range(R_small):
            current_mask = F.max_pool2d(current_mask, kernel_size=3, stride=1, padding=1)
            accum += current_mask

        grad_mask = accum / float(R_small)

        # Upscale back to original resolution (bilinear gives smooth gradient)
        grad_mask = F.interpolate(grad_mask, size=(h, w), mode='bilinear', align_corners=False)

        return torch.clamp(grad_mask, torch.tensor(0.0, device=self.device), torch.tensor(1.0, device=self.device))
def filter_features_by_mask(kps: torch.Tensor, desc: torch.Tensor, invalid_mask: torch.Tensor, H: int, W: int):
    """
    invalid_mask: (B, 1, H, W) tensor where True/nonzero is invalid.
    Keeps features where mask is False/0 (valid region).
    kps: keypoints in normalized coordinates [-1, 1] or pixel coordinates

    Fixed-size version: instead of boolean indexing (NonZero),
    set invalid keypoints to -2.0 and descriptors to 0.0 in-place.
    """
    kps_x = kps[0, :, 0]
    kps_y = kps[0, :, 1]
    # 正規化座標 [-1, 1] からピクセル座標に変換
    xs = ((kps_x + 1) / 2 * W).round().long()
    ys = ((kps_y + 1) / 2 * H).round().long()

    xs = torch.clamp(xs, torch.tensor(0, device=kps.device), torch.tensor(W - 1, device=kps.device))
    ys = torch.clamp(ys, torch.tensor(0, device=kps.device), torch.tensor(H - 1, device=kps.device))

    # invalid_mask が True/nonzero の場所は無効領域なので、False/0 の場所をキープ
    mask_vals = invalid_mask[0, 0, ys, xs]
    keep = (mask_vals == 0)  # [N]
    keep_f = keep.float().unsqueeze(0).unsqueeze(-1)  # [1, N, 1]
    # 無効なキーポイントはパディング値(-2.0)に、記述子は0にマスク
    kps = kps * keep_f + (-2.0) * (1.0 - keep_f)
    desc = desc * keep_f
    return kps, desc

def reset_canvas_orientation(canvas: torch.Tensor, canvas_mask: torch.Tensor, H_to_canvas: torch.Tensor, T: torch.Tensor, S: torch.Tensor):
    """
    Warp canvas back.
    canvas: (B, C, H, W)
    """
    cw, ch = canvas.shape[-1], canvas.shape[-2]
    device = canvas.device
    
    # 1. Old Transform (Current -> Old Canvas)
    # P_old = S @ T_old
    P = S @ T
    
    # H_total_old = P_old @ H_to_canvas
    H_total_old = P @ H_to_canvas
    H_old_inv = inverse_3x3_onnx(H_total_old.unsqueeze(0))
    M = P @ H_old_inv
    
    # 4. Warp

    packed = torch.cat([canvas, canvas_mask], dim=1).float()
    if _USE_CUDA_OPS and not torch.onnx.is_in_onnx_export():
        warped_packed = _cuda_warp(packed, M, ch, cw, mode=0).to(canvas.dtype)
    else:
        warped_packed = warp_perspective_onnx(
            packed, (M + torch.eye(3, device=device)*1e-6), dsize=(ch, cw), mode='nearest', padding_mode='zeros'
        ).to(canvas.dtype)

    canvas_chs = canvas.shape[1]
    mask_chs = canvas_chs +canvas_mask.shape[1]

    warped_canvas = warped_packed[:, :canvas_chs]
    warped_mask = warped_packed[:, canvas_chs:mask_chs]
    warped_mask = (warped_mask > 0).to(torch.uint8) * 255
    
    return warped_canvas, warped_mask

def inverse_3x3_onnx(M: torch.Tensor) -> torch.Tensor:
    """
    ユーザー様のinvmatロジックを3x3に特化してベクトル化した実装。
    ループを使わないため、ONNXのTracerWarningを回避し、高速に動作します。
    M: [B, 3, 3] or [3, 3]
    """
    
    # 各要素の抽出
    a11, a12, a13 = M[:, 0, 0], M[:, 0, 1], M[:, 0, 2]
    a21, a22, a23 = M[:, 1, 0], M[:, 1, 1], M[:, 1, 2]
    a31, a32, a33 = M[:, 2, 0], M[:, 2, 1], M[:, 2, 2]

    # 行列式の計算 (det)
    det = (a11 * (a22 * a33 - a23 * a32) -
           a12 * (a21 * a33 - a23 * a31) +
           a13 * (a21 * a32 - a22 * a31))

    # 余因子行列の各要素 (alcof / adj)
    # 転置を考慮して直接代入
    inv = torch.stack([
        torch.stack([a22 * a33 - a23 * a32, a13 * a32 - a12 * a33, a12 * a23 - a13 * a22], dim=-1),
        torch.stack([a23 * a31 - a21 * a33, a11 * a33 - a13 * a31, a13 * a21 - a11 * a23], dim=-1),
        torch.stack([a21 * a32 - a22 * a31, a12 * a31 - a11 * a32, a11 * a22 - a12 * a21], dim=-1)
    ], dim=1)

    return inv / det.unsqueeze(-1).unsqueeze(-1)

def _weighted_normalize_points(
    pts: torch.Tensor, w: torch.Tensor, eps: float = 1e-8
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Weighted Hartley normalization.
    Args:
        pts: [B, N, 2] float32 on CUDA
        w  : [B, N] non-negative weights
    Returns:
        pts_n: [B, N, 2] normalized points
        T    : [B, 3, 3] similarity transform s.t. [x_n,y_n,1]^T = T [x,y,1]^T
    """
    B, N, _ = pts.shape
    w = w.clamp_min(torch.tensor(0.0, device=pts.device))
    w_sum = w.sum(dim=1, keepdim=True).clamp_min(torch.tensor(eps, device=pts.device))  # [B,1]

    # weighted centroid
    cx = (w * pts[..., 0]).sum(dim=1, keepdim=True) / w_sum  # [B,1]
    cy = (w * pts[..., 1]).sum(dim=1, keepdim=True) / w_sum  # [B,1]
    c = torch.cat([cx, cy], dim=1)  # [B,2]

    # weighted mean distance to centroid
    dxy = pts - c.unsqueeze(1)  # [B,N,2]
    dist = torch.sqrt((dxy * dxy).sum(dim=2) + eps)  # [B,N]
    mean_dist = (w * dist).sum(dim=1, keepdim=True) / w_sum  # [B,1]
    mean_dist = mean_dist.clamp_min(torch.tensor(eps, device=pts.device))

    s = (torch.sqrt(torch.tensor(2.0, device=pts.device, dtype=pts.dtype)) / mean_dist)  # [B,1]
    s = s.squeeze(1)  # [B]

    # Build T
    T = torch.zeros((B, 3, 3), device=pts.device, dtype=pts.dtype)
    T[:, 0, 0] = s
    T[:, 1, 1] = s
    T[:, 0, 2] = -s * c[:, 0]
    T[:, 1, 2] = -s * c[:, 1]
    T[:, 2, 2] = 1.0

    # Apply T to points: x_n = s*(x-cx), y_n = s*(y-cy)
    pts_n = dxy * s.view(B, 1, 1)

    return pts_n, T


def find_homography_dlt_onnx(
    pts0: torch.Tensor,
    pts1: torch.Tensor,
    w: torch.Tensor,
    eps: float = 1e-8,
    damping: float = 1e-6,
) -> torch.Tensor:
    """
    Weighted homography estimation (no RANSAC), ONNX-friendly torch-only implementation.

    Solves for H (3x3) from correspondences pts0 -> pts1 using weighted linear least squares
    with constraint h33 = 1 (8 unknowns).
    Uses weighted point normalization for stability, then denormalizes.

    Args:
        pts0: [N,2] or [B,N,2] (float) on CUDA
        pts1: [N,2] or [B,N,2] (float) on CUDA
        w   : [N]   or [B,N]   (float) non-negative weights on CUDA
        eps: small constant
        damping: Tikhonov regularization for the normal equation (improves stability)

    Returns:
        H: [B,3,3] (or [3,3] if input was unbatched)
    """
    # --- shape normalize to batched ---
    unbatched = True
    pts0 = pts0.unsqueeze(0)
    pts1 = pts1.unsqueeze(0)
    w = w.unsqueeze(0)
    B, N, _ = pts0.shape

    # clamp weights, avoid all-zero
    w = w.clamp_min(torch.tensor(0.0, device=pts0.device))
    w_sum = w.sum(dim=1, keepdim=True).clamp_min(torch.tensor(eps, device=pts0.device))
    w = w / w_sum  # scale-invariant (optional but helps conditioning)

    # --- weighted normalization ---
    p0n, T0 = _weighted_normalize_points(pts0, w, eps=eps)  # [B,N,2], [B,3,3]
    p1n, T1 = _weighted_normalize_points(pts1, w, eps=eps)

    x = p0n[..., 0]  # [B,N]
    y = p0n[..., 1]
    u = p1n[..., 0]
    v = p1n[..., 1]

    ones = torch.ones((B, N), device=pts0.device, dtype=pts0.dtype)
    zeros = torch.zeros((B, N), device=pts0.device, dtype=pts0.dtype)

    M1 = torch.stack([x, y, ones, zeros, zeros, zeros, -u * x, -u * y], dim=2)  # [B,N,8]
    M2 = torch.stack([zeros, zeros, zeros, x, y, ones, -v * x, -v * y], dim=2)  # [B,N,8]
    rhs1 = u.unsqueeze(2)  # [B,N,1]
    rhs2 = v.unsqueeze(2)  # [B,N,1]

    M = torch.cat([M1, M2], dim=1)           # [B,2N,8]
    rhs = torch.cat([rhs1, rhs2], dim=1)     # [B,2N,1]

    # Apply weights: each correspondence contributes 2 equations with same weight
    w2 = torch.sqrt(w + eps)                 # [B,N]
    w_rows = torch.cat([w2, w2], dim=1).unsqueeze(2)  # [B,2N,1]
    Mw = M * w_rows                          # [B,2N,8]
    bw = rhs * w_rows                        # [B,2N,1]

    # Normal equation: (Mw^T Mw + λI) h = Mw^T bw
    Mt = Mw.transpose(1, 2)                  # [B,8,2N]
    AtA = Mt @ Mw                            # [B,8,8]
    Atb = Mt @ bw                            # [B,8,1]

    # Damping (Tikhonov)
    I = torch.eye(8, device=pts0.device, dtype=pts0.dtype).unsqueeze(0).expand(B, -1, -1)  # [B,8,8]
    AtA = AtA + damping * I

    # Solve
    # torch.linalg.solve is often exportable; your exporter/runtime chokes, swap to inv @ Atb.
    h = torch.matmul(invmat(AtA), Atb)        # [B,8,1]
    h = h.squeeze(2)                         # [B,8]

    # Assemble Hn (normalized)
    Hn = torch.zeros((B, 3, 3), device=pts0.device, dtype=pts0.dtype)
    Hn[:, 0, 0] = h[:, 0]
    Hn[:, 0, 1] = h[:, 1]
    Hn[:, 0, 2] = h[:, 2]
    Hn[:, 1, 0] = h[:, 3]
    Hn[:, 1, 1] = h[:, 4]
    Hn[:, 1, 2] = h[:, 5]
    Hn[:, 2, 0] = h[:, 6]
    Hn[:, 2, 1] = h[:, 7]
    Hn[:, 2, 2] = 1.0

    # Denormalize: H = inv(T1) @ Hn @ T0
    T1_inv = inverse_3x3_onnx(T1)
    H = T1_inv @ Hn @ T0

    # Scale so that H[2,2] == 1 (optional, nice for consistency)
    H22 = H[:, 2, 2].unsqueeze(1).unsqueeze(2).clamp_min(torch.tensor(eps, device=pts0.device))
    H = H / H22

    return H[0]

def invmat(M):
    """Inverse of batched positive-definite matrix via Gauss-Jordan elimination.

    No pivoting needed for positive definite matrices (guaranteed by
    Tikhonov regularization in find_homography_dlt_onnx).
    All ops (matmul, mul, div, sub) run on CUDA — eliminates 65 Det
    CPU ops and ~170 associated CPU ops from the cofactor approach.

    Numerically more stable than Faddeev-LeVerrier because each
    elimination step uses current matrix values directly without
    accumulating errors through matrix power iterations.
    """
    B, N, _ = M.shape
    I_mat = torch.eye(N, device=M.device, dtype=M.dtype).unsqueeze(0).expand_as(M)
    aug = torch.cat([M, I_mat], dim=2)  # [B, N, 2N]

    for k in range(N):
        # Normalize pivot row
        pivot = aug[:, k:k+1, k:k+1]  # [B, 1, 1]
        aug_k = aug[:, k:k+1, :] / pivot  # [B, 1, 2N]

        # Elimination factors for all rows (column k)
        factors = aug[:, :, k:k+1]  # [B, N, 1]

        # Mask: zero factor at pivot row k (don't subtract from itself)
        row_mask = torch.ones(N, device=M.device, dtype=M.dtype)
        row_mask[k] = 0.0
        row_mask = row_mask.view(1, N, 1)  # [1, N, 1]
        factors = factors * row_mask

        # Subtract factor * normalized_pivot_row from all non-pivot rows
        aug = aug - factors @ aug_k  # [B, N, 2N]

        # Replace pivot row with normalized version
        sel = 1.0 - row_mask  # [1, N, 1]: 1 at row k, 0 elsewhere
        aug = aug * row_mask + aug_k * sel

    return aug[:, :, N:]  # Right half is the inverse

def get_pixel_to_normalized_transform(H: int, W: int, device: torch.device, dtype: torch.dtype):
    """
    ピクセル座標 [0, W-1] x [0, H-1] を 正規化座標 [-1, 1] に変換する3x3行列を作成
    """
    transform = torch.eye(3, device=device, dtype=dtype).unsqueeze(0) # [1, 3, 3]
    
    transform[:, 0, 0] = 2.0 / (W - 1)
    transform[:, 0, 2] = -1.0
    transform[:, 1, 1] = 2.0 / (H - 1)
    transform[:, 1, 2] = -1.0
        
    return transform

def normalize_homography(M, src_size, dst_size):
    """
    ピクセル座標系のホモグラフィ行列 M (src -> dst) を
    正規化座標系の行列 M_norm (src_norm -> dst_norm) に変換する。
    Formula: M_norm = N_dst @ M @ inv(N_src)
    """
    B = M.shape[0]
    H_src, W_src = src_size
    H_dst, W_dst = dst_size
    device = M.device
    dtype = M.dtype

    # 正規化行列 N (pixel -> norm)
    N_src = get_pixel_to_normalized_transform(H_src, W_src, device, dtype) # [1, 3, 3]
    N_dst = get_pixel_to_normalized_transform(H_dst, W_dst, device, dtype) # [1, 3, 3]
    
    # N_src の逆行列 (norm -> pixel)
    # 対角行列+平行移動だけなので解析的に計算可能だが、簡単のためinverseを使う
    # ここではN_srcはバッチ1なので計算負荷は低い
    N_src_inv = inverse_3x3_onnx(N_src.expand(B, -1, -1))
    
    # M_norm = N_dst * M * N_src^-1
    # [B, 3, 3]
    M_norm = torch.matmul(torch.matmul(N_dst, M), N_src_inv)
    return M_norm

def create_meshgrid(H: int, W: int, device: torch.device):
    """
    [-1, 1] のメッシュグリッドを作成する (align_corners=True準拠)
    """
    xs = torch.linspace(-1, 1, W, device=device)
    ys = torch.linspace(-1, 1, H, device=device)
    # meshgrid args indexing='ij' is default for some versions, but we need xy behavior
    # torch.meshgrid behavior changed, so usually returns (y, x). 
    # We want (x, y) for grid.
    y, x = torch.meshgrid(ys, xs, indexing='ij')
    return torch.stack([x, y], dim=-1) # [H, W, 2]

def transform_points(trans_01, points_1):
    """
    正規化された点群にホモグラフィ行列を適用する
    Args:
        trans_01: [B, 3, 3] 変換行列
        points_1: [B, H, W, 2] 入力座標
    Returns:
        points_0: [B, H, W, 2] 変換後の座標 (透視除算済み)
    """
    B, H, W, _ = points_1.shape
    
    # [B, H, W, 3] (x, y, 1)
    points_1_h = torch.cat([points_1, torch.ones(B, H, W, 1, device=points_1.device, dtype=points_1.dtype)], dim=-1)
    
    # 行列演算のために形状変更: [B, H*W, 3]
    points_1_h_flat = points_1_h.view(B, -1, 3)
    
    # 変換: P_out = H @ P_in^T -> (B, 3, 3) @ (B, 3, N) -> (B, 3, N)
    # あるいは P_out^T = P_in @ H^T
    points_0_h_flat = torch.matmul(points_1_h_flat, trans_01.transpose(1, 2)) # [B, N, 3]
    
    # 透視除算 (x/z, y/z)
    # zが0に近い場合の安定性のため clamp を推奨
    eps = 1e-8
    z = points_0_h_flat[..., 2:3]
    # z = torch.where(torch.abs(z) < eps, torch.sign(z) * eps, z) # ONNXによっては条件分岐が重い場合がある
    z = z.clamp(min=torch.tensor(eps, device=points_1.device)) # 簡易的なゼロ除算回避(必要に応じて)

    xy = points_0_h_flat[..., 0:2] / z
    
    return xy.view(B, H, W, 2)

def warp_perspective_onnx(src, M, dsize, mode='nearest', padding_mode='zeros'):
    """
    warp_perspective のONNX対応版
    Args:
        src: [B, C, H, W]
        M: [B, 3, 3] Homography matrix (pixel space, src -> dst)
        dsize: (h_out, w_out)
        mode: 'bilinear' or 'nearest'
        padding_mode: 'zeros', 'border', or 'reflection'
    """
    B, _, H, W = src.size()
    h_out, w_out = dsize
    dst_norm_trans_src_norm = normalize_homography(M, (H, W), (h_out, w_out))  # [B, 3, 3]

    # 2. 逆行列を計算して dst(norm) -> src(norm) を得る
    src_norm_trans_dst_norm = inverse_3x3_onnx(dst_norm_trans_src_norm)  # [B, 3, 3]

    # 3. Destinationのメッシュグリッド作成 [-1, 1]
    # [1, h_out, w_out, 2] -> [B, h_out, w_out, 2]
    grid = (
        create_meshgrid(h_out, w_out, device=src.device)
        .to(src.dtype)
        .unsqueeze(0)
        .expand(B, h_out, w_out, 2)
    )
    
    # 4. グリッド座標を変換 (dst -> src)
    # これにより、dst画像の各ピクセルが参照すべきsrc画像の位置が計算される
    grid = transform_points(src_norm_trans_dst_norm, grid)
    
    # 5. サンプリング
    # align_corners=True は create_meshgrid の -1~1 設定と整合させるため必須
    return F.grid_sample(src, grid, align_corners=True, mode=mode, padding_mode=padding_mode)

def get_gaussian_kernel_5x5(device, channels=3):
    """5x5 Gaussian Kernel for Pyramid Construction"""
    kernel = torch.tensor([
        [1., 4., 6., 4., 1.],
        [4., 16., 24., 16., 4.],
        [6., 24., 36., 24., 6.],
        [4., 16., 24., 16., 4.],
        [1., 4., 6., 4., 1.]
    ], device=device)
    kernel = kernel / 256.0
    # Reshape to [C, 1, 5, 5] for depthwise conv
    kernel = kernel.view(1, 1, 5, 5).repeat(channels, 1, 1, 1)
    return kernel

def translation_matrix_from_offset(offset_xy, device=None):
    return torch.tensor([[1.0, 0.0, offset_xy[0]], [0.0, 1.0, offset_xy[1]], [0.0, 0.0, 1.0]], dtype=torch.float32, device=device)

def fast_gradient_mask(mask: torch.Tensor, radius: int, scale_factor: float = 1.0) -> torch.Tensor:
    h, w = mask.shape[-2:]
    
    # 小さい画像を作成
    mask_small = F.interpolate(mask, scale_factor=scale_factor, mode='bilinear', align_corners=False)
    mask_small = (mask_small > 0.5).float()

    r_small = int(radius * scale_factor)
    
    current_mask = mask_small
    accum = torch.zeros_like(mask_small)
    
    for _ in range(r_small):
        # 1px 外側に広げる (3x3 MaxPool = Dilation)
        current_mask = F.max_pool2d(current_mask, kernel_size=3, stride=1, padding=1)        
        # 加算する
        accum += current_mask
    
    grad_small = accum / r_small
    grad_mask = F.interpolate(grad_small, size=(h, w), mode='bilinear', align_corners=False)
    
    return torch.clamp(grad_mask, torch.tensor(0.0, device=mask.device), torch.tensor(1.0, device=mask.device))

def laplacian_var(img: torch.Tensor) -> torch.Tensor:
    # 1. Define the standard Laplacian kernel
    kernel = torch.tensor([
        [0, 1, 0], 
        [1, -4, 1], 
        [0, 1, 0]
    ], dtype=img.dtype, device=img.device)

    # 2. Reshape kernel to (Out, In/Groups, kH, kW)
    c = img.shape[1]
    kernel = kernel.view(1, 1, 3, 3).repeat(c, 1, 1, 1)
    # 3. Apply Convolution
    laplacian = F.conv2d(img, kernel, padding=1, groups=c)
    return laplacian.var(unbiased=False)

def pad_and_cat(t1, t2, padding_value=0.0):
    # F.pad の引数が Tensor だと古い opset でコケることがあるので、
    # 安全策として max_n - n1 が 0 の場合もそのまま通すロジックにする
    t1 = F.pad(t1, (0, 0, 0, 2048 - t1.shape[1]), value=padding_value)
    t2 = F.pad(t2, (0, 0, 0, 2048 - t2.shape[1]), value=padding_value)
    
    return torch.cat([t1, t2], dim=0)

def rad2deg_onnx(tensor):
    """torch.rad2deg の代替"""
    return tensor * (180.0 / math.pi)

def atan2_onnx(y, x):
    """
    torch.atan2 の代替
    ONNXのAtan2オペレータが使えない環境向けの近似実装。
    厳密には torch.atan2 を使うのがベストですが、
    サポートされていない場合は arctan + 象限補正 で計算します。
    """
    # ゼロ除算回避
    eps = 1e-6
    # xの符号と絶対値に基づくatan計算
    ans = torch.atan(y / (x + eps * torch.sign(x) + eps)) # 簡易的な安定化
    
    # 象限補正 (x < 0 の場合、piを加算/減算)
    # x > 0: atan(y/x)
    # x < 0, y >= 0: atan(y/x) + pi
    # x < 0, y < 0: atan(y/x) - pi
    
    pi = math.pi
    
    # 条件分岐を torch.where で記述
    ans = torch.where((x < 0) & (y >= 0), ans + pi, ans)
    ans = torch.where((x < 0) & (y < 0), ans - pi, ans)
    
    return ans
