"""Utility helpers that power the full stitching+segmentation pipeline (PyTorch/Kornia version).

Assumption:
    All image inputs are torch.Tensor with shape (B, C, H, W).
    - B: Batch size (usually 1)
    - C: Channels (3 for RGB, 1 for Gray/Mask)
    - Value range: Generally [0.0, 1.0] for float tensors or [0, 255] for byte tensors.
"""

import torch
import torch.nn.functional as F
import kornia
import numpy as np # Used only for minimal logic (e.g. palette generation lists) if strictly needed, mostly avoided.
import matplotlib.pyplot as plt
# -----------------------------------------------------------------------------
# 1. ROI & Preprocessing
# -----------------------------------------------------------------------------

def compute_static_roi(frame: torch.Tensor, thr: float = 30.0/255.0):
    """
    Threshold-based ROI finder.
    frame: (B, 3, H, W) float [0, 1] or byte [0, 255]
    """
    # グレースケール変換 (B, 3, H, W) -> (B, 1, H, W)
    if frame.shape[1] == 3:
        gray = kornia.color.rgb_to_grayscale(frame)
    else:
        gray = frame

    # Normalize if necessary for comparison
    if gray.max() > 1.0:
        gray_norm = gray / 255.0
        thr_val = thr
    else:
        gray_norm = gray
        thr_val = thr

    # マスク作成 (B, 1, H, W)
    mask = torch.zeros_like(gray, dtype=torch.uint8)
    mask[gray_norm > thr_val] = 255

    # Non-zero coordinates
    # torch.nonzero returns [b, c, y, x] indices
    coords = torch.nonzero(mask > 0)

    h, w = frame.shape[-2:]

    # Bounding Rect計算 (Batch次元を考慮せず、単純に全Batchの合算領域とする簡易実装)
    # coords: N x 4 (b, c, y, x)
    ys = coords[:, 2]
    xs = coords[:, 3]
    
    y_min, y_max = ys.min().item(), ys.max().item()
    x_min, x_max = xs.min().item(), xs.max().item()
    
    rect_w = x_max - x_min + 1
    rect_h = y_max - y_min + 1
    
    return (x_min, y_min, rect_w, rect_h)

def equalize_hist_rgb(img_rgb: torch.Tensor) -> torch.Tensor:
    """
    Custom vignette correction + "equalization".
    img_rgb: (B, 3, H, W), expected value range [0, 255] or [0, 1].
    Assumes logic is based on 0-255 inputs as per original code constants.
    """
    is_float = img_rgb.max() <= 1.0
    if is_float:
        img_proc = img_rgb * 255.0
    else:
        img_proc = img_rgb.float()

    B, C, H, W = img_proc.shape
    device = img_proc.device

    # Grid creation
    x = torch.linspace(-1, 1, W, device=device)
    y = torch.linspace(-1, 1, H, device=device)
    yv, xv = torch.meshgrid(y, x, indexing='ij')
    
    sigma = 0.95
    vignette = torch.exp(-(xv**2 + yv**2) / (2 * sigma**2))
    vignette = vignette / vignette.max()
    
    # (1.0 - vignette)**2 * 64
    vignette_correction = (1.0 - vignette).pow(2) * 64
    vignette_correction = torch.clamp(vignette_correction, 0, 128)
    
    # Broadcasting add
    # vignette_correction is (H, W), img_proc is (B, 3, H, W)
    img_corr = img_proc + vignette_correction.unsqueeze(0).unsqueeze(0)
    img_corr = torch.clamp(img_corr, 0, 255)

    if is_float:
        return img_corr / 255.0
    else:
        return img_corr.to(torch.uint8)


# -----------------------------------------------------------------------------
# 2. Segmentation & Depth
# -----------------------------------------------------------------------------


def blend_segment(overlay: torch.Tensor, seg_color: torch.Tensor, mask: torch.Tensor, color: torch.Tensor, alpha: float) -> None:
    """
    overlay: (C, H, W)
    mask: (H, W) bool
    color: (3,)
    """
    if mask is None or not mask.any():
        return
        
    color = color.to(overlay.device).to(overlay.dtype)
    
    # mask broadcasting: (H, W) -> (3, H, W)
    mask_exp = mask.unsqueeze(0).expand_as(overlay)

    if alpha <= 0:
        overlay.masked_fill_(mask_exp, 0)
        # Manually adding color since masked_fill takes scalar
        # Or better:
        overlay[:, mask] = color.unsqueeze(1)
        seg_color[:, mask] = color.unsqueeze(1)
        return

    # Alpha blending
    current_vals = overlay[:, mask] # (3, N)
    target_vals = color.unsqueeze(1).expand_as(current_vals)
    
    blended = (1 - alpha) * current_vals + alpha * target_vals
    overlay[:, mask] = blended.to(overlay.dtype)
    seg_color[:, mask] = target_vals.to(seg_color.dtype)


def get_color_palette(model, seg_map: torch.Tensor) -> torch.Tensor:
    # Logic remains similar, but output Tensor
    device = seg_map.device
    if model is not None:
        palette = getattr(model.config, "palette", None)
        if palette is None:
            palette = getattr(model.config, "color_palette", None)
        if palette is not None and len(palette) > 0:
            return torch.tensor(palette, dtype=torch.uint8, device=device).reshape(-1, 3)
            
    # Dynamic generation
    max_label = int(seg_map.max().item()) if seg_map.numel() > 0 and seg_map.max() >= 0 else 0
    num_colors = max(12, max_label + 1)
    
    # Vectorized palette generation usually not worth it for small N, simple list comprehension is fine
    palette = []
    for i in range(num_colors):
        r = (i * 41 + 73) % 256
        g = (i * 67 + 151) % 256
        b = (i * 89 + 29) % 256
        palette.append([b, g, r]) # BGR order? Original code implies BGR.
        
    return torch.tensor(palette, dtype=torch.uint8, device=device)


def get_tool_class_ids(model, cfg):
    # Pure python logic, returns list of ints.
    if model is None:
        return []
    if getattr(cfg, "tool_label_ids", None):
        return [int(i) for i in cfg.tool_label_ids]
    id2label = getattr(model.config, "id2label", None)
    tool_ids = []
    if id2label:
        for k, v in id2label.items():
            try:
                k_int = int(k)
            except Exception:
                k_int = k
            label = str(v).lower()
            if any(kw in label for kw in getattr(cfg, "tool_label_keywords", [])):
                if isinstance(k_int, int):
                    tool_ids.append(k_int)
    return tool_ids


def compute_depth_mask(depth_map: torch.Tensor, cfg) -> torch.Tensor:
    """
    depth_map: (B, 1, H, W)
    """
        
    valid = torch.isfinite(depth_map)
    if not valid.any():
        return None
        
    depth_valid = depth_map[valid]
    
    # Percentile logic
    # torch.quantile requires float input
    thresh = torch.quantile(depth_map[valid], cfg.depth_close_percentile / 100.0)
    
    mask = torch.zeros_like(depth_map, dtype=torch.uint8)
    mask[(valid) & (depth_map >= thresh)] = 255
    
    # Morphological Closing
    # Kernel 5x5 ellipse
    # Kornia closing
    kernel = torch.ones(5, 5, device=depth_map.device) # Ellipse approx by Rect or custom kernel
    mask_closed = kornia.morphology.closing(mask.float(), kernel)
    
    return (mask_closed > 0).to(torch.uint8) * 255


def merge_masks(*masks):
    combined = None
    for mask in masks:
        if mask is None:
            continue
            
        # Ensure format
        if mask.dim() == 2:
            m = mask
        elif mask.dim() == 3 and mask.shape[0] == 1:
            m = mask.squeeze(0)
        else:
            m = mask
            
        if m.dtype != torch.uint8:
            m = m.to(torch.uint8)
            
        combined = m.clone() if combined is None else torch.maximum(combined, m)
        
    return combined


# -----------------------------------------------------------------------------
# 3. Canvas & Transforms
# -----------------------------------------------------------------------------

def translation_matrix_from_shift(shift, device=None):
    dx, dy = shift
    return torch.tensor([[1.0, 0.0, -dx], [0.0, 1.0, -dy], [0.0, 0.0, 1.0]], dtype=torch.float32, device=device)


def warp_with_transform(img: torch.Tensor, H: torch.Tensor, output_shape=None, interpolation='bilinear', border_mode='zeros', border_value=0):
    """
    Kornia warp wrapper.
    img: (B, C, H, W)
    H: (3, 3) or (B, 3, 3)
    """
    if img is None or H is None:
        return img
    
    orig_dim = img.dim()
    if orig_dim == 2:
        img = img.unsqueeze(0).unsqueeze(0)
    elif orig_dim == 3:
        img = img.unsqueeze(0)

    B, C, H_src, W_src = img.shape
    if output_shape is None:
        out_h, out_w = H_src, W_src
    else:
        out_h, out_w = output_shape
        
    # Kornia expects H to be (B, 3, 3)
    if H.dim() == 2:
        H_batch = H.unsqueeze(0).repeat(B, 1, 1)
    else:
        H_batch = H
        
    # Map interpolation strings
    mode = interpolation # 'bilinear', 'nearest' are valid in kornia
    if mode == 0: mode = 'nearest' # simplistic mapping if cv2 constant passed
    if mode == 1: mode = 'bilinear'
    
    # Map padding strings
    padding = 'reflection' if border_mode == 'reflect' else 'zeros' # 'zeros' maps to border_value=0 usually
    if border_mode == 'constant': padding = 'zeros' # kornia doesn't support arbitrary fill value easily in simple warp, defaults 0
    orig_dtype = img.dtype
    warped = kornia.geometry.transform.warp_perspective(
        img.float(), H_batch, dsize=(int(out_h), int(out_w)), mode=mode, padding_mode=padding, align_corners=True
    )
    
    if orig_dtype == torch.uint8:
        warped = warped.to(torch.uint8)

    if orig_dim == 2:
        return warped.squeeze(0).squeeze(0)
    if orig_dim == 3:
        return warped.squeeze(0)
    return warped


def compute_optical_flow_mask(flow_tensor: torch.Tensor, base_mask: torch.Tensor, cfg):
    """
    Calculates statistics and motion mask from a PRE-CALCULATED flow tensor.
    
    Arguments:
        flow_tensor: (B, 2, H, W) torch.Tensor (Previously calculated by NeuFlow/Raft)
                     This replaces (prev_gray, curr_gray) inputs.
        base_mask: (H, W) or (B, 1, H, W)
    """
    stats = {
        "valid_ratio": 0.0,
        "motion_ratio": 1.0,
        "shift": (0.0, 0.0),
        "shift_mag": float("inf"),
        "med_res": 0.0,
        "res_thresh": 0.0,
        "med_mag": 0.0,
        "mag_thresh": 0.0,
    }

    if flow_tensor is None:
        # Return dummy assuming shape
        return None, (0.0, 0.0), None, stats

    # Dimensions
    B, _, H, W = flow_tensor.shape
    device = flow_tensor.device
    
    dx = flow_tensor[:, 0, ...]
    dy = flow_tensor[:, 1, ...]
    mag = torch.sqrt(dx**2 + dy**2)

    # Base Mask Handling
    if base_mask is not None:
        if base_mask.dim() == 2:
            base_mask = base_mask.unsqueeze(0) # (1, H, W)
        # Assuming base_mask 0 is valid
        valid_mask = (base_mask == 0)
    else:
        valid_mask = torch.ones((B, H, W), dtype=torch.bool, device=device)

    valid_mask &= torch.isfinite(dx) & torch.isfinite(dy)
    
    valid_count = valid_mask.sum().item()
    total_pixels = valid_mask.numel()
    
    stats["valid_ratio"] = valid_count / total_pixels if total_pixels > 0 else 0.0
    
    if stats["valid_ratio"] < cfg.optflow_min_valid_ratio:
        return torch.zeros((H, W), dtype=torch.uint8, device=device), (0.0, 0.0), flow_tensor, stats

    # Extract valid values (Flattened)
    dx_valid = dx[valid_mask]
    dy_valid = dy[valid_mask]
    
    med_dx = dx_valid.median()
    med_dy = dy_valid.median()
    
    # Residuals
    residual = torch.sqrt((dx - med_dx)**2 + (dy - med_dy)**2)
    residual_valid = residual[valid_mask]
    mag_valid = mag[valid_mask]
    
    med_res = residual_valid.median()
    mad_res = (residual_valid - med_res).abs().median() + 1e-6
    res_thresh = med_res + cfg.optflow_residual_factor * mad_res
    
    med_mag = mag_valid.median()
    mad_mag = (mag_valid - med_mag).abs().median() + 1e-6
    mag_thresh = med_mag + cfg.optflow_magnitude_factor * mad_mag
    
    # Motion Mask Construction
    raw_mask = (residual > res_thresh) | (mag > mag_thresh) # (B, H, W)
    
    # Morphology (Close -> Open)
    kernel = torch.ones(5, 5, device=device)
    m_float = raw_mask.float().unsqueeze(1) # (B, 1, H, W)
    m_closed = kornia.morphology.closing(m_float, kernel)
    m_opened = kornia.morphology.opening(m_closed, kernel)
    
    motion_mask = (m_opened > 0.5).squeeze(1).to(torch.uint8) * 255
    motion_ratio = (motion_mask > 0).float().mean().item()
    
    shift = (med_dx.item(), med_dy.item())
    
    stats.update({
        "motion_ratio": motion_ratio,
        "shift": shift,
        "shift_mag": float(np.hypot(shift[0], shift[1])),
        "med_res": med_res.item(),
        "res_thresh": res_thresh.item(),
        "med_mag": med_mag.item(),
        "mag_thresh": mag_thresh.item(),
    })
    
    return motion_mask.squeeze(0), shift, flow_tensor, stats


def filter_features_by_mask(feats: dict, invalid_mask: torch.Tensor) -> dict:
    """
    invalid_mask: (H, W) or (B, 1, H, W) tensor where >0 is invalid.
    """
    kps = feats["keypoints"][0] # (N, 2)
    if kps.numel() == 0:
        return feats
        
    # kps is Tensor, no need to detach/cpu
    h, w = invalid_mask.shape[-2:]
    
    xs = torch.clamp(kps[:, 0].round().long(), 0, w - 1)
    ys = torch.clamp(kps[:, 1].round().long(), 0, h - 1)
    
    # invalid_mask might be (B, 1, H, W) or (H, W)
    if invalid_mask.dim() == 4:
        mask_2d = invalid_mask[0, 0]
    elif invalid_mask.dim() == 3:
        mask_2d = invalid_mask[0]
    else:
        mask_2d = invalid_mask
        
    keep = mask_2d[ys, xs] == 0 # assuming 0 is keep
    
    if keep.all():
        return feats
        
    return _select_features_batch0(feats, keep)


def _select_features_batch0(feats: dict, keep_bool: torch.Tensor) -> dict:
    new_feats = {}
    for k, v in feats.items():
        if k == "image_size":
            new_feats[k] = v
        else:
            # v is typically (1, N, ...)
            if v.shape[1] == keep_bool.shape[0]:
                 new_feats[k] = v[:, keep_bool]
            else:
                 new_feats[k] = v # Should not happen for aligned features
    return new_feats


def paste_current_to_canvas_forward(
    canvas,
    canvas_mask,
    H_to_canvas,
    offset_xy,
    current_img,
    tool_mask,
    cfg,
    alpha_overlap=0.80,
    update_mode: str = "full",
):
    """
    All inputs are Tensors (B, C, H, W) or similar.
    H_to_canvas: (3, 3)
                matrix prep: 0.05 ms
                matrix prod: 0.04 ms
    resize mask + mask tool: 0.04 ms
                 warp image: 0.98 ms
           make & warp mask: 14.07 ms
             warp tool_mask: 13.73 ms
         bool logic+overlap: 0.08 ms
         masked_scatter new: 15.31 ms
         fast_gradient_mask: 0.29 ms
            warp grad alpha: 0.96 ms
                 blend calc: 0.05 ms
              blend overlap: 27.55 ms
                update mask: 0.02 ms
    """
    device = current_img.device
    scale = getattr(cfg, "canvas_superres_scale", 1.0)

    canvas4model = canvas.clone()
    canvas4model_mask = canvas_mask.clone()
    
    # Translation Matrix
    T_center = translation_matrix_from_offset(offset_xy, device=device)
    S_canvas = torch.tensor([[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32, device=device)
    
    # Apply canvas super-res scale first, then translate to the canvas center.
    H_total = S_canvas @ T_center @ H_to_canvas.to(device)
    
    img_to_warp = current_img.clone()
    trim_px = max(0, int(getattr(cfg, "canvas_border_trim_px", 0)))
    
    # Resize mask if needed
    resized_mask = None
    if tool_mask is not None:
        resized_mask = tool_mask.to(torch.uint8).squeeze(0)
        # Mask out tool in input
        img_to_warp.masked_fill_(resized_mask > 0, 0)
    ch, cw = canvas.shape[-2:]

    # Create & Warp Mask
    mask = torch.ones(current_img.shape[-2:], dtype=torch.float32, device=device) # (H, W)
    if resized_mask is not None:
        mask[resized_mask.squeeze(0) > 0] = 0
        
    if trim_px > 0:
        mask[:trim_px, :] = 0
        mask[-trim_px:, :] = 0
        mask[:, :trim_px] = 0
        mask[:, -trim_px:] = 0
        
    mask_b = mask.unsqueeze(0).unsqueeze(0) # (1, 1, H, W)

    # ---- Batch warp (image + mask + tool_mask) to reduce overhead ----
    # Kornia warp expects (B, C, H, W). We pack everything into channels and warp once.
    if resized_mask is not None:
        rm_b = resized_mask.float().unsqueeze(0)
        if rm_b.dim() == 3:
            rm_b = rm_b.unsqueeze(0)  # (1, 1, H, W)
    else:
        rm_b = torch.zeros_like(mask_b)

    c_img = img_to_warp.shape[1]
    packed = torch.cat([img_to_warp.float(), mask_b, rm_b], dim=1)  # (1, C+2, H, W)
    warped_packed = kornia.geometry.transform.warp_perspective(
        packed,
        (H_total.unsqueeze(0) + torch.eye(3, device=device)*1e-6),
        dsize=(ch, cw),
        mode='nearest',
        padding_mode='zeros',
    )

    warped = warped_packed[:, :c_img].to(canvas.dtype)
    warped_mask = warped_packed[:, c_img:c_img + 1]
    warped_tool_mask = warped_packed[:, c_img + 1:c_img + 2]

    # Bool Logic
    wm_bool = warped_mask > 0.5 # currentがいる領域
    cm_bool = canvas_mask > 0 # canvasのある領域
    wtm_bool = warped_tool_mask > 0.5 # tool他maskのある領域
    
    overlap = wm_bool &cm_bool & (~wtm_bool)
    only_new = wm_bool & (~cm_bool) & (~wtm_bool) # canvasになかった領域

    # Blur等で「前フレームcanvasをベースに、新規領域だけ」反映したい場合
    # - overlap領域は一切更新しない（ブレたフレームの色を混ぜない）
    # - maskは current footprint を入れて bbox 等に使えるようにする
    if str(update_mode).lower() in ("only_new", "only-new", "onlynew"):
        if only_new.any():
            canvas.masked_scatter_(only_new.expand_as(canvas), warped[only_new.expand_as(warped)])
            canvas4model.masked_scatter_(only_new.expand_as(canvas4model), warped[only_new.expand_as(warped)])
        canvas_mask[wm_bool] = 255
        canvas4model_mask[wm_bool] = 255
        return canvas, canvas_mask, canvas4model, canvas4model_mask

    # Update Canvas
    if only_new.any():
        canvas.masked_scatter_(only_new.expand_as(canvas), warped[only_new.expand_as(warped)])
    if overlap.any():
        # mask_b (1, 1, H, W) を複製
        grad_mask = mask_b.clone()
        grad_mask[grad_mask > 0] = 1.0 
        grad_mask[..., 0, :] = 0  # 上端
        grad_mask[..., -1, :] = 0 # 下端
        grad_mask[..., :, 0] = 0  # 左端
        grad_mask[..., :, -1] = 0 # 右端

        # grad maskを反転
        grad_mask = 1 - grad_mask
        # Approximate Distance Transform using Kornia
        dist_transform = fast_gradient_mask(grad_mask, radius=cfg.gradient_radius)
        dist_transform = 1 - dist_transform

        # デバッグ: 最大値が0より大きいか確認
        # print(f"Max distance: {dist_transform.max().item()}")
        if dist_transform.max() > 0:
            grad_alpha = dist_transform / dist_transform.max()
        else:
            # 万が一計算できない場合は、全体を1.0（ベタ塗り）にするフォールバック
            grad_alpha = torch.ones_like(dist_transform)
            
        warped_grad_alpha = kornia.geometry.transform.warp_perspective(
            grad_alpha,
            (H_total.unsqueeze(0) + torch.eye(3, device=device)*1e-6),
            dsize=(ch, cw),
            mode='bilinear' # アルファマップは滑らかにしたいのでbilinear推奨
        )
        blended_alpha = alpha_overlap * warped_grad_alpha
        
        overlap_expanded = overlap.expand_as(canvas)       
        alpha_expanded = blended_alpha.expand_as(canvas)   
        
        c_vals = canvas[overlap_expanded].float()
        w_vals = warped[overlap_expanded].float()
        a_vals = alpha_expanded[overlap_expanded]
        
        # ブレンド計算
        blended_vals = c_vals * (1.0 - a_vals) + w_vals * a_vals
        
        # Canvasに書き戻し
        canvas[overlap_expanded] = blended_vals.to(canvas.dtype)
        canvas4model[overlap_expanded] = c_vals.to(canvas4model.dtype)

        

    # Initialization case
    if not overlap.any() and not only_new.any():
        canvas.fill_(0)
        canvas_mask.fill_(0)
        
        # Resize current to canvas scale
        # F.interpolate
        new_h = int(round(current_img.shape[-2] * scale))
        new_w = int(round(current_img.shape[-1] * scale))
        
        hi_img = F.interpolate(current_img.float(), size=(new_h, new_w), mode='bilinear', align_corners=False).to(canvas.dtype)
        
        off_x = int(round(offset_xy[0]))
        off_y = int(round(offset_xy[1]))
        
        # Safe slicing
        y2 = min(off_y + new_h, ch)
        x2 = min(off_x + new_w, cw)
        
        h_slice = y2 - off_y
        w_slice = x2 - off_x
        
        if h_slice > 0 and w_slice > 0:
            canvas[..., off_y:y2, off_x:x2] = hi_img[..., :h_slice, :w_slice]
            canvas_mask[..., off_y:y2, off_x:x2] = 255
            # initialization case: canvas4model is equal to canvas
        return canvas, canvas_mask, canvas, canvas_mask

    # Update Mask
    canvas_mask[wm_bool] = 255
    canvas4model_mask[wm_bool] = 255
    return canvas, canvas_mask, canvas4model, canvas4model_mask


def invert_canvas_valid_mask(canvas_mask: torch.Tensor) -> torch.Tensor:
    if canvas_mask is None:
        return None
    # 255 is valid, 0 is invalid -> return 0 valid, 255 invalid
    return torch.where(canvas_mask > 0, torch.tensor(0, device=canvas_mask.device, dtype=torch.uint8), torch.tensor(255, device=canvas_mask.device, dtype=torch.uint8))


def extract_canvas_features(extractor, canvas_img: torch.Tensor, cfg, invalid_mask: torch.Tensor = None):
    # extractor assumed to handle Tensor input
    if extractor is None or canvas_img is None:
        return None
    with torch.no_grad():
        # to_lightglue_gray_tensor functionality is likely:
        # 1. rgb to gray
        # 2. normalize
        # Assuming canvas_img is (B, 3, H, W)
        tensor = kornia.color.rgb_to_grayscale(canvas_img).float() / 255.0
        feats = extractor.extract(tensor)
    
    if invalid_mask is not None:
        feats = filter_features_by_mask(feats, invalid_mask)
    return feats


def convert_homography_to_raw_space(H_stab: torch.Tensor, prev_transform: torch.Tensor, curr_transform: torch.Tensor) -> torch.Tensor:
    if H_stab is None:
        return None
    device = H_stab.device
    if prev_transform is None:
        prev_transform = torch.eye(3, device=device)
    if curr_transform is None:
        curr_transform = torch.eye(3, device=device)
        
    #try:
    #    curr_inv = torch.linalg.inv(curr_transform)
    #except RuntimeError:
    #    return None
    
    prev_inv = torch.linalg.inv(prev_transform)


    # Convert homography defined in stabilized coords to raw coords:
    # x_curr_raw = curr_inv * H_stab * prev_transform * x_prev_raw
    #return curr_inv @ H_stab @ prev_transform
    return prev_inv @ H_stab @ curr_transform


def translation_matrix_from_offset(offset_xy, device=None):
    off_x, off_y = offset_xy
    return torch.tensor([[1.0, 0.0, float(off_x)], [0.0, 1.0, float(off_y)], [0.0, 0.0, 1.0]], dtype=torch.float32, device=device)


def shear_angle_from_homography(H: torch.Tensor) -> float:
    """Return absolute shear angle (degrees)."""
    # QR decomp
    if abs(H[2, 2]) < 1e-8:
        return 0.0
    A = H[:2, :2] / H[2, 2]
    Q, R = torch.linalg.qr(A.to(torch.float32)) # CPU for QR usually safer/faster for 3x3
    k = R[0, 1] / (R[1, 1] + 1e-8)
    rad = torch.atan(k).abs()
    return torch.rad2deg(rad).item()


def rotate_angle_from_homography(H: torch.Tensor) -> float:
    """Return absolute rotation angle (degrees)."""
    rad = torch.atan2(H[1, 0], H[0, 0]).abs()
    return torch.rad2deg(rad).item()


def scale_factor_from_homography(H: torch.Tensor) -> float:
    if abs(H[2, 2]) < 1e-8:
        return 1.0
    A = H[:2, :2] / H[2, 2]
    A = A.to(torch.float32)
    # SVD
    U, S, Vh = torch.linalg.svd(A)
    scale_major = float(S.max().item())
    scale_minor = float(S.min().item())
    if scale_minor <= 1e-8:
        return scale_major
    return float(max(scale_major, 1.0 / scale_minor))


def reset_canvas_orientation(canvas: torch.Tensor, canvas_mask: torch.Tensor, H_to_canvas: torch.Tensor, frame_shape, cfg, old_offset_xy):
    """
    Warp canvas back.
    canvas: (B, C, H, W)
    """
    cw, ch = canvas.shape[-1], canvas.shape[-2]
    device = canvas.device
    
    scale = float(getattr(cfg, "canvas_superres_scale", 1.0))
    S = torch.tensor([[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32, device=device)
    
    # 1. Old Transform (Current -> Old Canvas)
    # P_old = S @ T_old
    T_old = translation_matrix_from_offset(old_offset_xy, device=device)
    P_old = S @ T_old
    
    # H_total_old = P_old @ H_to_canvas
    H_total_old = P_old @ H_to_canvas
    
    # 2. New Transform (Current -> New Canvas, Rotation Removed)
    # New offset should be based on unscaled canvas size to work with S @ T
    frame_h, frame_w = frame_shape[-2:]
    
    base_cw = int(cw / scale) if scale > 0 else cw
    base_ch = int(ch / scale) if scale > 0 else ch
    
    offset_x = base_cw // 2 - frame_w // 2
    offset_y = base_ch // 2 - frame_h // 2
    
    T_new = translation_matrix_from_offset((offset_x, offset_y), device=device)
    P_new = S @ T_new
    
    # H_total_new = P_new (since H_to_canvas is reset to Identity)
    
    # 3. Warp Matrix M (Old Canvas -> New Canvas)
    # x_old = H_total_old @ x_curr
    # x_new = P_new @ x_curr
    # x_curr = inv(H_total_old) @ x_old
    # x_new = P_new @ inv(H_total_old) @ x_old
    
    try:
        H_old_inv = torch.linalg.inv(H_total_old)
    except RuntimeError:
        H_old_inv = torch.eye(3, device=device)
        
    M = P_new @ H_old_inv
    
    # 4. Warp

    packed = torch.cat([canvas, canvas_mask], dim=1).float()
    warped_packed = kornia.geometry.transform.warp_perspective(
        packed, (M.unsqueeze(0) + torch.eye(3, device=device)*1e-6), dsize=(ch, cw), mode='nearest', padding_mode='zeros'
    ).to(canvas.dtype)

    canvas_chs = canvas.shape[1]
    mask_chs = canvas_mask.shape[1]
    warped_canvas = warped_packed[:, :canvas_chs]
    warped_mask = warped_packed[:, canvas_chs:canvas_chs + mask_chs]
    warped_mask = (warped_mask > 0).to(torch.uint8) * 255
    
    return warped_canvas, warped_mask, (offset_x, offset_y)


def estimate_camera_shift(flow_tensor: torch.Tensor, base_mask: torch.Tensor, cfg):
    """
    Calculates median shift from flow tensor.
    flow_tensor: (B, 2, H, W)
    """
    if flow_tensor is None:
        return (0.0, 0.0)
        
    dx = flow_tensor[:, 0, ...]
    dy = flow_tensor[:, 1, ...]
    
    if base_mask is not None:
        if base_mask.dim() == 2:
            base_mask = base_mask.unsqueeze(0)
        valid_mask = (base_mask == 0)
    else:
        valid_mask = torch.ones_like(dx, dtype=torch.bool)
        
    valid_mask &= torch.isfinite(dx) & torch.isfinite(dy)
    
    total = valid_mask.numel()
    valid = valid_mask.sum().item()
    
    if valid == 0 or (total > 0 and valid / total < cfg.optflow_min_valid_ratio):
        return (0.0, 0.0)
        
    med_dx = dx[valid_mask].median().item()
    med_dy = dy[valid_mask].median().item()
    
    return (med_dx, med_dy)


import torch
import torch.nn.functional as F
import kornia
import numpy as np

def poisson_blend_roi(source, target, mask, num_iters=100):
    """
    PyTorchのみを使用したPoisson Blending（Membrane Interpolation法）。
    Sourceの勾配を維持しつつ、Targetの色調に合わせます。
    
    Args:
        source (Tensor): (C, H, W)
        target (Tensor): (C, H, W)
        mask (Tensor): (1, H, W) 1.0 inside, 0.0 outside
        num_iters (int): 反復回数。差分法なので50-100回程度で十分収束します。
    """
    # 境界条件のための初期化
    # 誤差（Target - Source）を計算
    diff = target - source
    
    # マスクの侵食（Erosion）を行い、境界ピクセルを特定
    # 境界（Mask=0）の値は固定し、内部（Mask=1）の値を拡散させる
    
    # ラプラス方程式のソルバー（平均化フィルタによる反復解法）
    # kernel: 上下左右の平均を取る畳み込み核
    kernel = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]], 
                          device=source.device, dtype=source.dtype) / 4.0
    kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(source.shape[0], 1, 1, 1) # (C, 1, 3, 3)

    # 初期値: 内部は平均値などで埋めると収束が早いが、ゼロスタートでもOK
    # ここでは境界のdiffを内部に滑らかに広げる問題を解く
    correction = diff.clone() 
    
    # Maskの調整: 計算対象領域を1、境界固定領域を0にする
    # (H, W) -> (1, C, H, W)
    mask_expanded = mask.expand_as(source).unsqueeze(0)
    
    # 入力を (1, C, H, W) に
    correction_input = correction.unsqueeze(0)
    
    # 固定部分（境界条件）
    boundary_val = diff.unsqueeze(0)
    
    for _ in range(num_iters):
        # 隣接画素の平均を計算
        avg = F.conv2d(correction_input, kernel, padding=1, groups=source.shape[0])
        
        # マスク内部（1の部分）は計算値(avg)で更新、外部（0の部分）は固定値(boundary_val)に戻す
        correction_input = avg * mask_expanded + boundary_val * (1 - mask_expanded)
        
    # 補正後の画像を生成: Source + 滑らかな色差
    # correction_inputは (1, C, H, W) なので squeeze
    blended = source + correction_input.squeeze(0)
    
    # 最終的にマスク範囲外はTargetのままにする（数値誤差防止）
    final = blended * mask + target * (1 - mask)
    
    return final.clamp(0, 255) # 必要に応じて範囲制限

def paste_current_to_canvas_forward_poisson(
    canvas,
    canvas_mask,
    H_to_canvas,
    offset_xy,
    current_img,
    tool_mask,
    cfg,
    alpha_overlap=0.45,
    update_mode: str = "full",
):
    """
    Poisson Blendingを用いた貼り付け処理
    """
    device = current_img.device
    scale = getattr(cfg, "canvas_superres_scale", 1.0)
    canvas4model = canvas.clone()
    canvas4model_mask = canvas_mask.clone()
    
    # --- 1. Warpの準備 ---
    T_center = translation_matrix_from_offset(offset_xy, device=device)
    S_canvas = torch.tensor([[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32, device=device)
    
    H_total = T_center @ S_canvas @ H_to_canvas.to(device)
    
    img_to_warp = current_img.clone()
    trim_px = max(0, int(getattr(cfg, "canvas_border_trim_px", 0)))
    
    # MaskのResizeと適用
    resized_mask = None
    if tool_mask is not None:
        tm = tool_mask
        if tm.dim() == 2: tm = tm.unsqueeze(0).unsqueeze(0)
        elif tm.dim() == 3: tm = tm.unsqueeze(0)

        if tm.shape[-2:] != img_to_warp.shape[-2:]:
            resized_mask = F.interpolate(tm.float(), size=img_to_warp.shape[-2:], mode='nearest')
        else:
            resized_mask = tm
        resized_mask = resized_mask.to(torch.uint8)
        if resized_mask.dim() == 4: resized_mask = resized_mask.squeeze(0)
        if resized_mask.dim() == 3: resized_mask = resized_mask.squeeze(0)
        
        img_to_warp.masked_fill_(resized_mask > 0, 0)

    # --- 2. Warp実行 ---
    ch, cw = canvas.shape[-2:]

    # Mask生成
    mask = torch.ones(current_img.shape[-2:], dtype=torch.float32, device=device)
    if resized_mask is not None:
        mask[resized_mask > 0] = 0
        
    if trim_px > 0:
        mask[:trim_px, :] = 0
        mask[-trim_px:, :] = 0
        mask[:, :trim_px] = 0
        mask[:, -trim_px:] = 0
        
    mask_b = mask.unsqueeze(0).unsqueeze(0) # (1, 1, H, W)

    # ---- Batch warp (image + geometry mask + tool mask) to reduce overhead ----
    # All share H_total, dsize and nearest mode -> pack into channels and warp once.
    if resized_mask is not None:
        rm_b = resized_mask.float().unsqueeze(0)
        if rm_b.dim() == 3:
            rm_b = rm_b.unsqueeze(0)  # (1, 1, H, W)
    else:
        rm_b = torch.zeros_like(mask_b)

    c_img = img_to_warp.shape[1]
    packed = torch.cat([img_to_warp.float(), mask_b, rm_b], dim=1)  # (1, C+2, H, W)
    warped_packed = kornia.geometry.transform.warp_perspective(
        packed,
        (H_total.unsqueeze(0) + torch.eye(3, device=device)*1e-6),
        dsize=(ch, cw),
        mode='nearest',
        padding_mode='zeros',
    )
    warped = warped_packed[:, :c_img].to(canvas.dtype)
    warped_mask = warped_packed[:, c_img:c_img + 1]
    warped_tool_mask = warped_packed[:, c_img + 1:c_img + 2]

    # 領域判定
    wm_bool = warped_mask > 0.5 
    cm_bool = canvas_mask > 0 
    wtm_bool = warped_tool_mask > 0.5 
    
    overlap = wm_bool & cm_bool & (~wtm_bool)
    only_new = wm_bool & (~cm_bool) & (~wtm_bool)

    if str(update_mode).lower() in ("only_new", "only-new", "onlynew"):
        # only_new だけ更新（overlapは更新しない）
        if only_new.any():
            canvas.masked_scatter_(only_new.expand_as(canvas), warped[only_new.expand_as(warped)])
            canvas4model.masked_scatter_(only_new.expand_as(canvas4model), warped[only_new.expand_as(warped)])
        canvas_mask[wm_bool] = 255
        canvas4model_mask[wm_bool] = 255
        return canvas, canvas_mask, canvas4model, canvas4model_mask

    # --- 3. 描画処理 (Poisson Blending) ---

    # Case A: 初期化 (キャンバスが空の場合など)
    if not overlap.any() and not only_new.any():
        # キャンバスが空なら単純コピーでOK（または従来の初期化ロジック）
        # ここではWarp結果を利用する形に統一します
        if wm_bool.any():
            canvas.masked_scatter_(wm_bool.expand_as(canvas), warped[wm_bool.expand_as(warped)])
            canvas_mask[wm_bool] = 255
        return canvas, canvas_mask, canvas, canvas_mask

    # Step 1: 新しい領域 (only_new) を先にコピー
    # これにより、overlap領域のブレンド計算時に「新しい外側の色」が境界条件として使われます
    if only_new.any():
        canvas.masked_scatter_(only_new.expand_as(canvas), warped[only_new.expand_as(warped)])
        canvas4model.masked_scatter_(only_new.expand_as(canvas4model), warped[only_new.expand_as(warped)])
    # Step 2: 重なり領域 (overlap) に対して Poisson Blending を適用
    if overlap.any():
        # ROI (Region of Interest) の計算
        # 計算量を減らすため、更新が必要な矩形領域だけ切り出します
        rows, cols = torch.where(overlap.squeeze(0).squeeze(0))
        if len(rows) > 0:
            min_y, max_y = rows.min().item(), rows.max().item()
            min_x, max_x = cols.min().item(), cols.max().item()
            
            # マージンを追加 (境界条件を取得するため)
            margin = 8 
            min_y = max(0, min_y - margin)
            max_y = min(ch, max_y + margin)
            min_x = max(0, min_x - margin)
            max_x = min(cw, max_x + margin)
            
            # Crop Tensors (Squeeze batch dim for solver)
            # warped: (1, C, H, W) -> roi: (C, H_roi, W_roi)
            src_roi = warped[0, :, min_y:max_y, min_x:max_x]
            dst_roi = canvas[0, :, min_y:max_y, min_x:max_x].float() # 計算のためfloatに
            
            # Currentが無効な場所(warped_mask==0)の影響を排除する処理
            # warped_mask: (1, 1, H, W)
            wmask_roi = warped_mask[0, 0, min_y:max_y, min_x:max_x] # (H_roi, W_roi)
            
            # src_roi を float 化して補正
            src_roi_f = src_roi.float()
            
            # Currentが無効な場所は Source = Target とすることで Diff = 0 (境界条件としての影響を無効化)
            invalid_current = (wmask_roi < 0.5) # bool
            if invalid_current.any():
                invalid_current_exp = invalid_current.unsqueeze(0).expand_as(src_roi_f)
                src_roi_f[invalid_current_exp] = dst_roi[invalid_current_exp]

            # Mask: Overlap部分のみを1とする (1.0 = Poissonで更新する場所)
            mask_roi = overlap[0, 0, min_y:max_y, min_x:max_x].float().unsqueeze(0) # (1, H_roi, W_roi)
            
            # --- Poisson Blending 実行 ---
            # ここでは「Sourceのテクスチャ + Targetの色」を目指す
            try:
                blended_roi = poisson_blend_roi(src_roi_f, dst_roi, mask_roi, num_iters=50)
                
                # Canvasに書き戻し (Overlap部分のみ更新)
                # blended_roiはfloatなので元のdtypeに戻す
                blended_roi = blended_roi.to(canvas.dtype)
                
                canvas_roi = canvas[..., min_y:max_y, min_x:max_x]
                canvas4model_roi = canvas4model[..., min_y:max_y, min_x:max_x]
                overlap_roi = overlap[..., min_y:max_y, min_x:max_x]
                
                canvas_roi.masked_scatter_(overlap_roi.expand_as(canvas_roi), blended_roi.unsqueeze(0)[overlap_roi.expand_as(canvas_roi)])
                canvas4model_roi.masked_scatter_(overlap_roi.expand_as(canvas4model_roi), blended_roi.unsqueeze(0)[overlap_roi.expand_as(canvas4model_roi)])
            except Exception as e:
                print(f"Blending Error: {e}")
                # フォールバック：単純上書き
                canvas.masked_scatter_(overlap.expand_as(canvas), warped[overlap.expand_as(warped)])
                canvas4model.masked_scatter_(overlap.expand_as(canvas4model), warped[overlap.expand_as(warped)])
    # Update Mask
    canvas_mask[wm_bool] = 255
    canvas4model_mask[wm_bool] = 255
    return canvas, canvas_mask, canvas4model, canvas4model_mask
# Helper (依存関係維持)
def translation_matrix_from_offset(offset_xy, device):
    tx, ty = offset_xy[0], offset_xy[1]
    return torch.tensor([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]], dtype=torch.float32, device=device)

def build_gaussian_pyramid(tensor, levels=5):
    pyramid = [tensor]
    for _ in range(int(levels - 1)):
        tensor = kornia.geometry.transform.pyr_down(tensor)
        pyramid.append(tensor)
    return pyramid

def build_laplacian_pyramid(gaussian_pyramid):
    laplacian_pyramid = []
    levels = len(gaussian_pyramid)
    for i in range(levels - 1):
        current = gaussian_pyramid[i]
        next_up = kornia.geometry.transform.pyr_up(gaussian_pyramid[i+1])
        if next_up.shape[-2:] != current.shape[-2:]:
            next_up = F.interpolate(next_up, size=current.shape[-2:], mode='nearest')
        laplacian = current - next_up
        laplacian_pyramid.append(laplacian)
    laplacian_pyramid.append(gaussian_pyramid[-1])
    return laplacian_pyramid

def reconstruct_from_laplacian(laplacian_pyramid):
    current = laplacian_pyramid[-1]
    for i in range(len(laplacian_pyramid) - 2, -1, -1):
        up_current = kornia.geometry.transform.pyr_up(current)
        h_next = laplacian_pyramid[i]
        if up_current.shape[-2:] != h_next.shape[-2:]:
            up_current = F.interpolate(up_current, size=h_next.shape[-2:], mode='nearest')
        current = h_next + up_current
    return current

def translation_matrix_from_offset(offset_xy, device):
    tx, ty = offset_xy[0], offset_xy[1]
    return torch.tensor([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]], dtype=torch.float32, device=device)

# --- Main Function with Mask Softening ---
def paste_current_to_canvas_forward_multiband(
    canvas, canvas_mask, H_to_canvas, offset_xy, current_img, tool_mask, cfg, 
    levels=4, 
    update_mode: str = "full",
):
    """
    Multiband Blending with Mask Softening (Erosion + Gaussian Blur).
    """
    device = current_img.device
    scale = getattr(cfg, "canvas_superres_scale", 1.0)
    canvas4model = canvas.clone()
    canvas4model_mask = canvas_mask.clone()
    
    # --- 1. Preparation & Warping ---
    T_center = translation_matrix_from_offset(offset_xy, device=device)
    S_canvas = torch.tensor([[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32, device=device)
    H_total = S_canvas @ T_center @ H_to_canvas.to(device)
    
    img_to_warp = current_img.clone()
    trim_px = max(0, int(getattr(cfg, "canvas_border_trim_px", 0)))
    
    # Mask out tool in input image to prevent color bleeding
    resized_mask = None
    if tool_mask is not None:
        resized_mask = tool_mask.to(torch.uint8).squeeze(0)
        img_to_warp.masked_fill_(resized_mask > 0, 0)

    ch, cw = canvas.shape[-2:]

    # Warp Masks (geometry mask + tool mask)
    mask = torch.ones(current_img.shape[-2:], dtype=torch.float32, device=device)
    if resized_mask is not None:
        mask[resized_mask.squeeze(0) > 0] = 0
    if trim_px > 0:
        mask[:trim_px, :] = 0; mask[-trim_px:, :] = 0; mask[:, :trim_px] = 0; mask[:, -trim_px:] = 0
        
    mask_b = mask.unsqueeze(0).unsqueeze(0)

    # ---- Batch warp (image + geometry mask + tool mask) to reduce overhead ----
    # All share H_total, dsize and nearest mode -> pack into channels and warp once.
    if resized_mask is not None:
        rm_b = resized_mask.float().unsqueeze(0)
        if rm_b.dim() == 3:
            rm_b = rm_b.unsqueeze(0)  # (1, 1, H, W)
    else:
        rm_b = torch.zeros_like(mask_b)

    c_img = img_to_warp.shape[1]
    packed = torch.cat([img_to_warp.float(), mask_b, rm_b], dim=1)  # (1, C+2, H, W)
    warped_packed = kornia.geometry.transform.warp_perspective(
        packed, (H_total.unsqueeze(0) + torch.eye(3, device=device)*1e-6), dsize=(ch, cw), mode='nearest', padding_mode='zeros'
    )
    warped = warped_packed[:, :c_img].to(canvas.dtype)
    warped_mask = warped_packed[:, c_img:c_img + 1]
    warped_tool_mask = warped_packed[:, c_img + 1:c_img + 2]

    # Logic Masks (Binary)
    wm_bool = warped_mask > 0.5
    cm_bool = canvas_mask > 0
    wtm_bool = warped_tool_mask > 0.5
    
    valid_new_region = wm_bool & (~wtm_bool)
    
    overlap = valid_new_region & cm_bool
    only_new = valid_new_region & (~cm_bool)

    if str(update_mode).lower() in ("only_new", "only-new", "onlynew"):
        if only_new.any():
            canvas.masked_scatter_(only_new.expand_as(canvas), warped[only_new.expand_as(warped)])
            canvas4model.masked_scatter_(only_new.expand_as(canvas4model), warped[only_new.expand_as(warped)])
        canvas_mask[valid_new_region] = 255
        canvas4model_mask[valid_new_region] = 255
        return canvas, canvas_mask, canvas4model, canvas4model_mask

    # Initialization case
    if not cm_bool.any():
        canvas.masked_scatter_(valid_new_region.expand_as(canvas), warped[valid_new_region.expand_as(warped)])
        canvas_mask[valid_new_region] = 255
        return canvas, canvas_mask, canvas, canvas_mask

    # Update non-overlapping area
    if only_new.any():
        canvas.masked_scatter_(only_new.expand_as(canvas), warped[only_new.expand_as(warped)])
        canvas4model.masked_scatter_(only_new.expand_as(canvas4model), warped[only_new.expand_as(warped)])
    
    # --- 2. Multiband Blending Logic with Mask Smoothing ---
    if overlap.any():
        # Optimization: ROI Clipping
        rows, cols = torch.where(valid_new_region.squeeze(0).squeeze(0))
        if len(rows) > 0:
            y_min, y_max = rows.min().item(), rows.max().item() + 1
            x_min, x_max = cols.min().item(), cols.max().item() + 1
            
            # ROI Padding
            pad = 2 ** levels
            y_min = int(max(0, y_min - pad)); x_min = int(max(0, x_min - pad))
            y_max = int(min(ch, y_max + pad)); x_max = int(min(cw, x_max + pad))
            
            # Crop ROI
            roi_canvas = canvas[..., y_min:y_max, x_min:x_max].float()
            roi_warped = warped[..., y_min:y_max, x_min:x_max].float()
            
            # Base Mask from Geometry
            roi_mask = valid_new_region[..., y_min:y_max, x_min:x_max].float()
            
            # マスクの強度が下がりすぎないように正規化する場合もありますが、
            # ブレンド用としては0~1のグラデーションが重要なのでそのままでOK
            
            # Build Pyramids
            pyr_canvas = build_gaussian_pyramid(roi_canvas, levels)
            pyr_warped = build_gaussian_pyramid(roi_warped, levels)
            pyr_mask = build_gaussian_pyramid(roi_mask, levels)
            
            lap_canvas = build_laplacian_pyramid(pyr_canvas)
            lap_warped = build_laplacian_pyramid(pyr_warped)
            
            # Blend
            blended_pyr = []
            for l_c, l_w, m in zip(lap_canvas, lap_warped, pyr_mask):
                if m.shape[-2:] != l_c.shape[-2:]:
                    m = F.interpolate(m, size=l_c.shape[-2:], mode='bilinear')
                blend = l_w * m + l_c * (1.0 - m)
                blended_pyr.append(blend)
            
            roi_blended = reconstruct_from_laplacian(blended_pyr)
            roi_blended = torch.clamp(roi_blended, 0, 255)
            
            # Paste back using the processed mask as alpha to prevent hard edges in paste-back
            # (ブレンド結果の書き戻しも、softなmaskを使って既存Canvasと混ぜることで、ROI境界の線を防ぐ)
            # もしcanvasが空だった場所に書き戻すなら直接代入で良いですが、
            # overlap領域なので、再構築誤差を隠すためにもマスク合成推奨
            canvas_crop = canvas[..., y_min:y_max, x_min:x_max]
            
            paste_alpha_radius = int(getattr(cfg, "gradient_radius", 201))
            
            roi_mask_bin = (roi_mask > 0.5).float()
            inv_mask = 1.0 - roi_mask_bin  # outside=1, inside=0
            # ROI crop境界も「外側」とみなして必ずフェード帯ができるようにする
            inv_mask[..., 0, :] = 1.0
            inv_mask[..., -1, :] = 1.0
            inv_mask[..., :, 0] = 1.0
            inv_mask[..., :, -1] = 1.0
            
            # fast_gradient_mask は「1の領域から外側に減衰」なので、反転してROI内側へ増加するalphaにする
            h_roi, w_roi = inv_mask.shape[-2:]
            paste_alpha_radius = int(min(paste_alpha_radius, max(1, min(h_roi, w_roi) - 1)))
            dist_out = fast_gradient_mask(inv_mask, radius=paste_alpha_radius)
            grad_in = 1.0 - dist_out  # outside≈0, inside→1
            
            # roi_mask（将来のblur/erosionなど）も掛けて、境界のカットを避ける
            update_weight = torch.clamp(grad_in * roi_mask, 0.0, 1.0)
            if update_weight.shape[-2:] != canvas_crop.shape[-2:]:
                update_weight = F.interpolate(update_weight, size=canvas_crop.shape[-2:], mode='bilinear', align_corners=False)
            
            blended_result = canvas_crop * (1.0 - update_weight) + roi_blended.to(canvas.dtype) * update_weight
            canvas[..., y_min:y_max, x_min:x_max] = blended_result
            canvas4model[..., y_min:y_max, x_min:x_max] = roi_blended.to(canvas4model.dtype)

    # Update Mask
    canvas_mask[valid_new_region] = 255
    canvas4model_mask[valid_new_region] = 255
    return canvas, canvas_mask, canvas4model, canvas4model_mask

def fast_gradient_mask(mask: torch.Tensor, radius: int, scale_factor: float = 0.125) -> torch.Tensor:
    """
    マスクの輪郭から外側に向かって、値が直線的(Linear)に0に落ちていく
    グラデーションマスクを作成する。
    
    Args:
        mask: 入力バイナリマスク (B, C, H, W) [0 or 1]
        radius: グラデーションを広げる距離 (px)
        scale_factor: 小さい画像のスケールファクター
    """
    h, w = mask.shape[-2:]
    
    # 小さい画像を作成
    mask_small = F.interpolate(mask, scale_factor=scale_factor, mode='bilinear', align_corners=False)
    mask_small = (mask_small > 0.5).float()

    r_small = int(radius * scale_factor)
    if r_small < 1: r_small = 1
    
    current_mask = mask_small
    accum = torch.zeros_like(mask_small)
    
    for _ in range(r_small):
        # 1px 外側に広げる (3x3 MaxPool = Dilation)
        current_mask = F.max_pool2d(current_mask, kernel_size=3, stride=1, padding=1)        
        # 加算する
        accum += current_mask
    
    grad_small = accum / r_small
    grad_mask = F.interpolate(grad_small, size=(h, w), mode='bilinear', align_corners=False)
    
    return torch.clamp(grad_mask, 0.0, 1.0)

def laplacian_var(img: torch.Tensor) -> torch.Tensor:
    """
    Calculates the variance of the Laplacian of an image in PyTorch.
    Equivalent to cv2.Laplacian(img, cv2.CV_64F).var()
    Args:
        img (torch.Tensor): Input image of shape (1, 3, H, W) or (B, C, H, W)
    
    Returns:
        torch.Tensor: The variance scalar.
    """
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
