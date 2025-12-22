import torch
import cv2
import torch.nn.functional as F
from tqdm import tqdm

video_path = "/mnt/data/data11/share/TLH/standardized_videos/001510725.mp4"
start_frame = 122000
end_frame = 122500

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


cap = cv2.VideoCapture(video_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# cv2.VideoWriter needs int(W), int(H)
frame_shape = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
out = cv2.VideoWriter(f'laplacian_var_{start_frame}_{end_frame}.mp4', fourcc, 30.0, frame_shape)
frame_idx = 0
for frame_idx in tqdm(range(start_frame, end_frame)):
    #cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).to(torch.float32)
    lap_var = laplacian_var(frame_tensor)
    lap_var_scalar = float(lap_var.item())
    # Put lap_var text on the image (convert to BGR)
    text = f"LapVar: {lap_var_scalar:.2f}"
    frame_vis = frame.copy()
    cv2.putText(
        frame_vis,
        text,
        (10, 30),  # position
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,       # font scale
        (0, 0, 255),  # red color in BGR
        2,         # thickness
        cv2.LINE_AA
    )
    out.write(frame_vis)
out.release()
cap.release()




def extract_scope_mask(self, torch_frame: torch.Tensor):
    """
    画像から内視鏡の円/楕円領域のパラメータを推定する
    Return: mask (x, y) - 1が内視鏡視野
    """
    import kornia
    gray = kornia.color.rgb_to_grayscale(torch_frame)
    # 1. 二値化 (閾値は環境に合わせて調整。10-30あたりが一般的)
    thresh = (gray > 15).float()
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
