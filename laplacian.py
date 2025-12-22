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