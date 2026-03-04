"""Custom CUDA kernels for stitch_seg hot-path operations.

JIT-compiles the C++/CUDA extensions on first import. Falls back to pure
PyTorch if compilation fails (e.g. no nvcc, wrong CUDA version).
"""

import os
import torch

_ext = None
_load_failed = False

def _load_extension():
    global _ext, _load_failed
    if _ext is not None:
        return _ext
    if _load_failed:
        return None

    try:
        from torch.utils.cpp_extension import load
        csrc_dir = os.path.join(os.path.dirname(__file__), "csrc")
        _ext = load(
            name="stitch_cuda_ops",
            sources=[
                os.path.join(csrc_dir, "bindings.cpp"),
                os.path.join(csrc_dir, "warp_perspective.cu"),
                os.path.join(csrc_dir, "distance_transform.cu"),
            ],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
        return _ext
    except Exception as e:
        print(f"[stitch_seg] CUDA extension build failed, falling back to PyTorch ops: {e}")
        _load_failed = True
        return None


def fused_warp_perspective(
    src: torch.Tensor,
    M: torch.Tensor,
    h_out: int,
    w_out: int,
    mode: int = 1,
) -> torch.Tensor:
    """Fused warp perspective: replaces normalize_homography + inverse_3x3 +
    create_meshgrid + transform_points + grid_sample in one CUDA kernel.

    Args:
        src:  [B, C, H_src, W_src] float32 CUDA tensor
        M:    [B, 3, 3] pixel-space homography (src->dst direction)
        h_out, w_out: output spatial dimensions
        mode: 0 = nearest, 1 = bilinear

    Returns:
        [B, C, h_out, w_out] float32 CUDA tensor
    """
    ext = _load_extension()
    if ext is None:
        raise RuntimeError("CUDA extension not available")
    return ext.fused_warp_perspective(src, M, h_out, w_out, mode)


def fast_gradient_mask_cuda(
    inv_mask: torch.Tensor,
    radius: int = 201,
    downscale: int = 4,
) -> torch.Tensor:
    """Fast gradient mask via iterated dilation.

    Replaces the 50-iteration F.max_pool2d loop with fused CUDA kernels
    that do 10 dilation steps per launch using shared memory.

    Args:
        inv_mask: [1, 1, H, W] float32 CUDA tensor (binary mask)
        radius:   dilation radius (default 201)
        downscale: spatial downscale factor (default 4)

    Returns:
        [1, 1, H, W] float32 gradient mask in [0, 1]
    """
    ext = _load_extension()
    if ext is None:
        raise RuntimeError("CUDA extension not available")
    return ext.fast_gradient_mask(inv_mask, radius, downscale)


def is_available() -> bool:
    """Check whether the CUDA extension can be loaded."""
    return _load_extension() is not None
