#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// ---- 3x3 inverse via Cramer's rule (device helper) ----
// Input:  M[9] row-major (M[row*3+col])
// Output: Minv[9] row-major
__device__ void inv3x3(const float* __restrict__ M, float* __restrict__ Minv) {
    float a = M[0], b = M[1], c = M[2];
    float d = M[3], e = M[4], f = M[5];
    float g = M[6], h = M[7], i = M[8];

    float det = a * (e * i - f * h)
              - b * (d * i - f * g)
              + c * (d * h - e * g);

    float inv_det = 1.0f / (det + 1e-10f);

    Minv[0] = (e * i - f * h) * inv_det;
    Minv[1] = (c * h - b * i) * inv_det;
    Minv[2] = (b * f - c * e) * inv_det;
    Minv[3] = (f * g - d * i) * inv_det;
    Minv[4] = (a * i - c * g) * inv_det;
    Minv[5] = (c * d - a * f) * inv_det;
    Minv[6] = (d * h - e * g) * inv_det;
    Minv[7] = (b * g - a * h) * inv_det;
    Minv[8] = (a * e - b * d) * inv_det;
}

// ---- Main warp kernel ----
// Each thread computes one output pixel across all channels.
// M is the pixel-space homography src->dst. We invert it to get dst->src mapping.
// mode: 0=nearest, 1=bilinear
template <int MODE>
__global__ void warp_kernel(
    const float* __restrict__ src,  // [B, C, H_src, W_src]
    const float* __restrict__ M_inv, // [B, 3, 3] pre-inverted
    float* __restrict__ dst,        // [B, C, H_dst, W_dst]
    int B, int C, int H_src, int W_src, int H_dst, int W_dst)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x; // x in dst
    int iy = blockIdx.y * blockDim.y + threadIdx.y; // y in dst
    int ib = blockIdx.z;                            // batch

    if (ix >= W_dst || iy >= H_dst || ib >= B)
        return;

    // Load M_inv for this batch element
    const float* Mi = M_inv + ib * 9;
    float m00 = Mi[0], m01 = Mi[1], m02 = Mi[2];
    float m10 = Mi[3], m11 = Mi[4], m12 = Mi[5];
    float m20 = Mi[6], m21 = Mi[7], m22 = Mi[8];

    // Map dst pixel (ix, iy) back to src coordinates
    float dx = (float)ix;
    float dy = (float)iy;
    float w  = m20 * dx + m21 * dy + m22;
    float inv_w = 1.0f / (w + 1e-10f);
    float sx = (m00 * dx + m01 * dy + m02) * inv_w;
    float sy = (m10 * dx + m11 * dy + m12) * inv_w;

    int dst_idx_base = (ib * C) * H_dst * W_dst + iy * W_dst + ix;
    int src_plane = H_src * W_src;
    int dst_plane = H_dst * W_dst;

    if constexpr (MODE == 0) {
        // Nearest
        int isx = __float2int_rn(sx);
        int isy = __float2int_rn(sy);
        if (isx >= 0 && isx < W_src && isy >= 0 && isy < H_src) {
            int src_base = ib * C * src_plane + isy * W_src + isx;
            for (int c = 0; c < C; c++) {
                dst[dst_idx_base + c * dst_plane] = src[src_base + c * src_plane];
            }
        } else {
            for (int c = 0; c < C; c++) {
                dst[dst_idx_base + c * dst_plane] = 0.0f;
            }
        }
    } else {
        // Bilinear
        if (sx >= 0.0f && sx < (float)(W_src - 1) && sy >= 0.0f && sy < (float)(H_src - 1)) {
            int x0 = __float2int_rd(sx);
            int y0 = __float2int_rd(sy);
            int x1 = x0 + 1;
            int y1 = y0 + 1;
            float fx = sx - (float)x0;
            float fy = sy - (float)y0;

            float w00 = (1.0f - fx) * (1.0f - fy);
            float w01 = fx * (1.0f - fy);
            float w10 = (1.0f - fx) * fy;
            float w11 = fx * fy;

            int src_base = ib * C * src_plane;
            for (int c = 0; c < C; c++) {
                int off = src_base + c * src_plane;
                float val = w00 * src[off + y0 * W_src + x0]
                          + w01 * src[off + y0 * W_src + x1]
                          + w10 * src[off + y1 * W_src + x0]
                          + w11 * src[off + y1 * W_src + x1];
                dst[dst_idx_base + c * dst_plane] = val;
            }
        } else if (sx >= -1.0f && sx < (float)W_src && sy >= -1.0f && sy < (float)H_src) {
            // Border case: some neighbors out of bounds -> clamp
            int x0 = __float2int_rd(sx);
            int y0 = __float2int_rd(sy);
            int x1 = x0 + 1;
            int y1 = y0 + 1;
            float fx = sx - (float)x0;
            float fy = sy - (float)y0;

            // Clamp
            int cx0 = max(0, min(x0, W_src - 1));
            int cy0 = max(0, min(y0, H_src - 1));
            int cx1 = max(0, min(x1, W_src - 1));
            int cy1 = max(0, min(y1, H_src - 1));

            // Zero weight for out-of-bounds (zeros padding)
            float valid00 = (x0 >= 0 && y0 >= 0) ? 1.0f : 0.0f;
            float valid01 = (x1 < W_src && y0 >= 0) ? 1.0f : 0.0f;
            float valid10 = (x0 >= 0 && y1 < H_src) ? 1.0f : 0.0f;
            float valid11 = (x1 < W_src && y1 < H_src) ? 1.0f : 0.0f;

            float w00 = (1.0f - fx) * (1.0f - fy) * valid00;
            float w01 = fx * (1.0f - fy) * valid01;
            float w10 = (1.0f - fx) * fy * valid10;
            float w11 = fx * fy * valid11;

            int src_base = ib * C * src_plane;
            for (int c = 0; c < C; c++) {
                int off = src_base + c * src_plane;
                float val = w00 * src[off + cy0 * W_src + cx0]
                          + w01 * src[off + cy0 * W_src + cx1]
                          + w10 * src[off + cy1 * W_src + cx0]
                          + w11 * src[off + cy1 * W_src + cx1];
                dst[dst_idx_base + c * dst_plane] = val;
            }
        } else {
            for (int c = 0; c < C; c++) {
                dst[dst_idx_base + c * dst_plane] = 0.0f;
            }
        }
    }
}

// ---- Invert 3x3 kernel (1 block, 1 thread per batch element) ----
__global__ void invert_3x3_kernel(
    const float* __restrict__ M, // [B, 3, 3]
    float* __restrict__ Minv,    // [B, 3, 3]
    int B)
{
    int ib = blockIdx.x * blockDim.x + threadIdx.x;
    if (ib >= B) return;
    inv3x3(M + ib * 9, Minv + ib * 9);
}

// ---- Host function ----
torch::Tensor fused_warp_perspective(
    torch::Tensor src,   // [B, C, H_src, W_src] float32 CUDA
    torch::Tensor M,     // [B, 3, 3] pixel-space homography (src->dst)
    int h_out, int w_out,
    int mode)            // 0=nearest, 1=bilinear
{
    TORCH_CHECK(src.is_cuda(), "src must be CUDA");
    TORCH_CHECK(M.is_cuda(), "M must be CUDA");
    TORCH_CHECK(src.dim() == 4, "src must be [B,C,H,W]");
    TORCH_CHECK(M.dim() == 3 && M.size(1) == 3 && M.size(2) == 3, "M must be [B,3,3]");

    src = src.contiguous().to(torch::kFloat32);
    M = M.contiguous().to(torch::kFloat32);

    int B = src.size(0);
    int C = src.size(1);
    int H_src = src.size(2);
    int W_src = src.size(3);

    // Allocate output
    auto dst = torch::zeros({B, C, h_out, w_out}, src.options());

    // Allocate and compute M_inv on GPU
    auto M_inv = torch::empty_like(M);
    {
        int threads = min(B, 256);
        int blocks = (B + threads - 1) / threads;
        invert_3x3_kernel<<<blocks, threads>>>(
            M.data_ptr<float>(), M_inv.data_ptr<float>(), B);
    }

    // Launch warp kernel
    dim3 block(16, 16);
    dim3 grid(
        (w_out + block.x - 1) / block.x,
        (h_out + block.y - 1) / block.y,
        B);

    if (mode == 0) {
        warp_kernel<0><<<grid, block>>>(
            src.data_ptr<float>(), M_inv.data_ptr<float>(),
            dst.data_ptr<float>(),
            B, C, H_src, W_src, h_out, w_out);
    } else {
        warp_kernel<1><<<grid, block>>>(
            src.data_ptr<float>(), M_inv.data_ptr<float>(),
            dst.data_ptr<float>(),
            B, C, H_src, W_src, h_out, w_out);
    }

    return dst;
}
