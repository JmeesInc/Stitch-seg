#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// ---- Iterated 3x3 max-dilation kernel with shared memory ----
// Each launch does `K` steps of 3x3 max dilation on a 2D float mask.
// Uses shared memory tile with K-pixel halo for data reuse.
//
// TILE = blockDim (e.g., 16x16)
// Shared memory = (TILE + 2*K) x (TILE + 2*K) floats
// After K sequential dilation steps within shared memory, the center
// TILE region is written back.

// We template on K (number of dilation steps per launch) for unrolling.
template <int TILE_X, int TILE_Y, int K>
__global__ void dilate_and_accumulate_kernel(
    const float* __restrict__ input,  // [H, W]
    float* __restrict__ accum,        // [H, W] accumulate dilated masks
    float* __restrict__ output,       // [H, W] final dilated state
    int H, int W)
{
    // Shared memory tile dimensions
    constexpr int SM_W = TILE_X + 2 * K;
    constexpr int SM_H = TILE_Y + 2 * K;

    extern __shared__ float smem[];
    // We use two buffers to ping-pong
    float* buf0 = smem;
    float* buf1 = smem + SM_W * SM_H;

    // Global position of this tile's top-left corner
    int tile_x0 = blockIdx.x * TILE_X;
    int tile_y0 = blockIdx.y * TILE_Y;

    // Each thread loads multiple elements to fill the shared memory tile
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int num_threads_x = blockDim.x;
    int num_threads_y = blockDim.y;

    // Load input into buf0 (with halo)
    for (int sy = ty; sy < SM_H; sy += num_threads_y) {
        for (int sx = tx; sx < SM_W; sx += num_threads_x) {
            int gx = tile_x0 - K + sx;
            int gy = tile_y0 - K + sy;
            float val = 0.0f;
            if (gx >= 0 && gx < W && gy >= 0 && gy < H) {
                val = input[gy * W + gx];
            }
            buf0[sy * SM_W + sx] = val;
        }
    }
    __syncthreads();

    // Perform K iterations of 3x3 max dilation in shared memory
    float* src = buf0;
    float* dst_buf = buf1;

    for (int step = 0; step < K; step++) {
        // After each dilation step, the valid region shrinks by 1 pixel on each side.
        // Step `step` reads from region with halo (K-step), writes to region with halo (K-step-1).
        // For the accumulation, we need the value at the center tile position after this step.

        int halo = K - step - 1; // remaining halo after this step
        int valid_w = TILE_X + 2 * halo;
        int valid_h = TILE_Y + 2 * halo;

        // Each thread handles multiple output positions within the valid region
        for (int oy = ty; oy < valid_h; oy += num_threads_y) {
            for (int ox = tx; ox < valid_w; ox += num_threads_x) {
                int sx = ox + (step + 1) - 1; // offset into src buffer
                int sy = oy + (step + 1) - 1;
                // 3x3 max
                float m = src[sy * SM_W + sx]; // center
                // SAFETY: sx >= step, sy >= step; sx+1 <= TILE_X+2*K-step-2, sy+1 <= ...
                // We access 3x3 neighborhood around (sx, sy) in src
                float v;
                v = src[(sy - 1) * SM_W + (sx - 1)]; m = fmaxf(m, v);
                v = src[(sy - 1) * SM_W + (sx    )]; m = fmaxf(m, v);
                v = src[(sy - 1) * SM_W + (sx + 1)]; m = fmaxf(m, v);
                v = src[(sy    ) * SM_W + (sx - 1)]; m = fmaxf(m, v);
                v = src[(sy    ) * SM_W + (sx + 1)]; m = fmaxf(m, v);
                v = src[(sy + 1) * SM_W + (sx - 1)]; m = fmaxf(m, v);
                v = src[(sy + 1) * SM_W + (sx    )]; m = fmaxf(m, v);
                v = src[(sy + 1) * SM_W + (sx + 1)]; m = fmaxf(m, v);

                // Write to dst_buf at the appropriate offset
                // dst_buf valid region starts 1 pixel inward from src
                dst_buf[(oy + 1) * SM_W + (ox + 1)] = m;
            }
        }
        __syncthreads();

        // Accumulate: each thread adds the dilated value for its tile positions
        // After step `step`, the center TILE values are at offset (step+1) in dst_buf
        for (int py = ty; py < TILE_Y; py += num_threads_y) {
            for (int px = tx; px < TILE_X; px += num_threads_x) {
                int gx = tile_x0 + px;
                int gy = tile_y0 + py;
                if (gx < W && gy < H) {
                    int sm_x = px + K; // center of tile in the original buffer = K offset
                    int sm_y = py + K;
                    // But after step+1 dilations, the tile center has shifted by (step+1) from buf start
                    // Actually, dst_buf was written starting at offset 1 from src's valid start.
                    // The tile center in dst_buf after step `step` is at:
                    //   sm_x = px + (K - step - 1) + 1 = px + K - step
                    //   -> No, let's reconsider...
                    // The valid region of dst_buf starts at offset 1 from src's valid start.
                    // src's valid region for step `step` starts at offset `step` from buf0.
                    // So dst_buf's valid region starts at offset `step+1` from buf0.
                    // The tile center pixels (px, py) in global correspond to
                    //   buf0 offset: (px + K, py + K)
                    // After step+1 dilations, these are at dst_buf offset: (px + K, py + K)
                    // because we wrote dst_buf at (oy+1, ox+1) where oy/ox loop over
                    // the valid region starting from 0, which maps to src offset (step, step).
                    // oy = py + K - (step+1), ox = px + K - (step+1)
                    // dst_buf position = (oy+1, ox+1) = (py+K-step, px+K-step)
                    // Actually let me just use the simpler calculation:
                    int dst_sm_x = px + K - step;
                    int dst_sm_y = py + K - step;
                    atomicAdd(&accum[gy * W + gx], dst_buf[dst_sm_y * SM_W + dst_sm_x]);
                }
            }
        }
        __syncthreads();

        // Swap buffers
        float* tmp = src;
        src = dst_buf;
        dst_buf = tmp;
    }

    // Write final dilated state to output
    for (int py = ty; py < TILE_Y; py += num_threads_y) {
        for (int px = tx; px < TILE_X; px += num_threads_x) {
            int gx = tile_x0 + px;
            int gy = tile_y0 + py;
            if (gx < W && gy < H) {
                int dst_sm_x = px + K - (K - 1); // = px + 1... no
                // After K steps, the center tile in `src` (which is the last written buffer) is at:
                // offset from buf start: K (unchanged, because we track the center)
                // Actually the last write was to `dst_buf` (now `src` after swap), and
                // the center tile is at (px + K - (K-1), py + K - (K-1)) = (px+1, py+1)
                // Wait, let me reconsider with K steps.
                // After step 0: dst_buf center = (px + K - 0, py + K - 0) -- nope
                // Let me use a cleaner approach: just read from `src` at the known center offset.
                // After K dilations from buf0 offset K, the center pixel stays at offset K
                // in the result (since each dilation step removes 1 halo).
                // In src (the last dst_buf, swapped), the tile center is at:
                //   position = K - (number of offsets applied)
                // Actually, for K steps: the output valid start is at offset K from buf0 start.
                // But our ping-pong places it differently. Let me just directly track:
                // After K steps, the last dst_buf (now src) was written with offset mapping
                // where the tile center (px+K, py+K in buf0) maps to position
                // (px + K - K, py + K - K) = (px, py) ... plus the +1 from each dst_buf write...
                // No, let me think more carefully.
                //
                // Simplification: after the K-step loop, the final dilated mask for the tile
                // center pixels is stored in accum (accumulated) and we need the final state for
                // the output. But we already wrote to accum during the loop.
                // For the output (final dilated state), read from src at the right offset.
                //
                // After step k (0-indexed), dst_buf center = (px + K - k, py + K - k)
                // After all K steps (last step = K-1), center = (px + K - (K-1), py + K - (K-1))
                //                                              = (px + 1, py + 1)
                output[gy * W + gx] = src[(py + 1) * SM_W + (px + 1)];
            }
        }
    }
}


// ---- Simpler approach: just iterate max_pool in a loop on the GPU ----
// This avoids shared memory complexity and still fuses 50 kernel launches
// into much fewer. The key insight: we can do multiple dilation steps
// in a single kernel by reading/writing through global memory with double buffering.
// But the simpler approach is to just accumulate in one fused kernel.

// Actually, let's use an even simpler and more robust approach:
// A single kernel that does K iterations of 3x3 max-dilation in shared memory
// with proper halo management. But given the complexity above, let's use a
// straightforward iteration approach with a small K (like 5-10) per launch.

// ---- Simple iterative dilation + accumulation ----
// Each kernel launch does ONE dilation step and adds the result to accum.
__global__ void dilate_step_and_accum(
    const float* __restrict__ current, // [H, W]
    float* __restrict__ next,          // [H, W]
    float* __restrict__ accum,         // [H, W]
    int H, int W)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    // 3x3 max
    float m = current[y * W + x];

    #pragma unroll
    for (int dy = -1; dy <= 1; dy++) {
        int ny = y + dy;
        if (ny >= 0 && ny < H) {
            #pragma unroll
            for (int dx = -1; dx <= 1; dx++) {
                int nx = x + dx;
                if (nx >= 0 && nx < W) {
                    m = fmaxf(m, current[ny * W + nx]);
                }
            }
        }
    }

    next[y * W + x] = m;
    accum[y * W + x] += m;
}

// ---- Multi-step dilation with shared memory (K steps per launch) ----
// TILE_SIZE: block dimension (threads per axis)
// K: number of dilation steps per kernel launch
// Shared memory: 2 * (TILE_SIZE + 2*K)^2 floats for double buffering
template <int K>
__global__ void dilate_multi_step_accum(
    const float* __restrict__ current, // [H, W]
    float* __restrict__ output,        // [H, W] state after K dilations
    float* __restrict__ accum,         // [H, W] accumulate each step's result
    int H, int W)
{
    constexpr int TILE = 16;
    constexpr int SM_DIM = TILE + 2 * K;

    __shared__ float buf[2][SM_DIM][SM_DIM];

    int tile_x0 = blockIdx.x * TILE;
    int tile_y0 = blockIdx.y * TILE;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Load input into buf[0] with K-pixel halo
    for (int sy = ty; sy < SM_DIM; sy += TILE) {
        for (int sx = tx; sx < SM_DIM; sx += TILE) {
            int gx = tile_x0 - K + sx;
            int gy = tile_y0 - K + sy;
            float val = 0.0f;
            if (gx >= 0 && gx < W && gy >= 0 && gy < H) {
                val = current[gy * W + gx];
            }
            buf[0][sy][sx] = val;
        }
    }
    __syncthreads();

    int cur = 0;

    for (int step = 0; step < K; step++) {
        int nxt = 1 - cur;
        // After `step` dilations, the valid halo remaining is K - step.
        // We dilate from buf[cur] to buf[nxt].
        // The region we need to compute in buf[nxt] has halo K - step - 1.
        int halo = K - step - 1;
        int region_w = TILE + 2 * halo;
        int region_h = TILE + 2 * halo;
        // In buf[cur], the valid data starts at offset `step` from buf origin.
        // So the center of the tile in buf[cur] is at (tx + K, ty + K).
        // For the output region in buf[nxt], the data starts at offset `step + 1`.

        for (int ry = ty; ry < region_h; ry += TILE) {
            for (int rx = tx; rx < region_w; rx += TILE) {
                // Position in buf[cur]
                int bx = rx + step + 1; // center of output region in buf
                int by = ry + step + 1;

                // 3x3 max in buf[cur] centered at (bx, by)
                float m = buf[cur][by][bx];
                m = fmaxf(m, buf[cur][by - 1][bx - 1]);
                m = fmaxf(m, buf[cur][by - 1][bx    ]);
                m = fmaxf(m, buf[cur][by - 1][bx + 1]);
                m = fmaxf(m, buf[cur][by    ][bx - 1]);
                m = fmaxf(m, buf[cur][by    ][bx + 1]);
                m = fmaxf(m, buf[cur][by + 1][bx - 1]);
                m = fmaxf(m, buf[cur][by + 1][bx    ]);
                m = fmaxf(m, buf[cur][by + 1][bx + 1]);

                // Write to buf[nxt] at same position
                buf[nxt][by][bx] = m;
            }
        }
        __syncthreads();

        // Accumulate tile center values to accum
        // Tile center in buf[nxt] is at (tx + K, ty + K)
        if (tx < TILE && ty < TILE) {
            int gx = tile_x0 + tx;
            int gy = tile_y0 + ty;
            if (gx < W && gy < H) {
                accum[gy * W + gx] += buf[nxt][ty + K][tx + K];
            }
        }
        __syncthreads();

        cur = nxt;
    }

    // Write final dilated state
    if (tx < TILE && ty < TILE) {
        int gx = tile_x0 + tx;
        int gy = tile_y0 + ty;
        if (gx < W && gy < H) {
            output[gy * W + gx] = buf[cur][ty + K][tx + K];
        }
    }
}


// ---- Host function ----
// Replaces the Python fast_gradient_mask:
//   1. Downscale by `downscale` via avg_pool
//   2. Threshold > 0.3
//   3. R_small = radius // downscale iterations of 3x3 max dilation, accumulating
//   4. grad = accum / R_small
//   5. Upscale back, clamp [0, 1]
torch::Tensor fast_gradient_mask(
    torch::Tensor inv_mask,  // [1, 1, H, W] float32 CUDA
    int radius,              // e.g. 201
    int downscale)           // e.g. 4
{
    TORCH_CHECK(inv_mask.is_cuda(), "inv_mask must be CUDA");
    inv_mask = inv_mask.contiguous().to(torch::kFloat32);

    int H = inv_mask.size(2);
    int W = inv_mask.size(3);

    // 1. Downscale via avg_pool2d
    auto mask_small = torch::avg_pool2d(inv_mask, {downscale, downscale}, {downscale, downscale});
    mask_small = (mask_small > 0.3f).to(torch::kFloat32);

    int Hs = mask_small.size(2);
    int Ws = mask_small.size(3);
    int R_small = radius / downscale; // integer division

    // Get 2D view
    auto current = mask_small.squeeze(0).squeeze(0); // [Hs, Ws]
    auto accum = torch::zeros({Hs, Ws}, current.options());

    // Choose strategy based on R_small
    // For K=25 multi-step kernel: shared memory = 2 * (16 + 50)^2 * 4 = ~35KB (fits 48KB)
    // For K=10: shared memory = 2 * (16 + 20)^2 * 4 = ~10KB
    // We'll use K=10 and launch R_small/10 times (5 launches for R_small=50)

    constexpr int K = 10;
    constexpr int TILE = 16;
    constexpr int SM_DIM = TILE + 2 * K;
    size_t smem_size = 2 * SM_DIM * SM_DIM * sizeof(float);

    dim3 block(TILE, TILE);
    dim3 grid((Ws + TILE - 1) / TILE, (Hs + TILE - 1) / TILE);

    auto next_state = torch::zeros_like(current);

    int full_launches = R_small / K;
    int remaining = R_small % K;

    for (int i = 0; i < full_launches; i++) {
        dilate_multi_step_accum<K><<<grid, block, smem_size>>>(
            current.data_ptr<float>(),
            next_state.data_ptr<float>(),
            accum.data_ptr<float>(),
            Hs, Ws);
        // Swap current <-> next_state for the next launch
        std::swap(current, next_state);
    }

    // Handle remaining steps one at a time
    for (int i = 0; i < remaining; i++) {
        dilate_step_and_accum<<<grid, block>>>(
            current.data_ptr<float>(),
            next_state.data_ptr<float>(),
            accum.data_ptr<float>(),
            Hs, Ws);
        std::swap(current, next_state);
    }

    // Normalize
    auto grad_small = accum / (float)R_small;
    grad_small = grad_small.unsqueeze(0).unsqueeze(0); // [1, 1, Hs, Ws]

    // Upscale back to original size
    auto grad = torch::nn::functional::interpolate(
        grad_small,
        torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>{H, W})
            .mode(torch::kBilinear)
            .align_corners(false));

    // Clamp [0, 1]
    return grad.clamp(0.0f, 1.0f);
}
