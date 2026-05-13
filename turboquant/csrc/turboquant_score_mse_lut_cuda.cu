#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>


namespace {

// ============================================================
// 2-bit MSE-only LUT fused score kernel
//
// Formula:
//
//   score[b,h,t]
//     = mse_norm[b,h,t]
//       * sum_d q_rot[b,h,d] * centroid[index[b,h,t,d]]
//
// LUT rewrite:
//
//   lut[d, c] = q_rot[d] * centroid[c]
//
//   score[b,h,t]
//     = mse_norm[b,h,t]
//       * sum_d lut[d, index[b,h,t,d]]
//
// Inputs:
//   q_rot:                  [B,H,D]
//   packed_mse_indices_t:   [B,H,packed_D,T]
//   mse_norms:              [B,H,T]
//   centroids:              [4]
//
// Output:
//   out_scores:             [B,H,1,T]
//
// Grid:
//   blockIdx.x = token tile
//   blockIdx.y = head
//   blockIdx.z = batch
//
// One block handles:
//   one (b,h), 256 token positions.
//
// Shared memory:
//   LUT[D * 4]
// ============================================================

__global__ void turboquant_mse_lut_score_transposed_kernel(
    const float* __restrict__ q_rot,                  // [B,H,D]
    const uint8_t* __restrict__ packed_mse_indices_t, // [B,H,packed_D,T]
    const float* __restrict__ mse_norms,              // [B,H,T]
    const float* __restrict__ centroids,              // [4]
    float* __restrict__ out_scores,                   // [B,H,1,T]
    int B,
    int H,
    int T,
    int D,
    int packed_D
) {
    extern __shared__ float shared_lut[];

    const int b =
        static_cast<int>(blockIdx.z);

    const int h =
        static_cast<int>(blockIdx.y);

    const int token_base =
        static_cast<int>(blockIdx.x) * blockDim.x;

    const int t =
        token_base + threadIdx.x;

    const int64_t q_base =
        (static_cast<int64_t>(b) * H + h) * D;

    // ============================================================
    // Build LUT:
    //
    //   shared_lut[d * 4 + c]
    //     = q_rot[b,h,d] * centroids[c]
    //
    // Total entries:
    //   D * 4 = 128 * 4 = 512 floats
    // ============================================================
    const int lut_size =
        D * 4;

    for (
        int linear_idx = threadIdx.x;
        linear_idx < lut_size;
        linear_idx += blockDim.x
    ) {
        const int d =
            linear_idx / 4;

        const int c =
            linear_idx % 4;

        shared_lut[linear_idx] =
            q_rot[q_base + d] *
            centroids[c];
    }

    __syncthreads();

    if (t >= T) {
        return;
    }

    const int64_t norm_idx =
        (static_cast<int64_t>(b) * H + h) * T + t;

    float acc =
        0.0f;

    // ============================================================
    // Accumulate LUT-based score
    //
    // packed_mse_indices_t layout:
    //   [B,H,packed_D,T]
    //
    // One byte stores 4 x 2-bit centroid ids.
    // ============================================================
    for (int pd = 0; pd < packed_D; ++pd) {
        const int64_t packed_idx =
            (((static_cast<int64_t>(b) * H + h) * packed_D + pd) * T + t);

        const uint8_t packed_byte =
            packed_mse_indices_t[packed_idx];

        const int c0 =
            static_cast<int>((packed_byte >> 0) & 0x3);

        const int c1 =
            static_cast<int>((packed_byte >> 2) & 0x3);

        const int c2 =
            static_cast<int>((packed_byte >> 4) & 0x3);

        const int c3 =
            static_cast<int>((packed_byte >> 6) & 0x3);

        const int d0 =
            pd * 4 + 0;

        const int d1 =
            pd * 4 + 1;

        const int d2 =
            pd * 4 + 2;

        const int d3 =
            pd * 4 + 3;

        acc +=
            shared_lut[d0 * 4 + c0];

        acc +=
            shared_lut[d1 * 4 + c1];

        acc +=
            shared_lut[d2 * 4 + c2];

        acc +=
            shared_lut[d3 * 4 + c3];
    }

    const float score =
        mse_norms[norm_idx] *
        acc;

    const int64_t out_idx =
        (static_cast<int64_t>(b) * H + h) * T + t;

    out_scores[out_idx] =
        score;
}

} // namespace


// ============================================================
// Host launcher
// ============================================================

torch::Tensor turboquant_mse_lut_score_transposed_cuda(
    torch::Tensor q_rot,
    torch::Tensor packed_mse_indices_t,
    torch::Tensor mse_norms,
    torch::Tensor centroids
) {
    TORCH_CHECK(
        q_rot.is_cuda(),
        "q_rot must be CUDA"
    );

    TORCH_CHECK(
        packed_mse_indices_t.is_cuda(),
        "packed_mse_indices_t must be CUDA"
    );

    TORCH_CHECK(
        mse_norms.is_cuda(),
        "mse_norms must be CUDA"
    );

    TORCH_CHECK(
        centroids.is_cuda(),
        "centroids must be CUDA"
    );

    TORCH_CHECK(
        q_rot.dtype() == torch::kFloat32,
        "q_rot must be float32"
    );

    TORCH_CHECK(
        packed_mse_indices_t.dtype() == torch::kUInt8,
        "packed_mse_indices_t must be uint8"
    );

    TORCH_CHECK(
        mse_norms.dtype() == torch::kFloat32,
        "mse_norms must be float32"
    );

    TORCH_CHECK(
        centroids.dtype() == torch::kFloat32,
        "centroids must be float32"
    );

    TORCH_CHECK(
        q_rot.dim() == 3,
        "q_rot must be [B,H,D]"
    );

    TORCH_CHECK(
        packed_mse_indices_t.dim() == 4,
        "packed_mse_indices_t must be [B,H,packed_D,T]"
    );

    TORCH_CHECK(
        mse_norms.dim() == 3,
        "mse_norms must be [B,H,T]"
    );

    TORCH_CHECK(
        centroids.dim() == 1 &&
        centroids.numel() == 4,
        "centroids must be [4]"
    );

    TORCH_CHECK(
        q_rot.is_contiguous(),
        "q_rot must be contiguous"
    );

    TORCH_CHECK(
        packed_mse_indices_t.is_contiguous(),
        "packed_mse_indices_t must be contiguous"
    );

    TORCH_CHECK(
        mse_norms.is_contiguous(),
        "mse_norms must be contiguous"
    );

    TORCH_CHECK(
        centroids.is_contiguous(),
        "centroids must be contiguous"
    );

    const int B =
        static_cast<int>(q_rot.size(0));

    const int H =
        static_cast<int>(q_rot.size(1));

    const int D =
        static_cast<int>(q_rot.size(2));

    const int packed_D =
        static_cast<int>(packed_mse_indices_t.size(2));

    const int T =
        static_cast<int>(packed_mse_indices_t.size(3));

    TORCH_CHECK(
        packed_D * 4 == D,
        "packed_D mismatch: packed_D*4 must equal D"
    );

    TORCH_CHECK(
        packed_mse_indices_t.size(0) == B,
        "packed_mse_indices_t B mismatch"
    );

    TORCH_CHECK(
        packed_mse_indices_t.size(1) == H,
        "packed_mse_indices_t H mismatch"
    );

    TORCH_CHECK(
        mse_norms.size(0) == B,
        "mse_norms B mismatch"
    );

    TORCH_CHECK(
        mse_norms.size(1) == H,
        "mse_norms H mismatch"
    );

    TORCH_CHECK(
        mse_norms.size(2) == T,
        "mse_norms T mismatch"
    );

    auto out =
        torch::empty(
            {B, H, 1, T},
            torch::TensorOptions()
                .dtype(torch::kFloat32)
                .device(q_rot.device())
        );

    constexpr int threads =
        256;

    const int token_tiles =
        (T + threads - 1) / threads;

    const dim3 grid(
        token_tiles,
        H,
        B
    );

    const size_t shared_bytes =
        static_cast<size_t>(D * 4) *
        sizeof(float);

    turboquant_mse_lut_score_transposed_kernel<<<
        grid,
        threads,
        shared_bytes,
        at::cuda::getDefaultCUDAStream()
    >>>(
        q_rot.data_ptr<float>(),
        packed_mse_indices_t.data_ptr<uint8_t>(),
        mse_norms.data_ptr<float>(),
        centroids.data_ptr<float>(),
        out.data_ptr<float>(),
        B,
        H,
        T,
        D,
        packed_D
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}
