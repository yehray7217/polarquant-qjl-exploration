#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>


namespace {

// ============================================================
// 4-bit MSE-only LUT fused score kernel
//
// Formula:
//
//   score[b,h,t]
//     = mse_norm[b,h,t]
//       * sum_d q_rot[b,h,d] * centroid[index[b,h,t,d]]
//
// LUT:
//
//   lut[d, c] = q_rot[d] * centroid[c]
//
// where c ∈ {0, ..., 15}
//
// Inputs:
//   q_rot:                       [B,H,D]
//   packed_mse_indices_t:        [B,H,packed_D,T]
//   mse_norms:                   [B,H,T]
//   centroids:                   [16]
//
// Output:
//   out_scores:                  [B,H,1,T]
//
// packed_D = D / 2
//
// One byte stores:
//   low  nibble -> coordinate d0
//   high nibble -> coordinate d1
// ============================================================

__global__ void turboquant_mse_lut_4bit_score_transposed_kernel(
    const float* __restrict__ q_rot,                  // [B,H,D]
    const uint8_t* __restrict__ packed_indices_t,     // [B,H,packed_D,T]
    const float* __restrict__ mse_norms,              // [B,H,T]
    const float* __restrict__ centroids,              // [16]
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

    // ------------------------------------------------------------
    // Build LUT:
    //
    // shared_lut[d * 16 + c]
    //   = q_rot[b,h,d] * centroids[c]
    // ------------------------------------------------------------
    const int lut_size =
        D * 16;

    for (
        int linear_idx = threadIdx.x;
        linear_idx < lut_size;
        linear_idx += blockDim.x
    ) {
        const int d =
            linear_idx / 16;

        const int c =
            linear_idx % 16;

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

    // ------------------------------------------------------------
    // Accumulate packed 4-bit ids.
    //
    // Layout:
    //   [B,H,packed_D,T]
    //
    // One byte -> 2 coordinates.
    // ------------------------------------------------------------
    for (int pd = 0; pd < packed_D; ++pd) {
        const int64_t packed_idx =
            (((static_cast<int64_t>(b) * H + h) * packed_D + pd) * T + t);

        const uint8_t packed_byte =
            packed_indices_t[packed_idx];

        const int c0 =
            static_cast<int>(packed_byte & 0x0F);

        const int c1 =
            static_cast<int>((packed_byte >> 4) & 0x0F);

        const int d0 =
            pd * 2 + 0;

        const int d1 =
            pd * 2 + 1;

        acc +=
            shared_lut[d0 * 16 + c0];

        acc +=
            shared_lut[d1 * 16 + c1];
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


torch::Tensor turboquant_mse_lut_4bit_score_transposed_cuda(
    torch::Tensor q_rot,
    torch::Tensor packed_indices_t,
    torch::Tensor mse_norms,
    torch::Tensor centroids
) {
    TORCH_CHECK(q_rot.is_cuda(), "q_rot must be CUDA");
    TORCH_CHECK(packed_indices_t.is_cuda(), "packed_indices_t must be CUDA");
    TORCH_CHECK(mse_norms.is_cuda(), "mse_norms must be CUDA");
    TORCH_CHECK(centroids.is_cuda(), "centroids must be CUDA");

    TORCH_CHECK(q_rot.dtype() == torch::kFloat32, "q_rot must be float32");
    TORCH_CHECK(packed_indices_t.dtype() == torch::kUInt8, "packed_indices_t must be uint8");
    TORCH_CHECK(mse_norms.dtype() == torch::kFloat32, "mse_norms must be float32");
    TORCH_CHECK(centroids.dtype() == torch::kFloat32, "centroids must be float32");

    TORCH_CHECK(q_rot.dim() == 3, "q_rot must be [B,H,D]");
    TORCH_CHECK(packed_indices_t.dim() == 4, "packed_indices_t must be [B,H,packed_D,T]");
    TORCH_CHECK(mse_norms.dim() == 3, "mse_norms must be [B,H,T]");
    TORCH_CHECK(centroids.dim() == 1 && centroids.numel() == 16, "centroids must be [16]");

    TORCH_CHECK(q_rot.is_contiguous(), "q_rot must be contiguous");
    TORCH_CHECK(packed_indices_t.is_contiguous(), "packed_indices_t must be contiguous");
    TORCH_CHECK(mse_norms.is_contiguous(), "mse_norms must be contiguous");
    TORCH_CHECK(centroids.is_contiguous(), "centroids must be contiguous");

    const int B =
        static_cast<int>(q_rot.size(0));

    const int H =
        static_cast<int>(q_rot.size(1));

    const int D =
        static_cast<int>(q_rot.size(2));

    const int packed_D =
        static_cast<int>(packed_indices_t.size(2));

    const int T =
        static_cast<int>(packed_indices_t.size(3));

    TORCH_CHECK(
        packed_D * 2 == D,
        "packed_D mismatch: packed_D*2 must equal D"
    );

    TORCH_CHECK(
        packed_indices_t.size(0) == B,
        "packed_indices_t B mismatch"
    );

    TORCH_CHECK(
        packed_indices_t.size(1) == H,
        "packed_indices_t H mismatch"
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
        static_cast<size_t>(D * 16) *
        sizeof(float);

    turboquant_mse_lut_4bit_score_transposed_kernel<<<
        grid,
        threads,
        shared_bytes,
        at::cuda::getDefaultCUDAStream()
    >>>(
        q_rot.data_ptr<float>(),
        packed_indices_t.data_ptr<uint8_t>(),
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
