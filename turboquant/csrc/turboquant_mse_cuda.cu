#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <vector>


namespace {

// ============================================================
// Fused MSE centroid assignment + 2-bit packing + x_hat_rot
//
// Inputs:
//   x_norm:    [N, D] float32
//   norms:     [N] float32
//   centroids: [4] float32
//
// Outputs:
//   packed_indices: [N, D/4] uint8
//   x_hat_rot:      [N, D] float32
//
// Each thread processes 4 coordinates and writes:
//   - one packed uint8
//   - four x_hat_rot float values
// ============================================================

__device__ __forceinline__ int nearest_centroid_idx(
    float x,
    const float c0,
    const float c1,
    const float c2,
    const float c3
) {
    float best_dist = fabsf(x - c0);
    int best_idx = 0;

    float d1 = fabsf(x - c1);
    if (d1 < best_dist) {
        best_dist = d1;
        best_idx = 1;
    }

    float d2 = fabsf(x - c2);
    if (d2 < best_dist) {
        best_dist = d2;
        best_idx = 2;
    }

    float d3 = fabsf(x - c3);
    if (d3 < best_dist) {
        best_idx = 3;
    }

    // Strict '<' preserves torch.argmin-style first-index tie behavior.
    return best_idx;
}


__global__ void mse_assign_pack_reconstruct_rot_kernel(
    const float* __restrict__ x_norm,
    const float* __restrict__ norms,
    const float* __restrict__ centroids,
    uint8_t* __restrict__ packed_indices,
    float* __restrict__ x_hat_rot,
    int64_t N,
    int64_t D
) {
    const int64_t packed_D = D / 4;
    const int64_t out_idx =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    const int64_t total_packed = N * packed_D;

    if (out_idx >= total_packed) {
        return;
    }

    const int64_t n = out_idx / packed_D;
    const int64_t packed_col = out_idx % packed_D;
    const int64_t d_base = packed_col * 4;

    const int64_t row_base = n * D;

    const float c0 = centroids[0];
    const float c1 = centroids[1];
    const float c2 = centroids[2];
    const float c3 = centroids[3];

    const float norm = norms[n];

    const float x0 = x_norm[row_base + d_base + 0];
    const float x1 = x_norm[row_base + d_base + 1];
    const float x2 = x_norm[row_base + d_base + 2];
    const float x3 = x_norm[row_base + d_base + 3];

    const int i0 = nearest_centroid_idx(x0, c0, c1, c2, c3);
    const int i1 = nearest_centroid_idx(x1, c0, c1, c2, c3);
    const int i2 = nearest_centroid_idx(x2, c0, c1, c2, c3);
    const int i3 = nearest_centroid_idx(x3, c0, c1, c2, c3);

    packed_indices[out_idx] =
        static_cast<uint8_t>(
            (i0 & 0x3) |
            ((i1 & 0x3) << 2) |
            ((i2 & 0x3) << 4) |
            ((i3 & 0x3) << 6)
        );

    const float centroid_values[4] = {c0, c1, c2, c3};

    x_hat_rot[row_base + d_base + 0] = centroid_values[i0] * norm;
    x_hat_rot[row_base + d_base + 1] = centroid_values[i1] * norm;
    x_hat_rot[row_base + d_base + 2] = centroid_values[i2] * norm;
    x_hat_rot[row_base + d_base + 3] = centroid_values[i3] * norm;
}

} // namespace


// ============================================================
// C++ entry
// ============================================================

std::vector<torch::Tensor> mse_assign_pack_reconstruct_rot_cuda(
    torch::Tensor x_norm,
    torch::Tensor norms,
    torch::Tensor centroids
) {
    TORCH_CHECK(
        x_norm.is_cuda(),
        "x_norm must be a CUDA tensor"
    );

    TORCH_CHECK(
        norms.is_cuda(),
        "norms must be a CUDA tensor"
    );

    TORCH_CHECK(
        centroids.is_cuda(),
        "centroids must be a CUDA tensor"
    );

    TORCH_CHECK(
        x_norm.scalar_type() == torch::kFloat32,
        "x_norm must be float32"
    );

    TORCH_CHECK(
        norms.scalar_type() == torch::kFloat32,
        "norms must be float32"
    );

    TORCH_CHECK(
        centroids.scalar_type() == torch::kFloat32,
        "centroids must be float32"
    );

    TORCH_CHECK(
        x_norm.dim() == 2,
        "x_norm must be [N,D]"
    );

    TORCH_CHECK(
        norms.dim() == 1,
        "norms must be [N]"
    );

    TORCH_CHECK(
        centroids.dim() == 1 && centroids.numel() == 4,
        "centroids must be shape [4]"
    );

    TORCH_CHECK(
        x_norm.is_contiguous(),
        "x_norm must be contiguous"
    );

    TORCH_CHECK(
        norms.is_contiguous(),
        "norms must be contiguous"
    );

    TORCH_CHECK(
        centroids.is_contiguous(),
        "centroids must be contiguous"
    );

    const int64_t N = x_norm.size(0);
    const int64_t D = x_norm.size(1);

    TORCH_CHECK(
        norms.size(0) == N,
        "norm count mismatch"
    );

    TORCH_CHECK(
        D % 4 == 0,
        "D must be divisible by 4"
    );

    auto packed_indices = torch::empty(
        {N, D / 4},
        torch::TensorOptions()
            .dtype(torch::kUInt8)
            .device(x_norm.device())
    );

    auto x_hat_rot = torch::empty(
        {N, D},
        torch::TensorOptions()
            .dtype(torch::kFloat32)
            .device(x_norm.device())
    );

    const int64_t total_packed = N * (D / 4);

    constexpr int threads = 256;
    const int blocks = static_cast<int>(
        (total_packed + threads - 1) / threads
    );

    mse_assign_pack_reconstruct_rot_kernel<<<
        blocks,
        threads,
        0,
        at::cuda::getDefaultCUDAStream()
    >>>(
        x_norm.data_ptr<float>(),
        norms.data_ptr<float>(),
        centroids.data_ptr<float>(),
        packed_indices.data_ptr<uint8_t>(),
        x_hat_rot.data_ptr<float>(),
        N,
        D
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {
        packed_indices,
        x_hat_rot,
    };
}
