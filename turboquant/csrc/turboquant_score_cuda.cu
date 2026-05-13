#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>


namespace {

__global__ void turboquant_decode_score_kernel(
    const float* __restrict__ q_rot,                 // [B,H,D]
    const float* __restrict__ sq,                    // [B,H,M]
    const uint8_t* __restrict__ packed_mse_indices,  // [B,H,capacity,D/4]
    const float* __restrict__ mse_norms,             // [B,H,capacity]
    const uint8_t* __restrict__ packed_sign_bits,    // [B,H,capacity,M/8]
    const float* __restrict__ residual_norms,        // [B,H,capacity]
    const float* __restrict__ centroids,             // [4]
    float* __restrict__ out_scores,                  // [B,H,T]
    int B,
    int H,
    int T,           // active seq_len
    int capacity,    // physical cache stride
    int D,
    int M,
    int packed_D,
    int packed_M
) {
    const int linear_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = B * H * T;

    if (linear_idx >= total) {
        return;
    }

    const int t = linear_idx % T;
    const int bh = linear_idx / T;
    const int h = bh % H;
    const int b = bh / H;

    const int q_base = (b * H + h) * D;
    const int sq_base = (b * H + h) * M;

    // Buffer tensors are physically [B,H,capacity,...]
    const int buffer_token_base =
        (b * H + h) * capacity + t;

    // Output scores are logically [B,H,T]
    const int output_token_base =
        (b * H + h) * T + t;

    const int packed_idx_base =
        buffer_token_base * packed_D;

    const int packed_sign_base =
        buffer_token_base * packed_M;

    // ============================================================
    // MSE contribution:
    //
    //   mse_norm * <q_rot, centroid_lookup(indices)>
    //
    // Packed layout:
    //   1 byte = 4 x 2-bit codes
    // ============================================================

    float mse_acc = 0.0f;

    for (int byte_idx = 0; byte_idx < packed_D; ++byte_idx) {
        const uint8_t packed_byte =
            packed_mse_indices[packed_idx_base + byte_idx];

        const int d0 = byte_idx * 4 + 0;
        const int d1 = byte_idx * 4 + 1;
        const int d2 = byte_idx * 4 + 2;
        const int d3 = byte_idx * 4 + 3;

        const uint8_t c0 = (packed_byte >> 0) & 0x3;
        const uint8_t c1 = (packed_byte >> 2) & 0x3;
        const uint8_t c2 = (packed_byte >> 4) & 0x3;
        const uint8_t c3 = (packed_byte >> 6) & 0x3;

        mse_acc +=
            q_rot[q_base + d0] *
            centroids[static_cast<int>(c0)];

        mse_acc +=
            q_rot[q_base + d1] *
            centroids[static_cast<int>(c1)];

        mse_acc +=
            q_rot[q_base + d2] *
            centroids[static_cast<int>(c2)];

        mse_acc +=
            q_rot[q_base + d3] *
            centroids[static_cast<int>(c3)];
    }

    const float mse_part =
        mse_norms[buffer_token_base] * mse_acc;

    // ============================================================
    // QJL residual contribution:
    //
    //   sqrt(pi/2)/M
    //   * residual_norm
    //   * <Sq, sign_bits>
    //
    // Packed layout:
    //   1 byte = 8 x 1-bit signs
    // ============================================================

    float qjl_acc = 0.0f;

    for (int byte_idx = 0; byte_idx < packed_M; ++byte_idx) {
        const uint8_t packed_byte =
            packed_sign_bits[packed_sign_base + byte_idx];

        const int m0 = byte_idx * 8 + 0;
        const int m1 = byte_idx * 8 + 1;
        const int m2 = byte_idx * 8 + 2;
        const int m3 = byte_idx * 8 + 3;
        const int m4 = byte_idx * 8 + 4;
        const int m5 = byte_idx * 8 + 5;
        const int m6 = byte_idx * 8 + 6;
        const int m7 = byte_idx * 8 + 7;

        const float s0 =
            ((packed_byte >> 0) & 0x1) ? 1.0f : -1.0f;

        const float s1 =
            ((packed_byte >> 1) & 0x1) ? 1.0f : -1.0f;

        const float s2 =
            ((packed_byte >> 2) & 0x1) ? 1.0f : -1.0f;

        const float s3 =
            ((packed_byte >> 3) & 0x1) ? 1.0f : -1.0f;

        const float s4 =
            ((packed_byte >> 4) & 0x1) ? 1.0f : -1.0f;

        const float s5 =
            ((packed_byte >> 5) & 0x1) ? 1.0f : -1.0f;

        const float s6 =
            ((packed_byte >> 6) & 0x1) ? 1.0f : -1.0f;

        const float s7 =
            ((packed_byte >> 7) & 0x1) ? 1.0f : -1.0f;

        qjl_acc += sq[sq_base + m0] * s0;
        qjl_acc += sq[sq_base + m1] * s1;
        qjl_acc += sq[sq_base + m2] * s2;
        qjl_acc += sq[sq_base + m3] * s3;
        qjl_acc += sq[sq_base + m4] * s4;
        qjl_acc += sq[sq_base + m5] * s5;
        qjl_acc += sq[sq_base + m6] * s6;
        qjl_acc += sq[sq_base + m7] * s7;
    }

    constexpr float SQRT_PI_OVER_2 =
        1.2533141373155001f;

    const float qjl_scale =
        SQRT_PI_OVER_2 / static_cast<float>(M);

    const float qjl_part =
        qjl_scale *
        residual_norms[buffer_token_base] *
        qjl_acc;

    out_scores[output_token_base] =
        mse_part + qjl_part;
}

} // namespace


torch::Tensor turboquant_decode_score_cuda(
    torch::Tensor q_rot,
    torch::Tensor sq,
    torch::Tensor packed_mse_indices,
    torch::Tensor mse_norms,
    torch::Tensor packed_qjl_sign_bits,
    torch::Tensor residual_norms,
    torch::Tensor centroids,
    int64_t seq_len
) {
    const int B =
        static_cast<int>(q_rot.size(0));

    const int H =
        static_cast<int>(q_rot.size(1));

    const int D =
        static_cast<int>(q_rot.size(2));

    const int M =
        static_cast<int>(sq.size(2));

    const int capacity =
        static_cast<int>(packed_mse_indices.size(2));

    const int T =
        static_cast<int>(seq_len);

    const int packed_D =
        static_cast<int>(packed_mse_indices.size(3));

    const int packed_M =
        static_cast<int>(packed_qjl_sign_bits.size(3));

    auto scores = torch::empty(
        {B, H, T},
        torch::TensorOptions()
            .dtype(torch::kFloat32)
            .device(q_rot.device())
    );

    const int total =
        B * H * T;

    constexpr int threads = 256;

    const int blocks =
        (total + threads - 1) / threads;

    turboquant_decode_score_kernel<<<
        blocks,
        threads,
        0,
        at::cuda::getDefaultCUDAStream()
    >>>(
        q_rot.data_ptr<float>(),
        sq.data_ptr<float>(),
        packed_mse_indices.data_ptr<uint8_t>(),
        mse_norms.data_ptr<float>(),
        packed_qjl_sign_bits.data_ptr<uint8_t>(),
        residual_norms.data_ptr<float>(),
        centroids.data_ptr<float>(),
        scores.data_ptr<float>(),
        B,
        H,
        T,
        capacity,
        D,
        M,
        packed_D,
        packed_M
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return scores;
}