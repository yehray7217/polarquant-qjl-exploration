#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>


namespace {

// ============================================================
// Transposed-layout packed TurboQuant score kernel
//
// Layout:
//   packed_mse_indices_t:   [B,H,packed_D,T]
//   packed_qjl_sign_bits_t:[B,H,packed_M,T]
//
// One thread computes one score:
//   output[b,h,0,t]
//
// This layout is score-friendly:
//   for fixed packed feature j,
//   adjacent threads in a warp read adjacent token positions.
// ============================================================

__device__ __forceinline__ float centroid_from_2bit(
    uint8_t idx,
    const float* __restrict__ centroids
) {
    return centroids[idx & 0x3];
}


__global__ void turboquant_decode_score_transposed_kernel(
    const float* __restrict__ q_rot,                    // [B,H,D]
    const float* __restrict__ q_sketch,                 // [B,H,M]
    const uint8_t* __restrict__ packed_mse_indices_t,   // [B,H,packed_D,T]
    const float* __restrict__ mse_norms,                // [B,H,T]
    const uint8_t* __restrict__ packed_qjl_sign_bits_t, // [B,H,packed_M,T]
    const float* __restrict__ qjl_residual_norms,       // [B,H,T]
    const float* __restrict__ centroids,                // [4]
    float* __restrict__ out_scores,                     // [B,H,1,T]
    int B,
    int H,
    int T,
    int D,
    int M,
    int packed_D,
    int packed_M
) {
    const int64_t global_idx =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    const int64_t total =
        static_cast<int64_t>(B) * H * T;

    if (global_idx >= total) {
        return;
    }

    const int t =
        static_cast<int>(global_idx % T);

    const int h =
        static_cast<int>((global_idx / T) % H);

    const int b =
        static_cast<int>(global_idx / (static_cast<int64_t>(H) * T));

    const int64_t q_rot_base =
        (static_cast<int64_t>(b) * H + h) * D;

    const int64_t q_sketch_base =
        (static_cast<int64_t>(b) * H + h) * M;

    const int64_t norm_base =
        (static_cast<int64_t>(b) * H + h) * T + t;

    // ============================================================
    // MSE 2-bit contribution
    // ============================================================

    float mse_acc = 0.0f;

    for (int pd = 0; pd < packed_D; ++pd) {
        const int64_t packed_idx =
            (((static_cast<int64_t>(b) * H + h) * packed_D + pd) * T + t);

        const uint8_t packed_byte =
            packed_mse_indices_t[packed_idx];

        const uint8_t c0 =
            (packed_byte >> 0) & 0x3;

        const uint8_t c1 =
            (packed_byte >> 2) & 0x3;

        const uint8_t c2 =
            (packed_byte >> 4) & 0x3;

        const uint8_t c3 =
            (packed_byte >> 6) & 0x3;

        const int d0 = pd * 4 + 0;
        const int d1 = pd * 4 + 1;
        const int d2 = pd * 4 + 2;
        const int d3 = pd * 4 + 3;

        mse_acc +=
            q_rot[q_rot_base + d0] *
            centroid_from_2bit(c0, centroids);

        mse_acc +=
            q_rot[q_rot_base + d1] *
            centroid_from_2bit(c1, centroids);

        mse_acc +=
            q_rot[q_rot_base + d2] *
            centroid_from_2bit(c2, centroids);

        mse_acc +=
            q_rot[q_rot_base + d3] *
            centroid_from_2bit(c3, centroids);
    }

    const float mse_part =
        mse_norms[norm_base] * mse_acc;

    // ============================================================
    // QJL residual contribution:
    //
    //   sqrt(pi/2)/M
    //   * residual_norm
    //   * <Sq, sign_bits>
    // ============================================================

    float qjl_acc = 0.0f;

    for (int pm = 0; pm < packed_M; ++pm) {
        const int64_t packed_idx =
            (((static_cast<int64_t>(b) * H + h) * packed_M + pm) * T + t);

        const uint8_t packed_byte =
            packed_qjl_sign_bits_t[packed_idx];

        const int m0 = pm * 8 + 0;
        const int m1 = pm * 8 + 1;
        const int m2 = pm * 8 + 2;
        const int m3 = pm * 8 + 3;
        const int m4 = pm * 8 + 4;
        const int m5 = pm * 8 + 5;
        const int m6 = pm * 8 + 6;
        const int m7 = pm * 8 + 7;

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

        qjl_acc += q_sketch[q_sketch_base + m0] * s0;
        qjl_acc += q_sketch[q_sketch_base + m1] * s1;
        qjl_acc += q_sketch[q_sketch_base + m2] * s2;
        qjl_acc += q_sketch[q_sketch_base + m3] * s3;
        qjl_acc += q_sketch[q_sketch_base + m4] * s4;
        qjl_acc += q_sketch[q_sketch_base + m5] * s5;
        qjl_acc += q_sketch[q_sketch_base + m6] * s6;
        qjl_acc += q_sketch[q_sketch_base + m7] * s7;
    }

    constexpr float SQRT_PI_OVER_2 =
        1.2533141373155001f;

    const float qjl_scale =
        SQRT_PI_OVER_2 / static_cast<float>(M);

    const float qjl_part =
        qjl_scale *
        qjl_residual_norms[norm_base] *
        qjl_acc;

    const int64_t out_idx =
        (static_cast<int64_t>(b) * H + h) * T + t;

    out_scores[out_idx] =
        mse_part + qjl_part;
}


// ============================================================
// Transposed-layout packed TurboQuant score kernel
// with shared-memory staged query vectors.
//
// Grid:
//   blockIdx.x = token tile
//   blockIdx.y = head
//   blockIdx.z = batch
//
// Each block serves:
//   one (b, h), 256 consecutive tokens.
//
// Shared memory:
//   q_rot[D]
//   q_sketch[M]
//
// This avoids re-reading the same q_rot / q_sketch vectors
// from global memory for every token thread.
// ============================================================

__global__ void turboquant_decode_score_transposed_sharedq_kernel(
    const float* __restrict__ q_rot,                    // [B,H,D]
    const float* __restrict__ q_sketch,                 // [B,H,M]
    const uint8_t* __restrict__ packed_mse_indices_t,   // [B,H,packed_D,T]
    const float* __restrict__ mse_norms,                // [B,H,T]
    const uint8_t* __restrict__ packed_qjl_sign_bits_t, // [B,H,packed_M,T]
    const float* __restrict__ qjl_residual_norms,       // [B,H,T]
    const float* __restrict__ centroids,                // [4]
    float* __restrict__ out_scores,                     // [B,H,1,T]
    int B,
    int H,
    int T,
    int D,
    int M,
    int packed_D,
    int packed_M
) {
    extern __shared__ float shared_q[];

    float* sh_q_rot =
        shared_q;

    float* sh_q_sketch =
        shared_q + D;

    const int b =
        static_cast<int>(blockIdx.z);

    const int h =
        static_cast<int>(blockIdx.y);

    const int token_base =
        static_cast<int>(blockIdx.x) * blockDim.x;

    const int t =
        token_base + threadIdx.x;

    const int64_t q_rot_base =
        (static_cast<int64_t>(b) * H + h) * D;

    const int64_t q_sketch_base =
        (static_cast<int64_t>(b) * H + h) * M;

    // ------------------------------------------------------------
    // Cooperative query staging:
    //   load q_rot[D] once per block
    //   load q_sketch[M] once per block
    // ------------------------------------------------------------

    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        sh_q_rot[d] =
            q_rot[q_rot_base + d];
    }

    for (int m = threadIdx.x; m < M; m += blockDim.x) {
        sh_q_sketch[m] =
            q_sketch[q_sketch_base + m];
    }

    __syncthreads();

    if (t >= T) {
        return;
    }

    const int64_t norm_base =
        (static_cast<int64_t>(b) * H + h) * T + t;

    // ============================================================
    // MSE 2-bit contribution
    // ============================================================

    float mse_acc = 0.0f;

    for (int pd = 0; pd < packed_D; ++pd) {
        const int64_t packed_idx =
            (((static_cast<int64_t>(b) * H + h) * packed_D + pd) * T + t);

        const uint8_t packed_byte =
            packed_mse_indices_t[packed_idx];

        const uint8_t c0 =
            (packed_byte >> 0) & 0x3;

        const uint8_t c1 =
            (packed_byte >> 2) & 0x3;

        const uint8_t c2 =
            (packed_byte >> 4) & 0x3;

        const uint8_t c3 =
            (packed_byte >> 6) & 0x3;

        const int d0 = pd * 4 + 0;
        const int d1 = pd * 4 + 1;
        const int d2 = pd * 4 + 2;
        const int d3 = pd * 4 + 3;

        mse_acc +=
            sh_q_rot[d0] *
            centroids[static_cast<int>(c0)];

        mse_acc +=
            sh_q_rot[d1] *
            centroids[static_cast<int>(c1)];

        mse_acc +=
            sh_q_rot[d2] *
            centroids[static_cast<int>(c2)];

        mse_acc +=
            sh_q_rot[d3] *
            centroids[static_cast<int>(c3)];
    }

    const float mse_part =
        mse_norms[norm_base] * mse_acc;

    // ============================================================
    // QJL residual contribution:
    //
    //   sqrt(pi/2)/M
    //   * residual_norm
    //   * <Sq, sign_bits>
    // ============================================================

    float qjl_acc = 0.0f;

    for (int pm = 0; pm < packed_M; ++pm) {
        const int64_t packed_idx =
            (((static_cast<int64_t>(b) * H + h) * packed_M + pm) * T + t);

        const uint8_t packed_byte =
            packed_qjl_sign_bits_t[packed_idx];

        const int m0 = pm * 8 + 0;
        const int m1 = pm * 8 + 1;
        const int m2 = pm * 8 + 2;
        const int m3 = pm * 8 + 3;
        const int m4 = pm * 8 + 4;
        const int m5 = pm * 8 + 5;
        const int m6 = pm * 8 + 6;
        const int m7 = pm * 8 + 7;

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

        qjl_acc += sh_q_sketch[m0] * s0;
        qjl_acc += sh_q_sketch[m1] * s1;
        qjl_acc += sh_q_sketch[m2] * s2;
        qjl_acc += sh_q_sketch[m3] * s3;
        qjl_acc += sh_q_sketch[m4] * s4;
        qjl_acc += sh_q_sketch[m5] * s5;
        qjl_acc += sh_q_sketch[m6] * s6;
        qjl_acc += sh_q_sketch[m7] * s7;
    }

    constexpr float SQRT_PI_OVER_2 =
        1.2533141373155001f;

    const float qjl_scale =
        SQRT_PI_OVER_2 / static_cast<float>(M);

    const float qjl_part =
        qjl_scale *
        qjl_residual_norms[norm_base] *
        qjl_acc;

    const int64_t out_idx =
        (static_cast<int64_t>(b) * H + h) * T + t;

    out_scores[out_idx] =
        mse_part + qjl_part;
}

} // namespace


// ============================================================
// Host launcher: transposed packed layout
// ============================================================

torch::Tensor turboquant_decode_score_transposed_cuda(
    torch::Tensor q_rot,
    torch::Tensor q_sketch,
    torch::Tensor packed_mse_indices_t,
    torch::Tensor mse_norms,
    torch::Tensor packed_qjl_sign_bits_t,
    torch::Tensor qjl_residual_norms,
    torch::Tensor centroids
) {
    TORCH_CHECK(
        q_rot.is_cuda(),
        "q_rot must be CUDA"
    );

    TORCH_CHECK(
        q_sketch.is_cuda(),
        "q_sketch must be CUDA"
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
        packed_qjl_sign_bits_t.is_cuda(),
        "packed_qjl_sign_bits_t must be CUDA"
    );

    TORCH_CHECK(
        qjl_residual_norms.is_cuda(),
        "qjl_residual_norms must be CUDA"
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
        q_sketch.dtype() == torch::kFloat32,
        "q_sketch must be float32"
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
        packed_qjl_sign_bits_t.dtype() == torch::kUInt8,
        "packed_qjl_sign_bits_t must be uint8"
    );

    TORCH_CHECK(
        qjl_residual_norms.dtype() == torch::kFloat32,
        "qjl_residual_norms must be float32"
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
        q_sketch.dim() == 3,
        "q_sketch must be [B,H,M]"
    );

    TORCH_CHECK(
        packed_mse_indices_t.dim() == 4,
        "packed_mse_indices_t must be [B,H,packed_D,T]"
    );

    TORCH_CHECK(
        packed_qjl_sign_bits_t.dim() == 4,
        "packed_qjl_sign_bits_t must be [B,H,packed_M,T]"
    );

    TORCH_CHECK(
        mse_norms.dim() == 3,
        "mse_norms must be [B,H,T]"
    );

    TORCH_CHECK(
        qjl_residual_norms.dim() == 3,
        "qjl_residual_norms must be [B,H,T]"
    );

    TORCH_CHECK(
        centroids.numel() == 4,
        "centroids must have 4 elements"
    );

    TORCH_CHECK(
        q_rot.is_contiguous(),
        "q_rot must be contiguous"
    );

    TORCH_CHECK(
        q_sketch.is_contiguous(),
        "q_sketch must be contiguous"
    );

    TORCH_CHECK(
        packed_mse_indices_t.is_contiguous(),
        "packed_mse_indices_t must be contiguous"
    );

    TORCH_CHECK(
        packed_qjl_sign_bits_t.is_contiguous(),
        "packed_qjl_sign_bits_t must be contiguous"
    );

    TORCH_CHECK(
        mse_norms.is_contiguous(),
        "mse_norms must be contiguous"
    );

    TORCH_CHECK(
        qjl_residual_norms.is_contiguous(),
        "qjl_residual_norms must be contiguous"
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

    const int M =
        static_cast<int>(q_sketch.size(2));

    const int packed_D =
        static_cast<int>(packed_mse_indices_t.size(2));

    const int T =
        static_cast<int>(packed_mse_indices_t.size(3));

    const int packed_M =
        static_cast<int>(packed_qjl_sign_bits_t.size(2));

    TORCH_CHECK(
        packed_D * 4 == D,
        "packed_D mismatch"
    );

    TORCH_CHECK(
        packed_M * 8 == M,
        "packed_M mismatch"
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

    TORCH_CHECK(
        qjl_residual_norms.size(0) == B,
        "qjl_residual_norms B mismatch"
    );

    TORCH_CHECK(
        qjl_residual_norms.size(1) == H,
        "qjl_residual_norms H mismatch"
    );

    TORCH_CHECK(
        qjl_residual_norms.size(2) == T,
        "qjl_residual_norms T mismatch"
    );

    auto out =
        torch::empty(
            {B, H, 1, T},
            torch::TensorOptions()
                .dtype(torch::kFloat32)
                .device(q_rot.device())
        );

    const int64_t total =
        static_cast<int64_t>(B) * H * T;

    constexpr int threads =
        256;

    const int blocks =
        static_cast<int>(
            (total + threads - 1) / threads
        );

    turboquant_decode_score_transposed_kernel<<<
        blocks,
        threads,
        0,
        at::cuda::getDefaultCUDAStream()
    >>>(
        q_rot.data_ptr<float>(),
        q_sketch.data_ptr<float>(),
        packed_mse_indices_t.data_ptr<uint8_t>(),
        mse_norms.data_ptr<float>(),
        packed_qjl_sign_bits_t.data_ptr<uint8_t>(),
        qjl_residual_norms.data_ptr<float>(),
        centroids.data_ptr<float>(),
        out.data_ptr<float>(),
        B,
        H,
        T,
        D,
        M,
        packed_D,
        packed_M
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}


// ============================================================
// Host launcher: transposed packed layout + shared staged query
// ============================================================

torch::Tensor turboquant_decode_score_transposed_sharedq_cuda(
    torch::Tensor q_rot,
    torch::Tensor q_sketch,
    torch::Tensor packed_mse_indices_t,
    torch::Tensor mse_norms,
    torch::Tensor packed_qjl_sign_bits_t,
    torch::Tensor qjl_residual_norms,
    torch::Tensor centroids
) {
    TORCH_CHECK(
        q_rot.is_cuda(),
        "q_rot must be CUDA"
    );

    TORCH_CHECK(
        q_sketch.is_cuda(),
        "q_sketch must be CUDA"
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
        packed_qjl_sign_bits_t.is_cuda(),
        "packed_qjl_sign_bits_t must be CUDA"
    );

    TORCH_CHECK(
        qjl_residual_norms.is_cuda(),
        "qjl_residual_norms must be CUDA"
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
        q_sketch.dtype() == torch::kFloat32,
        "q_sketch must be float32"
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
        packed_qjl_sign_bits_t.dtype() == torch::kUInt8,
        "packed_qjl_sign_bits_t must be uint8"
    );

    TORCH_CHECK(
        qjl_residual_norms.dtype() == torch::kFloat32,
        "qjl_residual_norms must be float32"
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
        q_sketch.dim() == 3,
        "q_sketch must be [B,H,M]"
    );

    TORCH_CHECK(
        packed_mse_indices_t.dim() == 4,
        "packed_mse_indices_t must be [B,H,packed_D,T]"
    );

    TORCH_CHECK(
        packed_qjl_sign_bits_t.dim() == 4,
        "packed_qjl_sign_bits_t must be [B,H,packed_M,T]"
    );

    TORCH_CHECK(
        mse_norms.dim() == 3,
        "mse_norms must be [B,H,T]"
    );

    TORCH_CHECK(
        qjl_residual_norms.dim() == 3,
        "qjl_residual_norms must be [B,H,T]"
    );

    TORCH_CHECK(
        centroids.numel() == 4,
        "centroids must have 4 elements"
    );

    TORCH_CHECK(
        q_rot.is_contiguous(),
        "q_rot must be contiguous"
    );

    TORCH_CHECK(
        q_sketch.is_contiguous(),
        "q_sketch must be contiguous"
    );

    TORCH_CHECK(
        packed_mse_indices_t.is_contiguous(),
        "packed_mse_indices_t must be contiguous"
    );

    TORCH_CHECK(
        packed_qjl_sign_bits_t.is_contiguous(),
        "packed_qjl_sign_bits_t must be contiguous"
    );

    TORCH_CHECK(
        mse_norms.is_contiguous(),
        "mse_norms must be contiguous"
    );

    TORCH_CHECK(
        qjl_residual_norms.is_contiguous(),
        "qjl_residual_norms must be contiguous"
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

    const int M =
        static_cast<int>(q_sketch.size(2));

    const int packed_D =
        static_cast<int>(packed_mse_indices_t.size(2));

    const int T =
        static_cast<int>(packed_mse_indices_t.size(3));

    const int packed_M =
        static_cast<int>(packed_qjl_sign_bits_t.size(2));

    TORCH_CHECK(
        packed_D * 4 == D,
        "packed_D mismatch"
    );

    TORCH_CHECK(
        packed_M * 8 == M,
        "packed_M mismatch"
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

    TORCH_CHECK(
        qjl_residual_norms.size(0) == B,
        "qjl_residual_norms B mismatch"
    );

    TORCH_CHECK(
        qjl_residual_norms.size(1) == H,
        "qjl_residual_norms H mismatch"
    );

    TORCH_CHECK(
        qjl_residual_norms.size(2) == T,
        "qjl_residual_norms T mismatch"
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
        static_cast<size_t>(D + M) * sizeof(float);

    turboquant_decode_score_transposed_sharedq_kernel<<<
        grid,
        threads,
        shared_bytes,
        at::cuda::getDefaultCUDAStream()
    >>>(
        q_rot.data_ptr<float>(),
        q_sketch.data_ptr<float>(),
        packed_mse_indices_t.data_ptr<uint8_t>(),
        mse_norms.data_ptr<float>(),
        packed_qjl_sign_bits_t.data_ptr<uint8_t>(),
        qjl_residual_norms.data_ptr<float>(),
        centroids.data_ptr<float>(),
        out.data_ptr<float>(),
        B,
        H,
        T,
        D,
        M,
        packed_D,
        packed_M
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}