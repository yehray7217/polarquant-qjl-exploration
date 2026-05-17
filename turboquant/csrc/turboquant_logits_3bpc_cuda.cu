#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAException.h>

#include <cmath>
#include <cstdint>

namespace {

constexpr int D = 128;
constexpr int M = 64;
constexpr int META_BYTES = 32;
constexpr int THREADS = 256;  // 8 warps / CTA
constexpr float QJL_CORRECTION_SCALE = 0.375f;


__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(
            0xffffffff,
            val,
            offset
        );
    }
    return val;
}


__global__ void polarquant_3bpc_fused_logits_decode_b1q1_warp8_meta32_kernel(
    const float* __restrict__ q,
    const float* __restrict__ q_projected,

    const uint8_t* __restrict__ packed_meta32,
    const half* __restrict__ radii,

    const float* __restrict__ cos_l1,
    const float* __restrict__ sin_l1,
    const float* __restrict__ cos_l2,
    const float* __restrict__ sin_l2,
    const float* __restrict__ cos_l3,
    const float* __restrict__ sin_l3,
    const float* __restrict__ cos_l4,
    const float* __restrict__ sin_l4,

    const half* __restrict__ qjl_norms,

    float* __restrict__ out,

    int H,
    int T
) {
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;  // 0..7
    const int lane = tid & 31;     // 0..31

    const int h_idx = static_cast<int>(blockIdx.y);
    const int t_idx = static_cast<int>(blockIdx.x) * 8 + warp_id;

    __shared__ float sh_q[D];
    __shared__ float sh_qproj[M];

    const int64_t q_offset = static_cast<int64_t>(h_idx) * D;
    const int64_t qproj_offset = static_cast<int64_t>(h_idx) * M;

    if (tid < D) {
        sh_q[tid] = q[q_offset + tid];
    }

    if (tid < M) {
        sh_qproj[tid] = q_projected[qproj_offset + tid];
    }

    __syncthreads();

    if (h_idx >= H || t_idx >= T) {
        return;
    }

    const int64_t key_linear = static_cast<int64_t>(h_idx) * T + t_idx;
    const uint8_t* meta_ptr = packed_meta32 + key_linear * META_BYTES;
    const half* radii_ptr = radii + key_linear * 8;

    // meta32: 8 x uint32 words.
    // words 0..3: L1 2-bit (16 B)
    // word 4:     L2 1-bit (4 B)
    // word 5:     L3 1-bit (2 B), L4 1-bit (1 B), padding (1 B)
    // words 6..7: QJL signs M=64 (8 B)
    uint32_t meta_word_local = 0;
    if (lane < 8) {
        meta_word_local =
            reinterpret_cast<const uint32_t*>(meta_ptr)[lane];
    }

    float polar_acc = 0.0f;

    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int d = lane + 32 * k;

        const int block16 = d >> 4;  // 0..7
        const int local16 = d & 15;

        const int l4_idx = block16;
        const int l3_idx = block16 * 2 + (local16 >> 3);
        const int l2_idx = block16 * 4 + (local16 >> 2);
        const int l1_idx = block16 * 8 + (local16 >> 1);

        // L1: 64 codes x 2 bits = 16 B = words 0..3.
        const int l1_byte_idx = l1_idx >> 2;
        const int l1_word_owner = l1_byte_idx >> 2;
        const int l1_byte_in_word = l1_byte_idx & 3;

        const uint32_t l1_word = __shfl_sync(
            0xffffffff,
            meta_word_local,
            l1_word_owner
        );

        const uint8_t l1_byte = static_cast<uint8_t>(
            (l1_word >> (l1_byte_in_word * 8)) & 0xFFu
        );

        const uint8_t c1 = static_cast<uint8_t>(
            (l1_byte >> ((l1_idx & 3) * 2)) & 0x03u
        );

        // L2: 32 codes x 1 bit = one uint32, word 4.
        const uint32_t l2_word = __shfl_sync(
            0xffffffff,
            meta_word_local,
            4
        );

        const uint8_t c2 = static_cast<uint8_t>(
            (l2_word >> l2_idx) & 0x01u
        );

        // L3/L4 share word 5:
        // bits  0..15: L3 codes
        // bits 16..23: L4 codes
        const uint32_t l3l4_word = __shfl_sync(
            0xffffffff,
            meta_word_local,
            5
        );

        const uint8_t c3 = static_cast<uint8_t>(
            (l3l4_word >> l3_idx) & 0x01u
        );

        const uint8_t c4 = static_cast<uint8_t>(
            (l3l4_word >> (16 + l4_idx)) & 0x01u
        );

        const bool use_sin_l1 = (local16 & 1) != 0;
        const bool use_sin_l2 = (local16 & 2) != 0;
        const bool use_sin_l3 = (local16 & 4) != 0;
        const bool use_sin_l4 = (local16 & 8) != 0;

        const float f1 = use_sin_l1 ? sin_l1[c1] : cos_l1[c1];
        const float f2 = use_sin_l2 ? sin_l2[c2] : cos_l2[c2];
        const float f3 = use_sin_l3 ? sin_l3[c3] : cos_l3[c3];
        const float f4 = use_sin_l4 ? sin_l4[c4] : cos_l4[c4];

        const float radius = __half2float(radii_ptr[block16]);
        const float reconstructed_k = radius * f4 * f3 * f2 * f1;

        polar_acc += sh_q[d] * reconstructed_k;
    }

    float qjl_acc = 0.0f;

    // M=64. Each lane handles two sketch coordinates.
    #pragma unroll
    for (int k = 0; k < 2; ++k) {
        const int sketch_idx = lane + 32 * k;
        const float qproj_val = sh_qproj[sketch_idx];

        const int qjl_byte_idx = sketch_idx >> 3;
        const int qjl_word_owner = 6 + (qjl_byte_idx >> 2);
        const int qjl_byte_in_word = qjl_byte_idx & 3;

        const uint32_t qjl_word = __shfl_sync(
            0xffffffff,
            meta_word_local,
            qjl_word_owner
        );

        const uint8_t qjl_byte = static_cast<uint8_t>(
            (qjl_word >> (qjl_byte_in_word * 8)) & 0xFFu
        );

        const uint8_t qjl_bit = static_cast<uint8_t>(
            (qjl_byte >> (sketch_idx & 7)) & 0x01u
        );

        qjl_acc += qjl_bit ? qproj_val : -qproj_val;
    }

    const float polar_sum = warp_reduce_sum(polar_acc);
    const float qjl_sum = warp_reduce_sum(qjl_acc);

    if (lane == 0) {
        const float residual_norm = __half2float(qjl_norms[key_linear]);

        const float qjl_scale =
            QJL_CORRECTION_SCALE
            * sqrtf(3.14159265358979323846f / 2.0f)
            / sqrtf(static_cast<float>(M));

        const float final_logit =
            polar_sum
            + qjl_scale * residual_norm * qjl_sum;

        const int64_t out_linear = static_cast<int64_t>(h_idx) * T + t_idx;
        out[out_linear] = final_logit;
    }
}

} // namespace


torch::Tensor polarquant_3bpc_fused_logits_cuda(
    torch::Tensor q,
    torch::Tensor q_projected,
    torch::Tensor packed_meta32,
    torch::Tensor radii,
    torch::Tensor cos_l1,
    torch::Tensor sin_l1,
    torch::Tensor cos_l2,
    torch::Tensor sin_l2,
    torch::Tensor cos_l3,
    torch::Tensor sin_l3,
    torch::Tensor cos_l4,
    torch::Tensor sin_l4,
    torch::Tensor qjl_norms
) {
    TORCH_CHECK(q.is_cuda(), "q must be CUDA");
    TORCH_CHECK(q_projected.is_cuda(), "q_projected must be CUDA");
    TORCH_CHECK(packed_meta32.is_cuda(), "packed_meta32 must be CUDA");
    TORCH_CHECK(radii.is_cuda(), "radii must be CUDA");
    TORCH_CHECK(qjl_norms.is_cuda(), "qjl_norms must be CUDA");

    TORCH_CHECK(q.dtype() == torch::kFloat32, "q must be float32");
    TORCH_CHECK(q_projected.dtype() == torch::kFloat32, "q_projected must be float32");
    TORCH_CHECK(packed_meta32.dtype() == torch::kUInt8, "packed_meta32 must be uint8");
    TORCH_CHECK(radii.dtype() == torch::kFloat16, "radii must be float16");
    TORCH_CHECK(qjl_norms.dtype() == torch::kFloat16, "qjl_norms must be float16");

    TORCH_CHECK(q.dim() == 4, "q must be [B,H,Q,128]");
    TORCH_CHECK(q.size(0) == 1, "3bpc fast path currently requires B=1");
    TORCH_CHECK(q.size(2) == 1, "3bpc fast path currently requires Q=1");
    TORCH_CHECK(q.size(3) == D, "q last dim must be 128");

    TORCH_CHECK(q_projected.dim() == 4, "q_projected must be [B,H,Q,64]");
    TORCH_CHECK(q_projected.size(0) == q.size(0), "q_projected B mismatch");
    TORCH_CHECK(q_projected.size(1) == q.size(1), "q_projected H mismatch");
    TORCH_CHECK(q_projected.size(2) == q.size(2), "q_projected Q mismatch");
    TORCH_CHECK(q_projected.size(3) == M, "q_projected last dim must be 64");

    TORCH_CHECK(packed_meta32.dim() == 4, "packed_meta32 must be [B,H,T,32]");
    TORCH_CHECK(packed_meta32.size(0) == q.size(0), "packed_meta32 B mismatch");
    TORCH_CHECK(packed_meta32.size(1) == q.size(1), "packed_meta32 H mismatch");
    TORCH_CHECK(packed_meta32.size(3) == META_BYTES, "packed_meta32 last dim must be 32");

    TORCH_CHECK(radii.dim() == 4, "radii must be [B,H,T,8]");
    TORCH_CHECK(radii.size(0) == q.size(0), "radii B mismatch");
    TORCH_CHECK(radii.size(1) == q.size(1), "radii H mismatch");
    TORCH_CHECK(radii.size(2) == packed_meta32.size(2), "radii T mismatch");
    TORCH_CHECK(radii.size(3) == 8, "radii last dim must be 8");

    TORCH_CHECK(qjl_norms.dim() == 3, "qjl_norms must be [B,H,T]");
    TORCH_CHECK(qjl_norms.size(0) == q.size(0), "qjl_norms B mismatch");
    TORCH_CHECK(qjl_norms.size(1) == q.size(1), "qjl_norms H mismatch");
    TORCH_CHECK(qjl_norms.size(2) == packed_meta32.size(2), "qjl_norms T mismatch");

    TORCH_CHECK(cos_l1.numel() == 4 && sin_l1.numel() == 4, "L1 trig tables must have 4 entries");
    TORCH_CHECK(cos_l2.numel() == 2 && sin_l2.numel() == 2, "L2 trig tables must have 2 entries");
    TORCH_CHECK(cos_l3.numel() == 2 && sin_l3.numel() == 2, "L3 trig tables must have 2 entries");
    TORCH_CHECK(cos_l4.numel() == 2 && sin_l4.numel() == 2, "L4 trig tables must have 2 entries");

    const int64_t B = q.size(0);
    const int64_t H = q.size(1);
    const int64_t Q = q.size(2);
    const int64_t T = packed_meta32.size(2);

    auto out = torch::empty(
        {B, H, Q, T},
        torch::TensorOptions()
            .device(q.device())
            .dtype(torch::kFloat32)
    );

    const dim3 block(THREADS);
    const int64_t token_groups = (T + 7) / 8;

    TORCH_CHECK(token_groups <= 2147483647LL, "meta32 grid.x exceeds CUDA limit");
    TORCH_CHECK(H <= 65535, "meta32 grid.y exceeds CUDA limit");

    const dim3 grid(
        static_cast<unsigned int>(token_groups),
        static_cast<unsigned int>(H),
        1
    );

    polarquant_3bpc_fused_logits_decode_b1q1_warp8_meta32_kernel<<<grid, block>>>(
        q.contiguous().data_ptr<float>(),
        q_projected.contiguous().data_ptr<float>(),
        packed_meta32.contiguous().data_ptr<uint8_t>(),
        reinterpret_cast<const half*>(radii.contiguous().data_ptr<at::Half>()),
        cos_l1.contiguous().data_ptr<float>(),
        sin_l1.contiguous().data_ptr<float>(),
        cos_l2.contiguous().data_ptr<float>(),
        sin_l2.contiguous().data_ptr<float>(),
        cos_l3.contiguous().data_ptr<float>(),
        sin_l3.contiguous().data_ptr<float>(),
        cos_l4.contiguous().data_ptr<float>(),
        sin_l4.contiguous().data_ptr<float>(),
        reinterpret_cast<const half*>(qjl_norms.contiguous().data_ptr<at::Half>()),
        out.data_ptr<float>(),
        static_cast<int>(H),
        static_cast<int>(T)
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "polarquant_3bpc_fused_logits_cuda",
        &polarquant_3bpc_fused_logits_cuda,
        "PolarQuant aligned ~3bpc fused logits CUDA fast path (meta32, M=64)"
    );
}
