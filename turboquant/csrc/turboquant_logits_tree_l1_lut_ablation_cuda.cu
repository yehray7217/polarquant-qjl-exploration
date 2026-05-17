#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAException.h>

#include <cmath>
#include <cstdint>

namespace {

constexpr int M = 128;
constexpr int META_BYTES = 64;
constexpr int THREADS = 256;   // 8 warps / CTA
constexpr int L1_LUT_PAIRS = 64;
constexpr int L1_LUT_CODES = 16;
constexpr float QJL_CORRECTION_SCALE = 0.375f;


__device__ __forceinline__ uint8_t byte_from_u32(
    uint32_t word,
    int byte_idx
) {
    return static_cast<uint8_t>(
        (word >> (byte_idx * 8)) & 0xFFu
    );
}


__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}


__device__ __forceinline__ uint8_t load_l1_code_meta64(
    uint32_t meta_word_local,
    int l1_idx
) {
    const int byte_idx = l1_idx >> 1;
    const int word_owner = byte_idx >> 2;
    const int byte_in_word = byte_idx & 3;

    const uint32_t word = __shfl_sync(
        0xffffffff,
        meta_word_local,
        word_owner
    );

    const uint8_t packed_byte = byte_from_u32(word, byte_in_word);
    return static_cast<uint8_t>(
        (packed_byte >> ((l1_idx & 1) * 4)) & 0x0Fu
    );
}


__device__ __forceinline__ uint8_t load_l2_code_meta64(
    uint32_t meta_word_local,
    int l2_idx
) {
    const int byte_idx = l2_idx >> 2;
    const int word_owner = 8 + (byte_idx >> 2);
    const int byte_in_word = byte_idx & 3;

    const uint32_t word = __shfl_sync(
        0xffffffff,
        meta_word_local,
        word_owner
    );

    const uint8_t packed_byte = byte_from_u32(word, byte_in_word);
    return static_cast<uint8_t>(
        (packed_byte >> ((l2_idx & 3) * 2)) & 0x03u
    );
}


__device__ __forceinline__ uint8_t load_l3_code_meta64(
    uint32_t meta_word_local,
    int l3_idx
) {
    const int byte_idx = l3_idx >> 2;
    const uint32_t word = __shfl_sync(
        0xffffffff,
        meta_word_local,
        10
    );

    const uint8_t packed_byte = byte_from_u32(word, byte_idx);
    return static_cast<uint8_t>(
        (packed_byte >> ((l3_idx & 3) * 2)) & 0x03u
    );
}


__device__ __forceinline__ uint8_t load_l4_code_meta64(
    uint32_t meta_word_local,
    int l4_idx
) {
    const int byte_idx = l4_idx >> 2;
    const uint32_t word = __shfl_sync(
        0xffffffff,
        meta_word_local,
        11
    );

    const uint8_t packed_byte = byte_from_u32(word, byte_idx);
    return static_cast<uint8_t>(
        (packed_byte >> ((l4_idx & 3) * 2)) & 0x03u
    );
}


__device__ __forceinline__ uint8_t load_qjl_bit_meta64(
    uint32_t meta_word_local,
    int sketch_idx
) {
    const int qjl_byte_idx = sketch_idx >> 3;
    const int word_owner = 12 + (qjl_byte_idx >> 2);
    const int byte_in_word = qjl_byte_idx & 3;

    const uint32_t word = __shfl_sync(
        0xffffffff,
        meta_word_local,
        word_owner
    );

    const uint8_t qjl_byte = byte_from_u32(word, byte_in_word);
    return static_cast<uint8_t>(
        (qjl_byte >> (sketch_idx & 7)) & 0x01u
    );
}


__device__ __forceinline__ float polar_l1_lut_tree_sum(
    uint32_t meta_word_local,
    const half* __restrict__ radii_ptr,
    const float* __restrict__ l1_factor_lut,
    const float* __restrict__ cos_l2,
    const float* __restrict__ sin_l2,
    const float* __restrict__ cos_l3,
    const float* __restrict__ sin_l3,
    const float* __restrict__ cos_l4,
    const float* __restrict__ sin_l4,
    int h_idx,
    int lane
) {
    const int block16 = lane >> 2;
    const int lane4 = lane & 3;

    const int l1_pair_base = block16 * 8 + lane4 * 2;
    const int l1_idx_a = l1_pair_base;
    const int l1_idx_b = l1_pair_base + 1;

    const uint8_t c1a = load_l1_code_meta64(meta_word_local, l1_idx_a);
    const uint8_t c1b = load_l1_code_meta64(meta_word_local, l1_idx_b);

    const int64_t l1_base_a =
        (static_cast<int64_t>(h_idx) * L1_LUT_PAIRS + l1_idx_a)
        * L1_LUT_CODES;
    const int64_t l1_base_b =
        (static_cast<int64_t>(h_idx) * L1_LUT_PAIRS + l1_idx_b)
        * L1_LUT_CODES;

    const float s1a = l1_factor_lut[l1_base_a + c1a];
    const float s1b = l1_factor_lut[l1_base_b + c1b];

    const int l2_idx = block16 * 4 + lane4;
    const uint8_t c2 = load_l2_code_meta64(meta_word_local, l2_idx);

    const float s2 =
        s1a * cos_l2[c2]
        + s1b * sin_l2[c2];

    const int l3_idx = block16 * 2 + (lane4 >> 1);
    const uint8_t c3 = load_l3_code_meta64(meta_word_local, l3_idx);

    const float s2_right = __shfl_down_sync(
        0xffffffff,
        s2,
        1
    );

    float s3 = 0.0f;
    if ((lane4 & 1) == 0) {
        s3 =
            s2 * cos_l3[c3]
            + s2_right * sin_l3[c3];
    }

    const uint8_t c4 = load_l4_code_meta64(meta_word_local, block16);
    const float s3_right = __shfl_down_sync(
        0xffffffff,
        s3,
        2
    );

    float polar_block_contrib = 0.0f;
    if (lane4 == 0) {
        const float s4 =
            s3 * cos_l4[c4]
            + s3_right * sin_l4[c4];

        const float radius = __half2float(radii_ptr[block16]);
        polar_block_contrib = radius * s4;
    }

    return warp_reduce_sum(polar_block_contrib);
}


__device__ __forceinline__ float qjl_residual_sum(
    uint32_t meta_word_local,
    const float* __restrict__ sh_qproj,
    const half* __restrict__ qjl_norms,
    int64_t key_linear,
    int lane
) {
    float qjl_acc = 0.0f;

    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int sketch_idx = lane + 32 * k;
        const float qproj_val = sh_qproj[sketch_idx];
        const uint8_t qjl_bit = load_qjl_bit_meta64(
            meta_word_local,
            sketch_idx
        );
        qjl_acc += qjl_bit ? qproj_val : -qproj_val;
    }

    const float qjl_sum = warp_reduce_sum(qjl_acc);

    if (lane == 0) {
        const float residual_norm = __half2float(qjl_norms[key_linear]);
        const float qjl_scale =
            QJL_CORRECTION_SCALE
            * sqrtf(3.14159265358979323846f / 2.0f)
            / sqrtf(static_cast<float>(M));
        return qjl_scale * residual_norm * qjl_sum;
    }

    return 0.0f;
}


__global__ void turboquant_polar_tree_l1_lut_polar_only_decode_b1q1_warp8_meta64_kernel(
    const uint8_t* __restrict__ packed_meta,
    const half* __restrict__ radii,
    const float* __restrict__ l1_factor_lut,
    const float* __restrict__ cos_l2,
    const float* __restrict__ sin_l2,
    const float* __restrict__ cos_l3,
    const float* __restrict__ sin_l3,
    const float* __restrict__ cos_l4,
    const float* __restrict__ sin_l4,
    float* __restrict__ out,
    int H,
    int T
) {
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    const int h_idx = static_cast<int>(blockIdx.y);
    const int t_idx = static_cast<int>(blockIdx.x) * 8 + warp_id;

    if (h_idx >= H || t_idx >= T) {
        return;
    }

    const int64_t key_linear = static_cast<int64_t>(h_idx) * T + t_idx;
    const uint8_t* meta_ptr = packed_meta + key_linear * META_BYTES;
    const half* radii_ptr = radii + key_linear * 8;

    uint32_t meta_word_local = 0;
    if (lane < 16) {
        meta_word_local = reinterpret_cast<const uint32_t*>(meta_ptr)[lane];
    }

    const float polar_sum = polar_l1_lut_tree_sum(
        meta_word_local,
        radii_ptr,
        l1_factor_lut,
        cos_l2,
        sin_l2,
        cos_l3,
        sin_l3,
        cos_l4,
        sin_l4,
        h_idx,
        lane
    );

    if (lane == 0) {
        const int64_t out_linear =
            static_cast<int64_t>(h_idx) * T + t_idx;
        out[out_linear] = polar_sum;
    }
}


__global__ void turboquant_polar_tree_l1_lut_qjl_only_decode_b1q1_warp8_meta64_kernel(
    const float* __restrict__ q_projected,
    const uint8_t* __restrict__ packed_meta,
    const half* __restrict__ qjl_norms,
    float* __restrict__ out,
    int H,
    int T
) {
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    const int h_idx = static_cast<int>(blockIdx.y);
    const int t_idx = static_cast<int>(blockIdx.x) * 8 + warp_id;

    __shared__ float sh_qproj[M];

    const int64_t qproj_offset = static_cast<int64_t>(h_idx) * M;
    if (tid < M) {
        sh_qproj[tid] = q_projected[qproj_offset + tid];
    }

    __syncthreads();

    if (h_idx >= H || t_idx >= T) {
        return;
    }

    const int64_t key_linear = static_cast<int64_t>(h_idx) * T + t_idx;
    const uint8_t* meta_ptr = packed_meta + key_linear * META_BYTES;

    uint32_t meta_word_local = 0;
    if (lane < 16) {
        meta_word_local = reinterpret_cast<const uint32_t*>(meta_ptr)[lane];
    }

    const float qjl_only_contribution = qjl_residual_sum(
        meta_word_local,
        sh_qproj,
        qjl_norms,
        key_linear,
        lane
    );

    if (lane == 0) {
        const int64_t out_linear =
            static_cast<int64_t>(h_idx) * T + t_idx;
        out[out_linear] = qjl_only_contribution;
    }
}


void validate_meta_common(
    torch::Tensor packed_meta
) {
    TORCH_CHECK(packed_meta.is_cuda(), "packed_meta must be CUDA");
    TORCH_CHECK(packed_meta.dtype() == torch::kUInt8, "packed_meta must be uint8");
    TORCH_CHECK(packed_meta.dim() == 4, "packed_meta must be [B,H,T,64]");
    TORCH_CHECK(packed_meta.size(0) == 1, "ablation kernels require B=1");
    TORCH_CHECK(packed_meta.size(3) == META_BYTES, "packed_meta last dim must be 64");
}


torch::Tensor allocate_out_from_meta(
    torch::Tensor packed_meta
) {
    return torch::empty(
        {1, packed_meta.size(1), 1, packed_meta.size(2)},
        torch::TensorOptions()
            .device(packed_meta.device())
            .dtype(torch::kFloat32)
    );
}


void validate_grid(int64_t H, int64_t T) {
    const int64_t token_groups = (T + 7) / 8;
    TORCH_CHECK(token_groups <= 2147483647LL, "ablation grid.x exceeds CUDA limit");
    TORCH_CHECK(H <= 65535, "ablation grid.y exceeds CUDA limit");
}

} // namespace


torch::Tensor turboquant_polar_tree_l1_lut_polar_only_cuda(
    torch::Tensor packed_meta,
    torch::Tensor radii,
    torch::Tensor l1_factor_lut,
    torch::Tensor cos_l2,
    torch::Tensor sin_l2,
    torch::Tensor cos_l3,
    torch::Tensor sin_l3,
    torch::Tensor cos_l4,
    torch::Tensor sin_l4
) {
    validate_meta_common(packed_meta);

    TORCH_CHECK(radii.is_cuda(), "radii must be CUDA");
    TORCH_CHECK(radii.dtype() == torch::kFloat16, "radii must be float16");
    TORCH_CHECK(radii.dim() == 4, "radii must be [B,H,T,8]");
    TORCH_CHECK(radii.size(0) == 1, "radii B mismatch");
    TORCH_CHECK(radii.size(1) == packed_meta.size(1), "radii H mismatch");
    TORCH_CHECK(radii.size(2) == packed_meta.size(2), "radii T mismatch");
    TORCH_CHECK(radii.size(3) == 8, "radii last dim must be 8");

    TORCH_CHECK(l1_factor_lut.is_cuda(), "l1_factor_lut must be CUDA");
    TORCH_CHECK(l1_factor_lut.dtype() == torch::kFloat32, "l1_factor_lut must be float32");
    TORCH_CHECK(l1_factor_lut.dim() == 3, "l1_factor_lut must be [H,64,16]");
    TORCH_CHECK(l1_factor_lut.size(0) == packed_meta.size(1), "l1_factor_lut H mismatch");
    TORCH_CHECK(l1_factor_lut.size(1) == L1_LUT_PAIRS, "l1_factor_lut pair dim must be 64");
    TORCH_CHECK(l1_factor_lut.size(2) == L1_LUT_CODES, "l1_factor_lut code dim must be 16");

    TORCH_CHECK(cos_l2.numel() == 4 && sin_l2.numel() == 4, "L2 trig tables must have 4 entries");
    TORCH_CHECK(cos_l3.numel() == 4 && sin_l3.numel() == 4, "L3 trig tables must have 4 entries");
    TORCH_CHECK(cos_l4.numel() == 4 && sin_l4.numel() == 4, "L4 trig tables must have 4 entries");

    const int64_t H = packed_meta.size(1);
    const int64_t T = packed_meta.size(2);
    validate_grid(H, T);

    auto out = allocate_out_from_meta(packed_meta);

    const dim3 block(THREADS);
    const dim3 grid(
        static_cast<unsigned int>((T + 7) / 8),
        static_cast<unsigned int>(H),
        1
    );

    turboquant_polar_tree_l1_lut_polar_only_decode_b1q1_warp8_meta64_kernel<<<grid, block>>>(
        packed_meta.contiguous().data_ptr<uint8_t>(),
        reinterpret_cast<const half*>(radii.contiguous().data_ptr<at::Half>()),
        l1_factor_lut.contiguous().data_ptr<float>(),
        cos_l2.contiguous().data_ptr<float>(),
        sin_l2.contiguous().data_ptr<float>(),
        cos_l3.contiguous().data_ptr<float>(),
        sin_l3.contiguous().data_ptr<float>(),
        cos_l4.contiguous().data_ptr<float>(),
        sin_l4.contiguous().data_ptr<float>(),
        out.data_ptr<float>(),
        static_cast<int>(H),
        static_cast<int>(T)
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}


torch::Tensor turboquant_polar_tree_l1_lut_qjl_only_cuda(
    torch::Tensor q_projected,
    torch::Tensor packed_meta,
    torch::Tensor qjl_norms
) {
    validate_meta_common(packed_meta);

    TORCH_CHECK(q_projected.is_cuda(), "q_projected must be CUDA");
    TORCH_CHECK(q_projected.dtype() == torch::kFloat32, "q_projected must be float32");
    TORCH_CHECK(q_projected.dim() == 4, "q_projected must be [B,H,Q,128]");
    TORCH_CHECK(q_projected.size(0) == 1, "q_projected B mismatch");
    TORCH_CHECK(q_projected.size(1) == packed_meta.size(1), "q_projected H mismatch");
    TORCH_CHECK(q_projected.size(2) == 1, "q_projected Q must be 1");
    TORCH_CHECK(q_projected.size(3) == M, "q_projected last dim must be 128");

    TORCH_CHECK(qjl_norms.is_cuda(), "qjl_norms must be CUDA");
    TORCH_CHECK(qjl_norms.dtype() == torch::kFloat16, "qjl_norms must be float16");
    TORCH_CHECK(qjl_norms.dim() == 3, "qjl_norms must be [B,H,T]");
    TORCH_CHECK(qjl_norms.size(0) == 1, "qjl_norms B mismatch");
    TORCH_CHECK(qjl_norms.size(1) == packed_meta.size(1), "qjl_norms H mismatch");
    TORCH_CHECK(qjl_norms.size(2) == packed_meta.size(2), "qjl_norms T mismatch");

    const int64_t H = packed_meta.size(1);
    const int64_t T = packed_meta.size(2);
    validate_grid(H, T);

    auto out = allocate_out_from_meta(packed_meta);

    const dim3 block(THREADS);
    const dim3 grid(
        static_cast<unsigned int>((T + 7) / 8),
        static_cast<unsigned int>(H),
        1
    );

    turboquant_polar_tree_l1_lut_qjl_only_decode_b1q1_warp8_meta64_kernel<<<grid, block>>>(
        q_projected.contiguous().data_ptr<float>(),
        packed_meta.contiguous().data_ptr<uint8_t>(),
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
        "turboquant_polar_tree_l1_lut_polar_only_cuda",
        &turboquant_polar_tree_l1_lut_polar_only_cuda,
        "Polar-only ablation for PolarQuant tree L1-LUT score"
    );
    m.def(
        "turboquant_polar_tree_l1_lut_qjl_only_cuda",
        &turboquant_polar_tree_l1_lut_qjl_only_cuda,
        "QJL-only ablation for PolarQuant tree L1-LUT score"
    );
}
