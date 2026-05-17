#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAException.h>

#include <cmath>
#include <cstdint>

namespace {

constexpr int META_BYTES = 64;
constexpr int THREADS = 256;   // 8 warps / CTA
constexpr int L1_LUT_PAIRS = 64;
constexpr int L1_LUT_CODES = 16;


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


__device__ __forceinline__ float tree_finish_meta_codes_radius(
    float s2,
    uint32_t meta_word_local,
    const half* __restrict__ radii_ptr,
    const float* __restrict__ cos_l3,
    const float* __restrict__ sin_l3,
    const float* __restrict__ cos_l4,
    const float* __restrict__ sin_l4,
    int block16,
    int lane4
) {
    const int l3_idx = block16 * 2 + (lane4 >> 1);
    const uint8_t c3 = load_l3_code_meta64(meta_word_local, l3_idx);

    const float s2_right = __shfl_down_sync(0xffffffff, s2, 1);

    float s3 = 0.0f;
    if ((lane4 & 1) == 0) {
        s3 = s2 * cos_l3[c3] + s2_right * sin_l3[c3];
    }

    const uint8_t c4 = load_l4_code_meta64(meta_word_local, block16);
    const float s3_right = __shfl_down_sync(0xffffffff, s3, 2);

    float contrib = 0.0f;
    if (lane4 == 0) {
        const float s4 = s3 * cos_l4[c4] + s3_right * sin_l4[c4];
        const float radius = __half2float(radii_ptr[block16]);
        contrib = radius * s4;
    }

    return warp_reduce_sum(contrib);
}


__device__ __forceinline__ float tree_finish_meta_codes_unit_radius(
    float s2,
    uint32_t meta_word_local,
    const float* __restrict__ cos_l3,
    const float* __restrict__ sin_l3,
    const float* __restrict__ cos_l4,
    const float* __restrict__ sin_l4,
    int block16,
    int lane4
) {
    const int l3_idx = block16 * 2 + (lane4 >> 1);
    const uint8_t c3 = load_l3_code_meta64(meta_word_local, l3_idx);

    const float s2_right = __shfl_down_sync(0xffffffff, s2, 1);

    float s3 = 0.0f;
    if ((lane4 & 1) == 0) {
        s3 = s2 * cos_l3[c3] + s2_right * sin_l3[c3];
    }

    const uint8_t c4 = load_l4_code_meta64(meta_word_local, block16);
    const float s3_right = __shfl_down_sync(0xffffffff, s3, 2);

    float contrib = 0.0f;
    if (lane4 == 0) {
        const float s4 = s3 * cos_l4[c4] + s3_right * sin_l4[c4];
        contrib = s4;
    }

    return warp_reduce_sum(contrib);
}


__device__ __forceinline__ float tree_finish_direct_codes_radius(
    float s2,
    const uint8_t* __restrict__ l3_codes_ptr,
    const uint8_t* __restrict__ l4_codes_ptr,
    const half* __restrict__ radii_ptr,
    const float* __restrict__ cos_l3,
    const float* __restrict__ sin_l3,
    const float* __restrict__ cos_l4,
    const float* __restrict__ sin_l4,
    int block16,
    int lane4
) {
    const int l3_idx = block16 * 2 + (lane4 >> 1);
    const uint8_t c3 = l3_codes_ptr[l3_idx];

    const float s2_right = __shfl_down_sync(0xffffffff, s2, 1);

    float s3 = 0.0f;
    if ((lane4 & 1) == 0) {
        s3 = s2 * cos_l3[c3] + s2_right * sin_l3[c3];
    }

    const uint8_t c4 = l4_codes_ptr[block16];
    const float s3_right = __shfl_down_sync(0xffffffff, s3, 2);

    float contrib = 0.0f;
    if (lane4 == 0) {
        const float s4 = s3 * cos_l4[c4] + s3_right * sin_l4[c4];
        const float radius = __half2float(radii_ptr[block16]);
        contrib = radius * s4;
    }

    return warp_reduce_sum(contrib);
}


__global__ void turboquant_polar_l1_lut_no_factor_global_load_decode_b1q1_warp8_meta64_kernel(
    const uint8_t* __restrict__ packed_meta,
    const half* __restrict__ radii,
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

    const int block16 = lane >> 2;
    const int lane4 = lane & 3;

    const int l1_pair_base = block16 * 8 + lane4 * 2;
    const int l1_idx_a = l1_pair_base;
    const int l1_idx_b = l1_pair_base + 1;

    const uint8_t c1a = load_l1_code_meta64(meta_word_local, l1_idx_a);
    const uint8_t c1b = load_l1_code_meta64(meta_word_local, l1_idx_b);

    // Synthetic register-only surrogates keep dependency on c1a/c1b while
    // removing the factor LUT global loads.
    const float s1a = (static_cast<float>(c1a) + 1.0f) * 0.03125f;
    const float s1b = (static_cast<float>(c1b) + 1.0f) * 0.03125f;

    const int l2_idx = block16 * 4 + lane4;
    const uint8_t c2 = load_l2_code_meta64(meta_word_local, l2_idx);

    const float s2 = s1a * cos_l2[c2] + s1b * sin_l2[c2];

    const float polar_sum = tree_finish_meta_codes_radius(
        s2,
        meta_word_local,
        radii_ptr,
        cos_l3,
        sin_l3,
        cos_l4,
        sin_l4,
        block16,
        lane4
    );

    if (lane == 0) {
        const int64_t out_linear = static_cast<int64_t>(h_idx) * T + t_idx;
        out[out_linear] = polar_sum;
    }
}


__global__ void turboquant_polar_l1_lut_no_radii_global_load_decode_b1q1_warp8_meta64_kernel(
    const uint8_t* __restrict__ packed_meta,
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

    uint32_t meta_word_local = 0;
    if (lane < 16) {
        meta_word_local = reinterpret_cast<const uint32_t*>(meta_ptr)[lane];
    }

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

    const float s2 = s1a * cos_l2[c2] + s1b * sin_l2[c2];

    const float polar_sum = tree_finish_meta_codes_unit_radius(
        s2,
        meta_word_local,
        cos_l3,
        sin_l3,
        cos_l4,
        sin_l4,
        block16,
        lane4
    );

    if (lane == 0) {
        const int64_t out_linear = static_cast<int64_t>(h_idx) * T + t_idx;
        out[out_linear] = polar_sum;
    }
}


__global__ void turboquant_polar_l1_lut_direct_u8_codes_decode_b1q1_warp8_kernel(
    const uint8_t* __restrict__ l1_codes,
    const uint8_t* __restrict__ l2_codes,
    const uint8_t* __restrict__ l3_codes,
    const uint8_t* __restrict__ l4_codes,
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

    const uint8_t* l1_codes_ptr = l1_codes + key_linear * 64;
    const uint8_t* l2_codes_ptr = l2_codes + key_linear * 32;
    const uint8_t* l3_codes_ptr = l3_codes + key_linear * 16;
    const uint8_t* l4_codes_ptr = l4_codes + key_linear * 8;
    const half* radii_ptr = radii + key_linear * 8;

    const int block16 = lane >> 2;
    const int lane4 = lane & 3;

    const int l1_pair_base = block16 * 8 + lane4 * 2;
    const int l1_idx_a = l1_pair_base;
    const int l1_idx_b = l1_pair_base + 1;

    const uint8_t c1a = l1_codes_ptr[l1_idx_a];
    const uint8_t c1b = l1_codes_ptr[l1_idx_b];

    const int64_t l1_base_a =
        (static_cast<int64_t>(h_idx) * L1_LUT_PAIRS + l1_idx_a)
        * L1_LUT_CODES;
    const int64_t l1_base_b =
        (static_cast<int64_t>(h_idx) * L1_LUT_PAIRS + l1_idx_b)
        * L1_LUT_CODES;

    const float s1a = l1_factor_lut[l1_base_a + c1a];
    const float s1b = l1_factor_lut[l1_base_b + c1b];

    const int l2_idx = block16 * 4 + lane4;
    const uint8_t c2 = l2_codes_ptr[l2_idx];

    const float s2 = s1a * cos_l2[c2] + s1b * sin_l2[c2];

    const float polar_sum = tree_finish_direct_codes_radius(
        s2,
        l3_codes_ptr,
        l4_codes_ptr,
        radii_ptr,
        cos_l3,
        sin_l3,
        cos_l4,
        sin_l4,
        block16,
        lane4
    );

    if (lane == 0) {
        const int64_t out_linear = static_cast<int64_t>(h_idx) * T + t_idx;
        out[out_linear] = polar_sum;
    }
}


void validate_meta(
    torch::Tensor packed_meta
) {
    TORCH_CHECK(packed_meta.is_cuda(), "packed_meta must be CUDA");
    TORCH_CHECK(packed_meta.dtype() == torch::kUInt8, "packed_meta must be uint8");
    TORCH_CHECK(packed_meta.dim() == 4, "packed_meta must be [B,H,T,64]");
    TORCH_CHECK(packed_meta.size(0) == 1, "stall probes require B=1");
    TORCH_CHECK(packed_meta.size(3) == META_BYTES, "packed_meta last dim must be 64");
}


void validate_radii(
    torch::Tensor radii,
    int64_t H,
    int64_t T
) {
    TORCH_CHECK(radii.is_cuda(), "radii must be CUDA");
    TORCH_CHECK(radii.dtype() == torch::kFloat16, "radii must be float16");
    TORCH_CHECK(radii.dim() == 4, "radii must be [B,H,T,8]");
    TORCH_CHECK(radii.size(0) == 1, "radii B mismatch");
    TORCH_CHECK(radii.size(1) == H, "radii H mismatch");
    TORCH_CHECK(radii.size(2) == T, "radii T mismatch");
    TORCH_CHECK(radii.size(3) == 8, "radii last dim must be 8");
}


void validate_lut(
    torch::Tensor l1_factor_lut,
    int64_t H
) {
    TORCH_CHECK(l1_factor_lut.is_cuda(), "l1_factor_lut must be CUDA");
    TORCH_CHECK(l1_factor_lut.dtype() == torch::kFloat32, "l1_factor_lut must be float32");
    TORCH_CHECK(l1_factor_lut.dim() == 3, "l1_factor_lut must be [H,64,16]");
    TORCH_CHECK(l1_factor_lut.size(0) == H, "l1_factor_lut H mismatch");
    TORCH_CHECK(l1_factor_lut.size(1) == L1_LUT_PAIRS, "l1_factor_lut pair dim must be 64");
    TORCH_CHECK(l1_factor_lut.size(2) == L1_LUT_CODES, "l1_factor_lut code dim must be 16");
}


void validate_trig(
    torch::Tensor cos_l2,
    torch::Tensor sin_l2,
    torch::Tensor cos_l3,
    torch::Tensor sin_l3,
    torch::Tensor cos_l4,
    torch::Tensor sin_l4
) {
    TORCH_CHECK(cos_l2.numel() == 4 && sin_l2.numel() == 4, "L2 trig tables must have 4 entries");
    TORCH_CHECK(cos_l3.numel() == 4 && sin_l3.numel() == 4, "L3 trig tables must have 4 entries");
    TORCH_CHECK(cos_l4.numel() == 4 && sin_l4.numel() == 4, "L4 trig tables must have 4 entries");
}


void validate_codes(
    torch::Tensor l1_codes,
    torch::Tensor l2_codes,
    torch::Tensor l3_codes,
    torch::Tensor l4_codes
) {
    for (auto t : {l1_codes, l2_codes, l3_codes, l4_codes}) {
        TORCH_CHECK(t.is_cuda(), "direct code tensors must be CUDA");
        TORCH_CHECK(t.dtype() == torch::kUInt8, "direct code tensors must be uint8");
        TORCH_CHECK(t.dim() == 4, "direct code tensors must be [B,H,T,C]");
        TORCH_CHECK(t.size(0) == 1, "direct code tensors require B=1");
    }

    TORCH_CHECK(l1_codes.size(3) == 64, "l1_codes last dim must be 64");
    TORCH_CHECK(l2_codes.size(3) == 32, "l2_codes last dim must be 32");
    TORCH_CHECK(l3_codes.size(3) == 16, "l3_codes last dim must be 16");
    TORCH_CHECK(l4_codes.size(3) == 8, "l4_codes last dim must be 8");

    TORCH_CHECK(l2_codes.size(1) == l1_codes.size(1), "l2 H mismatch");
    TORCH_CHECK(l3_codes.size(1) == l1_codes.size(1), "l3 H mismatch");
    TORCH_CHECK(l4_codes.size(1) == l1_codes.size(1), "l4 H mismatch");
    TORCH_CHECK(l2_codes.size(2) == l1_codes.size(2), "l2 T mismatch");
    TORCH_CHECK(l3_codes.size(2) == l1_codes.size(2), "l3 T mismatch");
    TORCH_CHECK(l4_codes.size(2) == l1_codes.size(2), "l4 T mismatch");
}


torch::Tensor allocate_out(torch::Tensor ref, int64_t H, int64_t T) {
    return torch::empty(
        {1, H, 1, T},
        torch::TensorOptions()
            .device(ref.device())
            .dtype(torch::kFloat32)
    );
}


void validate_grid(int64_t H, int64_t T) {
    const int64_t groups = (T + 7) / 8;
    TORCH_CHECK(groups <= 2147483647LL, "stall probe grid.x exceeds CUDA limit");
    TORCH_CHECK(H <= 65535, "stall probe grid.y exceeds CUDA limit");
}

} // namespace


torch::Tensor turboquant_polar_l1_lut_no_factor_global_load_cuda(
    torch::Tensor packed_meta,
    torch::Tensor radii,
    torch::Tensor cos_l2,
    torch::Tensor sin_l2,
    torch::Tensor cos_l3,
    torch::Tensor sin_l3,
    torch::Tensor cos_l4,
    torch::Tensor sin_l4
) {
    validate_meta(packed_meta);
    const int64_t H = packed_meta.size(1);
    const int64_t T = packed_meta.size(2);
    validate_radii(radii, H, T);
    validate_trig(cos_l2, sin_l2, cos_l3, sin_l3, cos_l4, sin_l4);
    validate_grid(H, T);

    auto out = allocate_out(packed_meta, H, T);
    const dim3 block(THREADS);
    const dim3 grid(static_cast<unsigned int>((T + 7) / 8), static_cast<unsigned int>(H), 1);

    turboquant_polar_l1_lut_no_factor_global_load_decode_b1q1_warp8_meta64_kernel<<<grid, block>>>(
        packed_meta.contiguous().data_ptr<uint8_t>(),
        reinterpret_cast<const half*>(radii.contiguous().data_ptr<at::Half>()),
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


torch::Tensor turboquant_polar_l1_lut_no_radii_global_load_cuda(
    torch::Tensor packed_meta,
    torch::Tensor l1_factor_lut,
    torch::Tensor cos_l2,
    torch::Tensor sin_l2,
    torch::Tensor cos_l3,
    torch::Tensor sin_l3,
    torch::Tensor cos_l4,
    torch::Tensor sin_l4
) {
    validate_meta(packed_meta);
    const int64_t H = packed_meta.size(1);
    const int64_t T = packed_meta.size(2);
    validate_lut(l1_factor_lut, H);
    validate_trig(cos_l2, sin_l2, cos_l3, sin_l3, cos_l4, sin_l4);
    validate_grid(H, T);

    auto out = allocate_out(packed_meta, H, T);
    const dim3 block(THREADS);
    const dim3 grid(static_cast<unsigned int>((T + 7) / 8), static_cast<unsigned int>(H), 1);

    turboquant_polar_l1_lut_no_radii_global_load_decode_b1q1_warp8_meta64_kernel<<<grid, block>>>(
        packed_meta.contiguous().data_ptr<uint8_t>(),
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


torch::Tensor turboquant_polar_l1_lut_direct_u8_codes_cuda(
    torch::Tensor l1_codes,
    torch::Tensor l2_codes,
    torch::Tensor l3_codes,
    torch::Tensor l4_codes,
    torch::Tensor radii,
    torch::Tensor l1_factor_lut,
    torch::Tensor cos_l2,
    torch::Tensor sin_l2,
    torch::Tensor cos_l3,
    torch::Tensor sin_l3,
    torch::Tensor cos_l4,
    torch::Tensor sin_l4
) {
    validate_codes(l1_codes, l2_codes, l3_codes, l4_codes);
    const int64_t H = l1_codes.size(1);
    const int64_t T = l1_codes.size(2);
    validate_radii(radii, H, T);
    validate_lut(l1_factor_lut, H);
    validate_trig(cos_l2, sin_l2, cos_l3, sin_l3, cos_l4, sin_l4);
    validate_grid(H, T);

    auto out = allocate_out(l1_codes, H, T);
    const dim3 block(THREADS);
    const dim3 grid(static_cast<unsigned int>((T + 7) / 8), static_cast<unsigned int>(H), 1);

    turboquant_polar_l1_lut_direct_u8_codes_decode_b1q1_warp8_kernel<<<grid, block>>>(
        l1_codes.contiguous().data_ptr<uint8_t>(),
        l2_codes.contiguous().data_ptr<uint8_t>(),
        l3_codes.contiguous().data_ptr<uint8_t>(),
        l4_codes.contiguous().data_ptr<uint8_t>(),
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


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "turboquant_polar_l1_lut_no_factor_global_load_cuda",
        &turboquant_polar_l1_lut_no_factor_global_load_cuda,
        "Polar-only L1-LUT stall probe: remove factor LUT global loads"
    );
    m.def(
        "turboquant_polar_l1_lut_no_radii_global_load_cuda",
        &turboquant_polar_l1_lut_no_radii_global_load_cuda,
        "Polar-only L1-LUT stall probe: remove radii global loads"
    );
    m.def(
        "turboquant_polar_l1_lut_direct_u8_codes_cuda",
        &turboquant_polar_l1_lut_direct_u8_codes_cuda,
        "Polar-only L1-LUT stall probe: direct uint8 angle codes"
    );
}
