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
constexpr int QJL_LANE_SIGN_BYTES = 16;
constexpr int THREADS = 256;   // 8 warps / CTA
constexpr int L2_LUT_GROUPS = 32;
constexpr int L2_LUT_CODES = 1024;
constexpr float QJL_CORRECTION_SCALE = 0.375f;
constexpr unsigned FULL_MASK = 0xffffffffu;


__device__ __forceinline__ uint8_t byte_from_u32(uint32_t word, int byte_idx) {
    return static_cast<uint8_t>((word >> (byte_idx * 8)) & 0xFFu);
}


__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    return val;
}


__device__ __forceinline__ uint8_t load_l1_code_meta64(uint32_t meta_word_local, int l1_idx) {
    const int byte_idx = l1_idx >> 1;
    const int word_owner = byte_idx >> 2;
    const int byte_in_word = byte_idx & 3;
    const uint32_t word = __shfl_sync(FULL_MASK, meta_word_local, word_owner);
    const uint8_t packed_byte = byte_from_u32(word, byte_in_word);
    return static_cast<uint8_t>((packed_byte >> ((l1_idx & 1) * 4)) & 0x0Fu);
}


__device__ __forceinline__ uint8_t load_l2_code_meta64(uint32_t meta_word_local, int l2_idx) {
    const int byte_idx = l2_idx >> 2;
    const int word_owner = 8 + (byte_idx >> 2);
    const int byte_in_word = byte_idx & 3;
    const uint32_t word = __shfl_sync(FULL_MASK, meta_word_local, word_owner);
    const uint8_t packed_byte = byte_from_u32(word, byte_in_word);
    return static_cast<uint8_t>((packed_byte >> ((l2_idx & 3) * 2)) & 0x03u);
}


__device__ __forceinline__ uint8_t load_l3_code_meta64(uint32_t meta_word_local, int l3_idx) {
    const int byte_idx = l3_idx >> 2;
    const uint32_t word = __shfl_sync(FULL_MASK, meta_word_local, 10);
    const uint8_t packed_byte = byte_from_u32(word, byte_idx);
    return static_cast<uint8_t>((packed_byte >> ((l3_idx & 3) * 2)) & 0x03u);
}


__device__ __forceinline__ uint8_t load_l4_code_meta64(uint32_t meta_word_local, int l4_idx) {
    const int byte_idx = l4_idx >> 2;
    const uint32_t word = __shfl_sync(FULL_MASK, meta_word_local, 11);
    const uint8_t packed_byte = byte_from_u32(word, byte_idx);
    return static_cast<uint8_t>((packed_byte >> ((l4_idx & 3) * 2)) & 0x03u);
}


__device__ __forceinline__ float l2_combined_s2_fp16(
    uint32_t meta_word_local,
    const half* __restrict__ l2_factor_lut_fp16,
    int h_idx,
    int block16,
    int lane4
) {
    const int l1_pair_base = block16 * 8 + lane4 * 2;
    const int l1_idx_a = l1_pair_base;
    const int l1_idx_b = l1_pair_base + 1;

    const uint8_t c1a = load_l1_code_meta64(meta_word_local, l1_idx_a);
    const uint8_t c1b = load_l1_code_meta64(meta_word_local, l1_idx_b);

    const int l2_idx = block16 * 4 + lane4;
    const uint8_t c2 = load_l2_code_meta64(meta_word_local, l2_idx);

    const int combo =
        (static_cast<int>(c2) << 8)
        | (static_cast<int>(c1b) << 4)
        | static_cast<int>(c1a);

    const int64_t l2_base =
        (static_cast<int64_t>(h_idx) * L2_LUT_GROUPS + l2_idx)
        * L2_LUT_CODES;

    return __half2float(l2_factor_lut_fp16[l2_base + combo]);
}


__device__ __forceinline__ float l2_combined_s2_fp16_combo_major_layout(
    uint32_t meta_word_local,
    const half* __restrict__ l2_factor_lut_combo_major_fp16,
    int h_idx,
    int block16,
    int lane4
) {
    // Exact-value factor LUT layout ablation.
    //
    // Baseline layout:
    //   [H, 32 groups, 1024 combos]
    //   flat index = (h * 32 + l2_idx) * 1024 + combo
    //
    // Candidate combo-major layout:
    //   [H, 1024 combos, 32 groups]
    //   flat index = (h * 1024 + combo) * 32 + l2_idx
    //
    // The LUT value is identical; only memory order changes.
    const int l1_pair_base = block16 * 8 + lane4 * 2;
    const int l1_idx_a = l1_pair_base;
    const int l1_idx_b = l1_pair_base + 1;

    const uint8_t c1a = load_l1_code_meta64(meta_word_local, l1_idx_a);
    const uint8_t c1b = load_l1_code_meta64(meta_word_local, l1_idx_b);

    const int l2_idx = block16 * 4 + lane4;
    const uint8_t c2 = load_l2_code_meta64(meta_word_local, l2_idx);

    const int combo =
        (static_cast<int>(c2) << 8)
        | (static_cast<int>(c1b) << 4)
        | static_cast<int>(c1a);

    const int64_t lut_idx =
        (static_cast<int64_t>(h_idx) * L2_LUT_CODES + combo)
        * L2_LUT_GROUPS
        + l2_idx;

    return __half2float(l2_factor_lut_combo_major_fp16[lut_idx]);
}


__device__ __forceinline__ float radii_pair_word_to_float(
    uint32_t packed_radii_pair,
    int block16
) {
    const uint16_t radius_bits = static_cast<uint16_t>(
        (block16 & 1)
            ? ((packed_radii_pair >> 16) & 0xFFFFu)
            : (packed_radii_pair & 0xFFFFu)
    );
    return __half2float(__ushort_as_half(radius_bits));
}


__device__ __forceinline__ float load_vectorized_radius_float(
    uint32_t radii_pair_word_local,
    int block16
) {
    const int pair_owner = block16 >> 1;
    const uint32_t pair_word = __shfl_sync(
        FULL_MASK,
        radii_pair_word_local,
        pair_owner
    );
    return radii_pair_word_to_float(pair_word, block16);
}


__device__ __forceinline__ float finish_tree_polar_with_radius(
    float s2,
    uint32_t meta_word_local,
    float radius_float,
    const float* __restrict__ cos_l3,
    const float* __restrict__ sin_l3,
    const float* __restrict__ cos_l4,
    const float* __restrict__ sin_l4,
    int block16,
    int lane4
) {
    const int l3_idx = block16 * 2 + (lane4 >> 1);
    const uint8_t c3 = load_l3_code_meta64(meta_word_local, l3_idx);

    const float s2_right = __shfl_down_sync(FULL_MASK, s2, 1);

    float s3 = 0.0f;
    if ((lane4 & 1) == 0) {
        s3 = s2 * cos_l3[c3] + s2_right * sin_l3[c3];
    }

    const uint8_t c4 = load_l4_code_meta64(meta_word_local, block16);
    const float s3_right = __shfl_down_sync(FULL_MASK, s3, 2);

    float polar_block_contrib = 0.0f;
    if (lane4 == 0) {
        const float s4 = s3 * cos_l4[c4] + s3_right * sin_l4[c4];
        polar_block_contrib = radius_float * s4;
    }

    return warp_reduce_sum(polar_block_contrib);
}


__device__ __forceinline__ float qjl_sum_lane_nibble(
    const uint8_t* __restrict__ lane_nibble_signs_ptr,
    const float* __restrict__ sh_qproj,
    int lane
) {
    const uint8_t packed_nibble_pair = lane_nibble_signs_ptr[lane >> 1];
    const uint8_t lane_nibble =
        (lane & 1)
            ? static_cast<uint8_t>((packed_nibble_pair >> 4) & 0x0Fu)
            : static_cast<uint8_t>(packed_nibble_pair & 0x0Fu);

    float qjl_acc = 0.0f;
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int sketch_idx = lane + 32 * k;
        const float qproj_val = sh_qproj[sketch_idx];
        const uint8_t sign_bit = static_cast<uint8_t>((lane_nibble >> k) & 0x01u);
        qjl_acc += sign_bit ? qproj_val : -qproj_val;
    }
    return warp_reduce_sum(qjl_acc);
}


__global__ void turboquant_polar_tree_l2_combined_lut_fp16_early_radii_polar_only_decode_b1q1_warp8_meta64_kernel(
    const uint8_t* __restrict__ packed_meta,
    const half* __restrict__ radii,
    const half* __restrict__ l2_factor_lut_fp16,
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

    const int block16 = lane >> 2;
    const int lane4 = lane & 3;

    float radius_float = 0.0f;
    if (lane4 == 0) {
        radius_float = __half2float(radii_ptr[block16]);
    }

    uint32_t meta_word_local = 0;
    if (lane < 16) {
        meta_word_local = reinterpret_cast<const uint32_t*>(meta_ptr)[lane];
    }

    const float s2 = l2_combined_s2_fp16(
        meta_word_local,
        l2_factor_lut_fp16,
        h_idx,
        block16,
        lane4
    );

    const float polar_sum = finish_tree_polar_with_radius(
        s2,
        meta_word_local,
        radius_float,
        cos_l3,
        sin_l3,
        cos_l4,
        sin_l4,
        block16,
        lane4
    );

    if (lane == 0) {
        out[static_cast<int64_t>(h_idx) * T + t_idx] = polar_sum;
    }
}


__global__ void turboquant_polar_tree_l2_combined_lut_fp16_vector_radii_polar_only_decode_b1q1_warp8_meta64_kernel(
    const uint8_t* __restrict__ packed_meta,
    const half* __restrict__ radii,
    const half* __restrict__ l2_factor_lut_fp16,
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

    const int block16 = lane >> 2;
    const int lane4 = lane & 3;

    uint32_t radii_pair_word_local = 0;
    if (lane < 4) {
        radii_pair_word_local =
            reinterpret_cast<const uint32_t*>(radii_ptr)[lane];
    }
    const float radius_float =
        load_vectorized_radius_float(radii_pair_word_local, block16);

    uint32_t meta_word_local = 0;
    if (lane < 16) {
        meta_word_local = reinterpret_cast<const uint32_t*>(meta_ptr)[lane];
    }

    const float s2 = l2_combined_s2_fp16(
        meta_word_local,
        l2_factor_lut_fp16,
        h_idx,
        block16,
        lane4
    );

    const float polar_sum = finish_tree_polar_with_radius(
        s2,
        meta_word_local,
        radius_float,
        cos_l3,
        sin_l3,
        cos_l4,
        sin_l4,
        block16,
        lane4
    );

    if (lane == 0) {
        out[static_cast<int64_t>(h_idx) * T + t_idx] = polar_sum;
    }
}


__global__ void turboquant_polar_tree_l2_combined_lut_fp16_early_radii_lane_nibble_fused_logits_decode_b1q1_warp8_meta64_kernel(
    const float* __restrict__ q_projected,
    const uint8_t* __restrict__ packed_meta,
    const half* __restrict__ radii,
    const half* __restrict__ l2_factor_lut_fp16,
    const float* __restrict__ cos_l3,
    const float* __restrict__ sin_l3,
    const float* __restrict__ cos_l4,
    const float* __restrict__ sin_l4,
    const uint8_t* __restrict__ lane_nibble_qjl_signs,
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
    const half* radii_ptr = radii + key_linear * 8;
    const uint8_t* lane_signs_ptr =
        lane_nibble_qjl_signs + key_linear * QJL_LANE_SIGN_BYTES;

    const int block16 = lane >> 2;
    const int lane4 = lane & 3;

    float radius_float = 0.0f;
    if (lane4 == 0) {
        radius_float = __half2float(radii_ptr[block16]);
    }

    uint32_t meta_word_local = 0;
    if (lane < 16) {
        meta_word_local = reinterpret_cast<const uint32_t*>(meta_ptr)[lane];
    }

    const float s2 = l2_combined_s2_fp16(
        meta_word_local,
        l2_factor_lut_fp16,
        h_idx,
        block16,
        lane4
    );

    const float polar_sum = finish_tree_polar_with_radius(
        s2,
        meta_word_local,
        radius_float,
        cos_l3,
        sin_l3,
        cos_l4,
        sin_l4,
        block16,
        lane4
    );

    const float qjl_sum = qjl_sum_lane_nibble(
        lane_signs_ptr,
        sh_qproj,
        lane
    );

    if (lane == 0) {
        const float residual_norm = __half2float(qjl_norms[key_linear]);
        const float qjl_scale =
            QJL_CORRECTION_SCALE
            * sqrtf(3.14159265358979323846f / 2.0f)
            / sqrtf(static_cast<float>(M));
        out[static_cast<int64_t>(h_idx) * T + t_idx] =
            polar_sum + qjl_scale * residual_norm * qjl_sum;
    }
}



__global__ void turboquant_polar_tree_l2_combined_lut_fp16_early_radii_early_qjl_norm_lane_nibble_fused_logits_decode_b1q1_warp8_meta64_kernel(
    const float* __restrict__ q_projected,
    const uint8_t* __restrict__ packed_meta,
    const half* __restrict__ radii,
    const half* __restrict__ l2_factor_lut_fp16,
    const float* __restrict__ cos_l3,
    const float* __restrict__ sin_l3,
    const float* __restrict__ cos_l4,
    const float* __restrict__ sin_l4,
    const uint8_t* __restrict__ lane_nibble_qjl_signs,
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
    const half* radii_ptr = radii + key_linear * 8;
    const uint8_t* lane_signs_ptr =
        lane_nibble_qjl_signs + key_linear * QJL_LANE_SIGN_BYTES;

    // The previous best path loaded qjl_norms[key_linear] immediately before
    // the final output write. Source-level NCU showed that this late half load
    // was the second-largest remaining L1TEX scoreboard source.
    //
    // Hoist it here so the load latency can overlap with:
    //   - early-radii Polar tree work
    //   - packed-meta decode
    //   - L2 combined factor LUT lookup
    //   - lane-nibble QJL sign decode and reduction
    float residual_norm_float = 0.0f;
    if (lane == 0) {
        residual_norm_float = __half2float(qjl_norms[key_linear]);
    }

    const int block16 = lane >> 2;
    const int lane4 = lane & 3;

    float radius_float = 0.0f;
    if (lane4 == 0) {
        radius_float = __half2float(radii_ptr[block16]);
    }

    uint32_t meta_word_local = 0;
    if (lane < 16) {
        meta_word_local = reinterpret_cast<const uint32_t*>(meta_ptr)[lane];
    }

    const float s2 = l2_combined_s2_fp16(
        meta_word_local,
        l2_factor_lut_fp16,
        h_idx,
        block16,
        lane4
    );

    const float polar_sum = finish_tree_polar_with_radius(
        s2,
        meta_word_local,
        radius_float,
        cos_l3,
        sin_l3,
        cos_l4,
        sin_l4,
        block16,
        lane4
    );

    const float qjl_sum = qjl_sum_lane_nibble(
        lane_signs_ptr,
        sh_qproj,
        lane
    );

    if (lane == 0) {
        const float qjl_scale =
            QJL_CORRECTION_SCALE
            * sqrtf(3.14159265358979323846f / 2.0f)
            / sqrtf(static_cast<float>(M));

        out[static_cast<int64_t>(h_idx) * T + t_idx] =
            polar_sum + qjl_scale * residual_norm_float * qjl_sum;
    }
}



__global__ void turboquant_polar_tree_l2_combined_lut_fp16_combo_major_layout_early_radii_early_qjl_norm_lane_nibble_fused_logits_decode_b1q1_warp8_meta64_kernel(
    const float* __restrict__ q_projected,
    const uint8_t* __restrict__ packed_meta,
    const half* __restrict__ radii,
    const half* __restrict__ l2_factor_lut_combo_major_fp16,
    const float* __restrict__ cos_l3,
    const float* __restrict__ sin_l3,
    const float* __restrict__ cos_l4,
    const float* __restrict__ sin_l4,
    const uint8_t* __restrict__ lane_nibble_qjl_signs,
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
    const half* radii_ptr = radii + key_linear * 8;
    const uint8_t* lane_signs_ptr =
        lane_nibble_qjl_signs + key_linear * QJL_LANE_SIGN_BYTES;

    // Keep the validated early qjl_norm optimization.
    float residual_norm_float = 0.0f;
    if (lane == 0) {
        residual_norm_float = __half2float(qjl_norms[key_linear]);
    }

    const int block16 = lane >> 2;
    const int lane4 = lane & 3;

    // Keep the validated early radii optimization.
    float radius_float = 0.0f;
    if (lane4 == 0) {
        radius_float = __half2float(radii_ptr[block16]);
    }

    uint32_t meta_word_local = 0;
    if (lane < 16) {
        meta_word_local = reinterpret_cast<const uint32_t*>(meta_ptr)[lane];
    }

    const float s2 = l2_combined_s2_fp16_combo_major_layout(
        meta_word_local,
        l2_factor_lut_combo_major_fp16,
        h_idx,
        block16,
        lane4
    );

    const float polar_sum = finish_tree_polar_with_radius(
        s2,
        meta_word_local,
        radius_float,
        cos_l3,
        sin_l3,
        cos_l4,
        sin_l4,
        block16,
        lane4
    );

    const float qjl_sum = qjl_sum_lane_nibble(
        lane_signs_ptr,
        sh_qproj,
        lane
    );

    if (lane == 0) {
        const float qjl_scale =
            QJL_CORRECTION_SCALE
            * sqrtf(3.14159265358979323846f / 2.0f)
            / sqrtf(static_cast<float>(M));

        out[static_cast<int64_t>(h_idx) * T + t_idx] =
            polar_sum + qjl_scale * residual_norm_float * qjl_sum;
    }
}


__global__ void turboquant_polar_tree_l2_combined_lut_fp16_vector_radii_lane_nibble_fused_logits_decode_b1q1_warp8_meta64_kernel(
    const float* __restrict__ q_projected,
    const uint8_t* __restrict__ packed_meta,
    const half* __restrict__ radii,
    const half* __restrict__ l2_factor_lut_fp16,
    const float* __restrict__ cos_l3,
    const float* __restrict__ sin_l3,
    const float* __restrict__ cos_l4,
    const float* __restrict__ sin_l4,
    const uint8_t* __restrict__ lane_nibble_qjl_signs,
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
    const half* radii_ptr = radii + key_linear * 8;
    const uint8_t* lane_signs_ptr =
        lane_nibble_qjl_signs + key_linear * QJL_LANE_SIGN_BYTES;

    const int block16 = lane >> 2;
    const int lane4 = lane & 3;

    uint32_t radii_pair_word_local = 0;
    if (lane < 4) {
        radii_pair_word_local =
            reinterpret_cast<const uint32_t*>(radii_ptr)[lane];
    }
    const float radius_float =
        load_vectorized_radius_float(radii_pair_word_local, block16);

    uint32_t meta_word_local = 0;
    if (lane < 16) {
        meta_word_local = reinterpret_cast<const uint32_t*>(meta_ptr)[lane];
    }

    const float s2 = l2_combined_s2_fp16(
        meta_word_local,
        l2_factor_lut_fp16,
        h_idx,
        block16,
        lane4
    );

    const float polar_sum = finish_tree_polar_with_radius(
        s2,
        meta_word_local,
        radius_float,
        cos_l3,
        sin_l3,
        cos_l4,
        sin_l4,
        block16,
        lane4
    );

    const float qjl_sum = qjl_sum_lane_nibble(
        lane_signs_ptr,
        sh_qproj,
        lane
    );

    if (lane == 0) {
        const float residual_norm = __half2float(qjl_norms[key_linear]);
        const float qjl_scale =
            QJL_CORRECTION_SCALE
            * sqrtf(3.14159265358979323846f / 2.0f)
            / sqrtf(static_cast<float>(M));
        out[static_cast<int64_t>(h_idx) * T + t_idx] =
            polar_sum + qjl_scale * residual_norm * qjl_sum;
    }
}


void validate_meta(torch::Tensor packed_meta) {
    TORCH_CHECK(packed_meta.is_cuda(), "packed_meta must be CUDA");
    TORCH_CHECK(packed_meta.dtype() == torch::kUInt8, "packed_meta must be uint8");
    TORCH_CHECK(packed_meta.dim() == 4, "packed_meta must be [B,H,T,64]");
    TORCH_CHECK(packed_meta.size(0) == 1, "radii optimization kernels require B=1");
    TORCH_CHECK(packed_meta.size(3) == META_BYTES, "packed_meta last dim must be 64");
}


void validate_radii(torch::Tensor radii, int64_t H, int64_t T) {
    TORCH_CHECK(radii.is_cuda(), "radii must be CUDA");
    TORCH_CHECK(radii.dtype() == torch::kFloat16, "radii must be float16");
    TORCH_CHECK(radii.dim() == 4, "radii must be [B,H,T,8]");
    TORCH_CHECK(radii.size(0) == 1, "radii B mismatch");
    TORCH_CHECK(radii.size(1) == H, "radii H mismatch");
    TORCH_CHECK(radii.size(2) == T, "radii T mismatch");
    TORCH_CHECK(radii.size(3) == 8, "radii last dim must be 8");
}


void validate_l2_lut_fp16(torch::Tensor lut, int64_t H) {
    TORCH_CHECK(lut.is_cuda(), "l2_factor_lut_fp16 must be CUDA");
    TORCH_CHECK(lut.dtype() == torch::kFloat16, "l2_factor_lut_fp16 must be float16");
    TORCH_CHECK(lut.dim() == 3, "l2_factor_lut_fp16 must be [H,32,1024]");
    TORCH_CHECK(lut.size(0) == H, "l2_factor_lut_fp16 H mismatch");
    TORCH_CHECK(lut.size(1) == L2_LUT_GROUPS, "l2_factor_lut_fp16 group dim must be 32");
    TORCH_CHECK(lut.size(2) == L2_LUT_CODES, "l2_factor_lut_fp16 code dim must be 1024");
}


void validate_trig(
    torch::Tensor cos_l3,
    torch::Tensor sin_l3,
    torch::Tensor cos_l4,
    torch::Tensor sin_l4
) {
    TORCH_CHECK(cos_l3.numel() == 4 && sin_l3.numel() == 4, "L3 trig tables must have 4 entries");
    TORCH_CHECK(cos_l4.numel() == 4 && sin_l4.numel() == 4, "L4 trig tables must have 4 entries");
}


void validate_fused_extras(
    torch::Tensor q_projected,
    torch::Tensor lane_nibble_qjl_signs,
    torch::Tensor qjl_norms,
    int64_t H,
    int64_t T
) {
    TORCH_CHECK(q_projected.is_cuda(), "q_projected must be CUDA");
    TORCH_CHECK(q_projected.dtype() == torch::kFloat32, "q_projected must be float32");
    TORCH_CHECK(q_projected.dim() == 4, "q_projected must be [B,H,Q,128]");
    TORCH_CHECK(q_projected.size(0) == 1, "q_projected B mismatch");
    TORCH_CHECK(q_projected.size(1) == H, "q_projected H mismatch");
    TORCH_CHECK(q_projected.size(2) == 1, "q_projected Q must be 1");
    TORCH_CHECK(q_projected.size(3) == M, "q_projected last dim must be 128");

    TORCH_CHECK(lane_nibble_qjl_signs.is_cuda(), "lane_nibble_qjl_signs must be CUDA");
    TORCH_CHECK(lane_nibble_qjl_signs.dtype() == torch::kUInt8, "lane_nibble_qjl_signs must be uint8");
    TORCH_CHECK(lane_nibble_qjl_signs.dim() == 4, "lane_nibble_qjl_signs must be [B,H,T,16]");
    TORCH_CHECK(lane_nibble_qjl_signs.size(0) == 1, "lane_nibble_qjl_signs B mismatch");
    TORCH_CHECK(lane_nibble_qjl_signs.size(1) == H, "lane_nibble_qjl_signs H mismatch");
    TORCH_CHECK(lane_nibble_qjl_signs.size(2) == T, "lane_nibble_qjl_signs T mismatch");
    TORCH_CHECK(lane_nibble_qjl_signs.size(3) == QJL_LANE_SIGN_BYTES, "lane_nibble_qjl_signs last dim must be 16");

    TORCH_CHECK(qjl_norms.is_cuda(), "qjl_norms must be CUDA");
    TORCH_CHECK(qjl_norms.dtype() == torch::kFloat16, "qjl_norms must be float16");
    TORCH_CHECK(qjl_norms.dim() == 3, "qjl_norms must be [B,H,T]");
    TORCH_CHECK(qjl_norms.size(0) == 1, "qjl_norms B mismatch");
    TORCH_CHECK(qjl_norms.size(1) == H, "qjl_norms H mismatch");
    TORCH_CHECK(qjl_norms.size(2) == T, "qjl_norms T mismatch");
}



void validate_l2_lut_combo_major_fp16(torch::Tensor lut, int64_t H) {
    TORCH_CHECK(lut.is_cuda(), "l2_factor_lut_combo_major_fp16 must be CUDA");
    TORCH_CHECK(lut.dtype() == torch::kFloat16, "l2_factor_lut_combo_major_fp16 must be float16");
    TORCH_CHECK(lut.dim() == 3, "l2_factor_lut_combo_major_fp16 must be [H,1024,32]");
    TORCH_CHECK(lut.size(0) == H, "l2_factor_lut_combo_major_fp16 H mismatch");
    TORCH_CHECK(lut.size(1) == L2_LUT_CODES, "l2_factor_lut_combo_major_fp16 combo dim must be 1024");
    TORCH_CHECK(lut.size(2) == L2_LUT_GROUPS, "l2_factor_lut_combo_major_fp16 group dim must be 32");
}


torch::Tensor allocate_out(torch::Tensor ref, int64_t H, int64_t T) {
    return torch::empty(
        {1, H, 1, T},
        torch::TensorOptions().device(ref.device()).dtype(torch::kFloat32)
    );
}


void validate_grid(int64_t H, int64_t T) {
    const int64_t groups = (T + 7) / 8;
    TORCH_CHECK(groups <= 2147483647LL, "radii optimization grid.x exceeds CUDA limit");
    TORCH_CHECK(H <= 65535, "radii optimization grid.y exceeds CUDA limit");
}

} // namespace


torch::Tensor turboquant_polar_tree_l2_combined_lut_fp16_early_radii_polar_only_cuda(
    torch::Tensor packed_meta,
    torch::Tensor radii,
    torch::Tensor l2_factor_lut_fp16,
    torch::Tensor cos_l3,
    torch::Tensor sin_l3,
    torch::Tensor cos_l4,
    torch::Tensor sin_l4
) {
    validate_meta(packed_meta);
    const int64_t H = packed_meta.size(1);
    const int64_t T = packed_meta.size(2);
    validate_radii(radii, H, T);
    validate_l2_lut_fp16(l2_factor_lut_fp16, H);
    validate_trig(cos_l3, sin_l3, cos_l4, sin_l4);
    validate_grid(H, T);

    auto out = allocate_out(packed_meta, H, T);
    const dim3 block(THREADS);
    const dim3 grid(static_cast<unsigned int>((T + 7) / 8), static_cast<unsigned int>(H), 1);

    turboquant_polar_tree_l2_combined_lut_fp16_early_radii_polar_only_decode_b1q1_warp8_meta64_kernel<<<grid, block>>>(
        packed_meta.contiguous().data_ptr<uint8_t>(),
        reinterpret_cast<const half*>(radii.contiguous().data_ptr<at::Half>()),
        reinterpret_cast<const half*>(l2_factor_lut_fp16.contiguous().data_ptr<at::Half>()),
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


torch::Tensor turboquant_polar_tree_l2_combined_lut_fp16_vector_radii_polar_only_cuda(
    torch::Tensor packed_meta,
    torch::Tensor radii,
    torch::Tensor l2_factor_lut_fp16,
    torch::Tensor cos_l3,
    torch::Tensor sin_l3,
    torch::Tensor cos_l4,
    torch::Tensor sin_l4
) {
    validate_meta(packed_meta);
    const int64_t H = packed_meta.size(1);
    const int64_t T = packed_meta.size(2);
    validate_radii(radii, H, T);
    validate_l2_lut_fp16(l2_factor_lut_fp16, H);
    validate_trig(cos_l3, sin_l3, cos_l4, sin_l4);
    validate_grid(H, T);

    auto out = allocate_out(packed_meta, H, T);
    const dim3 block(THREADS);
    const dim3 grid(static_cast<unsigned int>((T + 7) / 8), static_cast<unsigned int>(H), 1);

    turboquant_polar_tree_l2_combined_lut_fp16_vector_radii_polar_only_decode_b1q1_warp8_meta64_kernel<<<grid, block>>>(
        packed_meta.contiguous().data_ptr<uint8_t>(),
        reinterpret_cast<const half*>(radii.contiguous().data_ptr<at::Half>()),
        reinterpret_cast<const half*>(l2_factor_lut_fp16.contiguous().data_ptr<at::Half>()),
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


torch::Tensor turboquant_polar_tree_l2_combined_lut_fp16_early_radii_lane_nibble_fused_logits_cuda(
    torch::Tensor q_projected,
    torch::Tensor packed_meta,
    torch::Tensor radii,
    torch::Tensor l2_factor_lut_fp16,
    torch::Tensor cos_l3,
    torch::Tensor sin_l3,
    torch::Tensor cos_l4,
    torch::Tensor sin_l4,
    torch::Tensor lane_nibble_qjl_signs,
    torch::Tensor qjl_norms
) {
    validate_meta(packed_meta);
    const int64_t H = packed_meta.size(1);
    const int64_t T = packed_meta.size(2);
    validate_radii(radii, H, T);
    validate_l2_lut_fp16(l2_factor_lut_fp16, H);
    validate_trig(cos_l3, sin_l3, cos_l4, sin_l4);
    validate_fused_extras(q_projected, lane_nibble_qjl_signs, qjl_norms, H, T);
    validate_grid(H, T);

    auto out = allocate_out(packed_meta, H, T);
    const dim3 block(THREADS);
    const dim3 grid(static_cast<unsigned int>((T + 7) / 8), static_cast<unsigned int>(H), 1);

    turboquant_polar_tree_l2_combined_lut_fp16_early_radii_lane_nibble_fused_logits_decode_b1q1_warp8_meta64_kernel<<<grid, block>>>(
        q_projected.contiguous().data_ptr<float>(),
        packed_meta.contiguous().data_ptr<uint8_t>(),
        reinterpret_cast<const half*>(radii.contiguous().data_ptr<at::Half>()),
        reinterpret_cast<const half*>(l2_factor_lut_fp16.contiguous().data_ptr<at::Half>()),
        cos_l3.contiguous().data_ptr<float>(),
        sin_l3.contiguous().data_ptr<float>(),
        cos_l4.contiguous().data_ptr<float>(),
        sin_l4.contiguous().data_ptr<float>(),
        lane_nibble_qjl_signs.contiguous().data_ptr<uint8_t>(),
        reinterpret_cast<const half*>(qjl_norms.contiguous().data_ptr<at::Half>()),
        out.data_ptr<float>(),
        static_cast<int>(H),
        static_cast<int>(T)
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}



torch::Tensor turboquant_polar_tree_l2_combined_lut_fp16_early_radii_early_qjl_norm_lane_nibble_fused_logits_cuda(
    torch::Tensor q_projected,
    torch::Tensor packed_meta,
    torch::Tensor radii,
    torch::Tensor l2_factor_lut_fp16,
    torch::Tensor cos_l3,
    torch::Tensor sin_l3,
    torch::Tensor cos_l4,
    torch::Tensor sin_l4,
    torch::Tensor lane_nibble_qjl_signs,
    torch::Tensor qjl_norms
) {
    validate_meta(packed_meta);
    const int64_t H = packed_meta.size(1);
    const int64_t T = packed_meta.size(2);
    validate_radii(radii, H, T);
    validate_l2_lut_fp16(l2_factor_lut_fp16, H);
    validate_trig(cos_l3, sin_l3, cos_l4, sin_l4);
    validate_fused_extras(q_projected, lane_nibble_qjl_signs, qjl_norms, H, T);
    validate_grid(H, T);

    auto out = allocate_out(packed_meta, H, T);
    const dim3 block(THREADS);
    const dim3 grid(static_cast<unsigned int>((T + 7) / 8), static_cast<unsigned int>(H), 1);

    turboquant_polar_tree_l2_combined_lut_fp16_early_radii_early_qjl_norm_lane_nibble_fused_logits_decode_b1q1_warp8_meta64_kernel<<<grid, block>>>(
        q_projected.contiguous().data_ptr<float>(),
        packed_meta.contiguous().data_ptr<uint8_t>(),
        reinterpret_cast<const half*>(radii.contiguous().data_ptr<at::Half>()),
        reinterpret_cast<const half*>(l2_factor_lut_fp16.contiguous().data_ptr<at::Half>()),
        cos_l3.contiguous().data_ptr<float>(),
        sin_l3.contiguous().data_ptr<float>(),
        cos_l4.contiguous().data_ptr<float>(),
        sin_l4.contiguous().data_ptr<float>(),
        lane_nibble_qjl_signs.contiguous().data_ptr<uint8_t>(),
        reinterpret_cast<const half*>(qjl_norms.contiguous().data_ptr<at::Half>()),
        out.data_ptr<float>(),
        static_cast<int>(H),
        static_cast<int>(T)
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}



torch::Tensor turboquant_polar_tree_l2_combined_lut_fp16_combo_major_layout_early_radii_early_qjl_norm_lane_nibble_fused_logits_cuda(
    torch::Tensor q_projected,
    torch::Tensor packed_meta,
    torch::Tensor radii,
    torch::Tensor l2_factor_lut_combo_major_fp16,
    torch::Tensor cos_l3,
    torch::Tensor sin_l3,
    torch::Tensor cos_l4,
    torch::Tensor sin_l4,
    torch::Tensor lane_nibble_qjl_signs,
    torch::Tensor qjl_norms
) {
    validate_meta(packed_meta);
    const int64_t H = packed_meta.size(1);
    const int64_t T = packed_meta.size(2);
    validate_radii(radii, H, T);
    validate_l2_lut_combo_major_fp16(l2_factor_lut_combo_major_fp16, H);
    validate_trig(cos_l3, sin_l3, cos_l4, sin_l4);
    validate_fused_extras(
        q_projected,
        lane_nibble_qjl_signs,
        qjl_norms,
        H,
        T
    );
    validate_grid(H, T);

    auto out = allocate_out(packed_meta, H, T);
    const dim3 block(THREADS);
    const dim3 grid(
        static_cast<unsigned int>((T + 7) / 8),
        static_cast<unsigned int>(H),
        1
    );

    turboquant_polar_tree_l2_combined_lut_fp16_combo_major_layout_early_radii_early_qjl_norm_lane_nibble_fused_logits_decode_b1q1_warp8_meta64_kernel<<<grid, block>>>(
        q_projected.contiguous().data_ptr<float>(),
        packed_meta.contiguous().data_ptr<uint8_t>(),
        reinterpret_cast<const half*>(radii.contiguous().data_ptr<at::Half>()),
        reinterpret_cast<const half*>(l2_factor_lut_combo_major_fp16.contiguous().data_ptr<at::Half>()),
        cos_l3.contiguous().data_ptr<float>(),
        sin_l3.contiguous().data_ptr<float>(),
        cos_l4.contiguous().data_ptr<float>(),
        sin_l4.contiguous().data_ptr<float>(),
        lane_nibble_qjl_signs.contiguous().data_ptr<uint8_t>(),
        reinterpret_cast<const half*>(qjl_norms.contiguous().data_ptr<at::Half>()),
        out.data_ptr<float>(),
        static_cast<int>(H),
        static_cast<int>(T)
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}


torch::Tensor turboquant_polar_tree_l2_combined_lut_fp16_vector_radii_lane_nibble_fused_logits_cuda(
    torch::Tensor q_projected,
    torch::Tensor packed_meta,
    torch::Tensor radii,
    torch::Tensor l2_factor_lut_fp16,
    torch::Tensor cos_l3,
    torch::Tensor sin_l3,
    torch::Tensor cos_l4,
    torch::Tensor sin_l4,
    torch::Tensor lane_nibble_qjl_signs,
    torch::Tensor qjl_norms
) {
    validate_meta(packed_meta);
    const int64_t H = packed_meta.size(1);
    const int64_t T = packed_meta.size(2);
    validate_radii(radii, H, T);
    validate_l2_lut_fp16(l2_factor_lut_fp16, H);
    validate_trig(cos_l3, sin_l3, cos_l4, sin_l4);
    validate_fused_extras(q_projected, lane_nibble_qjl_signs, qjl_norms, H, T);
    validate_grid(H, T);

    auto out = allocate_out(packed_meta, H, T);
    const dim3 block(THREADS);
    const dim3 grid(static_cast<unsigned int>((T + 7) / 8), static_cast<unsigned int>(H), 1);

    turboquant_polar_tree_l2_combined_lut_fp16_vector_radii_lane_nibble_fused_logits_decode_b1q1_warp8_meta64_kernel<<<grid, block>>>(
        q_projected.contiguous().data_ptr<float>(),
        packed_meta.contiguous().data_ptr<uint8_t>(),
        reinterpret_cast<const half*>(radii.contiguous().data_ptr<at::Half>()),
        reinterpret_cast<const half*>(l2_factor_lut_fp16.contiguous().data_ptr<at::Half>()),
        cos_l3.contiguous().data_ptr<float>(),
        sin_l3.contiguous().data_ptr<float>(),
        cos_l4.contiguous().data_ptr<float>(),
        sin_l4.contiguous().data_ptr<float>(),
        lane_nibble_qjl_signs.contiguous().data_ptr<uint8_t>(),
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
        "turboquant_polar_tree_l2_combined_lut_fp16_early_radii_polar_only_cuda",
        &turboquant_polar_tree_l2_combined_lut_fp16_early_radii_polar_only_cuda,
        "Polar-only FP16 L2 LUT with radii load hoisted early"
    );
    m.def(
        "turboquant_polar_tree_l2_combined_lut_fp16_vector_radii_polar_only_cuda",
        &turboquant_polar_tree_l2_combined_lut_fp16_vector_radii_polar_only_cuda,
        "Polar-only FP16 L2 LUT with vectorized warp radii loads"
    );
    m.def(
        "turboquant_polar_tree_l2_combined_lut_fp16_early_radii_lane_nibble_fused_logits_cuda",
        &turboquant_polar_tree_l2_combined_lut_fp16_early_radii_lane_nibble_fused_logits_cuda,
        "Full fused FP16 L2 LUT + lane-nibble QJL with radii load hoisted early"
    );
    m.def(
        "turboquant_polar_tree_l2_combined_lut_fp16_early_radii_early_qjl_norm_lane_nibble_fused_logits_cuda",
        &turboquant_polar_tree_l2_combined_lut_fp16_early_radii_early_qjl_norm_lane_nibble_fused_logits_cuda,
        "Full fused FP16 L2 LUT + lane-nibble QJL with early radii and early qjl norm load"
    );
    m.def(
        "turboquant_polar_tree_l2_combined_lut_fp16_combo_major_layout_early_radii_early_qjl_norm_lane_nibble_fused_logits_cuda",
        &turboquant_polar_tree_l2_combined_lut_fp16_combo_major_layout_early_radii_early_qjl_norm_lane_nibble_fused_logits_cuda,
        "Full fused FP16 L2 LUT with combo-major LUT layout + early radii + early qjl norm"
    );
    m.def(
        "turboquant_polar_tree_l2_combined_lut_fp16_vector_radii_lane_nibble_fused_logits_cuda",
        &turboquant_polar_tree_l2_combined_lut_fp16_vector_radii_lane_nibble_fused_logits_cuda,
        "Full fused FP16 L2 LUT + lane-nibble QJL with vectorized warp radii loads"
    );
}
