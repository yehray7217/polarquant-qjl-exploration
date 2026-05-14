#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAException.h>

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace {

constexpr int D = 128;
constexpr int M = 128;
constexpr int PACKED_QJL_M = M / 8;   // 16 bytes
constexpr int META_BYTES = 64;
constexpr int THREADS = 256;   // 8 warps / CTA

// Current calibrated correction scale for:
//   Polar Stage-1 (4,2,2,2)
//   QJL M=128
constexpr float QJL_CORRECTION_SCALE = 0.375f;


// ============================================================
// Helpers
// ============================================================

__device__ __forceinline__ uint8_t load_4bit_code(
    const uint8_t* packed,
    int code_idx
) {
    const int byte_idx = code_idx >> 1;
    const int shift = (code_idx & 1) * 4;

    return static_cast<uint8_t>(
        (packed[byte_idx] >> shift) & 0x0F
    );
}


__device__ __forceinline__ uint8_t load_2bit_code(
    const uint8_t* packed,
    int code_idx
) {
    const int byte_idx = code_idx >> 2;
    const int shift = (code_idx & 3) * 2;

    return static_cast<uint8_t>(
        (packed[byte_idx] >> shift) & 0x03
    );
}


__device__ __forceinline__ uint8_t byte_from_u32(
    uint32_t word,
    int byte_idx
) {
    return static_cast<uint8_t>(
        (word >> (byte_idx * 8)) & 0xFFu
    );
}


__device__ __forceinline__ float warp_reduce_sum(
    float val
) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(
            0xffffffff,
            val,
            offset
        );
    }

    return val;
}


__device__ __forceinline__ float signed_q_from_packed_qjl_bit(
    const uint8_t* packed_signs,
    int sketch_idx,
    float q_val
) {
    const int byte_idx = sketch_idx >> 3;
    const int bit_idx = sketch_idx & 7;

    const uint8_t byte_val =
        packed_signs[byte_idx];

    const uint8_t bit =
        (byte_val >> bit_idx) & 0x01;

    return bit ? q_val : -q_val;
}


// ============================================================
// Generic single-output helper
// ============================================================

__device__ __forceinline__ float compute_fused_logit_one_key_generic(
    const float* __restrict__ q,
    const float* __restrict__ q_projected,

    const uint8_t* __restrict__ packed_l1,
    const uint8_t* __restrict__ packed_l2,
    const uint8_t* __restrict__ packed_l3,
    const uint8_t* __restrict__ packed_l4,
    const half* __restrict__ radii,

    const float* __restrict__ cos_l1,
    const float* __restrict__ sin_l1,
    const float* __restrict__ cos_l2,
    const float* __restrict__ sin_l2,
    const float* __restrict__ cos_l3,
    const float* __restrict__ sin_l3,
    const float* __restrict__ cos_l4,
    const float* __restrict__ sin_l4,

    const uint8_t* __restrict__ packed_qjl_signs,
    const half* __restrict__ qjl_norms,

    int tid,
    int64_t q_offset,
    int64_t qproj_offset,
    int64_t key_linear
) {
    const uint8_t* l1_ptr =
        packed_l1 + key_linear * 32;

    const uint8_t* l2_ptr =
        packed_l2 + key_linear * 8;

    const uint8_t* l3_ptr =
        packed_l3 + key_linear * 4;

    const uint8_t* l4_ptr =
        packed_l4 + key_linear * 2;

    const half* radii_ptr =
        radii + key_linear * 8;

    const uint8_t* qjl_sign_ptr =
        packed_qjl_signs + key_linear * PACKED_QJL_M;

    // ------------------------------------------------------------
    // Polar Stage-1 contribution
    // ------------------------------------------------------------

    const int d =
        tid;

    const int block16 =
        d >> 4;

    const int local16 =
        d & 15;

    const int l4_idx =
        block16;

    const int l3_idx =
        block16 * 2
        + (local16 >> 3);

    const int l2_idx =
        block16 * 4
        + (local16 >> 2);

    const int l1_idx =
        block16 * 8
        + (local16 >> 1);

    const uint8_t c1 =
        load_4bit_code(
            l1_ptr,
            l1_idx
        );

    const uint8_t c2 =
        load_2bit_code(
            l2_ptr,
            l2_idx
        );

    const uint8_t c3 =
        load_2bit_code(
            l3_ptr,
            l3_idx
        );

    const uint8_t c4 =
        load_2bit_code(
            l4_ptr,
            l4_idx
        );

    const bool use_sin_l1 =
        (local16 & 1) != 0;

    const bool use_sin_l2 =
        (local16 & 2) != 0;

    const bool use_sin_l3 =
        (local16 & 4) != 0;

    const bool use_sin_l4 =
        (local16 & 8) != 0;

    const float f1 =
        use_sin_l1
        ? sin_l1[c1]
        : cos_l1[c1];

    const float f2 =
        use_sin_l2
        ? sin_l2[c2]
        : cos_l2[c2];

    const float f3 =
        use_sin_l3
        ? sin_l3[c3]
        : cos_l3[c3];

    const float f4 =
        use_sin_l4
        ? sin_l4[c4]
        : cos_l4[c4];

    const float radius =
        __half2float(
            radii_ptr[block16]
        );

    const float reconstructed_k =
        radius * f4 * f3 * f2 * f1;

    const float polar_partial =
        q[q_offset + d] * reconstructed_k;

    // ------------------------------------------------------------
    // QJL residual contribution
    // ------------------------------------------------------------

    const int sketch_idx =
        tid;

    const float qproj_val =
        q_projected[qproj_offset + sketch_idx];

    const float qjl_partial =
        signed_q_from_packed_qjl_bit(
            qjl_sign_ptr,
            sketch_idx,
            qproj_val
        );

    // ------------------------------------------------------------
    // Block reduction
    // ------------------------------------------------------------

    float polar_sum =
        warp_reduce_sum(
            polar_partial
        );

    float qjl_sum =
        warp_reduce_sum(
            qjl_partial
        );

    __shared__ float polar_warp_sums[4];
    __shared__ float qjl_warp_sums[4];

    const int lane =
        tid & 31;

    const int warp =
        tid >> 5;

    if (lane == 0) {
        polar_warp_sums[warp] =
            polar_sum;

        qjl_warp_sums[warp] =
            qjl_sum;
    }

    __syncthreads();

    if (warp == 0) {
        float polar_block_sum =
            (lane < 4)
            ? polar_warp_sums[lane]
            : 0.0f;

        float qjl_block_sum =
            (lane < 4)
            ? qjl_warp_sums[lane]
            : 0.0f;

        polar_block_sum =
            warp_reduce_sum(
                polar_block_sum
            );

        qjl_block_sum =
            warp_reduce_sum(
                qjl_block_sum
            );

        if (lane == 0) {
            const float residual_norm =
                __half2float(
                    qjl_norms[key_linear]
                );

            const float qjl_scale =
                QJL_CORRECTION_SCALE
                * sqrtf(3.14159265358979323846f / 2.0f)
                / sqrtf(static_cast<float>(M));

            return (
                polar_block_sum
                + qjl_scale
                  * residual_norm
                  * qjl_block_sum
            );
        }
    }

    return 0.0f;
}


// ============================================================
// Generic fallback kernel
// ============================================================

__global__ void turboquant_fused_logits_generic_kernel(
    const float* __restrict__ q,
    const float* __restrict__ q_projected,

    const uint8_t* __restrict__ packed_l1,
    const uint8_t* __restrict__ packed_l2,
    const uint8_t* __restrict__ packed_l3,
    const uint8_t* __restrict__ packed_l4,
    const half* __restrict__ radii,

    const float* __restrict__ cos_l1,
    const float* __restrict__ sin_l1,
    const float* __restrict__ cos_l2,
    const float* __restrict__ sin_l2,
    const float* __restrict__ cos_l3,
    const float* __restrict__ sin_l3,
    const float* __restrict__ cos_l4,
    const float* __restrict__ sin_l4,

    const uint8_t* __restrict__ packed_qjl_signs,
    const half* __restrict__ qjl_norms,

    float* __restrict__ out,

    int B,
    int H,
    int Q,
    int T
) {
    const int64_t output_linear =
        static_cast<int64_t>(blockIdx.x)
        + static_cast<int64_t>(blockIdx.y)
          * static_cast<int64_t>(gridDim.x);

    const int tid =
        threadIdx.x;

    if (tid >= THREADS) {
        return;
    }

    const int64_t total_outputs =
        static_cast<int64_t>(B)
        * static_cast<int64_t>(H)
        * static_cast<int64_t>(Q)
        * static_cast<int64_t>(T);

    if (output_linear >= total_outputs) {
        return;
    }

    int64_t tmp =
        output_linear;

    const int t_idx =
        static_cast<int>(tmp % T);
    tmp /= T;

    const int q_idx =
        static_cast<int>(tmp % Q);
    tmp /= Q;

    const int h_idx =
        static_cast<int>(tmp % H);
    tmp /= H;

    const int b_idx =
        static_cast<int>(tmp);

    const int64_t q_offset =
        (
            (
                static_cast<int64_t>(b_idx) * H
                + h_idx
            ) * Q
            + q_idx
        ) * D;

    const int64_t qproj_offset =
        (
            (
                static_cast<int64_t>(b_idx) * H
                + h_idx
            ) * Q
            + q_idx
        ) * M;

    const int64_t key_linear =
        (
            static_cast<int64_t>(b_idx) * H
            + h_idx
        ) * T
        + t_idx;

    const float final_logit =
        compute_fused_logit_one_key_generic(
            q,
            q_projected,

            packed_l1,
            packed_l2,
            packed_l3,
            packed_l4,
            radii,

            cos_l1,
            sin_l1,
            cos_l2,
            sin_l2,
            cos_l3,
            sin_l3,
            cos_l4,
            sin_l4,

            packed_qjl_signs,
            qjl_norms,

            tid,
            q_offset,
            qproj_offset,
            key_linear
        );

    if (tid == 0) {
        out[output_linear] =
            final_logit;
    }
}


// ============================================================
// Decode-specialized fast path:
//   B = 1, Q = 1
//
//   1 warp = 1 key logit
//   1 CTA  = 4 warps = 4 key logits
//
//   Reads compressed K metadata from packed_meta:
//     [B,H,T,64] uint8
//
// Packed-meta layout:
//   0  ~ 31 : L1 packed codes       32 B
//   32 ~ 39 : L2 packed codes        8 B
//   40 ~ 43 : L3 packed codes        4 B
//   44 ~ 45 : L4 packed codes        2 B
//   46 ~ 47 : padding                2 B
//   48 ~ 63 : QJL packed signs      16 B
// ============================================================

__global__ void turboquant_fused_logits_decode_b1q1_warp4_meta64_kernel(
    const float* __restrict__ q,
    const float* __restrict__ q_projected,

    const uint8_t* __restrict__ packed_meta,
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
    const int tid =
        threadIdx.x;

    const int warp_id =
        tid >> 5;          // 0..3

    const int lane =
        tid & 31;          // 0..31

    const int h_idx =
        static_cast<int>(blockIdx.y);

    const int t_idx =
        static_cast<int>(blockIdx.x) * 4
        + warp_id;

    // ------------------------------------------------------------
    // Query-side staging:
    // same q / q_projected reused by 4 warps = 4 key logits
    // ------------------------------------------------------------

    __shared__ float sh_q[D];
    __shared__ float sh_qproj[M];

    const int64_t q_offset =
        static_cast<int64_t>(h_idx) * D;

    const int64_t qproj_offset =
        static_cast<int64_t>(h_idx) * M;

    if (tid < D) {
        sh_q[tid] =
            q[q_offset + tid];
    }

    if (tid < M) {
        sh_qproj[tid] =
            q_projected[qproj_offset + tid];
    }

    __syncthreads();

    if (h_idx >= H || t_idx >= T) {
        return;
    }

    const int64_t key_linear =
        static_cast<int64_t>(h_idx) * T
        + t_idx;

    const uint8_t* meta_ptr =
        packed_meta + key_linear * META_BYTES;

    const half* radii_ptr =
        radii + key_linear * 8;

    // ------------------------------------------------------------
    // Read one 64-byte metadata blob as 16 uint32 words.
    // Lanes 0..15 perform the loads, all lanes reuse via shuffle.
    // ------------------------------------------------------------

    uint32_t meta_word_local = 0;

    if (lane < 16) {
        meta_word_local =
            reinterpret_cast<const uint32_t*>(meta_ptr)[lane];
    }

    float polar_acc =
        0.0f;

    float qjl_acc =
        0.0f;

    // ============================================================
    // Each lane handles 4 original coordinates:
    //   d = lane + 32*k, k=0..3
    // ============================================================

    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int d =
            lane + 32 * k;

        // --------------------------------------------------------
        // Polar Stage-1 contribution
        // --------------------------------------------------------

        const int block16 =
            d >> 4;          // 0..7

        const int local16 =
            d & 15;          // 0..15

        const int l4_idx =
            block16;

        const int l3_idx =
            block16 * 2
            + (local16 >> 3);

        const int l2_idx =
            block16 * 4
            + (local16 >> 2);

        const int l1_idx =
            block16 * 8
            + (local16 >> 1);

        // -------------------------
        // c1 from meta words 0..7
        // -------------------------

        const int l1_byte_idx =
            l1_idx >> 1;

        const int l1_word_owner =
            l1_byte_idx >> 2;

        const int l1_byte_in_word =
            l1_byte_idx & 3;

        const uint32_t l1_word =
            __shfl_sync(
                0xffffffff,
                meta_word_local,
                l1_word_owner
            );

        const uint8_t l1_byte =
            byte_from_u32(
                l1_word,
                l1_byte_in_word
            );

        const uint8_t c1 =
            static_cast<uint8_t>(
                (l1_byte >> ((l1_idx & 1) * 4)) & 0x0F
            );

        // -------------------------
        // c2 from meta words 8..9
        // -------------------------

        const int l2_byte_idx =
            l2_idx >> 2;

        const int l2_word_owner =
            8 + (l2_byte_idx >> 2);

        const int l2_byte_in_word =
            l2_byte_idx & 3;

        const uint32_t l2_word =
            __shfl_sync(
                0xffffffff,
                meta_word_local,
                l2_word_owner
            );

        const uint8_t l2_byte =
            byte_from_u32(
                l2_word,
                l2_byte_in_word
            );

        const uint8_t c2 =
            static_cast<uint8_t>(
                (l2_byte >> ((l2_idx & 3) * 2)) & 0x03
            );

        // -------------------------
        // c3 from meta word 10
        // -------------------------

        const int l3_byte_idx =
            l3_idx >> 2;

        const uint32_t l3_word =
            __shfl_sync(
                0xffffffff,
                meta_word_local,
                10
            );

        const uint8_t l3_byte =
            byte_from_u32(
                l3_word,
                l3_byte_idx
            );

        const uint8_t c3 =
            static_cast<uint8_t>(
                (l3_byte >> ((l3_idx & 3) * 2)) & 0x03
            );

        // -------------------------
        // c4 from meta word 11
        // Layout:
        //   bytes 44,45 = L4
        //   bytes 46,47 = padding
        // -------------------------

        const int l4_byte_idx =
            l4_idx >> 2;

        const uint32_t l4_word =
            __shfl_sync(
                0xffffffff,
                meta_word_local,
                11
            );

        const uint8_t l4_byte =
            byte_from_u32(
                l4_word,
                l4_byte_idx
            );

        const uint8_t c4 =
            static_cast<uint8_t>(
                (l4_byte >> ((l4_idx & 3) * 2)) & 0x03
            );

        const bool use_sin_l1 =
            (local16 & 1) != 0;

        const bool use_sin_l2 =
            (local16 & 2) != 0;

        const bool use_sin_l3 =
            (local16 & 4) != 0;

        const bool use_sin_l4 =
            (local16 & 8) != 0;

        const float f1 =
            use_sin_l1
            ? sin_l1[c1]
            : cos_l1[c1];

        const float f2 =
            use_sin_l2
            ? sin_l2[c2]
            : cos_l2[c2];

        const float f3 =
            use_sin_l3
            ? sin_l3[c3]
            : cos_l3[c3];

        const float f4 =
            use_sin_l4
            ? sin_l4[c4]
            : cos_l4[c4];

        const float radius =
            __half2float(
                radii_ptr[block16]
            );

        const float reconstructed_k =
            radius * f4 * f3 * f2 * f1;

        polar_acc +=
            sh_q[d] * reconstructed_k;

        // --------------------------------------------------------
        // QJL residual contribution
        // QJL sign words live at meta words 12..15
        // --------------------------------------------------------

        const int sketch_idx =
            d;

        const float qproj_val =
            sh_qproj[sketch_idx];

        const int qjl_byte_idx =
            sketch_idx >> 3;

        const int qjl_word_owner =
            12 + (qjl_byte_idx >> 2);

        const int qjl_byte_in_word =
            qjl_byte_idx & 3;

        const uint32_t qjl_word =
            __shfl_sync(
                0xffffffff,
                meta_word_local,
                qjl_word_owner
            );

        const uint8_t qjl_byte =
            byte_from_u32(
                qjl_word,
                qjl_byte_in_word
            );

        const uint8_t qjl_bit =
            static_cast<uint8_t>(
                (qjl_byte >> (sketch_idx & 7)) & 0x01
            );

        qjl_acc +=
            qjl_bit
            ? qproj_val
            : -qproj_val;
    }

    // ============================================================
    // Warp-only reduction:
    // one warp owns one key, so no cross-warp reduction needed
    // ============================================================

    float polar_sum =
        warp_reduce_sum(
            polar_acc
        );

    float qjl_sum =
        warp_reduce_sum(
            qjl_acc
        );

    if (lane == 0) {
        const float residual_norm =
            __half2float(
                qjl_norms[key_linear]
            );

        const float qjl_scale =
            QJL_CORRECTION_SCALE
            * sqrtf(3.14159265358979323846f / 2.0f)
            / sqrtf(static_cast<float>(M));

        const float final_logit =
            polar_sum
            + qjl_scale
              * residual_norm
              * qjl_sum;

        const int64_t out_linear =
            static_cast<int64_t>(h_idx) * T
            + t_idx;

        out[out_linear] =
            final_logit;
    }
}

// ============================================================
// Decode-specialized fast path:
//   B = 1, Q = 1
//
//   1 warp = 1 key logit
//   1 CTA  = 8 warps = 8 key logits
//
//   Reads compressed K metadata from packed_meta:
//     [B,H,T,64] uint8
// ============================================================

__global__ void turboquant_fused_logits_decode_b1q1_warp8_meta64_kernel(
    const float* __restrict__ q,
    const float* __restrict__ q_projected,

    const uint8_t* __restrict__ packed_meta,
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
    const int tid =
        threadIdx.x;

    const int warp_id =
        tid >> 5;          // 0..7

    const int lane =
        tid & 31;          // 0..31

    const int h_idx =
        static_cast<int>(blockIdx.y);

    const int t_idx =
        static_cast<int>(blockIdx.x) * 8
        + warp_id;

    // ------------------------------------------------------------
    // Query-side staging:
    // same q / q_projected reused by 8 warps = 8 key logits
    // ------------------------------------------------------------

    __shared__ float sh_q[D];
    __shared__ float sh_qproj[M];

    const int64_t q_offset =
        static_cast<int64_t>(h_idx) * D;

    const int64_t qproj_offset =
        static_cast<int64_t>(h_idx) * M;

    if (tid < D) {
        sh_q[tid] =
            q[q_offset + tid];
    }

    if (tid < M) {
        sh_qproj[tid] =
            q_projected[qproj_offset + tid];
    }

    __syncthreads();

    if (h_idx >= H || t_idx >= T) {
        return;
    }

    const int64_t key_linear =
        static_cast<int64_t>(h_idx) * T
        + t_idx;

    const uint8_t* meta_ptr =
        packed_meta + key_linear * META_BYTES;

    const half* radii_ptr =
        radii + key_linear * 8;

    // ------------------------------------------------------------
    // Read one 64-byte metadata blob as 16 uint32 words.
    // Lanes 0..15 perform the loads, all lanes reuse via shuffle.
    // ------------------------------------------------------------

    uint32_t meta_word_local = 0;

    if (lane < 16) {
        meta_word_local =
            reinterpret_cast<const uint32_t*>(meta_ptr)[lane];
    }

    float polar_acc =
        0.0f;

    float qjl_acc =
        0.0f;

    // ============================================================
    // Each lane handles 4 original coordinates:
    //   d = lane + 32*k, k=0..3
    // ============================================================

    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int d =
            lane + 32 * k;

        // --------------------------------------------------------
        // Polar Stage-1 contribution
        // --------------------------------------------------------

        const int block16 =
            d >> 4;

        const int local16 =
            d & 15;

        const int l4_idx =
            block16;

        const int l3_idx =
            block16 * 2
            + (local16 >> 3);

        const int l2_idx =
            block16 * 4
            + (local16 >> 2);

        const int l1_idx =
            block16 * 8
            + (local16 >> 1);

        // -------------------------
        // c1 from meta words 0..7
        // -------------------------

        const int l1_byte_idx =
            l1_idx >> 1;

        const int l1_word_owner =
            l1_byte_idx >> 2;

        const int l1_byte_in_word =
            l1_byte_idx & 3;

        const uint32_t l1_word =
            __shfl_sync(
                0xffffffff,
                meta_word_local,
                l1_word_owner
            );

        const uint8_t l1_byte =
            byte_from_u32(
                l1_word,
                l1_byte_in_word
            );

        const uint8_t c1 =
            static_cast<uint8_t>(
                (l1_byte >> ((l1_idx & 1) * 4)) & 0x0F
            );

        // -------------------------
        // c2 from meta words 8..9
        // -------------------------

        const int l2_byte_idx =
            l2_idx >> 2;

        const int l2_word_owner =
            8 + (l2_byte_idx >> 2);

        const int l2_byte_in_word =
            l2_byte_idx & 3;

        const uint32_t l2_word =
            __shfl_sync(
                0xffffffff,
                meta_word_local,
                l2_word_owner
            );

        const uint8_t l2_byte =
            byte_from_u32(
                l2_word,
                l2_byte_in_word
            );

        const uint8_t c2 =
            static_cast<uint8_t>(
                (l2_byte >> ((l2_idx & 3) * 2)) & 0x03
            );

        // -------------------------
        // c3 from meta word 10
        // -------------------------

        const int l3_byte_idx =
            l3_idx >> 2;

        const uint32_t l3_word =
            __shfl_sync(
                0xffffffff,
                meta_word_local,
                10
            );

        const uint8_t l3_byte =
            byte_from_u32(
                l3_word,
                l3_byte_idx
            );

        const uint8_t c3 =
            static_cast<uint8_t>(
                (l3_byte >> ((l3_idx & 3) * 2)) & 0x03
            );

        // -------------------------
        // c4 from meta word 11
        // -------------------------

        const int l4_byte_idx =
            l4_idx >> 2;

        const uint32_t l4_word =
            __shfl_sync(
                0xffffffff,
                meta_word_local,
                11
            );

        const uint8_t l4_byte =
            byte_from_u32(
                l4_word,
                l4_byte_idx
            );

        const uint8_t c4 =
            static_cast<uint8_t>(
                (l4_byte >> ((l4_idx & 3) * 2)) & 0x03
            );

        const bool use_sin_l1 =
            (local16 & 1) != 0;

        const bool use_sin_l2 =
            (local16 & 2) != 0;

        const bool use_sin_l3 =
            (local16 & 4) != 0;

        const bool use_sin_l4 =
            (local16 & 8) != 0;

        const float f1 =
            use_sin_l1
            ? sin_l1[c1]
            : cos_l1[c1];

        const float f2 =
            use_sin_l2
            ? sin_l2[c2]
            : cos_l2[c2];

        const float f3 =
            use_sin_l3
            ? sin_l3[c3]
            : cos_l3[c3];

        const float f4 =
            use_sin_l4
            ? sin_l4[c4]
            : cos_l4[c4];

        const float radius =
            __half2float(
                radii_ptr[block16]
            );

        const float reconstructed_k =
            radius * f4 * f3 * f2 * f1;

        polar_acc +=
            sh_q[d] * reconstructed_k;

        // --------------------------------------------------------
        // QJL residual contribution
        // QJL sign words live at meta words 12..15
        // --------------------------------------------------------

        const int sketch_idx =
            d;

        const float qproj_val =
            sh_qproj[sketch_idx];

        const int qjl_byte_idx =
            sketch_idx >> 3;

        const int qjl_word_owner =
            12 + (qjl_byte_idx >> 2);

        const int qjl_byte_in_word =
            qjl_byte_idx & 3;

        const uint32_t qjl_word =
            __shfl_sync(
                0xffffffff,
                meta_word_local,
                qjl_word_owner
            );

        const uint8_t qjl_byte =
            byte_from_u32(
                qjl_word,
                qjl_byte_in_word
            );

        const uint8_t qjl_bit =
            static_cast<uint8_t>(
                (qjl_byte >> (sketch_idx & 7)) & 0x01
            );

        qjl_acc +=
            qjl_bit
            ? qproj_val
            : -qproj_val;
    }

    // ============================================================
    // Warp-only reduction:
    // one warp owns one key
    // ============================================================

    float polar_sum =
        warp_reduce_sum(
            polar_acc
        );

    float qjl_sum =
        warp_reduce_sum(
            qjl_acc
        );

    if (lane == 0) {
        const float residual_norm =
            __half2float(
                qjl_norms[key_linear]
            );

        const float qjl_scale =
            QJL_CORRECTION_SCALE
            * sqrtf(3.14159265358979323846f / 2.0f)
            / sqrtf(static_cast<float>(M));

        const float final_logit =
            polar_sum
            + qjl_scale
              * residual_norm
              * qjl_sum;

        const int64_t out_linear =
            static_cast<int64_t>(h_idx) * T
            + t_idx;

        out[out_linear] =
            final_logit;
    }
}

} // namespace


// ============================================================
// Host entry
// ============================================================

torch::Tensor turboquant_fused_logits_cuda(
    torch::Tensor q,
    torch::Tensor q_projected,

    torch::Tensor packed_meta,

    torch::Tensor packed_l1,
    torch::Tensor packed_l2,
    torch::Tensor packed_l3,
    torch::Tensor packed_l4,
    torch::Tensor radii,

    torch::Tensor cos_l1,
    torch::Tensor sin_l1,
    torch::Tensor cos_l2,
    torch::Tensor sin_l2,
    torch::Tensor cos_l3,
    torch::Tensor sin_l3,
    torch::Tensor cos_l4,
    torch::Tensor sin_l4,

    torch::Tensor packed_qjl_signs,
    torch::Tensor qjl_norms
) {
    TORCH_CHECK(q.is_cuda(), "q must be CUDA");
    TORCH_CHECK(q_projected.is_cuda(), "q_projected must be CUDA");

    TORCH_CHECK(packed_l1.is_cuda(), "packed_l1 must be CUDA");
    TORCH_CHECK(packed_l2.is_cuda(), "packed_l2 must be CUDA");
    TORCH_CHECK(packed_l3.is_cuda(), "packed_l3 must be CUDA");
    TORCH_CHECK(packed_l4.is_cuda(), "packed_l4 must be CUDA");
    TORCH_CHECK(radii.is_cuda(), "radii must be CUDA");

    TORCH_CHECK(cos_l1.is_cuda() && sin_l1.is_cuda(), "level1 trig tables must be CUDA");
    TORCH_CHECK(cos_l2.is_cuda() && sin_l2.is_cuda(), "level2 trig tables must be CUDA");
    TORCH_CHECK(cos_l3.is_cuda() && sin_l3.is_cuda(), "level3 trig tables must be CUDA");
    TORCH_CHECK(cos_l4.is_cuda() && sin_l4.is_cuda(), "level4 trig tables must be CUDA");

    TORCH_CHECK(packed_qjl_signs.is_cuda(), "packed_qjl_signs must be CUDA");
    TORCH_CHECK(qjl_norms.is_cuda(), "qjl_norms must be CUDA");

    TORCH_CHECK(q.dtype() == torch::kFloat32, "q must be float32");
    TORCH_CHECK(q_projected.dtype() == torch::kFloat32, "q_projected must be float32");

    TORCH_CHECK(packed_l1.dtype() == torch::kUInt8, "packed_l1 must be uint8");
    TORCH_CHECK(packed_l2.dtype() == torch::kUInt8, "packed_l2 must be uint8");
    TORCH_CHECK(packed_l3.dtype() == torch::kUInt8, "packed_l3 must be uint8");
    TORCH_CHECK(packed_l4.dtype() == torch::kUInt8, "packed_l4 must be uint8");

    TORCH_CHECK(radii.dtype() == torch::kFloat16, "radii must be float16");

    TORCH_CHECK(cos_l1.dtype() == torch::kFloat32, "cos_l1 must be float32");
    TORCH_CHECK(sin_l1.dtype() == torch::kFloat32, "sin_l1 must be float32");
    TORCH_CHECK(cos_l2.dtype() == torch::kFloat32, "cos_l2 must be float32");
    TORCH_CHECK(sin_l2.dtype() == torch::kFloat32, "sin_l2 must be float32");
    TORCH_CHECK(cos_l3.dtype() == torch::kFloat32, "cos_l3 must be float32");
    TORCH_CHECK(sin_l3.dtype() == torch::kFloat32, "sin_l3 must be float32");
    TORCH_CHECK(cos_l4.dtype() == torch::kFloat32, "cos_l4 must be float32");
    TORCH_CHECK(sin_l4.dtype() == torch::kFloat32, "sin_l4 must be float32");

    TORCH_CHECK(packed_qjl_signs.dtype() == torch::kUInt8, "packed_qjl_signs must be uint8");
    TORCH_CHECK(qjl_norms.dtype() == torch::kFloat16, "qjl_norms must be float16");

    TORCH_CHECK(q.dim() == 4, "q must be [B,H,Q,128]");
    TORCH_CHECK(q.size(3) == D, "q last dim must be 128");

    TORCH_CHECK(q_projected.dim() == 4, "q_projected must be [B,H,Q,128]");
    TORCH_CHECK(q_projected.size(3) == M, "q_projected last dim must be 128");

    TORCH_CHECK(packed_l1.dim() == 4, "packed_l1 must be [B,H,T,32]");
    TORCH_CHECK(packed_l2.dim() == 4, "packed_l2 must be [B,H,T,8]");
    TORCH_CHECK(packed_l3.dim() == 4, "packed_l3 must be [B,H,T,4]");
    TORCH_CHECK(packed_l4.dim() == 4, "packed_l4 must be [B,H,T,2]");
    TORCH_CHECK(radii.dim() == 4, "radii must be [B,H,T,8]");

    TORCH_CHECK(packed_l1.size(3) == 32, "packed_l1 last dim must be 32");
    TORCH_CHECK(packed_l2.size(3) == 8, "packed_l2 last dim must be 8");
    TORCH_CHECK(packed_l3.size(3) == 4, "packed_l3 last dim must be 4");
    TORCH_CHECK(packed_l4.size(3) == 2, "packed_l4 last dim must be 2");
    TORCH_CHECK(radii.size(3) == 8, "radii last dim must be 8");

    TORCH_CHECK(packed_qjl_signs.dim() == 4, "packed_qjl_signs must be [B,H,T,16]");
    TORCH_CHECK(packed_qjl_signs.size(3) == PACKED_QJL_M, "packed_qjl_signs last dim must be 16");

    TORCH_CHECK(qjl_norms.dim() == 3, "qjl_norms must be [B,H,T]");

    TORCH_CHECK(cos_l1.numel() == 16, "cos_l1 must have 16 entries");
    TORCH_CHECK(sin_l1.numel() == 16, "sin_l1 must have 16 entries");

    TORCH_CHECK(cos_l2.numel() == 4, "cos_l2 must have 4 entries");
    TORCH_CHECK(sin_l2.numel() == 4, "sin_l2 must have 4 entries");

    TORCH_CHECK(cos_l3.numel() == 4, "cos_l3 must have 4 entries");
    TORCH_CHECK(sin_l3.numel() == 4, "sin_l3 must have 4 entries");

    TORCH_CHECK(cos_l4.numel() == 4, "cos_l4 must have 4 entries");
    TORCH_CHECK(sin_l4.numel() == 4, "sin_l4 must have 4 entries");

    const int64_t B =
        q.size(0);

    const int64_t H =
        q.size(1);

    const int64_t Q =
        q.size(2);

    const int64_t T =
        packed_l1.size(2);

    TORCH_CHECK(
        q_projected.size(0) == B &&
        q_projected.size(1) == H &&
        q_projected.size(2) == Q,
        "q_projected B/H/Q mismatch"
    );

    TORCH_CHECK(
        packed_l1.size(0) == B &&
        packed_l1.size(1) == H,
        "packed_l1 B/H mismatch"
    );

    TORCH_CHECK(
        packed_l2.size(0) == B &&
        packed_l2.size(1) == H &&
        packed_l2.size(2) == T,
        "packed_l2 shape mismatch"
    );

    TORCH_CHECK(
        packed_l3.size(0) == B &&
        packed_l3.size(1) == H &&
        packed_l3.size(2) == T,
        "packed_l3 shape mismatch"
    );

    TORCH_CHECK(
        packed_l4.size(0) == B &&
        packed_l4.size(1) == H &&
        packed_l4.size(2) == T,
        "packed_l4 shape mismatch"
    );

    TORCH_CHECK(
        radii.size(0) == B &&
        radii.size(1) == H &&
        radii.size(2) == T,
        "radii shape mismatch"
    );

    TORCH_CHECK(
        packed_qjl_signs.size(0) == B &&
        packed_qjl_signs.size(1) == H &&
        packed_qjl_signs.size(2) == T,
        "packed_qjl_signs shape mismatch"
    );

    TORCH_CHECK(
        qjl_norms.size(0) == B &&
        qjl_norms.size(1) == H &&
        qjl_norms.size(2) == T,
        "qjl_norms shape mismatch"
    );

    auto out = torch::empty(
        {B, H, Q, T},
        torch::TensorOptions()
            .device(q.device())
            .dtype(torch::kFloat32)
    );

    const dim3 block(THREADS);

    // ============================================================
    // Fast path:
    //   B = 1, Q = 1, packed_meta available
    // ============================================================

    const bool has_packed_meta =
        packed_meta.defined()
        && packed_meta.numel() > 0;

    if (B == 1 && Q == 1 && has_packed_meta) {
        TORCH_CHECK(packed_meta.is_cuda(), "packed_meta must be CUDA");
        TORCH_CHECK(packed_meta.dtype() == torch::kUInt8, "packed_meta must be uint8");
        TORCH_CHECK(packed_meta.dim() == 4, "packed_meta must be [B,H,T,64]");

        TORCH_CHECK(
            packed_meta.size(0) == B &&
            packed_meta.size(1) == H &&
            packed_meta.size(2) == T &&
            packed_meta.size(3) == META_BYTES,
            "packed_meta shape mismatch"
        );

        const int64_t token_groups =
            (T + 7) / 8;

        TORCH_CHECK(
            token_groups <= 2147483647LL,
            "meta64 grid.x exceeds CUDA limit: ",
            token_groups
        );

        TORCH_CHECK(
            H <= 65535,
            "meta64 grid.y exceeds CUDA limit: ",
            H
        );

        const dim3 grid(
            static_cast<unsigned int>(token_groups),
            static_cast<unsigned int>(H),
            1
        );

        turboquant_fused_logits_decode_b1q1_warp8_meta64_kernel<<<grid, block>>>(
            q.contiguous().data_ptr<float>(),
            q_projected.contiguous().data_ptr<float>(),

            packed_meta.contiguous().data_ptr<uint8_t>(),
            reinterpret_cast<const half*>(
                radii.contiguous().data_ptr<at::Half>()
            ),

            cos_l1.contiguous().data_ptr<float>(),
            sin_l1.contiguous().data_ptr<float>(),
            cos_l2.contiguous().data_ptr<float>(),
            sin_l2.contiguous().data_ptr<float>(),
            cos_l3.contiguous().data_ptr<float>(),
            sin_l3.contiguous().data_ptr<float>(),
            cos_l4.contiguous().data_ptr<float>(),
            sin_l4.contiguous().data_ptr<float>(),

            reinterpret_cast<const half*>(
                qjl_norms.contiguous().data_ptr<at::Half>()
            ),

            out.data_ptr<float>(),

            static_cast<int>(H),
            static_cast<int>(T)
        );

        C10_CUDA_KERNEL_LAUNCH_CHECK();

        return out;
    }

    // ============================================================
    // Generic fallback
    // ============================================================

    const int64_t total_outputs =
        B * H * Q * T;

    constexpr int64_t GRID_X_CAP = 65535;

    const int64_t grid_x =
        std::min<int64_t>(
            total_outputs,
            GRID_X_CAP
        );

    const int64_t grid_y =
        (total_outputs + grid_x - 1)
        / grid_x;

    TORCH_CHECK(
        grid_y <= 65535,
        "generic turboquant fused grid_y exceeds CUDA limit: ",
        grid_y
    );

    const dim3 grid(
        static_cast<unsigned int>(grid_x),
        static_cast<unsigned int>(grid_y)
    );

    turboquant_fused_logits_generic_kernel<<<grid, block>>>(
        q.contiguous().data_ptr<float>(),
        q_projected.contiguous().data_ptr<float>(),

        packed_l1.contiguous().data_ptr<uint8_t>(),
        packed_l2.contiguous().data_ptr<uint8_t>(),
        packed_l3.contiguous().data_ptr<uint8_t>(),
        packed_l4.contiguous().data_ptr<uint8_t>(),

        reinterpret_cast<const half*>(
            radii.contiguous().data_ptr<at::Half>()
        ),

        cos_l1.contiguous().data_ptr<float>(),
        sin_l1.contiguous().data_ptr<float>(),
        cos_l2.contiguous().data_ptr<float>(),
        sin_l2.contiguous().data_ptr<float>(),
        cos_l3.contiguous().data_ptr<float>(),
        sin_l3.contiguous().data_ptr<float>(),
        cos_l4.contiguous().data_ptr<float>(),
        sin_l4.contiguous().data_ptr<float>(),

        packed_qjl_signs.contiguous().data_ptr<uint8_t>(),
        reinterpret_cast<const half*>(
            qjl_norms.contiguous().data_ptr<at::Half>()
        ),

        out.data_ptr<float>(),

        static_cast<int>(B),
        static_cast<int>(H),
        static_cast<int>(Q),
        static_cast<int>(T)
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "turboquant_fused_logits_cuda",
        &turboquant_fused_logits_cuda,
        "Fused TurboQuant final logits CUDA with meta64 fast path"
    );
}