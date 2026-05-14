#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAException.h>

#include <algorithm>
#include <cstdint>

namespace {

constexpr int D = 128;
constexpr int THREADS = 128;

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

__device__ __forceinline__ uint8_t load_3bit_code(
    const uint8_t* packed,
    int code_idx
) {
    // 8 codes -> 24 bits -> 3 bytes
    const int group_idx = code_idx >> 3;   // / 8
    const int local_idx = code_idx & 7;    // % 8

    const uint8_t* group_ptr =
        packed + group_idx * 3;

    const uint32_t packed24 =
        static_cast<uint32_t>(group_ptr[0]) |
        (static_cast<uint32_t>(group_ptr[1]) << 8) |
        (static_cast<uint32_t>(group_ptr[2]) << 16);

    const int shift = local_idx * 3;

    return static_cast<uint8_t>(
        (packed24 >> shift) & 0x07
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


__global__ void polar_stage1_score_kernel(
    const float* __restrict__ q,               // [B,H,Q,128]

    const uint8_t* __restrict__ packed_l1,     // [B,H,T,32]
    const uint8_t* __restrict__ packed_l2,     // [B,H,T,8]
    const uint8_t* __restrict__ packed_l3,     // [B,H,T,4]
    const uint8_t* __restrict__ packed_l4,     // [B,H,T,2]

    const half* __restrict__ radii,            // [B,H,T,8]

    const float* __restrict__ cos_l1,          // [16]
    const float* __restrict__ sin_l1,          // [16]

    const float* __restrict__ cos_l2,          // [4]
    const float* __restrict__ sin_l2,          // [4]

    const float* __restrict__ cos_l3,          // [4]
    const float* __restrict__ sin_l3,          // [4]

    const float* __restrict__ cos_l4,          // [4]
    const float* __restrict__ sin_l4,          // [4]

    float* __restrict__ out,                   // [B,H,Q,T]

    int B,
    int H,
    int Q,
    int T
) {
    const int64_t output_linear =
        static_cast<int64_t>(blockIdx.x)
        + static_cast<int64_t>(blockIdx.y)
          * static_cast<int64_t>(gridDim.x);

    const int d = threadIdx.x;

    if (d >= D) {
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

    // ------------------------------------------------------------
    // Decode output index:
    // out[b,h,q_idx,k_idx]
    // ------------------------------------------------------------

    int64_t tmp = output_linear;

    const int k_idx =
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

    // ------------------------------------------------------------
    // Offsets
    // ------------------------------------------------------------

    const int64_t q_offset =
        (
            (
                static_cast<int64_t>(b_idx) * H
                + h_idx
            ) * Q
            + q_idx
        ) * D;

    const int64_t k_linear =
        (
            static_cast<int64_t>(b_idx) * H
            + h_idx
        ) * T
        + k_idx;

    const uint8_t* l1_ptr =
        packed_l1 + k_linear * 32;

    const uint8_t* l2_ptr =
        packed_l2 + k_linear * 8;

    const uint8_t* l3_ptr =
        packed_l3 + k_linear * 4;

    const uint8_t* l4_ptr =
        packed_l4 + k_linear * 2;

    const half* radii_ptr =
        radii + k_linear * 8;

    // ------------------------------------------------------------
    // D=128, L=4 recursive polar topology
    //
    // Each 16-d block has:
    //   8 level-1 angles
    //   4 level-2 angles
    //   2 level-3 angles
    //   1 level-4 angle
    //   1 remaining radius
    // ------------------------------------------------------------

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

    // ------------------------------------------------------------
    // Path choices inside the recursive polar tree
    // ------------------------------------------------------------

    const bool use_sin_l1 =
        (local16 & 1) != 0;

    const bool use_sin_l2 =
        (local16 & 2) != 0;

    const bool use_sin_l3 =
        (local16 & 4) != 0;

    const bool use_sin_l4 =
        (local16 & 8) != 0;

    // ------------------------------------------------------------
    // Precomputed trig table lookup.
    //
    // Previous version called sinf/cosf in-kernel for every coordinate.
    // This version uses tiny precomputed tables:
    //   L1: 16 entries
    //   L2/L3/L4: 4 entries each
    // ------------------------------------------------------------

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

    const float reconstructed_x =
        radius * f4 * f3 * f2 * f1;

    const float product =
        q[q_offset + d] * reconstructed_x;

    // ------------------------------------------------------------
    // Block reduction over 128 D-coordinates.
    // 128 threads = 4 warps.
    // ------------------------------------------------------------

    float sum =
        warp_reduce_sum(product);

    __shared__ float warp_sums[4];

    const int lane =
        d & 31;

    const int warp =
        d >> 5;

    if (lane == 0) {
        warp_sums[warp] = sum;
    }

    __syncthreads();

    if (warp == 0) {
        float block_sum =
            (lane < 4)
            ? warp_sums[lane]
            : 0.0f;

        block_sum =
            warp_reduce_sum(block_sum);

        if (lane == 0) {
            out[output_linear] =
                block_sum;
        }
    }
}

} // namespace


torch::Tensor polar_stage1_score_cuda(
    torch::Tensor q,

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
    torch::Tensor sin_l4
) {
    TORCH_CHECK(
        q.is_cuda(),
        "q must be CUDA"
    );

    TORCH_CHECK(
        packed_l1.is_cuda(),
        "packed_l1 must be CUDA"
    );

    TORCH_CHECK(
        packed_l2.is_cuda(),
        "packed_l2 must be CUDA"
    );

    TORCH_CHECK(
        packed_l3.is_cuda(),
        "packed_l3 must be CUDA"
    );

    TORCH_CHECK(
        packed_l4.is_cuda(),
        "packed_l4 must be CUDA"
    );

    TORCH_CHECK(
        radii.is_cuda(),
        "radii must be CUDA"
    );

    TORCH_CHECK(
        cos_l1.is_cuda() &&
        sin_l1.is_cuda() &&
        cos_l2.is_cuda() &&
        sin_l2.is_cuda() &&
        cos_l3.is_cuda() &&
        sin_l3.is_cuda() &&
        cos_l4.is_cuda() &&
        sin_l4.is_cuda(),
        "all trig tables must be CUDA"
    );

    TORCH_CHECK(
        q.dtype() == torch::kFloat32,
        "q must be float32"
    );

    TORCH_CHECK(
        packed_l1.dtype() == torch::kUInt8,
        "packed_l1 must be uint8"
    );

    TORCH_CHECK(
        packed_l2.dtype() == torch::kUInt8,
        "packed_l2 must be uint8"
    );

    TORCH_CHECK(
        packed_l3.dtype() == torch::kUInt8,
        "packed_l3 must be uint8"
    );

    TORCH_CHECK(
        packed_l4.dtype() == torch::kUInt8,
        "packed_l4 must be uint8"
    );

    TORCH_CHECK(
        radii.dtype() == torch::kFloat16,
        "radii must be float16"
    );

    TORCH_CHECK(
        cos_l1.dtype() == torch::kFloat32,
        "cos_l1 must be float32"
    );

    TORCH_CHECK(
        sin_l1.dtype() == torch::kFloat32,
        "sin_l1 must be float32"
    );

    TORCH_CHECK(
        cos_l2.dtype() == torch::kFloat32,
        "cos_l2 must be float32"
    );

    TORCH_CHECK(
        sin_l2.dtype() == torch::kFloat32,
        "sin_l2 must be float32"
    );

    TORCH_CHECK(
        cos_l3.dtype() == torch::kFloat32,
        "cos_l3 must be float32"
    );

    TORCH_CHECK(
        sin_l3.dtype() == torch::kFloat32,
        "sin_l3 must be float32"
    );

    TORCH_CHECK(
        cos_l4.dtype() == torch::kFloat32,
        "cos_l4 must be float32"
    );

    TORCH_CHECK(
        sin_l4.dtype() == torch::kFloat32,
        "sin_l4 must be float32"
    );

    TORCH_CHECK(
        q.dim() == 4,
        "q must be [B,H,Q,128]"
    );

    TORCH_CHECK(
        q.size(3) == 128,
        "q last dim must be 128"
    );

    TORCH_CHECK(
        packed_l1.dim() == 4,
        "packed_l1 must be [B,H,T,32]"
    );

    TORCH_CHECK(
        packed_l2.dim() == 4,
        "packed_l2 must be [B,H,T,8]"
    );

    TORCH_CHECK(
        packed_l3.dim() == 4,
        "packed_l3 must be [B,H,T,4]"
    );

    TORCH_CHECK(
        packed_l4.dim() == 4,
        "packed_l4 must be [B,H,T,2]"
    );

    TORCH_CHECK(
        radii.dim() == 4,
        "radii must be [B,H,T,8]"
    );

    TORCH_CHECK(
        packed_l1.size(3) == 32,
        "packed_l1 last dim must be 32"
    );

    TORCH_CHECK(
        packed_l2.size(3) == 8,
        "packed_l2 last dim must be 8"
    );

    TORCH_CHECK(
        packed_l3.size(3) == 4,
        "packed_l3 last dim must be 4"
    );

    TORCH_CHECK(
        packed_l4.size(3) == 2,
        "packed_l4 last dim must be 2"
    );

    TORCH_CHECK(
        radii.size(3) == 8,
        "radii last dim must be 8"
    );

    TORCH_CHECK(
        cos_l1.numel() == 16,
        "cos_l1 must have 16 entries"
    );

    TORCH_CHECK(
        sin_l1.numel() == 16,
        "sin_l1 must have 16 entries"
    );

    TORCH_CHECK(
        cos_l2.numel() == 4,
        "cos_l2 must have 4 entries"
    );

    TORCH_CHECK(
        sin_l2.numel() == 4,
        "sin_l2 must have 4 entries"
    );

    TORCH_CHECK(
        cos_l3.numel() == 4,
        "cos_l3 must have 4 entries"
    );

    TORCH_CHECK(
        sin_l3.numel() == 4,
        "sin_l3 must have 4 entries"
    );

    TORCH_CHECK(
        cos_l4.numel() == 4,
        "cos_l4 must have 4 entries"
    );

    TORCH_CHECK(
        sin_l4.numel() == 4,
        "sin_l4 must have 4 entries"
    );

    const int64_t B =
        q.size(0);

    const int64_t H =
        q.size(1);

    const int64_t Q =
        q.size(2);

    const int64_t T =
        packed_l1.size(2);

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

    auto out = torch::empty(
        {B, H, Q, T},
        torch::TensorOptions()
            .device(q.device())
            .dtype(torch::kFloat32)
    );

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
        "polar_stage1_score_cuda grid_y exceeds CUDA limit: ",
        grid_y
    );

    const dim3 block(THREADS);

    const dim3 grid(
        static_cast<unsigned int>(grid_x),
        static_cast<unsigned int>(grid_y)
    );

    polar_stage1_score_kernel<<<grid, block>>>(
        q.contiguous().data_ptr<float>(),

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
        "polar_stage1_score_cuda",
        &polar_stage1_score_cuda,
        "Fused Polar Stage-1 score CUDA with precomputed trig tables"
    );
}