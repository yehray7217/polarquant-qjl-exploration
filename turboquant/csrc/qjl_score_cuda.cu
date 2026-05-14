#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAException.h>

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace {

constexpr int M = 128;
constexpr int PACKED_M = M / 8;
constexpr int THREADS = 128;

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

__device__ __forceinline__ float signed_q_from_packed_bit(
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

__global__ void qjl_packed_score_kernel(
    const float* __restrict__ q_projected,      // [B,H,Q,256]
    const uint8_t* __restrict__ packed_signs,   // [B,H,T,32]
    const half* __restrict__ norms,             // [B,H,T]
    float* __restrict__ out,                    // [B,H,Q,T]
    int B,
    int H,
    int Q,
    int T
) {
    const int64_t output_linear =
        static_cast<int64_t>(blockIdx.x)
        + static_cast<int64_t>(blockIdx.y)
          * static_cast<int64_t>(gridDim.x);

    const int tid = threadIdx.x;

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

    int64_t tmp = output_linear;

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

    const uint8_t* packed_ptr =
        packed_signs + key_linear * PACKED_M;

    // ------------------------------------------------------------
    // 128 threads handle 256 sketch coordinates.
    // Each thread processes:
    //   j0 = tid
    //   j1 = tid + 128
    // ------------------------------------------------------------

    const int j0 = tid;

    const float q0 =
        q_projected[qproj_offset + j0];

    float partial = signed_q_from_packed_bit(
        packed_ptr,
        j0,
        q0
    );

    // ------------------------------------------------------------
    // Reduce 128 partial sums.
    // 128 threads = 4 warps.
    // ------------------------------------------------------------

    float sum =
        warp_reduce_sum(partial);

    __shared__ float warp_sums[4];

    const int lane =
        tid & 31;

    const int warp =
        tid >> 5;

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
            const float norm =
                __half2float(norms[key_linear]);

        constexpr float QJL_CORRECTION_SCALE = 0.375f;

        const float scale =
            QJL_CORRECTION_SCALE
            * sqrtf(3.14159265358979323846f / 2.0f)
            / sqrtf(static_cast<float>(M));

            out[output_linear] =
                scale * norm * block_sum;
        }
    }
}

} // namespace


torch::Tensor qjl_packed_score_cuda(
    torch::Tensor q_projected,
    torch::Tensor packed_signs,
    torch::Tensor norms
) {
    TORCH_CHECK(
        q_projected.is_cuda(),
        "q_projected must be CUDA"
    );

    TORCH_CHECK(
        packed_signs.is_cuda(),
        "packed_signs must be CUDA"
    );

    TORCH_CHECK(
        norms.is_cuda(),
        "norms must be CUDA"
    );

    TORCH_CHECK(
        q_projected.dtype() == torch::kFloat32,
        "q_projected must be float32"
    );

    TORCH_CHECK(
        packed_signs.dtype() == torch::kUInt8,
        "packed_signs must be uint8"
    );

    TORCH_CHECK(
        norms.dtype() == torch::kFloat16,
        "norms must be float16"
    );

    TORCH_CHECK(
        q_projected.dim() == 4,
        "q_projected must be [B,H,Q,128]"
    );

    TORCH_CHECK(
        packed_signs.dim() == 4,
        "packed_signs must be [B,H,T,32]"
    );

    TORCH_CHECK(
        norms.dim() == 3,
        "norms must be [B,H,T]"
    );

    TORCH_CHECK(
        q_projected.size(3) == M,
        "q_projected last dim must be 128"
    );

    TORCH_CHECK(
        packed_signs.size(3) == PACKED_M,
        "packed_signs last dim must be 16"
    );

    const int64_t B =
        q_projected.size(0);

    const int64_t H =
        q_projected.size(1);

    const int64_t Q =
        q_projected.size(2);

    const int64_t T =
        packed_signs.size(2);

    TORCH_CHECK(
        packed_signs.size(0) == B
        && packed_signs.size(1) == H,
        "packed_signs B/H mismatch"
    );

    TORCH_CHECK(
        norms.size(0) == B
        && norms.size(1) == H
        && norms.size(2) == T,
        "norms shape mismatch"
    );

    auto out = torch::empty(
        {B, H, Q, T},
        torch::TensorOptions()
            .device(q_projected.device())
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
        "qjl_packed_score_cuda grid_y exceeds CUDA limit: ",
        grid_y
    );

    const dim3 block(THREADS);

    const dim3 grid(
        static_cast<unsigned int>(grid_x),
        static_cast<unsigned int>(grid_y)
    );

    qjl_packed_score_kernel<<<grid, block>>>(
        q_projected.contiguous().data_ptr<float>(),
        packed_signs.contiguous().data_ptr<uint8_t>(),
        reinterpret_cast<const half*>(
            norms.contiguous().data_ptr<at::Half>()
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
        "qjl_packed_score_cuda",
        &qjl_packed_score_cuda,
        "Fused QJL packed-sign score CUDA v2"
    );
}