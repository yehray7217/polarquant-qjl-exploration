#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAException.h>

#include <cmath>
#include <cstdint>

namespace {

constexpr int M = 128;
constexpr int SIGN_BYTES = 16;
constexpr int THREADS = 256;   // 8 warps / CTA
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


__device__ __forceinline__ float qjl_scale_value() {
    return QJL_CORRECTION_SCALE
        * sqrtf(3.14159265358979323846f / 2.0f)
        / sqrtf(static_cast<float>(M));
}


__global__ void turboquant_qjl_only_split_sign_words_decode_b1q1_warp8_kernel(
    const float* __restrict__ q_projected,
    const uint8_t* __restrict__ packed_qjl_signs,
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
    const uint8_t* signs_ptr = packed_qjl_signs + key_linear * SIGN_BYTES;

    uint32_t sign_word_local = 0;
    if (lane < 4) {
        sign_word_local =
            reinterpret_cast<const uint32_t*>(signs_ptr)[lane];
    }

    float qjl_acc = 0.0f;

    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int sketch_idx = lane + 32 * k;
        const float qproj_val = sh_qproj[sketch_idx];

        const int sign_byte_idx = sketch_idx >> 3;
        const int word_owner = sign_byte_idx >> 2;
        const int byte_in_word = sign_byte_idx & 3;

        const uint32_t sign_word = __shfl_sync(
            0xffffffff,
            sign_word_local,
            word_owner
        );

        const uint8_t sign_byte = byte_from_u32(
            sign_word,
            byte_in_word
        );

        const uint8_t sign_bit = static_cast<uint8_t>(
            (sign_byte >> (sketch_idx & 7)) & 0x01u
        );

        qjl_acc += sign_bit ? qproj_val : -qproj_val;
    }

    const float qjl_sum = warp_reduce_sum(qjl_acc);

    if (lane == 0) {
        const float residual_norm = __half2float(qjl_norms[key_linear]);
        const float final_value =
            qjl_scale_value() * residual_norm * qjl_sum;

        const int64_t out_linear =
            static_cast<int64_t>(h_idx) * T + t_idx;
        out[out_linear] = final_value;
    }
}


__global__ void turboquant_qjl_only_lane_nibble_signs_decode_b1q1_warp8_kernel(
    const float* __restrict__ q_projected,
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
    const uint8_t* signs_ptr =
        lane_nibble_qjl_signs + key_linear * SIGN_BYTES;

    const uint8_t packed_nibble_pair = signs_ptr[lane >> 1];
    const uint8_t lane_nibble =
        (lane & 1)
            ? static_cast<uint8_t>((packed_nibble_pair >> 4) & 0x0Fu)
            : static_cast<uint8_t>(packed_nibble_pair & 0x0Fu);

    float qjl_acc = 0.0f;

    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int sketch_idx = lane + 32 * k;
        const float qproj_val = sh_qproj[sketch_idx];
        const uint8_t sign_bit = static_cast<uint8_t>(
            (lane_nibble >> k) & 0x01u
        );
        qjl_acc += sign_bit ? qproj_val : -qproj_val;
    }

    const float qjl_sum = warp_reduce_sum(qjl_acc);

    if (lane == 0) {
        const float residual_norm = __half2float(qjl_norms[key_linear]);
        const float final_value =
            qjl_scale_value() * residual_norm * qjl_sum;

        const int64_t out_linear =
            static_cast<int64_t>(h_idx) * T + t_idx;
        out[out_linear] = final_value;
    }
}


void validate_common_inputs(
    torch::Tensor q_projected,
    torch::Tensor qjl_sign_layout,
    torch::Tensor qjl_norms
) {
    TORCH_CHECK(q_projected.is_cuda(), "q_projected must be CUDA");
    TORCH_CHECK(qjl_sign_layout.is_cuda(), "qjl_sign_layout must be CUDA");
    TORCH_CHECK(qjl_norms.is_cuda(), "qjl_norms must be CUDA");

    TORCH_CHECK(q_projected.dtype() == torch::kFloat32, "q_projected must be float32");
    TORCH_CHECK(qjl_sign_layout.dtype() == torch::kUInt8, "qjl_sign_layout must be uint8");
    TORCH_CHECK(qjl_norms.dtype() == torch::kFloat16, "qjl_norms must be float16");

    TORCH_CHECK(q_projected.dim() == 4, "q_projected must be [B,H,Q,128]");
    TORCH_CHECK(q_projected.size(0) == 1, "QJL sign-layout kernels require B=1");
    TORCH_CHECK(q_projected.size(2) == 1, "QJL sign-layout kernels require Q=1");
    TORCH_CHECK(q_projected.size(3) == M, "q_projected last dim must be 128");

    TORCH_CHECK(qjl_sign_layout.dim() == 4, "qjl_sign_layout must be [B,H,T,16]");
    TORCH_CHECK(qjl_sign_layout.size(0) == 1, "qjl_sign_layout B mismatch");
    TORCH_CHECK(qjl_sign_layout.size(1) == q_projected.size(1), "qjl_sign_layout H mismatch");
    TORCH_CHECK(qjl_sign_layout.size(3) == SIGN_BYTES, "qjl_sign_layout last dim must be 16");

    TORCH_CHECK(qjl_norms.dim() == 3, "qjl_norms must be [B,H,T]");
    TORCH_CHECK(qjl_norms.size(0) == 1, "qjl_norms B mismatch");
    TORCH_CHECK(qjl_norms.size(1) == q_projected.size(1), "qjl_norms H mismatch");
    TORCH_CHECK(qjl_norms.size(2) == qjl_sign_layout.size(2), "qjl_norms T mismatch");
}


torch::Tensor allocate_out(
    torch::Tensor q_projected,
    torch::Tensor qjl_sign_layout
) {
    return torch::empty(
        {1, q_projected.size(1), 1, qjl_sign_layout.size(2)},
        torch::TensorOptions()
            .device(q_projected.device())
            .dtype(torch::kFloat32)
    );
}


void validate_grid(int64_t H, int64_t T) {
    const int64_t token_groups = (T + 7) / 8;
    TORCH_CHECK(token_groups <= 2147483647LL, "QJL sign-layout grid.x exceeds CUDA limit");
    TORCH_CHECK(H <= 65535, "QJL sign-layout grid.y exceeds CUDA limit");
}

} // namespace


torch::Tensor turboquant_qjl_only_split_sign_words_cuda(
    torch::Tensor q_projected,
    torch::Tensor packed_qjl_signs,
    torch::Tensor qjl_norms
) {
    validate_common_inputs(
        q_projected,
        packed_qjl_signs,
        qjl_norms
    );

    const int64_t H = q_projected.size(1);
    const int64_t T = packed_qjl_signs.size(2);
    validate_grid(H, T);

    auto out = allocate_out(q_projected, packed_qjl_signs);

    const dim3 block(THREADS);
    const dim3 grid(
        static_cast<unsigned int>((T + 7) / 8),
        static_cast<unsigned int>(H),
        1
    );

    turboquant_qjl_only_split_sign_words_decode_b1q1_warp8_kernel<<<grid, block>>>(
        q_projected.contiguous().data_ptr<float>(),
        packed_qjl_signs.contiguous().data_ptr<uint8_t>(),
        reinterpret_cast<const half*>(qjl_norms.contiguous().data_ptr<at::Half>()),
        out.data_ptr<float>(),
        static_cast<int>(H),
        static_cast<int>(T)
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}


torch::Tensor turboquant_qjl_only_lane_nibble_signs_cuda(
    torch::Tensor q_projected,
    torch::Tensor lane_nibble_qjl_signs,
    torch::Tensor qjl_norms
) {
    validate_common_inputs(
        q_projected,
        lane_nibble_qjl_signs,
        qjl_norms
    );

    const int64_t H = q_projected.size(1);
    const int64_t T = lane_nibble_qjl_signs.size(2);
    validate_grid(H, T);

    auto out = allocate_out(q_projected, lane_nibble_qjl_signs);

    const dim3 block(THREADS);
    const dim3 grid(
        static_cast<unsigned int>((T + 7) / 8),
        static_cast<unsigned int>(H),
        1
    );

    turboquant_qjl_only_lane_nibble_signs_decode_b1q1_warp8_kernel<<<grid, block>>>(
        q_projected.contiguous().data_ptr<float>(),
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
        "turboquant_qjl_only_split_sign_words_cuda",
        &turboquant_qjl_only_split_sign_words_cuda,
        "QJL-only sign layout ablation: split sketch-major sign words"
    );
    m.def(
        "turboquant_qjl_only_lane_nibble_signs_cuda",
        &turboquant_qjl_only_lane_nibble_signs_cuda,
        "QJL-only sign layout ablation: lane-major sign nibbles"
    );
}
