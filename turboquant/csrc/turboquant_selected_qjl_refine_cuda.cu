#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAException.h>
#include <cmath>
#include <cstdint>

namespace {
constexpr int M = 128;
constexpr int QJL_LANE_SIGN_BYTES = 16;
constexpr int THREADS = 256;
constexpr float QJL_CORRECTION_SCALE = 0.375f;
constexpr unsigned FULL_MASK = 0xffffffffu;

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) val += __shfl_down_sync(FULL_MASK, val, offset);
    return val;
}

__device__ __forceinline__ float qjl_sum_lane_nibble(
    const uint8_t* __restrict__ lane_nibble_signs_ptr,
    const float* __restrict__ sh_qproj,
    int lane
) {
    const uint8_t packed_nibble_pair = lane_nibble_signs_ptr[lane >> 1];
    const uint8_t lane_nibble = (lane & 1)
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

__global__ void turboquant_selected_qjl_refine_topk_m128_kernel(
    const float* __restrict__ q_projected,
    const uint8_t* __restrict__ lane_nibble_qjl_signs,
    const half* __restrict__ qjl_norms,
    const float* __restrict__ polar_logits,
    const int64_t* __restrict__ selected_indices,
    float* __restrict__ selected_refined_logits,
    int H,
    int T,
    int K
) {
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int h_idx = static_cast<int>(blockIdx.y);
    const int k_idx = static_cast<int>(blockIdx.x) * 8 + warp_id;

    __shared__ float sh_qproj[M];
    const int64_t qproj_offset = static_cast<int64_t>(h_idx) * M;
    if (tid < M) sh_qproj[tid] = q_projected[qproj_offset + tid];
    __syncthreads();

    if (h_idx >= H || k_idx >= K) return;
    const int64_t selected_offset = static_cast<int64_t>(h_idx) * K + k_idx;
    const int64_t t_idx64 = selected_indices[selected_offset];
    if (t_idx64 < 0 || t_idx64 >= static_cast<int64_t>(T)) return;
    const int t_idx = static_cast<int>(t_idx64);
    const int64_t key_linear = static_cast<int64_t>(h_idx) * T + t_idx;
    const uint8_t* signs_ptr = lane_nibble_qjl_signs + key_linear * QJL_LANE_SIGN_BYTES;
    const float qjl_sum = qjl_sum_lane_nibble(signs_ptr, sh_qproj, lane);
    if (lane == 0) {
        const float residual_norm_float = __half2float(qjl_norms[key_linear]);
        const float qjl_scale = QJL_CORRECTION_SCALE
            * sqrtf(3.14159265358979323846f / 2.0f)
            / sqrtf(static_cast<float>(M));
        const float polar_score = polar_logits[key_linear];
        selected_refined_logits[selected_offset] = polar_score + qjl_scale * residual_norm_float * qjl_sum;
    }
}

void validate_selected_refine(
    torch::Tensor q_projected,
    torch::Tensor lane_nibble_qjl_signs,
    torch::Tensor qjl_norms,
    torch::Tensor polar_logits,
    torch::Tensor selected_indices
) {
    TORCH_CHECK(q_projected.is_cuda() && q_projected.dtype() == torch::kFloat32, "q_projected must be CUDA float32");
    TORCH_CHECK(q_projected.dim() == 4 && q_projected.size(0) == 1 && q_projected.size(2) == 1 && q_projected.size(3) == M, "q_projected must be [1,H,1,128]");
    const int64_t H = q_projected.size(1);
    TORCH_CHECK(lane_nibble_qjl_signs.is_cuda() && lane_nibble_qjl_signs.dtype() == torch::kUInt8, "lane_nibble_qjl_signs must be CUDA uint8");
    TORCH_CHECK(lane_nibble_qjl_signs.dim() == 4 && lane_nibble_qjl_signs.size(0) == 1 && lane_nibble_qjl_signs.size(1) == H && lane_nibble_qjl_signs.size(3) == QJL_LANE_SIGN_BYTES, "lane_nibble_qjl_signs must be [1,H,T,16]");
    const int64_t T = lane_nibble_qjl_signs.size(2);
    TORCH_CHECK(qjl_norms.is_cuda() && qjl_norms.dtype() == torch::kFloat16, "qjl_norms must be CUDA float16");
    TORCH_CHECK(qjl_norms.dim() == 3 && qjl_norms.size(0) == 1 && qjl_norms.size(1) == H && qjl_norms.size(2) == T, "qjl_norms must be [1,H,T]");
    TORCH_CHECK(polar_logits.is_cuda() && polar_logits.dtype() == torch::kFloat32, "polar_logits must be CUDA float32");
    TORCH_CHECK(polar_logits.dim() == 4 && polar_logits.size(0) == 1 && polar_logits.size(1) == H && polar_logits.size(2) == 1 && polar_logits.size(3) == T, "polar_logits must be [1,H,1,T]");
    TORCH_CHECK(selected_indices.is_cuda() && selected_indices.dtype() == torch::kInt64, "selected_indices must be CUDA int64");
    TORCH_CHECK(selected_indices.dim() == 4 && selected_indices.size(0) == 1 && selected_indices.size(1) == H && selected_indices.size(2) == 1, "selected_indices must be [1,H,1,K]");
}
}

torch::Tensor turboquant_selected_qjl_refine_topk_m128_cuda(
    torch::Tensor q_projected,
    torch::Tensor lane_nibble_qjl_signs,
    torch::Tensor qjl_norms,
    torch::Tensor polar_logits,
    torch::Tensor selected_indices
) {
    validate_selected_refine(q_projected, lane_nibble_qjl_signs, qjl_norms, polar_logits, selected_indices);
    const int64_t H = q_projected.size(1);
    const int64_t T = lane_nibble_qjl_signs.size(2);
    const int64_t K = selected_indices.size(3);
    TORCH_CHECK((K + 7) / 8 <= 2147483647LL, "K grid exceeds CUDA limit");
    TORCH_CHECK(H <= 65535, "H grid exceeds CUDA limit");
    auto out = torch::empty({1, H, 1, K}, torch::TensorOptions().device(q_projected.device()).dtype(torch::kFloat32));
    const dim3 block(THREADS);
    const dim3 grid(static_cast<unsigned int>((K + 7) / 8), static_cast<unsigned int>(H), 1);
    turboquant_selected_qjl_refine_topk_m128_kernel<<<grid, block>>>(
        q_projected.contiguous().data_ptr<float>(),
        lane_nibble_qjl_signs.contiguous().data_ptr<uint8_t>(),
        reinterpret_cast<const half*>(qjl_norms.contiguous().data_ptr<at::Half>()),
        polar_logits.contiguous().data_ptr<float>(),
        selected_indices.contiguous().data_ptr<int64_t>(),
        out.data_ptr<float>(),
        static_cast<int>(H), static_cast<int>(T), static_cast<int>(K)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "turboquant_selected_qjl_refine_topk_m128_cuda",
        &turboquant_selected_qjl_refine_topk_m128_cuda,
        "Refine Polar top-K logits with M=128 QJL correction on selected indices"
    );
}
