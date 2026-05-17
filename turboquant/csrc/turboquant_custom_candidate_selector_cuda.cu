#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <vector>
#include <cstdint>
#include <limits>
#include <cmath>
#include <climits>

namespace {

constexpr int GROUP_TOKENS = 128;
constexpr int WARP_THREADS = 32;
constexpr int VALUES_PER_LANE = 4;
constexpr int LOCAL_KEEP = 8;

constexpr int POOL_CHUNK = 256;
constexpr int POOL_KEEP = 32;
constexpr int FINAL_SORT_WIDTH = 1024;

constexpr unsigned FULL_MASK = 0xffffffffu;

__device__ __forceinline__ bool better_pair(
    float a_val,
    int a_idx,
    float b_val,
    int b_idx
) {
    return (a_val > b_val) || ((a_val == b_val) && (a_idx < b_idx));
}

template<int SORT_N>
__device__ __forceinline__ void bitonic_sort_descending(
    float* sh_vals,
    int* sh_indices
) {
    const int tid = static_cast<int>(threadIdx.x);

    for (int k = 2; k <= SORT_N; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            const int partner = tid ^ j;
            if (partner > tid) {
                const bool desc_half = ((tid & k) == 0);

                const float self_val = sh_vals[tid];
                const int self_idx = sh_indices[tid];
                const float other_val = sh_vals[partner];
                const int other_idx = sh_indices[partner];

                const bool other_better =
                    better_pair(other_val, other_idx, self_val, self_idx);
                const bool self_better =
                    better_pair(self_val, self_idx, other_val, other_idx);

                const bool do_swap =
                    (desc_half && other_better) ||
                    ((!desc_half) && self_better);

                if (do_swap) {
                    sh_vals[tid] = other_val;
                    sh_indices[tid] = other_idx;
                    sh_vals[partner] = self_val;
                    sh_indices[partner] = self_idx;
                }
            }
            __syncthreads();
        }
    }
}

__global__ void polar_local_warp_top8_kernel(
    const float* __restrict__ polar_logits,
    float* __restrict__ candidate_values,
    int32_t* __restrict__ candidate_indices,
    int H,
    int T,
    int groups_per_head
) {
    const int lane = static_cast<int>(threadIdx.x);
    const int group_idx = static_cast<int>(blockIdx.x);
    const int h_idx = static_cast<int>(blockIdx.y);

    if (lane >= WARP_THREADS || h_idx >= H || group_idx >= groups_per_head) {
        return;
    }

    const int group_base_t = group_idx * GROUP_TOKENS;
    float local_best_val = -INFINITY;
    int local_best_idx = INT_MAX;

    #pragma unroll
    for (int i = 0; i < VALUES_PER_LANE; ++i) {
        const int t_idx = group_base_t + lane + i * WARP_THREADS;
        if (t_idx < T) {
            const float v =
                polar_logits[(static_cast<int64_t>(h_idx) * T) + t_idx];
            if (better_pair(v, t_idx, local_best_val, local_best_idx)) {
                local_best_val = v;
                local_best_idx = t_idx;
            }
        }
    }

    // Approximate local selector:
    // keep top-8 among the 32 lane maxima for each 128-token group.
    #pragma unroll
    for (int keep = 0; keep < LOCAL_KEEP; ++keep) {
        float best_val = local_best_val;
        int best_idx = local_best_idx;
        int best_lane = lane;

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float other_val = __shfl_down_sync(FULL_MASK, best_val, offset);
            const int other_idx = __shfl_down_sync(FULL_MASK, best_idx, offset);
            const int other_lane = __shfl_down_sync(FULL_MASK, best_lane, offset);
            if (lane + offset < WARP_THREADS &&
                better_pair(other_val, other_idx, best_val, best_idx)) {
                best_val = other_val;
                best_idx = other_idx;
                best_lane = other_lane;
            }
        }

        const int winner_lane = __shfl_sync(FULL_MASK, best_lane, 0);
        const float winner_val = __shfl_sync(FULL_MASK, best_val, 0);
        const int winner_idx = __shfl_sync(FULL_MASK, best_idx, 0);

        if (lane == 0) {
            const int64_t out_offset =
                (static_cast<int64_t>(h_idx) * groups_per_head + group_idx)
                * LOCAL_KEEP
                + keep;
            candidate_values[out_offset] = winner_val;
            candidate_indices[out_offset] = static_cast<int32_t>(winner_idx);
        }

        if (lane == winner_lane) {
            local_best_val = -INFINITY;
            local_best_idx = INT_MAX;
        }
    }
}

__global__ void pooled_chunk_top32_bitonic_kernel(
    const float* __restrict__ candidate_values,
    const int32_t* __restrict__ candidate_indices,
    float* __restrict__ reduced_values,
    int32_t* __restrict__ reduced_indices,
    int H,
    int candidate_count,
    int chunks_per_head
) {
    const int tid = static_cast<int>(threadIdx.x);
    const int chunk_idx = static_cast<int>(blockIdx.x);
    const int h_idx = static_cast<int>(blockIdx.y);

    if (tid >= POOL_CHUNK || h_idx >= H || chunk_idx >= chunks_per_head) {
        return;
    }

    __shared__ float sh_vals[POOL_CHUNK];
    __shared__ int sh_indices[POOL_CHUNK];

    const int pos = chunk_idx * POOL_CHUNK + tid;
    const int64_t base = static_cast<int64_t>(h_idx) * candidate_count;

    if (pos < candidate_count) {
        sh_vals[tid] = candidate_values[base + pos];
        sh_indices[tid] = static_cast<int>(candidate_indices[base + pos]);
    } else {
        sh_vals[tid] = -INFINITY;
        sh_indices[tid] = INT_MAX;
    }
    __syncthreads();

    bitonic_sort_descending<POOL_CHUNK>(sh_vals, sh_indices);

    if (tid < POOL_KEEP) {
        const int64_t out_offset =
            (static_cast<int64_t>(h_idx) * chunks_per_head + chunk_idx)
            * POOL_KEEP
            + tid;
        reduced_values[out_offset] = sh_vals[tid];
        reduced_indices[out_offset] = static_cast<int32_t>(sh_indices[tid]);
    }
}

__global__ void final_top128_bitonic_kernel(
    const float* __restrict__ reduced_values,
    const int32_t* __restrict__ reduced_indices,
    int64_t* __restrict__ selected_indices,
    int H,
    int reduced_count,
    int K
) {
    const int tid = static_cast<int>(threadIdx.x);
    const int h_idx = static_cast<int>(blockIdx.x);

    if (tid >= FINAL_SORT_WIDTH || h_idx >= H) {
        return;
    }

    __shared__ float sh_vals[FINAL_SORT_WIDTH];
    __shared__ int sh_indices[FINAL_SORT_WIDTH];

    const int64_t base = static_cast<int64_t>(h_idx) * reduced_count;

    if (tid < reduced_count) {
        sh_vals[tid] = reduced_values[base + tid];
        sh_indices[tid] = static_cast<int>(reduced_indices[base + tid]);
    } else {
        sh_vals[tid] = -INFINITY;
        sh_indices[tid] = INT_MAX;
    }
    __syncthreads();

    bitonic_sort_descending<FINAL_SORT_WIDTH>(sh_vals, sh_indices);

    if (tid < K) {
        selected_indices[static_cast<int64_t>(h_idx) * K + tid] =
            static_cast<int64_t>(sh_indices[tid]);
    }
}

void validate_selector(torch::Tensor polar_logits, int64_t topk) {
    TORCH_CHECK(polar_logits.is_cuda(), "polar_logits must be CUDA");
    TORCH_CHECK(polar_logits.dtype() == torch::kFloat32, "polar_logits must be float32");
    TORCH_CHECK(
        polar_logits.dim() == 4 &&
        polar_logits.size(0) == 1 &&
        polar_logits.size(2) == 1,
        "polar_logits must be [1,H,1,T]"
    );
    TORCH_CHECK(topk > 0, "topk must be positive");
    TORCH_CHECK(topk <= 128, "current custom selector supports topk <= 128");
}

} // namespace

torch::Tensor turboquant_custom_candidate_topk_selector_cuda(
    torch::Tensor polar_logits,
    int64_t topk
) {
    validate_selector(polar_logits, topk);

    const int H = static_cast<int>(polar_logits.size(1));
    const int T = static_cast<int>(polar_logits.size(3));

    const int groups_per_head = (T + GROUP_TOKENS - 1) / GROUP_TOKENS;
    const int candidate_count = groups_per_head * LOCAL_KEEP;

    const int chunks_per_head = (candidate_count + POOL_CHUNK - 1) / POOL_CHUNK;
    const int reduced_count = chunks_per_head * POOL_KEEP;

    TORCH_CHECK(
        reduced_count <= FINAL_SORT_WIDTH,
        "current merge-v2 selector supports reduced candidate count <= ",
        FINAL_SORT_WIDTH,
        "; got ",
        reduced_count,
        ". This covers T <= 131072 with the current 128-token/top8 pooling."
    );

    auto values_opts = polar_logits.options().dtype(torch::kFloat32);
    auto idx32_opts = polar_logits.options().dtype(torch::kInt32);
    auto idx64_opts = polar_logits.options().dtype(torch::kInt64);

    auto candidate_values = torch::empty({H, candidate_count}, values_opts);
    auto candidate_indices = torch::empty({H, candidate_count}, idx32_opts);

    auto reduced_values = torch::empty({H, reduced_count}, values_opts);
    auto reduced_indices = torch::empty({H, reduced_count}, idx32_opts);

    auto selected_indices = torch::empty({1, H, 1, topk}, idx64_opts);

    const dim3 local_grid(groups_per_head, H, 1);
    const dim3 local_block(WARP_THREADS, 1, 1);
    polar_local_warp_top8_kernel<<<local_grid, local_block>>>(
        polar_logits.contiguous().data_ptr<float>(),
        candidate_values.data_ptr<float>(),
        candidate_indices.data_ptr<int32_t>(),
        H,
        T,
        groups_per_head
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const dim3 chunk_grid(chunks_per_head, H, 1);
    const dim3 chunk_block(POOL_CHUNK, 1, 1);
    pooled_chunk_top32_bitonic_kernel<<<chunk_grid, chunk_block>>>(
        candidate_values.data_ptr<float>(),
        candidate_indices.data_ptr<int32_t>(),
        reduced_values.data_ptr<float>(),
        reduced_indices.data_ptr<int32_t>(),
        H,
        candidate_count,
        chunks_per_head
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const dim3 final_grid(H, 1, 1);
    const dim3 final_block(FINAL_SORT_WIDTH, 1, 1);
    final_top128_bitonic_kernel<<<final_grid, final_block>>>(
        reduced_values.data_ptr<float>(),
        reduced_indices.data_ptr<int32_t>(),
        selected_indices.data_ptr<int64_t>(),
        H,
        reduced_count,
        static_cast<int>(topk)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return selected_indices;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "turboquant_custom_candidate_topk_selector_cuda",
        &turboquant_custom_candidate_topk_selector_cuda,
        "Custom three-stage Polar candidate top-K selector (CUDA): local pool, chunk top-32, final bitonic top-K"
    );
}
