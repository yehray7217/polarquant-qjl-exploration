#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAException.h>

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


__global__ void polar_stage1_reconstruct_kernel(
    const uint8_t* __restrict__ packed_l1,     // [B,H,T,32]
    const uint8_t* __restrict__ packed_l2,     // [B,H,T,8]
    const uint8_t* __restrict__ packed_l3,     // [B,H,T,4]
    const uint8_t* __restrict__ packed_l4,     // [B,H,T,2]
    const half* __restrict__ radii,            // [B,H,T,8]
    const float* __restrict__ centroids_l1,    // [16]
    const float* __restrict__ centroids_l2,    // [4]
    const float* __restrict__ centroids_l3,    // [4]
    const float* __restrict__ centroids_l4,    // [4]
    float* __restrict__ out,                   // [B,H,T,128]
    int B,
    int H,
    int T
) {
    const int token_linear = blockIdx.x;
    const int d = threadIdx.x;

    if (d >= D) {
        return;
    }

    int tmp = token_linear;

    const int t_idx = tmp % T;
    tmp /= T;

    const int h_idx = tmp % H;
    tmp /= H;

    const int b_idx = tmp;

    const int k_linear =
        ((b_idx * H + h_idx) * T + t_idx);

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
    // ------------------------------------------------------------

    const int block16 = d >> 4;       // 0..7
    const int local16 = d & 15;       // 0..15

    const int l4_idx =
        block16;

    const int l3_idx =
        block16 * 2 + (local16 >> 3);

    const int l2_idx =
        block16 * 4 + (local16 >> 2);

    const int l1_idx =
        block16 * 8 + (local16 >> 1);

    const uint8_t c1 =
        load_4bit_code(l1_ptr, l1_idx);

    const uint8_t c2 =
        load_2bit_code(l2_ptr, l2_idx);

    const uint8_t c3 =
        load_2bit_code(l3_ptr, l3_idx);

    const uint8_t c4 =
        load_2bit_code(l4_ptr, l4_idx);

    const float psi1 =
        centroids_l1[c1];

    const float psi2 =
        centroids_l2[c2];

    const float psi3 =
        centroids_l3[c3];

    const float psi4 =
        centroids_l4[c4];

    const bool use_sin_l1 =
        (local16 & 1) != 0;

    const bool use_sin_l2 =
        (local16 & 2) != 0;

    const bool use_sin_l3 =
        (local16 & 4) != 0;

    const bool use_sin_l4 =
        (local16 & 8) != 0;

    const float f1 =
        use_sin_l1 ? sinf(psi1) : cosf(psi1);

    const float f2 =
        use_sin_l2 ? sinf(psi2) : cosf(psi2);

    const float f3 =
        use_sin_l3 ? sinf(psi3) : cosf(psi3);

    const float f4 =
        use_sin_l4 ? sinf(psi4) : cosf(psi4);

    const float radius =
        __half2float(radii_ptr[block16]);

    const float reconstructed_x =
        radius * f4 * f3 * f2 * f1;

    out[k_linear * D + d] =
        reconstructed_x;
}

} // namespace


torch::Tensor polar_stage1_reconstruct_cuda(
    torch::Tensor packed_l1,
    torch::Tensor packed_l2,
    torch::Tensor packed_l3,
    torch::Tensor packed_l4,
    torch::Tensor radii,
    torch::Tensor centroids_l1,
    torch::Tensor centroids_l2,
    torch::Tensor centroids_l3,
    torch::Tensor centroids_l4
) {
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
        centroids_l1.dtype() == torch::kFloat32,
        "centroids_l1 must be float32"
    );
    TORCH_CHECK(
        centroids_l2.dtype() == torch::kFloat32,
        "centroids_l2 must be float32"
    );
    TORCH_CHECK(
        centroids_l3.dtype() == torch::kFloat32,
        "centroids_l3 must be float32"
    );
    TORCH_CHECK(
        centroids_l4.dtype() == torch::kFloat32,
        "centroids_l4 must be float32"
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

    const int64_t B =
        packed_l1.size(0);

    const int64_t H =
        packed_l1.size(1);

    const int64_t T =
        packed_l1.size(2);

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
        centroids_l1.numel() == 16,
        "centroids_l1 must have 16 entries"
    );
    TORCH_CHECK(
        centroids_l2.numel() == 4,
        "centroids_l2 must have 4 entries"
    );
    TORCH_CHECK(
        centroids_l3.numel() == 4,
        "centroids_l3 must have 4 entries"
    );
    TORCH_CHECK(
        centroids_l4.numel() == 4,
        "centroids_l4 must have 4 entries"
    );

    auto out = torch::empty(
        {B, H, T, D},
        torch::TensorOptions()
            .device(packed_l1.device())
            .dtype(torch::kFloat32)
    );

    const int total_tokens =
        static_cast<int>(B * H * T);

    const dim3 block(THREADS);
    const dim3 grid(total_tokens);

    polar_stage1_reconstruct_kernel<<<grid, block>>>(
        packed_l1.contiguous().data_ptr<uint8_t>(),
        packed_l2.contiguous().data_ptr<uint8_t>(),
        packed_l3.contiguous().data_ptr<uint8_t>(),
        packed_l4.contiguous().data_ptr<uint8_t>(),
        reinterpret_cast<const half*>(
            radii.contiguous().data_ptr<at::Half>()
        ),
        centroids_l1.contiguous().data_ptr<float>(),
        centroids_l2.contiguous().data_ptr<float>(),
        centroids_l3.contiguous().data_ptr<float>(),
        centroids_l4.contiguous().data_ptr<float>(),
        out.data_ptr<float>(),
        static_cast<int>(B),
        static_cast<int>(H),
        static_cast<int>(T)
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "polar_stage1_reconstruct_cuda",
        &polar_stage1_reconstruct_cuda,
        "Fused Polar Stage-1 reconstruction CUDA"
    );
}
