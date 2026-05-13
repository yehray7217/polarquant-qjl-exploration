#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>


namespace {

// ============================================================
// Pack 2-bit indices
//
// input:
//   indices: int64 tensor [..., D]
//   values expected in {0,1,2,3}
//
// output:
//   packed: uint8 tensor [..., D/4]
//
// Each CUDA thread writes one output byte:
//   byte = i0 | (i1 << 2) | (i2 << 4) | (i3 << 6)
// ============================================================

__global__ void pack_2bit_indices_int64_kernel(
    const int64_t* __restrict__ indices,
    uint8_t* __restrict__ packed,
    int64_t num_output_bytes
) {
    const int64_t out_idx =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (out_idx >= num_output_bytes) {
        return;
    }

    const int64_t in_base = out_idx * 4;

    const uint8_t i0 = static_cast<uint8_t>(indices[in_base + 0]) & 0x3;
    const uint8_t i1 = static_cast<uint8_t>(indices[in_base + 1]) & 0x3;
    const uint8_t i2 = static_cast<uint8_t>(indices[in_base + 2]) & 0x3;
    const uint8_t i3 = static_cast<uint8_t>(indices[in_base + 3]) & 0x3;

    packed[out_idx] =
        static_cast<uint8_t>(
            i0 |
            (i1 << 2) |
            (i2 << 4) |
            (i3 << 6)
        );
}


// ============================================================
// Pack 1-bit sign bits
//
// input:
//   sign_bits: float32 tensor [..., M]
//   convention:
//     value > 0 -> bit 1
//     else      -> bit 0
//
// output:
//   packed: uint8 tensor [..., M/8]
//
// Each CUDA thread writes one output byte.
// ============================================================

__global__ void pack_sign_bits_float_kernel(
    const float* __restrict__ sign_bits,
    uint8_t* __restrict__ packed,
    int64_t num_output_bytes
) {
    const int64_t out_idx =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (out_idx >= num_output_bytes) {
        return;
    }

    const int64_t in_base = out_idx * 8;

    const uint8_t b0 = sign_bits[in_base + 0] > 0.0f ? 1u : 0u;
    const uint8_t b1 = sign_bits[in_base + 1] > 0.0f ? 1u : 0u;
    const uint8_t b2 = sign_bits[in_base + 2] > 0.0f ? 1u : 0u;
    const uint8_t b3 = sign_bits[in_base + 3] > 0.0f ? 1u : 0u;
    const uint8_t b4 = sign_bits[in_base + 4] > 0.0f ? 1u : 0u;
    const uint8_t b5 = sign_bits[in_base + 5] > 0.0f ? 1u : 0u;
    const uint8_t b6 = sign_bits[in_base + 6] > 0.0f ? 1u : 0u;
    const uint8_t b7 = sign_bits[in_base + 7] > 0.0f ? 1u : 0u;

    packed[out_idx] =
        static_cast<uint8_t>(
            b0 |
            (b1 << 1) |
            (b2 << 2) |
            (b3 << 3) |
            (b4 << 4) |
            (b5 << 5) |
            (b6 << 6) |
            (b7 << 7)
        );
}

} // namespace


// ============================================================
// C++ entry: pack 2-bit indices
// ============================================================

torch::Tensor pack_2bit_indices_cuda(
    torch::Tensor indices
) {
    TORCH_CHECK(
        indices.is_cuda(),
        "indices must be a CUDA tensor"
    );

    TORCH_CHECK(
        indices.scalar_type() == torch::kInt64,
        "indices must be int64 / torch.long"
    );

    TORCH_CHECK(
        indices.is_contiguous(),
        "indices must be contiguous"
    );

    TORCH_CHECK(
        indices.dim() >= 1,
        "indices must have at least 1 dimension"
    );

    const int64_t D = indices.size(-1);

    TORCH_CHECK(
        D % 4 == 0,
        "last dimension of indices must be divisible by 4"
    );

    auto out_sizes = indices.sizes().vec();
    out_sizes.back() = D / 4;

    auto packed = torch::empty(
        out_sizes,
        torch::TensorOptions()
            .dtype(torch::kUInt8)
            .device(indices.device())
    );

    const int64_t num_output_bytes = packed.numel();

    constexpr int threads = 256;
    const int blocks = static_cast<int>(
        (num_output_bytes + threads - 1) / threads
    );

    pack_2bit_indices_int64_kernel<<<
        blocks,
        threads,
        0,
        at::cuda::getDefaultCUDAStream()
    >>>(
        indices.data_ptr<int64_t>(),
        packed.data_ptr<uint8_t>(),
        num_output_bytes
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return packed;
}


// ============================================================
// C++ entry: pack sign bits
// ============================================================

torch::Tensor pack_sign_bits_cuda(
    torch::Tensor sign_bits
) {
    TORCH_CHECK(
        sign_bits.is_cuda(),
        "sign_bits must be a CUDA tensor"
    );

    TORCH_CHECK(
        sign_bits.scalar_type() == torch::kFloat32,
        "sign_bits must be float32"
    );

    TORCH_CHECK(
        sign_bits.is_contiguous(),
        "sign_bits must be contiguous"
    );

    TORCH_CHECK(
        sign_bits.dim() >= 1,
        "sign_bits must have at least 1 dimension"
    );

    const int64_t M = sign_bits.size(-1);

    TORCH_CHECK(
        M % 8 == 0,
        "last dimension of sign_bits must be divisible by 8"
    );

    auto out_sizes = sign_bits.sizes().vec();
    out_sizes.back() = M / 8;

    auto packed = torch::empty(
        out_sizes,
        torch::TensorOptions()
            .dtype(torch::kUInt8)
            .device(sign_bits.device())
    );

    const int64_t num_output_bytes = packed.numel();

    constexpr int threads = 256;
    const int blocks = static_cast<int>(
        (num_output_bytes + threads - 1) / threads
    );

    pack_sign_bits_float_kernel<<<
        blocks,
        threads,
        0,
        at::cuda::getDefaultCUDAStream()
    >>>(
        sign_bits.data_ptr<float>(),
        packed.data_ptr<uint8_t>(),
        num_output_bytes
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return packed;
}
