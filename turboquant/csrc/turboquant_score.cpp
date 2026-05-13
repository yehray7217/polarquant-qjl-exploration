#include <torch/extension.h>

torch::Tensor turboquant_decode_score_cuda(
    torch::Tensor q_rot,                    // [B,H,D]
    torch::Tensor sq,                       // [B,H,M]
    torch::Tensor packed_mse_indices,       // [B,H,capacity,D/4]
    torch::Tensor mse_norms,                // [B,H,capacity]
    torch::Tensor packed_qjl_sign_bits,     // [B,H,capacity,M/8]
    torch::Tensor residual_norms,           // [B,H,capacity]
    torch::Tensor centroids,                // [4]
    int64_t seq_len
);

torch::Tensor pack_2bit_indices_cuda(
    torch::Tensor indices
);

torch::Tensor pack_sign_bits_cuda(
    torch::Tensor sign_bits
);

std::vector<torch::Tensor> mse_assign_pack_reconstruct_rot_cuda(
    torch::Tensor x_norm,
    torch::Tensor norms,
    torch::Tensor centroids
);

torch::Tensor turboquant_decode_score_transposed_cuda(
    torch::Tensor q_rot,
    torch::Tensor q_sketch,
    torch::Tensor packed_mse_indices_t,
    torch::Tensor mse_norms,
    torch::Tensor packed_qjl_sign_bits_t,
    torch::Tensor qjl_residual_norms,
    torch::Tensor centroids
);

torch::Tensor turboquant_decode_score_transposed_sharedq_cuda(
    torch::Tensor q_rot,
    torch::Tensor q_sketch,
    torch::Tensor packed_mse_indices_t,
    torch::Tensor mse_norms,
    torch::Tensor packed_qjl_sign_bits_t,
    torch::Tensor qjl_residual_norms,
    torch::Tensor centroids
);

torch::Tensor turboquant_mse_lut_score_transposed_cuda(
    torch::Tensor q_rot,
    torch::Tensor packed_mse_indices_t,
    torch::Tensor mse_norms,
    torch::Tensor centroids
);

torch::Tensor turboquant_mse_lut_1bit_score_transposed_cuda(
    torch::Tensor q_rot,
    torch::Tensor packed_mse_sign_bits_t,
    torch::Tensor mse_norms,
    torch::Tensor centroids
);

torch::Tensor turboquant_mse_lut_4bit_score_transposed_cuda(
    torch::Tensor q_rot,
    torch::Tensor packed_indices_t,
    torch::Tensor mse_norms,
    torch::Tensor centroids
);

torch::Tensor turboquant_decode_score(
    torch::Tensor q_rot,
    torch::Tensor sq,
    torch::Tensor packed_mse_indices,
    torch::Tensor mse_norms,
    torch::Tensor packed_qjl_sign_bits,
    torch::Tensor residual_norms,
    torch::Tensor centroids,
    int64_t seq_len
) {
    TORCH_CHECK(q_rot.is_cuda(), "q_rot must be CUDA tensor");
    TORCH_CHECK(sq.is_cuda(), "sq must be CUDA tensor");
    TORCH_CHECK(packed_mse_indices.is_cuda(), "packed_mse_indices must be CUDA tensor");
    TORCH_CHECK(mse_norms.is_cuda(), "mse_norms must be CUDA tensor");
    TORCH_CHECK(packed_qjl_sign_bits.is_cuda(), "packed_qjl_sign_bits must be CUDA tensor");
    TORCH_CHECK(residual_norms.is_cuda(), "residual_norms must be CUDA tensor");
    TORCH_CHECK(centroids.is_cuda(), "centroids must be CUDA tensor");

    TORCH_CHECK(q_rot.scalar_type() == torch::kFloat32, "q_rot must be float32");
    TORCH_CHECK(sq.scalar_type() == torch::kFloat32, "sq must be float32");
    TORCH_CHECK(mse_norms.scalar_type() == torch::kFloat32, "mse_norms must be float32");
    TORCH_CHECK(residual_norms.scalar_type() == torch::kFloat32, "residual_norms must be float32");
    TORCH_CHECK(centroids.scalar_type() == torch::kFloat32, "centroids must be float32");

    TORCH_CHECK(packed_mse_indices.scalar_type() == torch::kUInt8, "packed_mse_indices must be uint8");
    TORCH_CHECK(packed_qjl_sign_bits.scalar_type() == torch::kUInt8, "packed_qjl_sign_bits must be uint8");

    TORCH_CHECK(q_rot.dim() == 3, "q_rot must be [B,H,D]");
    TORCH_CHECK(sq.dim() == 3, "sq must be [B,H,M]");
    TORCH_CHECK(packed_mse_indices.dim() == 4, "packed_mse_indices must be [B,H,capacity,D/4]");
    TORCH_CHECK(mse_norms.dim() == 3, "mse_norms must be [B,H,capacity]");
    TORCH_CHECK(packed_qjl_sign_bits.dim() == 4, "packed_qjl_sign_bits must be [B,H,capacity,M/8]");
    TORCH_CHECK(residual_norms.dim() == 3, "residual_norms must be [B,H,capacity]");
    TORCH_CHECK(centroids.numel() == 4, "centroids must contain 4 values");

    TORCH_CHECK(q_rot.is_contiguous(), "q_rot must be contiguous");
    TORCH_CHECK(sq.is_contiguous(), "sq must be contiguous");
    TORCH_CHECK(packed_mse_indices.is_contiguous(), "packed_mse_indices buffer must be contiguous");
    TORCH_CHECK(mse_norms.is_contiguous(), "mse_norms buffer must be contiguous");
    TORCH_CHECK(packed_qjl_sign_bits.is_contiguous(), "packed_qjl_sign_bits buffer must be contiguous");
    TORCH_CHECK(residual_norms.is_contiguous(), "residual_norms buffer must be contiguous");
    TORCH_CHECK(centroids.is_contiguous(), "centroids must be contiguous");

    const auto B = q_rot.size(0);
    const auto H = q_rot.size(1);
    const auto D = q_rot.size(2);

    const auto B2 = sq.size(0);
    const auto H2 = sq.size(1);
    const auto M = sq.size(2);

    const auto B3 = packed_mse_indices.size(0);
    const auto H3 = packed_mse_indices.size(1);
    const auto capacity = packed_mse_indices.size(2);
    const auto packed_D = packed_mse_indices.size(3);

    const auto B4 = packed_qjl_sign_bits.size(0);
    const auto H4 = packed_qjl_sign_bits.size(1);
    const auto capacity2 = packed_qjl_sign_bits.size(2);
    const auto packed_M = packed_qjl_sign_bits.size(3);

    TORCH_CHECK(B == B2 && B == B3 && B == B4, "Batch mismatch");
    TORCH_CHECK(H == H2 && H == H3 && H == H4, "Head mismatch");
    TORCH_CHECK(capacity == capacity2, "Capacity mismatch");
    TORCH_CHECK(mse_norms.size(0) == B && mse_norms.size(1) == H && mse_norms.size(2) == capacity, "mse_norms shape mismatch");
    TORCH_CHECK(residual_norms.size(0) == B && residual_norms.size(1) == H && residual_norms.size(2) == capacity, "residual_norms shape mismatch");

    TORCH_CHECK(D == packed_D * 4, "D must equal packed_D * 4");
    TORCH_CHECK(M == packed_M * 8, "M must equal packed_M * 8");

    TORCH_CHECK(seq_len >= 0, "seq_len must be non-negative");
    TORCH_CHECK(seq_len <= capacity, "seq_len exceeds cache capacity");

    return turboquant_decode_score_cuda(
        q_rot,
        sq,
        packed_mse_indices,
        mse_norms,
        packed_qjl_sign_bits,
        residual_norms,
        centroids,
        seq_len
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "turboquant_decode_score",
        &turboquant_decode_score,
        "TurboQuant packed decode score CUDA with capacity-aware cache buffer"
    );

    m.def(
        "pack_2bit_indices",
        &pack_2bit_indices_cuda,
        "Pack int64 2-bit indices to uint8 CUDA"
    );

    m.def(
        "pack_sign_bits",
        &pack_sign_bits_cuda,
        "Pack float32 +/- sign values to uint8 CUDA"
    );

    m.def(
        "mse_assign_pack_reconstruct_rot",
        &mse_assign_pack_reconstruct_rot_cuda,
        "Fused MSE centroid assignment + packed indices + x_hat_rot CUDA"
    );
 
    m.def(
        "turboquant_decode_score_transposed",
        &turboquant_decode_score_transposed_cuda,
        "TurboQuant decode score CUDA with score-friendly transposed packed layout"
    );
    
    m.def(
        "turboquant_decode_score_transposed_sharedq",
        &turboquant_decode_score_transposed_sharedq_cuda,
        "TurboQuant transposed decode score CUDA with shared-memory staged query vectors"
    );

    m.def(
        "turboquant_mse_lut_score_transposed",
        &turboquant_mse_lut_score_transposed_cuda,
        "TurboQuant 2-bit MSE-only LUT fused score CUDA"
    );

    m.def(
        "turboquant_mse_lut_1bit_score_transposed",
        &turboquant_mse_lut_1bit_score_transposed_cuda,
        "TurboQuant 1-bit MSE-only LUT fused score CUDA"
    );

    m.def(
        "turboquant_mse_lut_4bit_score_transposed",
        &turboquant_mse_lut_4bit_score_transposed_cuda,
        "TurboQuant 4-bit MSE-only LUT fused score CUDA"
    );
}