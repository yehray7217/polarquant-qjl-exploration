from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from turboquant.mse_quant import (
    make_random_rotation,
    get_1bit_centroids,
)
from turboquant.cuda_packing import (
    pack_sign_bits_cuda,
)
from turboquant.cuda_score_mse_lut_1bit import (
    turboquant_mse_lut_1bit_score_transposed_cuda,
)


@torch.no_grad()
def main():
    device = "cuda:0"
    dtype = torch.float32

    B = 1
    H = 32
    T = 4096
    D = 128

    torch.manual_seed(0)

    rotation = make_random_rotation(
        d=D,
        device=device,
        dtype=dtype,
        seed=123,
    )

    centroids = get_1bit_centroids(
        d=D,
        device=device,
        dtype=dtype,
    )

    k = torch.randn(
        B,
        H,
        T,
        D,
        device=device,
        dtype=dtype,
    )

    q = torch.randn(
        B,
        H,
        1,
        D,
        device=device,
        dtype=dtype,
    )

    # ------------------------------------------------------------
    # Quantize K to 1-bit MSE sign codes
    # ------------------------------------------------------------
    k_flat = k.reshape(B * H * T, D)

    k_rot = k_flat @ rotation.T

    norms_flat = torch.linalg.vector_norm(
        k_rot,
        ord=2,
        dim=-1,
    )

    safe_norms = torch.clamp(
        norms_flat,
        min=torch.finfo(k_rot.dtype).eps,
    )

    k_norm = k_rot / safe_norms.unsqueeze(-1)

    k_norm_bhtd = k_norm.reshape(B, H, T, D).contiguous()

    packed_signs = pack_sign_bits_cuda(
        k_norm_bhtd
    )

    packed_signs_t = packed_signs.permute(
        0, 1, 3, 2
    ).contiguous()

    mse_norms = norms_flat.reshape(
        B, H, T
    ).contiguous()

    # ------------------------------------------------------------
    # Query rotation
    # ------------------------------------------------------------
    q_flat = q.squeeze(2)
    q_rot = q_flat @ rotation.T

    # ------------------------------------------------------------
    # Reference score
    #
    # bit 0 -> centroids[0]
    # bit 1 -> centroids[1]
    # ------------------------------------------------------------
    indices = (k_norm_bhtd >= 0).long()

    reconstructed_norm = centroids[indices]

    ref_scores = torch.einsum(
        "bhd,bhtd->bht",
        q_rot,
        reconstructed_norm,
    )

    ref_scores = (
        ref_scores *
        mse_norms
    ).unsqueeze(2)

    # ------------------------------------------------------------
    # CUDA LUT score
    # ------------------------------------------------------------
    lut_scores = turboquant_mse_lut_1bit_score_transposed_cuda(
        q_rot=q_rot,
        packed_mse_sign_bits_t=packed_signs_t,
        mse_norms=mse_norms,
        centroids=centroids,
    )

    diff = torch.abs(
        ref_scores - lut_scores
    )

    print("========== TurboQuant 1-bit MSE LUT score parity ==========")
    print(f"ref_scores.shape = {tuple(ref_scores.shape)}")
    print(f"lut_scores.shape = {tuple(lut_scores.shape)}")
    print(f"max_abs_diff     = {float(diff.max().item()):.6e}")
    print(f"mean_abs_diff    = {float(diff.mean().item()):.6e}")

    assert float(diff.max().item()) < 5e-5

    print("[PASS] TurboQuant 1-bit MSE LUT score parity passed.")


if __name__ == "__main__":
    main()
