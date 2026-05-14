from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from turboquant.polarquant import (
    recursive_polar_encode,
)
from turboquant.polarquant_quant import (
    DEFAULT_POLAR_BITS_BY_LEVEL,
    fit_polar_angle_codebooks_from_encodings,
)
from turboquant.polar_prod import (
    turboquant_polar_prod_quantize,
)
from turboquant.qjl import (
    make_gaussian_sketch,
)
from turboquant.polar_score_cuda import (
    polar_stage1_score_cuda,
)
from turboquant.qjl_score_cuda import (
    qjl_packed_score_cuda,
)
from turboquant.turboquant_logits_cuda import (
    turboquant_fused_logits_cuda,
)
from turboquant.packed_meta import (
    build_turboquant_packed_meta_blob,
)


@torch.no_grad()
def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    device = "cuda:0"

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    B = 1
    H = 32
    Q = 1
    T = 65
    D = 128
    M = 128
    L = 4
    N_calib = 4096

    # ------------------------------------------------------------
    # Build Polar codebooks
    # ------------------------------------------------------------
    x_calib = torch.randn(
        N_calib,
        D,
        device=device,
        dtype=torch.float32,
    )

    enc_calib = recursive_polar_encode(
        x_calib,
        num_levels=L,
    )

    codebooks = fit_polar_angle_codebooks_from_encodings(
        [enc_calib],
        bits_by_level=DEFAULT_POLAR_BITS_BY_LEVEL,
        max_iters=30,
        max_samples_per_level=200_000,
        seed=0,
    )

    # ------------------------------------------------------------
    # Build QJL sketch
    # ------------------------------------------------------------
    sketch = make_gaussian_sketch(
        d=D,
        m=M,
        device=device,
        dtype=torch.float32,
        seed=123,
    )

    # ------------------------------------------------------------
    # Synthetic K-cache and query
    # ------------------------------------------------------------
    k = torch.randn(
        B,
        H,
        T,
        D,
        device=device,
        dtype=torch.float32,
    )

    q = torch.randn(
        B,
        H,
        Q,
        D,
        device=device,
        dtype=torch.float32,
    )

    encoding = turboquant_polar_prod_quantize(
        x=k,
        codebooks=codebooks,
        sketch=sketch,
        num_levels=L,
    )

    packed = encoding.polar.packed_angles

    packed_qjl_signs = encoding.qjl_residual.packed_sign_bits.reshape(
        B,
        H,
        T,
        M // 8,
    )

    qjl_norms = encoding.qjl_residual.norms.reshape(
        B,
        H,
        T,
    )

    packed_meta = build_turboquant_packed_meta_blob(
        packed_l1=packed.level1_4bit,
        packed_l2=packed.level2_2bit,
        packed_l3=packed.level3_2bit,
        packed_l4=packed.level4_2bit,
        packed_qjl_signs=packed_qjl_signs,
    )

    q_projected = torch.matmul(
        q.to(torch.float32),
        sketch.T.to(torch.float32),
    )

    # ------------------------------------------------------------
    # Reference = separate kernels + add
    # ------------------------------------------------------------
    polar_scores = polar_stage1_score_cuda(
        q=q,
        packed_l1=packed.level1_4bit,
        packed_l2=packed.level2_2bit,
        packed_l3=packed.level3_2bit,
        packed_l4=packed.level4_2bit,
        radii=encoding.polar.radii,
        centroids_l1=codebooks.centroids[0],
        centroids_l2=codebooks.centroids[1],
        centroids_l3=codebooks.centroids[2],
        centroids_l4=codebooks.centroids[3],
    )

    qjl_scores = qjl_packed_score_cuda(
        q_projected=q_projected,
        packed_signs=packed_qjl_signs,
        norms=qjl_norms,
    )

    scores_ref = (
        polar_scores
        + qjl_scores
    )

    # ------------------------------------------------------------
    # Fused kernel with packed_meta fast path
    # ------------------------------------------------------------
    scores_fused = turboquant_fused_logits_cuda(
        q=q,
        q_projected=q_projected,

        packed_l1=packed.level1_4bit,
        packed_l2=packed.level2_2bit,
        packed_l3=packed.level3_2bit,
        packed_l4=packed.level4_2bit,
        radii=encoding.polar.radii,

        centroids_l1=codebooks.centroids[0],
        centroids_l2=codebooks.centroids[1],
        centroids_l3=codebooks.centroids[2],
        centroids_l4=codebooks.centroids[3],

        packed_qjl_signs=packed_qjl_signs,
        qjl_norms=qjl_norms,

        packed_meta=packed_meta,
    )

    diff = torch.abs(
        scores_ref.to(torch.float32)
        - scores_fused.to(torch.float32)
    )

    print("========== Fused TurboQuant final logits CUDA parity ==========")
    print(f"scores_ref.shape     = {tuple(scores_ref.shape)}")
    print(f"scores_fused.shape   = {tuple(scores_fused.shape)}")
    print(f"max_abs_diff         = {float(diff.max().item()):.6e}")
    print(f"mean_abs_diff        = {float(diff.mean().item()):.6e}")

    assert tuple(scores_fused.shape) == (
        B,
        H,
        Q,
        T,
    )

    assert float(diff.max().item()) < 5e-5

    print("[PASS] Fused TurboQuant final logits CUDA parity passed.")


if __name__ == "__main__":
    main()