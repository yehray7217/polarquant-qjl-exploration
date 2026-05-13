from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from turboquant.key_cache import TurboQuantKeyCache
from turboquant.mse_quant import (
    make_random_rotation,
    get_2bit_centroids,
)
from turboquant.qjl import (
    make_gaussian_sketch,
)
from turboquant.cuda_score import (
    turboquant_decode_score_cuda_from_cache,
)
from turboquant.cuda_score_transposed import (
    turboquant_decode_score_transposed_cuda,
)


@torch.no_grad()
def score_ref_from_cache(
    *,
    q_fp32: torch.Tensor,
    cache: TurboQuantKeyCache,
) -> torch.Tensor:
    return turboquant_decode_score_cuda_from_cache(
        query_states=q_fp32,
        cache=cache,
        layer_idx=0,
    )


@torch.no_grad()
def score_transposed(
    *,
    q: torch.Tensor,
    rotation: torch.Tensor,
    sketch: torch.Tensor,
    centroids: torch.Tensor,
    packed_mse_t: torch.Tensor,
    mse_norms: torch.Tensor,
    packed_qjl_t: torch.Tensor,
    qjl_norms: torch.Tensor,
) -> torch.Tensor:
    q_flat = q.to(torch.float32).squeeze(2)

    q_rot = q_flat @ rotation.T
    q_sketch = q_flat @ sketch.T

    return turboquant_decode_score_transposed_cuda(
        q_rot=q_rot,
        q_sketch=q_sketch,
        packed_mse_indices_t=packed_mse_t,
        mse_norms=mse_norms,
        packed_qjl_sign_bits_t=packed_qjl_t,
        qjl_residual_norms=qjl_norms,
        centroids=centroids,
    )


def print_diff(name: str, a: torch.Tensor, b: torch.Tensor) -> None:
    diff = torch.abs(a - b)
    print(f"----- {name} -----")
    print(f"ref abs max        = {float(a.abs().max().item()):.6e}")
    print(f"transposed abs max = {float(b.abs().max().item()):.6e}")
    print(f"max_abs_diff       = {float(diff.max().item()):.6e}")
    print(f"mean_abs_diff      = {float(diff.mean().item()):.6e}")
    print()


@torch.no_grad()
def main():
    device = "cuda:0"
    dtype = torch.float16

    B = 1
    H = 32
    T = 4096
    D = 128
    M = 256

    torch.manual_seed(0)

    rotation = make_random_rotation(
        d=D,
        device=device,
        dtype=torch.float32,
        seed=123,
    )

    centroids = get_2bit_centroids(
        d=D,
        device=device,
        dtype=torch.float32,
    )

    sketch = make_gaussian_sketch(
        d=D,
        m=M,
        device=device,
        dtype=torch.float32,
        seed=456,
    )

    cache = TurboQuantKeyCache(
        num_layers=1,
        rotation=rotation,
        centroids=centroids,
        sketch=sketch,
        max_cache_len=T,
    )

    k = torch.randn(
        B,
        H,
        T,
        D,
        device=device,
        dtype=dtype,
    )

    cache.append(
        layer_idx=0,
        key_states=k,
        value_states=None,
    )

    q = torch.randn(
        B,
        H,
        1,
        D,
        device=device,
        dtype=dtype,
    )
    q_fp32 = q.to(torch.float32)

    layer = cache.layers[0]

    packed_mse = layer.packed_mse_indices_buffer[:, :, :T, :]
    packed_qjl = layer.packed_qjl_sign_bits_buffer[:, :, :T, :]

    packed_mse_t = packed_mse.permute(0, 1, 3, 2).contiguous()
    packed_qjl_t = packed_qjl.permute(0, 1, 3, 2).contiguous()

    mse_norms_orig = layer.mse_norms_buffer[:, :, :T].contiguous()
    qjl_norms_orig = layer.qjl_residual_norms_buffer[:, :, :T].contiguous()

    zeros_mse = torch.zeros_like(mse_norms_orig)
    zeros_qjl = torch.zeros_like(qjl_norms_orig)

    # ============================================================
    # Case A: Full score
    # ============================================================
    ref_full = score_ref_from_cache(
        q_fp32=q_fp32,
        cache=cache,
    )

    t_full = score_transposed(
        q=q,
        rotation=rotation,
        sketch=sketch,
        centroids=centroids,
        packed_mse_t=packed_mse_t,
        mse_norms=mse_norms_orig,
        packed_qjl_t=packed_qjl_t,
        qjl_norms=qjl_norms_orig,
    )

    # ============================================================
    # Case B: MSE-only
    #
    # Temporarily zero cache QJL norms so existing kernel
    # also computes MSE-only.
    # ============================================================
    qjl_backup = layer.qjl_residual_norms_buffer[:, :, :T].clone()
    layer.qjl_residual_norms_buffer[:, :, :T].zero_()

    ref_mse_only = score_ref_from_cache(
        q_fp32=q_fp32,
        cache=cache,
    )

    layer.qjl_residual_norms_buffer[:, :, :T].copy_(qjl_backup)

    t_mse_only = score_transposed(
        q=q,
        rotation=rotation,
        sketch=sketch,
        centroids=centroids,
        packed_mse_t=packed_mse_t,
        mse_norms=mse_norms_orig,
        packed_qjl_t=packed_qjl_t,
        qjl_norms=zeros_qjl,
    )

    # ============================================================
    # Case C: QJL-only
    #
    # Temporarily zero cache MSE norms so existing kernel
    # also computes QJL-only.
    # ============================================================
    mse_backup = layer.mse_norms_buffer[:, :, :T].clone()
    layer.mse_norms_buffer[:, :, :T].zero_()

    ref_qjl_only = score_ref_from_cache(
        q_fp32=q_fp32,
        cache=cache,
    )

    layer.mse_norms_buffer[:, :, :T].copy_(mse_backup)

    t_qjl_only = score_transposed(
        q=q,
        rotation=rotation,
        sketch=sketch,
        centroids=centroids,
        packed_mse_t=packed_mse_t,
        mse_norms=zeros_mse,
        packed_qjl_t=packed_qjl_t,
        qjl_norms=qjl_norms_orig,
    )

    print("========== Transposed CUDA score component parity ==========")
    print_diff("FULL SCORE", ref_full, t_full)
    print_diff("MSE-ONLY", ref_mse_only, t_mse_only)
    print_diff("QJL-ONLY", ref_qjl_only, t_qjl_only)

    full_diff = torch.abs(ref_full - t_full)
    mse_diff = torch.abs(ref_mse_only - t_mse_only)
    qjl_diff = torch.abs(ref_qjl_only - t_qjl_only)

    print("========== Diagnosis ==========")
    if float(mse_diff.max().item()) < 5e-5:
        print("[OK] MSE transposed branch matches.")
    else:
        print("[BAD] MSE transposed branch mismatches.")

    if float(qjl_diff.max().item()) < 5e-5:
        print("[OK] QJL transposed branch matches.")
    else:
        print("[BAD] QJL transposed branch mismatches.")

    print(f"full max diff = {float(full_diff.max().item()):.6e}")


if __name__ == "__main__":
    main()
