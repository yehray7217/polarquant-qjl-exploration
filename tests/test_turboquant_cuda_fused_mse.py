from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from turboquant.mse_quant import (
    get_2bit_centroids,
)
from turboquant.cuda_mse import (
    mse_assign_pack_reconstruct_rot_cuda,
)
from turboquant.cuda_packing import (
    pack_2bit_indices_cuda,
)


@torch.no_grad()
def main():
    device = "cuda:0"
    dtype = torch.float32

    torch.manual_seed(0)

    N = 4096
    D = 128

    x_norm = torch.randn(
        N,
        D,
        device=device,
        dtype=dtype,
    )

    norms = torch.rand(
        N,
        device=device,
        dtype=dtype,
    ) + 1e-3

    centroids = get_2bit_centroids(
        d=D,
        device=device,
        dtype=dtype,
    )

    # ------------------------------------------------------------
    # PyTorch reference
    # ------------------------------------------------------------
    distances = torch.abs(
        x_norm.unsqueeze(-1) - centroids.view(1, 1, 4)
    )

    indices_ref = torch.argmin(
        distances,
        dim=-1,
    ).contiguous()

    packed_ref = pack_2bit_indices_cuda(
        indices_ref
    )

    x_hat_rot_ref = (
        centroids[indices_ref]
        * norms.unsqueeze(-1)
    )

    # ------------------------------------------------------------
    # CUDA fused
    # ------------------------------------------------------------
    packed_cuda, x_hat_rot_cuda = (
        mse_assign_pack_reconstruct_rot_cuda(
            x_norm=x_norm.contiguous(),
            norms=norms.contiguous(),
            centroids=centroids.contiguous(),
        )
    )

    packed_diff = torch.abs(
        packed_ref.to(torch.int16)
        - packed_cuda.to(torch.int16)
    )

    xhat_diff = torch.abs(
        x_hat_rot_ref - x_hat_rot_cuda
    )

    print("========== Fused MSE CUDA parity test ==========")
    print(f"x_norm.shape:                 {tuple(x_norm.shape)}")
    print(f"packed_ref.shape:             {tuple(packed_ref.shape)}")
    print(f"packed_cuda.shape:            {tuple(packed_cuda.shape)}")
    print(f"x_hat_rot_ref.shape:          {tuple(x_hat_rot_ref.shape)}")
    print(f"x_hat_rot_cuda.shape:         {tuple(x_hat_rot_cuda.shape)}")
    print()
    print(f"packed max diff:              {int(packed_diff.max().item())}")
    print(f"x_hat_rot max abs diff:       {float(xhat_diff.max().item()):.6e}")
    print(f"x_hat_rot mean abs diff:      {float(xhat_diff.mean().item()):.6e}")

    assert int(packed_diff.max().item()) == 0
    assert float(xhat_diff.max().item()) == 0.0

    print()
    print("[PASS] Fused MSE CUDA parity test passed.")


if __name__ == "__main__":
    main()