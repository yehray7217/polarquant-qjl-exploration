from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from turboquant.mse_quant import (
    make_random_rotation,
    get_2bit_centroids,
)
from turboquant.qjl import (
    make_gaussian_sketch,
)
from turboquant.prod_quant import (
    turboquant_prod_quantize_3bit,
    turboquant_prod_quantize_3bit_runtime_packed_qjl,
)
from turboquant.cuda_packing import (
    pack_2bit_indices_cuda,
    pack_sign_bits_cuda,
)


@torch.no_grad()
def main():
    device = "cuda:0"
    dtype = torch.float32

    torch.manual_seed(0)

    d = 128
    m = 256
    n = 4096

    x = torch.randn(
        n,
        d,
        device=device,
        dtype=dtype,
    )

    rotation = make_random_rotation(
        d=d,
        device=device,
        dtype=dtype,
        seed=123,
    )

    centroids = get_2bit_centroids(
        d=d,
        device=device,
        dtype=dtype,
    )

    sketch = make_gaussian_sketch(
        d=d,
        m=m,
        device=device,
        dtype=dtype,
        seed=456,
    )

    # ============================================================
    # Reference path
    # ============================================================
    ref = turboquant_prod_quantize_3bit(
        x=x,
        rotation=rotation,
        centroids=centroids,
        sketch=sketch,
    )

    ref_packed_mse = pack_2bit_indices_cuda(
        ref.mse.indices.contiguous()
    )

    ref_packed_qjl = pack_sign_bits_cuda(
        ref.qjl_residual.sign_bits.contiguous()
    )

    # ============================================================
    # Runtime fused path
    # ============================================================
    runtime = turboquant_prod_quantize_3bit_runtime_packed_qjl(
        x=x,
        rotation=rotation,
        centroids=centroids,
        sketch=sketch,
    )

    # ============================================================
    # Parity
    # ============================================================
    mse_packed_diff = torch.abs(
        ref_packed_mse.to(torch.int16) -
        runtime.packed_mse_indices.to(torch.int16)
    )

    mse_norm_diff = torch.abs(
        ref.mse.norms -
        runtime.mse_norms
    )

    qjl_packed_diff = torch.abs(
        ref_packed_qjl.to(torch.int16) -
        runtime.packed_qjl_sign_bits.to(torch.int16)
    )

    qjl_norm_diff = torch.abs(
        ref.qjl_residual.norms -
        runtime.qjl_residual_norms
    )

    print("========== Runtime fused MSE + packed-QJL parity test ==========")
    print(f"x.shape                         = {tuple(x.shape)}")
    print(f"reference packed MSE shape      = {tuple(ref_packed_mse.shape)}")
    print(f"runtime packed MSE shape        = {tuple(runtime.packed_mse_indices.shape)}")
    print(f"reference packed QJL shape      = {tuple(ref_packed_qjl.shape)}")
    print(f"runtime packed QJL shape        = {tuple(runtime.packed_qjl_sign_bits.shape)}")
    print()
    print(f"packed MSE max diff              = {int(mse_packed_diff.max().item())}")
    print(f"MSE norm max diff                = {float(mse_norm_diff.max().item()):.6e}")
    print(f"packed QJL sign max diff         = {int(qjl_packed_diff.max().item())}")
    print(f"QJL residual norm max diff       = {float(qjl_norm_diff.max().item()):.6e}")

    assert int(mse_packed_diff.max().item()) == 0
    assert float(mse_norm_diff.max().item()) == 0.0
    assert int(qjl_packed_diff.max().item()) == 0
    assert float(qjl_norm_diff.max().item()) == 0.0

    print()
    print("[PASS] Runtime fused MSE + packed-QJL parity test passed.")


if __name__ == "__main__":
    main()