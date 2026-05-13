from pathlib import Path
import sys
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

import turboquant_cuda


def elapsed_ms(fn, warmup: int = 10, iters: int = 100):
    """
    GPU-time measurement using CUDA events.
    """
    for _ in range(warmup):
        fn()

    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn()
    end.record()

    torch.cuda.synchronize()

    total_ms = start.elapsed_time(end)
    return total_ms / iters


@torch.no_grad()
def main():
    device = "cuda:0"

    # Match long decode benchmark scale approximately:
    # prompt_len=2048 + decode tail ~= 2176
    B = 1
    H = 32
    T = 2176
    D = 128
    M = 256

    packed_D = D // 4
    packed_M = M // 8

    torch.manual_seed(0)

    # ------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------
    q = torch.randn(
        B,
        H,
        D,
        device=device,
        dtype=torch.float32,
    )

    rotation = torch.randn(
        D,
        D,
        device=device,
        dtype=torch.float32,
    )

    sketch = torch.randn(
        M,
        D,
        device=device,
        dtype=torch.float32,
    )

    combined_query_transform = torch.cat(
        [rotation, sketch],
        dim=0,
    ).contiguous()

    packed_mse_indices = torch.randint(
        low=0,
        high=256,
        size=(B, H, T, packed_D),
        device=device,
        dtype=torch.uint8,
    )

    mse_norms = torch.rand(
        B,
        H,
        T,
        device=device,
        dtype=torch.float32,
    )

    packed_qjl_sign_bits = torch.randint(
        low=0,
        high=256,
        size=(B, H, T, packed_M),
        device=device,
        dtype=torch.uint8,
    )

    residual_norms = torch.rand(
        B,
        H,
        T,
        device=device,
        dtype=torch.float32,
    )

    centroids = torch.tensor(
        [-0.1334664, -0.0400399, 0.0400399, 0.1334664],
        device=device,
        dtype=torch.float32,
    )

    # ------------------------------------------------------------
    # Query transform variants
    # ------------------------------------------------------------

    def run_q_rot():
        return torch.matmul(
            q,
            rotation.T,
        )

    def run_sq():
        return torch.matmul(
            q,
            sketch.T,
        )

    def run_combined_transform():
        combined = torch.matmul(
            q,
            combined_query_transform.T,
        )

        q_rot_local = combined[..., :D]
        sq_local = combined[..., D:]

        return q_rot_local, sq_local

    # ------------------------------------------------------------
    # Build score-kernel inputs
    # ------------------------------------------------------------

    q_rot_separate = run_q_rot().contiguous()
    sq_separate = run_sq().contiguous()

    q_rot_combined, sq_combined = run_combined_transform()
    q_rot_combined = q_rot_combined.contiguous()
    sq_combined = sq_combined.contiguous()

    # Sanity: the two ways should be numerically identical or extremely close.
    q_rot_diff = torch.abs(q_rot_separate - q_rot_combined)
    sq_diff = torch.abs(sq_separate - sq_combined)

    q_rot_max_abs_diff = q_rot_diff.max().item()
    q_rot_mean_abs_diff = q_rot_diff.mean().item()

    sq_max_abs_diff = sq_diff.max().item()
    sq_mean_abs_diff = sq_diff.mean().item()

    # ------------------------------------------------------------
    # Score-kernel calls
    # ------------------------------------------------------------

    def run_cuda_score_separate_transform_inputs():
        return turboquant_cuda.turboquant_decode_score(
            q_rot,
            sq,
            packed_mse_indices,
            mse_norms,
            packed_qjl_sign_bits,
            residual_norms,
            centroids,
            T,
        )

    def run_cuda_score_combined_transform_inputs():
        return turboquant_cuda.turboquant_decode_score(
            q_rot,
            sq,
            packed_mse_indices,
            mse_norms,
            packed_qjl_sign_bits,
            residual_norms,
            centroids,
            T,
        )

    # ------------------------------------------------------------
    # Timing
    # ------------------------------------------------------------

    q_rot_ms = elapsed_ms(run_q_rot)
    sq_ms = elapsed_ms(run_sq)
    separate_query_transform_sum_ms = q_rot_ms + sq_ms

    combined_query_transform_ms = elapsed_ms(run_combined_transform)

    packed_score_kernel_ms = elapsed_ms(
        run_cuda_score_combined_transform_inputs
    )

    separate_score_path_ms = (
        separate_query_transform_sum_ms
        + packed_score_kernel_ms
    )

    combined_score_path_ms = (
        combined_query_transform_ms
        + packed_score_kernel_ms
    )

    transform_speedup_separate_over_combined = (
        separate_query_transform_sum_ms / combined_query_transform_ms
        if combined_query_transform_ms > 0 else None
    )

    total_score_path_speedup_separate_over_combined = (
        separate_score_path_ms / combined_score_path_ms
        if combined_score_path_ms > 0 else None
    )

    # ------------------------------------------------------------
    # Optional score equality sanity
    # ------------------------------------------------------------

    scores_from_separate = run_cuda_score_separate_transform_inputs()
    scores_from_combined = run_cuda_score_combined_transform_inputs()

    score_diff = torch.abs(
        scores_from_separate - scores_from_combined
    )

    score_max_abs_diff = score_diff.max().item()
    score_mean_abs_diff = score_diff.mean().item()

    # ------------------------------------------------------------
    # Report
    # ------------------------------------------------------------

    result = {
        "shape": {
            "B": B,
            "H": H,
            "T": T,
            "D": D,
            "M": M,
            "packed_D": packed_D,
            "packed_M": packed_M,
        },
        "query_transform_numerical_parity": {
            "q_rot_max_abs_diff": q_rot_max_abs_diff,
            "q_rot_mean_abs_diff": q_rot_mean_abs_diff,
            "sq_max_abs_diff": sq_max_abs_diff,
            "sq_mean_abs_diff": sq_mean_abs_diff,
        },
        "score_numerical_parity": {
            "score_max_abs_diff": score_max_abs_diff,
            "score_mean_abs_diff": score_mean_abs_diff,
        },
        "timing_ms_per_call": {
            "q_rot_matmul_ms": q_rot_ms,
            "sq_matmul_ms": sq_ms,
            "separate_query_transform_sum_ms": separate_query_transform_sum_ms,
            "combined_query_transform_ms": combined_query_transform_ms,
            "packed_score_kernel_ms": packed_score_kernel_ms,
            "separate_total_score_path_ms": separate_score_path_ms,
            "combined_total_score_path_ms": combined_score_path_ms,
        },
        "speedup": {
            "query_transform_speedup_separate_over_combined": (
                transform_speedup_separate_over_combined
            ),
            "total_score_path_speedup_separate_over_combined": (
                total_score_path_speedup_separate_over_combined
            ),
        },
        "fraction_of_combined_score_path": {
            "combined_query_transform": (
                combined_query_transform_ms / combined_score_path_ms
                if combined_score_path_ms > 0 else None
            ),
            "packed_score_kernel": (
                packed_score_kernel_ms / combined_score_path_ms
                if combined_score_path_ms > 0 else None
            ),
        },
    }

    print(
        "========== TurboQuant CUDA decode-score component benchmark =========="
    )
    print(json.dumps(result, indent=2))

    out_path = (
        "runs/svd_uniform_08/eval/"
        "bench_turboquant_cuda_score_components.json"
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print()
    print(f"[Save] {out_path}")
    print("[PASS] CUDA score component benchmark completed.")


if __name__ == "__main__":
    main()