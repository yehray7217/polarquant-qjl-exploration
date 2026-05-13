from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any


SWEEP_DIR = Path("runs/svd_uniform_08/eval/sweep_long_decode")
OUT_JSON = SWEEP_DIR / "summary_long_decode_sweep.json"
OUT_TSV = SWEEP_DIR / "summary_long_decode_sweep.tsv"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def safe_std(xs: list[float]) -> float:
    return stdev(xs) if len(xs) >= 2 else 0.0


def summarize_group(items: list[dict[str, Any]]) -> dict[str, Any]:
    baseline_tok = [float(x["summary"]["baseline_decode_tok_per_sec"]) for x in items]
    tq_tok = [float(x["summary"]["tq_cuda_decode_tok_per_sec"]) for x in items]
    ratio = [t / b for t, b in zip(tq_tok, baseline_tok)]
    speedup_field = [float(x["summary"]["decode_latency_speedup_fp16_over_tq"]) for x in items]

    baseline_kv = [int(x["summary"]["baseline_final_dense_kv_bytes"]) for x in items]
    tq_kv = [int(x["summary"]["tq_final_total_cache_bytes"]) for x in items]
    tq_k = [int(x["summary"]["tq_final_compressed_k_bytes"]) for x in items]
    tq_v = [int(x["summary"]["tq_final_fp16_v_bytes"]) for x in items]

    cache_ratio = [float(x["summary"]["overall_cache_ratio_tq_over_fp16"]) for x in items]
    cache_reduction = [float(x["summary"]["overall_cache_reduction_percent"]) for x in items]

    return {
        "num_runs": len(items),

        "baseline_decode_tok_per_sec_mean": mean(baseline_tok),
        "baseline_decode_tok_per_sec_std": safe_std(baseline_tok),

        "tq_cuda_decode_tok_per_sec_mean": mean(tq_tok),
        "tq_cuda_decode_tok_per_sec_std": safe_std(tq_tok),

        "tq_over_baseline_throughput_ratio_mean": mean(ratio),
        "tq_over_baseline_throughput_ratio_std": safe_std(ratio),

        "reported_decode_latency_speedup_fp16_over_tq_mean": mean(speedup_field),
        "reported_decode_latency_speedup_fp16_over_tq_std": safe_std(speedup_field),

        "baseline_final_dense_kv_bytes_mean": mean(baseline_kv),
        "tq_final_total_cache_bytes_mean": mean(tq_kv),
        "tq_final_compressed_k_bytes_mean": mean(tq_k),
        "tq_final_fp16_v_bytes_mean": mean(tq_v),

        "overall_cache_ratio_tq_over_fp16_mean": mean(cache_ratio),
        "overall_cache_reduction_percent_mean": mean(cache_reduction),
    }


def main() -> None:
    paths = sorted(SWEEP_DIR.glob("bench_p*_d*_r*.json"))

    if not paths:
        raise FileNotFoundError(f"No sweep JSON files found in {SWEEP_DIR}")

    groups: dict[tuple[int, int], list[dict[str, Any]]] = {}

    for path in paths:
        stem = path.stem
        # Example:
        # bench_p2048_d128_r2
        parts = stem.split("_")
        p = int(parts[1][1:])
        d = int(parts[2][1:])

        obj = load_json(path)
        groups.setdefault((p, d), []).append(obj)

    summary_rows: list[dict[str, Any]] = []

    for (prompt_len, decode_steps), items in sorted(groups.items()):
        row = {
            "prompt_len": prompt_len,
            "decode_steps": decode_steps,
        }
        row.update(summarize_group(items))
        summary_rows.append(row)

    full_summary = {
        "sweep_dir": str(SWEEP_DIR),
        "num_groups": len(summary_rows),
        "groups": summary_rows,
    }

    with OUT_JSON.open("w") as f:
        json.dump(full_summary, f, indent=2)

    columns = [
        "prompt_len",
        "decode_steps",
        "num_runs",
        "baseline_decode_tok_per_sec_mean",
        "baseline_decode_tok_per_sec_std",
        "tq_cuda_decode_tok_per_sec_mean",
        "tq_cuda_decode_tok_per_sec_std",
        "tq_over_baseline_throughput_ratio_mean",
        "tq_over_baseline_throughput_ratio_std",
        "overall_cache_reduction_percent_mean",
        "baseline_final_dense_kv_bytes_mean",
        "tq_final_total_cache_bytes_mean",
    ]

    with OUT_TSV.open("w") as f:
        f.write("\t".join(columns) + "\n")
        for row in summary_rows:
            f.write("\t".join(str(row[c]) for c in columns) + "\n")

    print("========== TurboQuant Long-Decode Sweep Summary ==========")
    for row in summary_rows:
        print()
        print(
            f"prompt={row['prompt_len']}, "
            f"decode={row['decode_steps']}, "
            f"runs={row['num_runs']}"
        )
        print(
            f"  baseline tok/s = "
            f"{row['baseline_decode_tok_per_sec_mean']:.4f} "
            f"± {row['baseline_decode_tok_per_sec_std']:.4f}"
        )
        print(
            f"  TQ CUDA tok/s  = "
            f"{row['tq_cuda_decode_tok_per_sec_mean']:.4f} "
            f"± {row['tq_cuda_decode_tok_per_sec_std']:.4f}"
        )
        print(
            f"  TQ / baseline  = "
            f"{row['tq_over_baseline_throughput_ratio_mean']:.4f} "
            f"± {row['tq_over_baseline_throughput_ratio_std']:.4f}"
        )
        print(
            f"  KV reduction   = "
            f"{row['overall_cache_reduction_percent_mean']:.2f}%"
        )

    print()
    print(f"[Save] {OUT_JSON}")
    print(f"[Save] {OUT_TSV}")
    print("[PASS] Sweep summary completed.")


if __name__ == "__main__":
    main()

