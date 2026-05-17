# Algorithmic CUDA exploration patch

This patch directly implements two algorithm-level CUDA experiments on top of the current best PolarQuant score path.

## A. Reduced QJL dimension, full fused kernels

- Reference: current M=128 fused score path.
- Candidates:
  - `M=64` compact QJL sign layout, **8 bytes/token**.
  - `M=32` compact QJL sign layout, **4 bytes/token**.
- Polar tree and factor LUT remain unchanged.
- Bench reports speed and score-quality metrics versus the M=128 score reference.

## B. Polar top-K selective QJL refinement

- Full sequence Polar-only logits: existing custom CUDA kernel.
- Candidate generation: `torch.topk` CUDA on Polar logits.
- Refinement: new custom CUDA kernel computing M=128 QJL residual correction only on selected top-K token indices.
- Bench reports full pipeline time, refinement-only time, and quality against full fused M=128 logits.

The selective path uses PyTorch's CUDA top-k selector rather than a custom selection kernel. The new contribution in this patch is the custom selected-QJL CUDA refinement kernel and the end-to-end timing harness.
