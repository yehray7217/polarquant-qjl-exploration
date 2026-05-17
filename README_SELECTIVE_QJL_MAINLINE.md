# Selective-QJL mainline

This patch promotes the algorithmic path

1. Polar-only retrieval
2. top-K candidate selection
3. selected QJL refinement

to an explicit mainline wrapper and benchmark.

## New wrapper

`turboquant/selective_qjl_pipeline.py`

It provides:

- `selective_qjl_sparse_topk_m128_cuda(...)`
  - preferred retrieval/rerank representation
  - returns dense Polar logits plus sparse selected refined logits

- `selective_qjl_dense_logits_topk_m128_cuda(...)`
  - compatibility helper
  - scatters selected refined logits back into a dense tensor

## New benchmark

`tests/bench_polarquant_selective_qjl_mainline.py`

Default:
- `topk=128`
- `quality_topk=32`

It compares:
- previous full fused Polar+QJL score kernel
- selective sparse pipeline
- selective dense-materialized compatibility path

## Current selection stage

Top-K candidate selection still uses `torch.topk` CUDA.
Prior experiments showed this is the next major bottleneck after the
selected QJL refinement kernel became negligible.
