# Custom CUDA candidate selector for Selective-QJL

This patch replaces `torch.topk` in the selective-QJL path with a first custom CUDA candidate selector.

## Selector design

1. Split Polar logits into 128-token groups.
2. One warp handles one group.
3. Each lane scans 4 tokens and keeps its lane maximum.
4. The warp emits the top-8 lane maxima as local candidates.
5. A per-head merge kernel computes exact top-K over the pooled candidates.

For `T=131072`, this reduces selection from:
- original space: 131072 scores/head
- pooled candidate space: 8192 scores/head

Default final K:
- `K=128`

## Important

The local stage is approximate. Evaluate:
- `full_top32_candidate_recall`
- `top32_overlap_vs_reference`
- `top1_agreement_vs_reference`

before treating it as the new mainline selector.
