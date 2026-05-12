# Generic KV Quantization Detour

This directory contains experiments that were created while exploring:
- uniform int8/int4 KV cache quantization
- custom compressed KV cache classes
- CUDA dequantization kernels
- fused compressed attention score kernels
- LLaMA attention monkey-patching

These experiments are NOT the main TurboQuant-on-SVD implementation.
They are preserved as potential baselines or future reference code.
