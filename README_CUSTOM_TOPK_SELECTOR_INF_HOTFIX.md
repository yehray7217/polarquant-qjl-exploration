# Custom top-K selector compile hotfix

Fixes NVCC compilation failure:
`identifier "CUDART_INF_F" is undefined`.

Change:
- `-CUDART_INF_F` -> `-INFINITY`
- adds `<cmath>` include

Only the CUDA source file is replaced.
