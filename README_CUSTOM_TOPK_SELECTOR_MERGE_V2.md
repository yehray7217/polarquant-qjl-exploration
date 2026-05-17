# Custom candidate selector merge-v2

Replaces the slow repeated-argmax final merge with:

1. Existing local pooling: 128-token group -> top-8 lane maxima.
2. New chunk compression: 256 pooled candidates -> exact top-32 via block bitonic sort.
3. New final merge: at most 1024 reduced candidates/head -> top-128 via one 1024-thread bitonic sort.

This keeps the previously validated local top-8 pooling, but removes the
128-pass full-pool repeated max scan.
