# NVFP4 Optimization Summary - Sunbathing under the GPT Sol

## Optimizations implemented

- Clustered large-T TMA launches:
  - Pure Linear: **5.0-5.3% faster** at `T=1024`.
  - LinearAdd: **2.4-3.4% faster** at `T=1024`.
  - AttentionInputProj: **5.2% faster** at `T=1024`.
  - GdnInputProj: **10.3% faster** at `T=1024`.
  - Fused SwiGLU: **6.6-8.7% faster** at `T=512/768/1024`.
- Fused SwiGLU cp.async route extended through `T=96`: **4.3-19.1% faster**.
- Fused SwiGLU TMA at `T=256/512/768`: **8.3-24.4% faster**.
- End-to-end Qwen3.8 NVFP4 prefill improvement: **1.19%** at `T=1024`.

## Rejected

- Direct register-to-global output stores: **1.6-7.0% slower**.
- Two-stage TMA pipeline: **1.6-1.7% slower**.
- Four-stage TMA pipeline: exceeds RTX 5090 shared-memory limits.
- `tcgen05`/TMEM: unsupported on `sm_120a`.
- Inline quantized MLP handoff and eight-chain decode GEMV: no stable benefit.

All retained routes pass numerical correctness and CUDA Graph replay checks.
