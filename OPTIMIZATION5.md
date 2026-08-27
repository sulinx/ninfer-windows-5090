# OPTIMIZATION5: Extending NVFP4-Style TMA / Cluster Optimizations to Non-NVFP4 Paths

This document is a findings-only prospectus for a future optimization campaign on NInfer's
non-NVFP4 execution paths (row-scaled FP8 and the grouped-int formats Q4/Q5/Q6/W8). It does not
propose or record any code change; it captures verified current-source facts so that a later session
can resume with a baseline-first plan. The evidence discipline follows `nvfp4_optimization.md`
(Phase 0): re-establish current-source attribution and measure complete public operations under CUDA
Graph replay, never isolated kernel time.

Status: **research only. Not started. No code modified.**

## 1. Scope and objective

The registered artifacts using non-NVFP4 kernels are the `groupwise-int` identities
(`qwen3.6-27b/groupwise-int`, `qwen3.6-35b-a3b/groupwise-int`, `qwen3.8-27b/groupwise-int`) and the
row-scaled FP8 projections inside `qwen3.8-27b/nvfp4` (attention and GDN projections use FP8; MLP
gate/up and down use NVFP4 — `docs/maintainer/qwen3.8-27b-artifact.md:69-100`).

The objective would be to apply the same hardware-level latency wins that NVFP4 already measured to
these other quant formats, without changing model semantics. This document records:
- what NVFP4 gained and the mechanism (evidence already held);
- the current non-NVFP4 kernel structure and its gap versus NVFP4;
- the feasibility of carrying TMA / cluster to each format given the persistent layouts;
- a prioritized, oracle-qualified, graph-replay-gated experiment order.

## 2. Executive summary

NVFP4's recent, retained large-T wins (`nvfp4_optimization.md:640-676`) come from two techniques at
the `T=1024` prefill point:

1. **TMA producer/consumer** — `cp.async.bulk.tensor.2d` with `mbarrier` `expect_tx` completion,
   replacing `cp.async` software staging (`src/ops/linear/nvfp4/nvfp4_w4a4_tma.*`).
2. **1x4x1 cluster placement** — placing the four `T=1024` token CTAs in one cluster, raising L2
   utilization (`cudaLaunchAttributeClusterDimension`; `nvfp4_w4a4_tma.cu:81-124`).

All retained NVFP4 measurements are on this exact hardware/toolchain (RTX 5090, `sm_120a`, CUDA 13.1)
and are qualified under repeated CUDA Graph replay, including the Windows by-value `__grid_constant__`
TMA descriptor carrier.

**The non-NVFP4 quantized GEMM paths use an older generation: `cp.async` staging plus warp-level
`ldmatrix` + `mma`. None of them use TMA or cluster launch.** This is the same structural gap NVFP4
had before its TMA + cluster work.

sm_120a rules out the Blackwell `tcgen05`/TMEM mainloop (CUDA 13.1 exposes it only for `sm_100`,
`sm_103`, `sm_110`; rejected in `nvfp4_optimization.md:280-283`). Therefore the realistic ceiling for
all formats is the same `mma.sync` + TMA + cluster combination NVFP4 uses. This bounds expectations:
TMA/cluster is an incremental win over the current `cp_async` kernels, not a tensor-pipe-class jump.

## 3. Evidence already held (NVFP4 reference)

Retained, complete-operation measurements from `nvfp4_optimization.md` (all at `T=1024`, cold-cache
and/or graph replay; exact line references):

| Scope | Gain | Source |
|---|---|---|
| Pure S3 Linear routes (1x4x1 cluster) | 2.3-4.1% | `nvfp4_optimization.md:649-653` |
| S3 LinearAdd residual / MLP down | 2.4% / 3.4% | `:655-657` |
| AttentionInputProj / GdnInputProj split-output | 5.2% / 10.3% | `:659-661` |
| Fused TMA LinearSwiGLU (T=512/768/1024) | 6.6% / 8.7% / 7.5% | `:663-666` |
| Public Engine prefill (cluster package on/off) | 1.19% (7574.0 vs 7484.9 tok/s pp1024) | `:673-676` |
| Large-T TMA stage sweep (S2/S3/S4) | S2 -1.6/-1.7%; S4 over smem; S3 retained | `:637-640` |

Two rejected experiments bound the future work (`nvfp4_optimization.md`):
- Output staging via direct per-lane BF16x2 global stores regressed 1.6-7.0%; wide/coalesced
  vector stores are required (`:633-635`).
- Native TMA multicast is unavailable on `sm_120a`; the cluster wins through higher L2 utilization,
  not lower traffic, so a DSM software fan-out is not indicated (`:649-653`).

## 4. Current non-NVFP4 kernel inventory

All tensor-phase paths dispatch centrally by `w.qtype` and run inside per-`(batch, frontier)`
captured CUDA Graphs. CUDA Graph capture/replay, `cp_async` staging, `ldmatrix`+`mma`, and per-shape
dispatch are **already common to all formats and are not transfer opportunities**.

### 4.1 FP8 (`FP8_E4M3FN_ROW_BF16S`, layout `RowScaleV1`)

Kernel home `src/ops/linear/fp8/`.

| Route | Mechanism | TMA | Cluster |
|---|---|---|---|
| A16 vocab (`fp8_a16_mma.cuh`) | `cp_async<16>` + `ldmatrix_x2` + `mma_bf16` | No | No |
| A8 production (`fp8_a8_mma.cuh:159,181,218,...`) | `cp_async<16>` + `ldmatrix_x4/x2` + `mma_fp8_e4m3` (`AllowA8`) | No | No |
| Fused SwiGLU (`linear_swiglu/fp8/*`) | Fused epilogue inside A8 mma / small-T | No | No |
| Fused residual (`linear_add/fp8/*`) | Fused residual epilogue mma / small-T | No | No |
| Projections (`attn_input_proj/fp8/*`, `gdn_input_proj/fp8/*`) | Fused QKV / GDN input projections | No | No |

Production A8 schedule (`fp8_a8_schedule.cuh:13-39`):
`Fp8MmaSchedule<64, 128, 128, 2, 4, 2, 2, Cache::cg, Cache::cg, PingPong, ...>`
— `BlockTokens=64`, `BlockRows=128`, `BlockK=128`, 2x4 warps, 2 stages, `MinBlocksPerSm=2`.

A8 engages at `tokens >= 12` (AttnInput), `>= 11` (GdnInput), `== 1 or >= 5` (MlpGateUp), `>= 25`
(Residual6144/Residual17408); Vocabulary stays A16 for every policy
(`fp8_dispatch.cpp:22-40`). Therefore `T=1024` prefill uses the A8 route. At `T=1024`,
`1024/64 = 16` token tiles per row-tile of 128.

### 4.2 W8 (`W8G32_F16S`, layout `RowSplitK128V1`)

Kernel home `src/ops/linear/w8/`. Dominant non-NVFP4 format for 35B-A3B (dense layers and MoE
gate/up; `src/targets/qwen3_6_35b_a3b/impl/variant.cpp:254,271,319`) and qwen3.8 endpoints.

| Route | Mechanism | TMA | Cluster |
|---|---|---|---|
| Large-T MMA (`w8_rowsplit_gemm_mma.cuh`) | 2-stage `cp_async<16>` + `ldmatrix` + `mma_bf16` (W8 decoded to BF16) | No | No |
| Fused SwiGLU epilogue | `w8_rowsplit_gemm_mma.cuh:293-402` (fused) | No | No |
| Fused residual epilogue | `w8_rowsplit_gemm_mma.cuh:403-471` (fused) | No | No |
| Split-K medium-T | `w8_rowsplit_gemm_medium_t_splitk.cuh` | No | No |
| Small-T / SIMT / GEMV | `w8_small_t*.cuh`, `w8_rowsplit_gemm_simt.cuh` (scalar decode, `mma_bf16`) | No | No |

MMA token-tile (`BN`) configs at large T are 64/96/112/128 with row-tile `BM` 32/48/64/128
(`w8_rowsplit_gemm_mma.cu:71-76`); e.g. `MmaR64C128 = W8RowSplitMmaGemmSchedule<64,128,64,16,2>`.
At `T=1024`, `1024/128 = 8` token tiles (or 16 at `BN=64`).

`linear_workspace_capacity_bytes` returns 0 for Q4/Q5/Q6/W8/BF16 (`src/ops/linear/linear.cpp:120-135`)
— a single fused kernel, no secondary workspace. FP8/NVFP4 carry nontrivial workspaces.

### 4.3 Q4 / Q5 / Q6 (`Q4G64_F16S` / `Q5G64_F16S` / `Q6G64_F16S`, layout `RowSplitK128V1`)

Kernel homes `src/ops/linear/q4/`, `q5/`, `q6/`. Used by 27B `groupwise-int` internals and 35B-A3B
MoE (routed experts: Q4/Q5 gate/up/down; Q6 output/vocabulary).

| Route | Mechanism | TMA | Cluster |
|---|---|---|---|
| `T==1` GEMV | Scalar CUDA-core nibble/fp16-mantissa decode, FP32 FMA, `cp_async`/PDL | No | No |
| Small-T SIMT | Scalar warp-per-row, `cp_async` double-buffer | No | No |
| Large-T MMA (`q4/q5/q6_rowsplit_gemm_mma.cuh`) | codes decoded to BF16 shared, `ldmatrix` + `mma_bf16` | No | No |
| Fused SwiGLU / residual | `linear_swiglu/{q4,q5,q6}/*`, `linear_add/{q5,q6}/*` | No | No |
| Attn / GDN input | `attn_input_proj/q4_q5/*` (`A16Only`) | No | No |

These formats have the most dispatch surface (many exact-T variants for the vocabulary head,
e.g. `q6_dispatch.cpp:7-63`), reflecting row-split + grouped scale handling rather than any TMA path.

## 5. Feasibility versus persistent layout

The physical layout determines whether TMA (and thus the proven cluster win) can reuse NVFP4's
producer/consumer structure. Layout geometry is defined in `src/artifact/storage_layouts.cpp`.

- **FP8 (`RowScaleV1`) — clean TMA target.** One contiguous `[rows, K]` code plane plus a per-row
  (`2` bytes/row) scale plane (`storage_layouts.cpp:199-217`). This is a dense byte matrix, *simpler*
  to tensor-map than NVFP4's packed 4-bit nibbles and per-16 scale blocks. A TMA route can bulk-copy
  the code plane and read row-scales directly; cluster placement at `T=1024` is a direct, low-effort
  first test since the A8 kernel already phases `T=1024` into 16 token tiles.

- **W8 / Q4 / Q5 / Q6 (`RowSplitK128V1`) — multi-plane challenge.** A low code plane, an optional
  high plane (Q5/Q6), and a separate 2-byte-per-group scale plane, with K padded to 128
  (`storage_layouts.cpp:129-168`). TMA applies naturally only to contiguous byte planes; the scale
  handling and the row-split groups mean either multi-plane TMA or TMA-for-codes + staged scales, and
  the current kernels deliberately decode to BF16 in shared memory (`mma_bf16`). File-level facts:
  * the mma kernel is a single fused kernel (no secondary workspace) and threads one token tile
    (`BM token = BN=64..128`) per block;
  * 1x4x1 cluster would group 4 token CTAs exactly like NVFP4 at `T=1024`.

- **NVFP4 (`BlockScaleK16M128x4V1`) — reference.** The only layout with a native 4-bit path
  (`storage_layouts.cpp:170-197`); its TMA descriptor construction (`nvfp4_w4a4_tma.cuh:56-64`) and
  Windows `__grid_constant__` byte carrier (`nvfp4_w4a4_tma.cuh:31-41`, `nvfp4_w4a4_tma.cu:77-81`)
  are the pattern any newly-added TMA route must reproduce for graph-replay safety.

### Windows / CUDA-Graph constraints for any new route

- Tensor-map descriptors must be passed by value in grid-constant storage; a host pointer is not
  replay-safe (`nvfp4_optimization.md:521-524`). Keep the `#ifdef _WIN32` carrier consistent between
  any `.cuh` kernel and launch site.
- All newly-added work must survive repeated CUDA Graph capture/replay (per-`(batch, frontier)`
  graphs); the workspace/addressing ownership rules in `nvfp4_optimization.md:568-571` apply.

## 6. Prioritized experiment order

Followed the NVFP4 Phase 0 discipline. Each step must measure complete public operations (including
activation quantization and post-processing where the format has it) at token points
`1, 2, 4, 5, 7, 8, 16, 17, 32, 33, 48, 49, 128, 256, 512, 768, 1024`, under both cold-cache operator
bench and CUDA Graph replay. Never tune from isolated kernel time.

1. **Current-source baseline.** Capture a route table of current winners and an attribution summary
   (`quantization | contraction | epilogue/post | launch/API`) for FP8 A8 and W8 large-T, at the
   token points above, on RTX 5090 / CUDA 13.1 / this Windows build.
2. **Low-effort 1x4x1 cluster on FP8 A8 (`T=1024`).** Reuse the proven cluster launch mechanism
   (`cudaLaunchKernelEx` + `cudaLaunchAttributeClusterDimension`) on the existing `fp8_mma_kernel`
   without a new memory pipeline. Gate: complete-Op improvement + graph replay + oracle.
3. **Cluster on W8 large-T MMA (`T=1024`).** Same probe on `w8_rowsplit_gemm_mma_kernel`. Gate:
   complete-Op improvement + graph replay + oracle. Expect the grouped-plane kernel to interact
   differently with L2 than NVFP4 (fewer token tiles, 8 vs 4).
4. **Full TMA producer/consumer for FP8 A8.** Port the NVFP4 structure (bulk tensor copy, mbarrier
   `expect_tx`, `__grid_constant__` carrier) to the clean `RowScaleV1` layout. Candidate geometry:
   AttnInput and GdnInput (split output), Residual6144, Residual17408. Gate as in `nvfp4_optimization.md` Phase 1.
5. **Full TMA for W8/groupwise** only if a step-3/4 win and profiler evidence support the extra
   multi-plane complexity (scale handling and K-pad 128). This is the highest-effort item and must
   not be attempted ahead of evidence.

Do not consider a persistent-layout change for any format unless profiler evidence proves the code
or scale layout remains a dominant bottleneck after the kernel-local and TMA/cluster work
(`nvfp4_optimization.md:538-554`).

## 7. Correctness and acceptance gates

Every accepted change must preserve or requalify (mirroring `nvfp4_optimization.md:555-611`):

- exact numeric-format decode (E4M3 for FP8; Q4/Q5/Q6 nibble/high-plane + fp16-mantissa; W8 group
  scale) with an independent oracle from represented public inputs and exact stored scales;
- BF16 semantic input/output boundaries of each public Op (the grouped-int weights dequantize to
  represent the BF16 math; a TMA route must not change the accumulation profile without requalifying);
- fused SwiGLU gate/up pairing and residual ordering;
- GDN convolution/publication/recurrent-state behavior where routes write split outputs;
- caller-owned workspace lifetime and address stability under CUDA Graphs;
- Windows by-value `__grid_constant__` TMA descriptor semantics;
- partial token-tile handling at non-multiple T and route-boundary continuity across `A16/A8`,
  `A16Only`, and mma/SIMT/GEMV crossovers.

Acceptance requires complete-operation or Engine-level improvement at the claimed scope; an isolated
kernel microbenchmark gain is insufficient.

## 8. Sources

- `nvfp4_optimization.md` (measured NVFP4 reference, Phase 0 discipline, gates).
- `src/artifact/storage_layouts.cpp` (persistent layouts: `RowScaleV1` :199-217, `RowSplitK128V1`
  :129-168, `BlockScaleK16M128x4V1` :170-197).
- `src/ops/linear/fp8/fp8_a8_mma.cuh`, `fp8_a8_schedule.cuh`, `fp8_dispatch.cpp`.
- `src/ops/linear/w8/w8_rowsplit_gemm_mma.cuh`, `w8_rowsplit_gemm_mma.cu`.
- `src/ops/linear/{q4,q5,q6}/` and `linear_swiglu/{fp8,w8,q4,q5,q6}/`, `linear_add/{fp8,w8,q5,q6}/`,
  `attn_input_proj/{fp8,w8,q4_q5}/`, `gdn_input_proj/{fp8,w8,q4_q5}/`.
- `src/ops/linear/nvfp4/nvfp4_w4a4_tma.cu`, `nvfp4_w4a4_tma.cuh` (TMA + cluster + Windows descriptor
  reference).
- `src/targets/qwen3_6_35b_a3b/impl/variant.cpp` (W8/Q4/Q5/Q6 binding).
