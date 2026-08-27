# NEW-PORT-PROGRESS.md

Tracking for the `windows-port-2` effort: bring all new upstream features, compile and test, then
port our optimizations and compare.

Branch: `windows-port-2` (based on `windows-port`)
Base commit: `a1042ea3`
Upstream merge-base: `a05746aa`
Upstream target: `origin/master` = `fbd04729` (69 new commits, `a05746aa..fbd04729`)
Merge commit: `4ad492ec` (merge of origin/master into windows-port-2, 2026-08-27)

## Phase 1 - Bring upstream new features

Status: COMPLETE (merge commit 4ad492ec)

### New upstream features being brought in
- `6183c9be` feat(attention): add fp8 kv cache support
- `17a7275f` feat(attention): add int8 kv hadamard rotation
- `f7bcd2ba` feat(engine): implement prefix caching resource scheduling
- `3a011c70` feat(engine): provision default host context cache
- `dda31c75` feat(runtime): add measured context cost scheduling
- `a87d1fc6` feat(engine): add qwen thinking budget control
- `94d0ef49` feat(engine): expose host work timing
- `26fe86dd` / `9d4cc6f9` / `c648a132` physical containers + device continuation lifecycle
- `d6af046a` / `020ca885` engine core rewrite (`engine_core.h`, `resource_manager.h`,
  `materialization_planner.h`, `scheduler.h`, replaces `concurrent_executor.h`)
- `a58a946c` / `e9c7046f` ops ownership consolidation (softmax attention, gdn)
- Second-tier fixes/perf (~55 more commits)

### Steps
- [x] Merge `origin/master` into `windows-port-2` (git merge origin/master, no fast-forward) -> 4ad492ec
- [x] Resolve conflicts (4): README.md, api_impl.h (took upstream, dropped dead `RequestPlan` type),
      both GDN replay test files. Took upstream architecture as authoritative for engine/runtime.
- [x] Re-apply / repair Win32 porting bits: audit confirms intact --
      reader.cpp MappedFile (CreateFileW, :188), request_log `_getpid`, console_log `localtime_s`,
      load_progress `_isatty`, acquire winsock2, src/CMakeLists ws2_32+UTF8PROC_STATIC, root
      CMakeLists NOMINMAX(:28), nvfp4 TMA `__grid_constant__` carrier (:25/31/35), api_impl
      explicit-body move ctors (SequencePlan :21, RequestBasePlan :87)
- [x] Verify engine/target wiring: upstream refactor renamed RequestPlan -> RequestBasePlan;
      `RequestBasePlan` + `PressurePlanningSession` explicit/default move members correct

## Phase 2 - Compile and test

Status: IN PROGRESS (root causes found)

- [x] Full Windows build (MSVC 14.41, +sm_120a, CUDA 13.1, static cudart full parallelism)
- [x] Root cause `0xC0000409`: host `.cpp` built without `+EHsc` (exceptions disabled) -> `throw`
      calls `fail-fast`. Fix: `/EHsc` added to base CMake (root CMakeLists MSVC block).
- [x] Root cause `ninfer_qwen3_6_frontend_test` `0xC0000409`: `core.autocrlf=true` converts the
      committed LF chat-template fixtures to CRLF on checkout, breaking the exact-SHA256
      `CompiledChatTemplate::resolve` digest; `create_component` throws invalid_argument, uncaught
      -> fail-fast. Fix: add `.gitattributes` forcing `tests/fixtures/**` to `text eol=lf`, and
      renormalize the working-tree fixture bytes to LF. LF-normalized digests match the constants
      exactly (thinking=e84f32a2..., reasoning=c3cf9e34...).
- [x] Second frontend crash: `test_explicit_leading_instruction_cache_boundary` and
      `test_media_admission_uses_aggregate_resources` read the Linux-only official tokenizer files.
      Fixed by gating all official-fixture sub-tests (8 total) behind
      `official_tokenizer_fixtures_present()`.
- [x] MSVC new-code incompatibilities: aligned_alloc, move-template explicit bodies, TMA/grid-constant.
- [x] `near`->`nearly_equals` rename in `test_frontend.cpp` (MSVC empty `near` macro from <windows.h>);
      `test_host_timing.cpp` already had it.
- [x] Run nvfp4 linear test binaries (PASS)
- [x] Run kv-cache + engine prefix/admission tests (PASS)
- [x] `ninfer_qwen3_6_frontend_test` PASS (14 tests green with `bin\` on PATH;
      official-fixture sub-tests skip when real model files absent)
- [ ] Refresh bin/ and distribution/

## Phase 3 - Port our optimizations and compare

Status: pending

- [ ] Carry NVFP4 TMA + cluster linear layer (untouched upstream, should re-apply cleanly)
- [ ] Carry quant GEMM tuning
- [ ] Re-establish performance baseline at token points
- [ ] Compare against new upstream default behavior

## Notes
- Upstream made zero changes to src/ops/linear**, linear_swiglu/**, or nvfp4/FP8/Q4/Q5/Q6 GEMM
  kernels, so our NVFP4 work re-carries cleanly.
- The engine/serve/runtime/attention/kv layer is where conflicts concentrate (second-gen rewrite).
