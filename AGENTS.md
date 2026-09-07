# AGENTS.md

These rules apply to the whole repository.

## Objective and scope

Complete the user's explicit deliverable within the applicable product and external contracts.
Choose a coherent solution with functional and numerical correctness, clear ownership, strong
architecture, and maximum performance at the requested scope. Do not sacrifice these goals to
reduce the diff or implementation effort. Evaluate complexity, maintenance cost, and verification
risk as engineering tradeoffs, not reasons to retain a known inferior design.

Before substantial work, identify the deliverable and its completion conditions. Work is relevant
when it completes that deliverable, preserves an applicable contract, resolves a material
uncertainty, or checks a realistic regression. A necessary redesign is in scope; unrelated cleanup,
hardening, compatibility, and benchmark campaigns are not. Address incidental findings when they
block the outcome or are inseparable from the selected implementation.

For analysis or design, deliver the explanation or design. For diagnosis, establish the cause and
supporting evidence; implement a fix when requested. For implementation, complete the selected
design across its affected implementations, callers, tests, tools, and active documentation.

The current product and architecture govern ordinary work. An explicit task may change them;
update the affected contracts and implementation together instead of treating the current design
as an immutable prohibition. Skills provide task-specific methods, not additional deliverables or
approval requirements beyond the user's instructions and the actual execution environment.

## Product and architecture

NInfer is a from-scratch C++/CUDA inference engine for maximum single-GPU performance on explicitly
registered artifacts. Current identities are `qwen3.6-27b/groupwise-int`, `qwen3.6-27b/nvfp4`,
`qwen3.8-27b/groupwise-int`, `qwen3.8-27b/nvfp4`, and `qwen3.6-35b-a3b/groupwise-int`.
The implementation targets `sm_120a` and is tuned on NVIDIA GeForce RTX 5090.

Generation uses one GPU, one resident model, startup-fixed concurrency of one to eight requests,
bounded FIFO ingress, no active-request preemption, and one compact decode batch per round.
Generation and offline CausalScoring use the same public `.ninfer` Engine route. Delivered
capabilities and commands are documented in `README.md`, the product guides, and executable
`--help`. Additional models, execution platforms, large-scale/preemptive continuous batching, and
priority/QoS require an explicit product change.

This is a local, single-owner project with trusted registered models, generated artifacts, and
local workflow. Do not derive requirements from a different deployment or trust model.

Keep these ownership boundaries visible when selecting a design:

- `.ninfer` is the only C++ product artifact; CLI, serving, and inference benchmarks use the public
  Engine. NInfer has no Python model-inference route or installed/exported C++ SDK.
- Core owns physical primitives and raw transfers; artifact owns generic framing and
  materialization; Ops own closed mathematical and state-transition implementations.
- The Qwen3.6 family owns shared frontend semantics and compile-time planning/Program algorithms.
  The 27B and 35B-A3B packages are peer Variants owning their identities, bindings, execution leaves,
  and Program instance storage. Programs share no mutable state or device allocation.
- Runtime owns common execution contracts and Engine publication policy; product/serving own input
  acquisition and protocol translation. Target packages do not acquire media or own transport.

Detailed family/package responsibilities and source ownership are defined in
[Engine architecture](docs/maintainer/engine-architecture.md). Read the relevant boundary before
changing it. Prefer explicit implementations for registered targets. Do not introduce generic model
graphs, family base classes, plugin discovery, string-driven execution, hidden device allocation,
runtime weight repacking, or placeholders for hypothetical targets without a product requirement.

## Change consistency

Project-owned APIs, CLIs, Python tools, fixtures, reports, formats, and documentation do not preserve
backward compatibility. When replacing behavior, remove superseded aliases, fallbacks, transition
branches, and their tests within the affected contract. Leave unrelated paths alone.

Advertised OpenAI and Anthropic protocol behavior is an external contract. Changes update the
affected schema tests and serving documentation together.

Keep stable requirements in their existing active reference. Temporary plans are useful only for
active work; remove them when completed or abandoned. Maintain one current authority rather than
parallel `final`, `v2`, or `new-design` documents.

## Verification and completion

Select evidence to support the changed behavior and material claims. Tests should protect supported
observable behavior, mathematical or state semantics, and realistic regressions, including plausible
boundary failures that have not occurred yet. Avoid tests that merely mirror implementation,
freeze private file/class organization, or increase coverage numbers.

For numerical changes, identify represented public inputs, the independent mathematical oracle,
semantic cast/quantization/state boundaries, output criteria, and relevant real model shapes. Each
floating-point Op uses a naive FP32/FP64 oracle; exact transforms/codecs use an exact oracle. Packed
inputs are independently decoded with their stored scales. Qualify production routes directly
against that oracle, not another kernel or plausible model output. Private arithmetic need not
reproduce unfused materializations unless an intermediate is an observable semantic boundary.
[Op development](docs/maintainer/op-development.md) defines the full qualification contract.

Measure performance at the claimed scope. An Op microbenchmark establishes an Op result, not an
end-to-end improvement. Use whole-inference profiling when an in-scope end-to-end attribution is
unresolved; use kernel profiling when an identified kernel question can change the decision. Reuse
applicable evidence and stop collecting once the relevant alternatives can be distinguished.

Choose the affected checks, rather than running this table as a checklist:

| Change | Typical evidence |
|---|---|
| Documentation | affected links/references and `git diff --check` |
| C++ runtime/API | affected build targets and behavioral tests |
| Python tooling | Python 3.11 `py_compile` and affected tests |
| Artifact framing/binding/conversion | affected contract tests; real artifact when semantics require it |
| CUDA mathematics | independent oracle at relevant shapes and route boundaries |
| Memory or lifetime | affected execution; sanitizer for a concrete lifetime question |
| Performance | measurement at the claimed scope; profiling only for unresolved attribution |
| Serving | affected schema tests and observable request/stream behavior |

Record the target, relevant hardware/toolchain, workload or command, and summarized result needed
to interpret a material claim. Hashes, clean worktrees, full command transcripts, raw report
inventories, and exact probabilistic outputs are not default requirements. Use exact comparison for
exact outputs, and appropriate numerical or behavioral criteria otherwise. State checks that could
not run and their implications.

Finish when the deliverable is usable, applicable contracts are satisfied, material claims have
sufficient evidence, relevant checks pass or their limitations are clear, and no known in-scope
issue blocks use. Expand or repeat verification only for new changes, failures, or unresolved risks
that could change the result. Supporting work is not an independent completion objective.

## Reference navigation

Read the authority relevant to the current decision; this is not a mandatory reading list.

| Decision | Entry point |
|---|---|
| Product capabilities and exact commands | `README.md`, executable `--help`; `docs/cli.md`, `docs/serving.md`, `docs/perplexity.md` |
| Execution, family/package ownership, scheduling, transactions, graphs | `docs/maintainer/engine-architecture.md` |
| Context resources, checkpoints, replicas; physical KV | `docs/maintainer/resource-scheduling-and-context-cache.md`; `docs/maintainer/paged-kv-cache.md` |
| Artifact, layout, codec, or exact target mathematics | model/artifact references linked from `docs/README.md` |
| Op contracts, implementation ownership, numerical/performance qualification | `docs/maintainer/op-development.md` |
| Test/benchmark commands and published performance | `tests/README.md`, `bench/README.md`, `docs/performance.md` |
| In-tree C++ interface | `include/ninfer/engine.h`, `include/ninfer/types.h` |

[Documentation map](docs/README.md) routes to narrower authorities when needed.

## Local operations

Use `cmake --build <build-dir> -j` by default. Adjust parallelism when actual resource pressure
causes failures or interferes with the task, and briefly explain why.

Use the selected Python 3.11 interpreter explicitly. On this machine it is
`/home/neroued/miniconda3/envs/py311/bin/python`; the default shell's `python3` may be a different
version. Use `python3` only after selecting the maintainer environment or checking its version.
Normal resources are `build/`, `out/qwen3_6_27b.ninfer`, its `.conversion.json` report, and
`profiles/ncu/`, `profiles/nsys/`, `profiles/bench/`; the local toolchain is CUDA 13.1.
Select model artifacts by explicit path, never glob order, modification time, or unqualified
“latest”. Source checkpoints and large artifacts are prerequisites; download or regenerate them
only when that work is in scope. Install or upgrade dependencies only when the task needs it.

Create commits only when requested. Use Conventional Commit subjects with concise lowercase types
such as `feat`, `fix`, `perf`, `bench`, `test`, `build`, `refactor`, `docs`, or `chore`.

## Windows build (native MSVC)

The `windows-port` branch adds native Windows build support. The implementation compiles with
Visual Studio 2022 (MSVC) and CUDA 13.1 targeting `sm_120a` (RTX 5090). DLL dependencies are
sourced from MSYS2 MinGW-w64 via MSVC import libraries generated from `.def` files.

### Prerequisites

| Component | Path / Source |
|---|---|
| Visual Studio 2022 Professional | `C:\Program Files\Microsoft Visual Studio\2022\Professional\` |
| CUDA Toolkit 13.1 | `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\` |
| CMake 3.28+ | `C:\bin\CMake\bin\cmake.exe` |
| Ninja | `C:\Python312\Scripts\ninja.exe` |
| MSYS2 (mingw-w64) | `C:\msys64\mingw64\` |
| Python 3.11+ | `python` (for HuggingFace model download) |

MSYS2 `mingw-w64` must have these packages installed (already present at `C:\msys64`):
- FFmpeg 7.x dev (libavformat, libavcodec, libavutil, libswscale headers + DLLs)
- libcurl 8.x dev (headers + DLL)
- pkg-config / pkgconf (not used directly; only for `.pc` reference files)

An additional FFmpeg dependency (`vvenc.dll`) lives at `C:\msys64\home\george\ffmpeg_deps\bin\`
and must exist at that path for the import-library generation scripts below.

### First-time setup

All build infrastructure (import libs, filtered headers) is committed to `windows-libs/`.
If MSYS2 FFmpeg/curl packages are updated, regenerate import libraries:

```pwsh
# From repository root, with MSVC lib.exe available:
$libExe = "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Tools\MSVC\14.41.34120\bin\Hostx64\x64\lib.exe"
$outDir = "L:\Temp\ninfer\windows-libs"
$msysDir = "C:\msys64\mingw64"

# 1. Generate MSVC import libraries from MSYS2 .def files
foreach ($pair in @(
    @("$msysDir\lib\avformat-62.def", "avformat.lib", "avformat-62.dll"),
    @("$msysDir\lib\avcodec-62.def", "avcodec.lib", "avcodec-62.dll"),
    @("$msysDir\lib\avutil-60.def", "avutil.lib", "avutil-60.dll"),
    @("$msysDir\lib\swscale-9.def", "swscale.lib", "swscale-9.dll")
)) {
    $defContent = "LIBRARY $($pair[2])`nEXPORTS`n" + (Get-Content $pair[0] -Raw)
    $defFile = "$outDir\$($pair[1].Replace('.lib','.def'))"
    Set-Content -Path $defFile -Value $defContent
    & $libExe /def:$defFile /out:$outDir\$($pair[1]) /machine:x64
}

# 2. Generate libcurl import library
& "$msysDir\bin\gendef.exe" "$msysDir\bin\libcurl-4.dll"
$defLines = Get-Content "libcurl-4.def"
$exports = @("LIBRARY libcurl-4.dll", "EXPORTS")
foreach ($line in $defLines) {
    $t = $line.Trim()
    if ($t -notmatch '^;' -and $t -ne '' -and $t -notmatch '^(LIBRARY|EXPORTS)') {
        $clean = $t -replace '@\d+$', ''
        if ($clean) { $exports += $clean }
    }
}
$exports -join "`n" | Set-Content -NoNewline -Path "$outDir\libcurl.def"
Add-Content -Path "$outDir\libcurl.def" -Value "`n"
& $libExe /def:$outDir\libcurl.def /out:$outDir\libcurl.lib /machine:x64
Remove-Item "libcurl-4.def" -ErrorAction SilentlyContinue

# 3. Create filtered header directories (avoid shadowing MSVC system headers)
$filteredDir = "$outDir\ffmpeg-include"
New-Item -ItemType Directory -Force -Path $filteredDir | Out-Null
@("libavcodec","libavformat","libavutil","libswscale","libswresample",
  "libavdevice","libavfilter","libpostproc") | ForEach-Object {
    Copy-Item -Recurse "$msysDir\include\$_" "$filteredDir\$_" -ErrorAction SilentlyContinue
}

$curlInc = "$outDir\curl-include"
New-Item -ItemType Directory -Force -Path "$curlInc\curl" | Out-Null
Copy-Item -Recurse "$msysDir\include\curl\*" "$curlInc\curl\"
```

### Configure and build

```pwsh
cmd /c '"C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\Tools\VsDevCmd.bat" -arch=amd64 && cmake -S L:\Temp\ninfer -B L:\Temp\ninfer\build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER="C:/Program Files/Microsoft Visual Studio/2022/Professional/VC/Tools/MSVC/14.41.34120/bin/Hostx64/x64/cl.exe" -DCMAKE_CXX_COMPILER="C:/Program Files/Microsoft Visual Studio/2022/Professional/VC/Tools/MSVC/14.41.34120/bin/Hostx64/x64/cl.exe" && cmake --build L:\Temp\ninfer\build --parallel'
```

After a successful build, copy executables and runtime DLLs to `bin/`:

```pwsh
New-Item -ItemType Directory -Force -Path L:\Temp\ninfer\bin | Out-Null
Copy-Item L:\Temp\ninfer\build\apps\ninfer.exe L:\Temp\ninfer\bin\
Copy-Item L:\Temp\ninfer\build\apps\ninfer-serve.exe L:\Temp\ninfer\bin\

# MSVC runtime
$msvcDll = "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Tools\MSVC\14.41.34120\bin\Hostx64\x64"
@("MSVCP140.dll","VCRUNTIME140.dll","VCRUNTIME140_1.dll") | ForEach-Object {
    Copy-Item "$msvcDll\$_" L:\Temp\ninfer\bin\
}

# FFmpeg + curl + transitive DLLs
$dllDir = "C:\msys64\mingw64\bin"
$vvencDir = "C:\msys64\home\george\ffmpeg_deps\bin"
@(
    # Direct deps
    "avformat-62.dll","avcodec-62.dll","avutil-60.dll","swscale-9.dll","libcurl-4.dll",
    # FFmpeg transitive
    "libbz2-1.dll","libiconv-2.dll","zlib1.dll","libwinpthread-1.dll","swresample-6.dll",
    "liblzma-5.dll","libaom.dll","libdav1d-7.dll","libfdk-aac-2.dll","libmp3lame-0.dll",
    "libopus-0.dll","libSvtAv1Enc-3.dll","libvpx-1.dll","libx264-164.dll","libx265-215.dll",
    # curl transitive
    "libbrotlidec.dll","libbrotlicommon.dll","libidn2-0.dll","libnghttp2-14.dll",
    "libnghttp3-9.dll","libpsl-5.dll","libssh2-1.dll","libzstd.dll",
    "libcrypto-3-x64.dll","libssl-3-x64.dll",
    # MinGW runtime transitive
    "libintl-8.dll","libstdc++-6.dll","libunistring-5.dll","libgcc_s_seh-1.dll"
) | ForEach-Object {
    if (Test-Path "$dllDir\$_") { Copy-Item "$dllDir\$_" L:\Temp\ninfer\bin\ }
}

# vvenc (separate FFmpeg dep)
Copy-Item "$vvencDir\vvenc.dll" L:\Temp\ninfer\bin\
```

The `bin\` directory is self-contained: run `ninfer.exe` or `ninfer-serve.exe` directly from it
without adjusting PATH.

### Download models

```pwsh
python -m pip install huggingface_hub
$env:PATH = "$env:APPDATA\Python\Python314\Scripts;$env:PATH"

hf download neroued/Qwen3.6-27B-NInfer qwen3_6_27b.ninfer --local-dir L:\Temp\ninfer\bin\models
hf download neroued/Qwen3.6-35B-A3B-NInfer qwen3_6_35b_a3b.ninfer --local-dir L:\Temp\ninfer\bin\models
hf download neroued/Qwen3.8-27B-NInfer qwen3_8_27b_nvfp4.ninfer --local-dir L:\Temp\ninfer\bin\models
```

The FP8/Qwen3.8 upstream additions (row-scaled FP8 ops, qwen3.8 nvfp4 converter/runtime) compile on
Windows with no extra porting; they add no `__grid_constant__` TMA descriptors or POSIX calls beyond
the files listed above. `qwen3_8_27b_nvfp4.ninfer` loads and serves through the same `.ninfer` route
after the sync above.

Expected hashes:

| File | Size | SHA-256 |
|---|---|---|
| `qwen3_6_27b.ninfer` | 17,495,365,888 | `74fac75f...` |
| `qwen3_6_35b_a3b.ninfer` | 22,783,246,080 | `5194407d...` |

### Run

`L:\Temp\ninfer\bin\run.cmd` starts the HTTP server on port 8080:

```cmd
ninfer-serve.exe models\qwen3_6_27b.ninfer --model-id qwen3.6-27b --host 0.0.0.0 --port 8080
ninfer-serve.exe models\qwen3_8_27b_nvfp4.ninfer --model-id qwen3.8-27b-nvfp4 --host 0.0.0.0 --port 8080
```

Edit `run.cmd` to add flags (e.g. `--vision`, `--spec mtp --draft-tokens 3`).

### Updating after upstream changes

All Windows port changes are committed on branch `windows-port`. After pulling from upstream:

```pwsh
git checkout windows-port
git pull origin master --rebase
# resolve conflicts if upstream changed the same files
cmd /c '"C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\Tools\VsDevCmd.bat" -arch=amd64 && cmake --build L:\Temp\ninfer\build --parallel'
Copy-Item L:\Temp\ninfer\build\apps\ninfer.exe L:\Temp\ninfer\bin\
Copy-Item L:\Temp\ninfer\build\apps\ninfer-serve.exe L:\Temp\ninfer\bin\
```

`windows-libs/`, DLLs, and models under `bin/` never need regeneration unless MSYS2 packages change.

### Ported files (changes from upstream Linux code)

| File | Change |
|---|---|
| `CMakeLists.txt` | Windows branch: define `PkgConfig::FFMPEG`/`LIBCURL` as imported INTERFACE targets bypassing pkg-config; add `NOMINMAX` |
| `src/CMakeLists.txt` | `UTF8PROC_STATIC` define; `ws2_32` link for Windows |
| `src/artifact/reader.cpp` | `MappedFile` class: Win32 `CreateFileW`/`CreateFileMappingW`/`MapViewOfFile`/`ReadFile` replacing POSIX `mmap`/`pread` |
| `src/serve/request_log.cpp` | `getpid()` → `_getpid()` from `<process.h>` |
| `tests/test_request_log.cpp` | Same `getpid` fix |
| `src/product/media_acquire/acquire.cpp` | BSD socket API (`arpa/inet.h` etc.) → Winsock2; wide-char `path::starts_with` fix |
| `src/product/load_progress/load_progress.cpp` | `isatty`/`STDERR_FILENO` → `_isatty`/`stderr_fileno` from `<io.h>` |
| `src/serve/console_log.cpp` | `localtime_r` → `localtime_s` (Win32 arg order) |
| `src/ops/linear/nvfp4/nvfp4_w4a4_tma.cuh` / `.cu` | `__grid_constant__` tensor-map descriptors carried by value as a plain 8-byte-aligned `Nvfp4W4a4TmaDescriptorBytes` on `_WIN32` (by-value `alignas(128)` aggregate fails MSVC C2719; a plain host-stack pointer is not GPU-readable by TMA and is invalid under CUDA Graph capture/replay). The kernel reinterprets the grid-constant carrier as the aligned struct |
| `src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_w4a4_tma.cuh` / `.cu` | Same by-value descriptor-carrier fix for the swiglu TMA kernel and its launch site |
| `src/targets/qwen3_6/impl/runtime/api_impl.h` | `SequencePlan`, `RequestPlan`, `RequestBasePlan` move ctor/assign: explicit `impl_(std::move(...))` body instead of `= default` (MSVC does not emit out-of-line defaulted template members) |

### Architecture notes

- **No pkg-config on Windows**: The `PkgConfig::FFMPEG` and `PkgConfig::LIBCURL` targets are
  defined as `INTERFACE IMPORTED` with hand-specified include dirs and link libraries. The Linux
  path (`find_package(PkgConfig)` / `pkg_check_modules`) is untouched.
- **Filtered headers**: `C:\msys64\mingw64\include` contains MinGW versions of `<windows.h>`,
  `<winsock2.h>`, `<winnt.h>` that are incompatible with MSVC. The filtered directories under
  `windows-libs/ffmpeg-include/` and `windows-libs/curl-include/` contain only the library headers,
  letting MSVC resolve system headers from the Windows SDK.
- **Import libraries**: MinGW `.dll.a` files are not MSVC-compatible. MSVC import libraries (`.lib`)
  are generated from `.def` export definition files using `lib.exe /def: /machine:x64`.
- **CUDA**: Linked statically (`cudart_static.lib`). The resulting executables have no runtime
  dependency on `cudart64_*.dll`.
- **`NOMINMAX`**: Required globally because CUDA/NVTX headers include `<windows.h>` which defines
  `min`/`max` macros that shadow `std::numeric_limits<>::max()`.
- **MSVC defaulted move templates**: An explicitly specialized `= default` move ctor/assign on a
  class template does not emit a linkable symbol when the move is used from another TU. Give these
  an explicit body (`impl_(std::move(other.impl_))`) on all platforms.
- **`__grid_constant__` by-value TMA descriptors**: MSVC rejects passing `alignas(128)` aggregates
  (containing `CUtensorMap`) by value (C2719), and CUDA 13.1 nvcc rejects `__grid_constant__` on a
  pointer parameter. Passing a plain host-stack pointer on Windows is broken: the GPU cannot source a
  tensor-map descriptor from a host address, and CUDA Graph capture records that host address and
  `re-reads it at replay after the stack is repurposed`, yielding the intermittent `illegal memory
  access` seen on the second nvfp4 request. Windows therefore carries the identical four maps by
  value as `Nvfp4W4a4TmaDescriptorBytes` (`alignas(8) CUtensorMap[4]`, `__grid_constant__`); the
  bytes live in grid-constant memory (a valid `cp.async.bulk.tensor` source) and a by-value
  parameter is snapshotted by graph capture, reproducing the Linux semantics. The kernel body
  reinterprets `&descriptor_value.maps[0]` as the aligned `Nvfp4W4a4TmaDescriptors`. Leave the
  `#ifdef _WIN32` branches in both `.cuh` kernels and their `.cu` launch sites consistent.
