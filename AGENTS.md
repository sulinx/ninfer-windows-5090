# AGENTS.md

## Product and architecture

NInfer is a C++/CUDA inference engine for maximum single-GPU performance, targeting `sm_120a`
on NVIDIA GeForce RTX 5090. Choose designs for functional and numerical correctness, clear
ownership, and the performance goal within the requested scope.

It implements `Qwen3_5ForCausalLM` and `Qwen3_5MoeForCausalLM`. Checkpoints and recipe combinations
of existing representations use the same architecture, binding and execution path; do not add
checkpoint-specific execution registration. Generation uses one GPU, one resident model, and
one to eight resident execution lanes fixed at startup.

V3 `.ninfer` is the only C++ product artifact. Generation, offline CausalScoring, CLI, serving, and
inference benchmarks use the public Engine. There is no Python model-inference route or
installed/exported C++ SDK.

This is a local, single-owner project with trusted local models, artifacts, and workflows.
Do not derive requirements from another deployment or trust model.

Model data is immutable; every Program owns its mutable state and device allocations. The loader
validates, uploads, and binds the stored representation; actual Parameters and Ops determine
execution support, without a whole-artifact capability registry.
Runtime owns execution and publication policy; product/serving own input acquisition and protocol
translation. Model code does not acquire media or own transport.

Keep architecture, binding, and execution explicit. Without a product requirement, do not introduce
generic model graphs, family base classes, plugin discovery, string-driven execution, hidden device
allocation, or runtime weight repacking. New mathematical architectures, execution platforms,
large-scale continuous batching, or priority/QoS require an explicit product change. When a task
changes product or architecture, update affected contracts and implementation together.

## Change policy

Project-owned contracts do not preserve backward compatibility. Replace behavior completely:
remove superseded aliases, fallbacks, transition branches, and their tests within the affected contract.
Advertised OpenAI and Anthropic protocols are external contracts; update affected schema tests
and serving documentation together.
Update stable requirements in their existing authoritative document; maintain one current authority.
Use Conventional Commit subjects with concise lowercase types when a commit is requested.

## Verification and reporting

Before changing Ops, numerical or state semantics, read
[Op development](docs/maintainer/op-development.md). Qualify affected production routes directly
against the contract's independent mathematical oracle or specified exact reference. Kernel parity
and plausible model output do not establish mathematical correctness.

Measure performance at the claimed scope. An Op microbenchmark establishes an Op result,
not an end-to-end improvement. For comparisons, establish the baseline, workloads, metrics,
aggregation, and acceptance criteria before evaluating results.
Distinguish new capability, fallback replacement, and improvement to an optimized implementation.
Report the baseline, hardware/toolchain, workloads or commands, run conditions, metrics, coverage,
result distribution, worst changes, and exceptions. Include small and unexplained regressions;
do not dismiss slowdowns as noise without evidence.

## References

Read [Engine architecture](docs/maintainer/engine-architecture.md) before changing execution or
ownership. [README](README.md) and executable `--help` define capabilities and exact commands.
The [documentation map](docs/README.md) routes to detailed contracts;
[Tests](tests/README.md) and [Benchmarks](bench/README.md) own their commands and execution details.

## Local environment

Use Python 3.11 via `/home/neroued/miniconda3/envs/py311/bin/python` explicitly; use `python3` only
after selecting the maintainer environment or checking its version.
The usual local model is `out/qwen3_8_27b_nvfp4.ninfer`. Select artifacts by explicit path, never glob order,
modification time, or unqualified “latest”. Source checkpoints and large artifacts are prerequisites;
download or regenerate them only when that work is in scope.

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
