#pragma once

// Windows/MSVC cannot pass an over-aligned aggregate by value. CUtensorMap is alignas(128), so
// every TMA descriptor struct inherits that alignment and MSVC rejects the kernel parameter with
// C2719 ("formal parameter with requested alignment of 128 won't be aligned"). That blocks the
// __grid_constant__ by-value tensor-map parameter the TMA kernels rely on.
//
// Windows therefore carries the identical descriptor bytes in a plain 8-byte-aligned wrapper.
// Passed by value, the bytes live in grid-constant memory, which is a valid tensor-map source for
// cp.async.bulk.tensor, and CUDA Graph capture snapshots the parameter bytes instead of recording
// a host stack address that replay would re-read after the stack is repurposed.
//
// On other platforms TmaDescriptorParam<D> is D itself and every helper is the identity, so kernel
// signatures, bodies and launch sites keep the upstream semantics byte for byte.

#include <cstring>
#include <type_traits>

#include <cuda.h>
#include <cuda_runtime.h>

namespace ninfer::ops::detail {

#ifdef _WIN32

template <class Descriptors>
struct alignas(8) TmaDescriptorParam {
    unsigned char bytes[sizeof(Descriptors)];
};

template <class Descriptors>
inline TmaDescriptorParam<Descriptors> make_tma_descriptor_param(const Descriptors& descriptors) {
    static_assert(!std::is_empty_v<Descriptors>);
    TmaDescriptorParam<Descriptors> param{};
    static_assert(sizeof(param.bytes) == sizeof(Descriptors));
    std::memcpy(param.bytes, &descriptors, sizeof(Descriptors));
    return param;
}

template <class Descriptors>
__device__ __forceinline__ const Descriptors&
tma_descriptors(const TmaDescriptorParam<Descriptors>& param) {
    return *reinterpret_cast<const Descriptors*>(param.bytes);
}

#else

template <class Descriptors>
using TmaDescriptorParam = Descriptors;

template <class Descriptors>
inline Descriptors make_tma_descriptor_param(const Descriptors& descriptors) {
    return descriptors;
}

template <class Descriptors>
__device__ __forceinline__ const Descriptors& tma_descriptors(const Descriptors& descriptors) {
    return descriptors;
}

#endif

} // namespace ninfer::ops::detail
