#include "artifact/file_io.h"

#include "artifact/framing.h"
#include "artifact/schema.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <limits>
#include <utility>

#include <fcntl.h>
#include <sys/stat.h>
#if defined(_MSC_VER)
#include <io.h>
#include <sys/types.h>
#else
#include <unistd.h>
#endif

#if defined(_MSC_VER)
// Windows/MSVC port of the POSIX positional-I/O helpers this file relies on.
// MSVC (x64, LLP64) provides open/close/read/fstat/_lseeki64 but not
// ssize_t, O_CLOEXEC, O_DIRECT, S_ISREG, or pread(); and its off_t is 32-bit.
typedef std::ptrdiff_t ssize_t;
using file_off_t = __int64;          // 64-bit file offset (off_t is only 32-bit on MSVC)
#ifndef O_CLOEXEC
#define O_CLOEXEC 0
#endif
#ifndef O_DIRECT
#define O_DIRECT 0
#endif
#ifndef S_ISREG
#define S_ISREG(m) (((m) & _S_IFMT) == _S_IFREG)
#endif

// POSIX pread() via lseek+read. read() caps the transfer at unsigned long;
// callers pass <= 64 MiB chunks, so no single read is truncated.
inline ssize_t pread(int fd, void* buffer, std::size_t count, file_off_t offset) {
    if (_lseeki64(fd, offset, SEEK_SET) < 0) { return -1; }
    return static_cast<ssize_t>(read(fd, buffer, static_cast<unsigned long>(count)));
}

// MSVC fstat() uses struct _stat, whose st_size is a 32-bit long: it fails
// with EOVERFLOW on files larger than 2 GiB (model artifacts are ~24 GiB).
// _fstat64 reports the real 64-bit size.
using file_stat_t = struct _stat64;
inline int file_stat(int fd, file_stat_t* out) { return ::_fstat64(fd, out); }
#else
using file_off_t = off_t;
using file_stat_t = struct stat;
inline int file_stat(int fd, file_stat_t* out) { return ::fstat(fd, out); }
#endif

// MSVC opens files in text mode unless O_BINARY is given: reads then stop at
// 0x1A (Ctrl+Z) and CRLF is rewritten to LF, which corrupts binary artifacts.
// POSIX has no O_BINARY, so it degrades to a no-op there.
#ifndef O_BINARY
#define O_BINARY 0
#endif

namespace ninfer::artifact {
namespace {

[[noreturn]] void fail(const std::filesystem::path& path, const char* operation) {
    throw ArtifactError(path.string() + ": " + operation + ": " + std::strerror(errno));
}

file_off_t file_offset(std::uint64_t offset) {
    if (offset > static_cast<std::uint64_t>(std::numeric_limits<file_off_t>::max())) {
        throw ArtifactError("file offset exceeds positional I/O range");
    }
    return static_cast<file_off_t>(offset);
}

} // namespace

InputFile::InputFile(std::filesystem::path path) : path_(std::move(path)) {
    fd_ = ::open(path_.string().c_str(), O_RDONLY | O_CLOEXEC | O_BINARY);
    if (fd_ < 0) { fail(path_, "open"); }

    file_stat_t status {};

    if (file_stat(fd_, &status) != 0) {
        const auto error = errno;
        ::close(fd_);
        fd_   = -1;
        errno = error;
        fail(path_, "fstat");
    }
    if (status.st_size < 0 || !S_ISREG(status.st_mode)) {
        ::close(fd_);
        fd_ = -1;
        throw ArtifactError(path_.string() + ": expected a regular file");
    }
    bytes_ = static_cast<std::uint64_t>(status.st_size);
}

InputFile::~InputFile() {
    if (direct_fd_ >= 0) { ::close(direct_fd_); }
    if (fd_ >= 0) { ::close(fd_); }
}

void InputFile::read_exact(std::uint64_t offset, std::span<std::byte> destination) const {
    if (offset > bytes_ || destination.size() > bytes_ - offset) {
        throw ArtifactError(path_.string() + ": read exceeds file length");
    }
    while (!destination.empty()) {
        const auto count = std::min<std::size_t>(destination.size(), 64ULL * 1024 * 1024);
        const auto read  = ::pread(fd_, destination.data(), count, file_offset(offset));
        if (read < 0) {
            if (errno == EINTR) { continue; }
            fail(path_, "pread");
        }
        if (!read) { throw ArtifactError(path_.string() + ": unexpected EOF"); }
        offset += static_cast<std::uint64_t>(read);
        destination = destination.subspan(static_cast<std::size_t>(read));
    }
}

std::size_t InputFile::read_direct(std::uint64_t offset, std::span<std::byte> destination) const {
    if (offset % kPayloadAlignment || destination.size() % kPayloadAlignment ||
        reinterpret_cast<std::uintptr_t>(destination.data()) % kPayloadAlignment ||
        destination.size() > static_cast<std::size_t>(std::numeric_limits<ssize_t>::max())) {
        throw ArtifactError(path_.string() + ": unaligned or oversized direct read");
    }
    if (destination.empty()) { return 0; }
    if (direct_fd_ < 0) {
        direct_fd_ = ::open(path_.string().c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT | O_BINARY);
        if (direct_fd_ < 0) { fail(path_, "open direct"); }
    }
    ssize_t read;
    do {
        read = ::pread(direct_fd_, destination.data(), destination.size(), file_offset(offset));
    } while (read < 0 && errno == EINTR);
    if (read < 0) { fail(path_, "direct pread"); }
    return static_cast<std::size_t>(read);
}

} // namespace ninfer::artifact
