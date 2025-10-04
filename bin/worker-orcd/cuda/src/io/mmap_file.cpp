/**
 * Memory-Mapped File I/O Implementation
 * 
 * Implements efficient memory-mapped access to GGUF files using mmap().
 * 
 * Spec: M0-W-1221
 */

#include "io/mmap_file.h"
#include "../cuda_error.h"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>
#include <sstream>

namespace worker {
namespace io {

MmapFile::MmapFile(const std::string& path, void* mapped_data, size_t file_size, int fd)
    : path_(path)
    , mapped_data_(mapped_data)
    , file_size_(file_size)
    , fd_(fd)
{
}

MmapFile::~MmapFile() {
    unmap();
}

MmapFile::MmapFile(MmapFile&& other) noexcept
    : path_(std::move(other.path_))
    , mapped_data_(other.mapped_data_)
    , file_size_(other.file_size_)
    , fd_(other.fd_)
{
    other.mapped_data_ = nullptr;
    other.file_size_ = 0;
    other.fd_ = -1;
}

MmapFile& MmapFile::operator=(MmapFile&& other) noexcept {
    if (this != &other) {
        unmap();
        
        path_ = std::move(other.path_);
        mapped_data_ = other.mapped_data_;
        file_size_ = other.file_size_;
        fd_ = other.fd_;
        
        other.mapped_data_ = nullptr;
        other.file_size_ = 0;
        other.fd_ = -1;
    }
    return *this;
}

void MmapFile::unmap() {
    if (mapped_data_ != nullptr) {
        munmap(mapped_data_, file_size_);
        mapped_data_ = nullptr;
    }
    
    if (fd_ >= 0) {
        close(fd_);
        fd_ = -1;
    }
    
    file_size_ = 0;
}

MmapFile MmapFile::open(const std::string& path) {
    // Open file
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        int err = errno;
        if (err == ENOENT) {
            throw CudaError::model_load_failed(
                "File not found: " + path
            );
        } else if (err == EACCES) {
            throw CudaError::model_load_failed(
                "Permission denied: " + path
            );
        } else {
            throw CudaError::model_load_failed(
                "Failed to open file: " + path + " (errno: " + std::to_string(err) + ")"
            );
        }
    }
    
    // Get file size
    struct stat st;
    if (fstat(fd, &st) != 0) {
        int err = errno;
        close(fd);
        throw CudaError::model_load_failed(
            "Failed to stat file: " + path + " (errno: " + std::to_string(err) + ")"
        );
    }
    
    size_t file_size = static_cast<size_t>(st.st_size);
    
    // Validate file size
    if (file_size == 0) {
        close(fd);
        throw CudaError::model_load_failed(
            "File is empty: " + path
        );
    }
    
    // Memory-map file
    void* addr = mmap(
        nullptr,           // Let kernel choose address
        file_size,         // Map entire file
        PROT_READ,         // Read-only access
        MAP_PRIVATE,       // Private copy-on-write (prevents file modification)
        fd,                // File descriptor
        0                  // Offset 0 (map from start)
    );
    
    if (addr == MAP_FAILED) {
        int err = errno;
        close(fd);
        
        if (err == ENOMEM) {
            throw CudaError::out_of_memory(
                "Insufficient memory to map file: " + path +
                " (size: " + std::to_string(file_size) + " bytes)"
            );
        } else if (err == EACCES) {
            throw CudaError::model_load_failed(
                "Permission denied for mmap: " + path
            );
        } else {
            throw CudaError::model_load_failed(
                "mmap failed for file: " + path + " (errno: " + std::to_string(err) + ")"
            );
        }
    }
    
    // Advise kernel about access pattern (sequential read)
    madvise(addr, file_size, MADV_SEQUENTIAL);
    
    return MmapFile(path, addr, file_size, fd);
}

const void* MmapFile::get_data_at(size_t offset) const {
    if (offset >= file_size_) {
        throw CudaError::invalid_parameter(
            "Offset " + std::to_string(offset) +
            " is beyond file size " + std::to_string(file_size_)
        );
    }
    
    return static_cast<const char*>(mapped_data_) + offset;
}

const void* MmapFile::get_tensor_data(size_t offset, size_t size) const {
    // Check for integer overflow
    if (size > SIZE_MAX - offset) {
        throw CudaError::invalid_parameter(
            "Integer overflow: offset=" + std::to_string(offset) +
            " size=" + std::to_string(size)
        );
    }
    
    // Check bounds
    if (offset + size > file_size_) {
        throw CudaError::invalid_parameter(
            "Tensor data extends beyond file: offset=" + std::to_string(offset) +
            " size=" + std::to_string(size) +
            " file_size=" + std::to_string(file_size_)
        );
    }
    
    return static_cast<const char*>(mapped_data_) + offset;
}

} // namespace io
} // namespace worker

// ---
// Implemented by Llama-Beta 🦙
