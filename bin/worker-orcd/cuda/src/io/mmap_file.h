/**
 * Memory-Mapped File I/O
 * 
 * Provides efficient memory-mapped access to GGUF files.
 * Enables zero-copy tensor data access and efficient chunked VRAM transfer.
 * 
 * Spec: M0-W-1221
 */

#ifndef WORKER_IO_MMAP_FILE_H
#define WORKER_IO_MMAP_FILE_H

#include <cstddef>
#include <string>
#include <memory>

namespace worker {
namespace io {

/**
 * Memory-mapped file for efficient GGUF access
 * 
 * Uses mmap() on Linux/macOS for zero-copy file access.
 * Automatically unmaps on destruction (RAII).
 */
class MmapFile {
public:
    /**
     * Open and memory-map a file
     * 
     * @param path Absolute path to file
     * @return MmapFile instance
     * @throws CudaError on failure (file not found, permission denied, mmap failed)
     */
    static MmapFile open(const std::string& path);
    
    /**
     * Destructor - unmaps file
     */
    ~MmapFile();
    
    // Non-copyable
    MmapFile(const MmapFile&) = delete;
    MmapFile& operator=(const MmapFile&) = delete;
    
    // Movable
    MmapFile(MmapFile&& other) noexcept;
    MmapFile& operator=(MmapFile&& other) noexcept;
    
    /**
     * Get base pointer to mapped data
     * 
     * @return Pointer to start of file
     */
    const void* data() const { return mapped_data_; }
    
    /**
     * Get file size
     * 
     * @return Size in bytes
     */
    size_t size() const { return file_size_; }
    
    /**
     * Get pointer to data at offset
     * 
     * @param offset Offset from start of file
     * @return Pointer to data at offset
     * @throws CudaError if offset >= file_size
     */
    const void* get_data_at(size_t offset) const;
    
    /**
     * Get pointer to tensor data at offset with size validation
     * 
     * @param offset Offset from start of file
     * @param size Size of data to access
     * @return Pointer to data at offset
     * @throws CudaError if offset + size > file_size
     */
    const void* get_tensor_data(size_t offset, size_t size) const;
    
    /**
     * Check if file is mapped
     * 
     * @return true if mapped, false otherwise
     */
    bool is_mapped() const { return mapped_data_ != nullptr; }
    
    /**
     * Get file path
     * 
     * @return Original file path
     */
    const std::string& path() const { return path_; }

private:
    /**
     * Private constructor (use open() factory)
     */
    MmapFile(const std::string& path, void* mapped_data, size_t file_size, int fd);
    
    /**
     * Unmap file (called by destructor and move assignment)
     */
    void unmap();
    
    std::string path_;
    void* mapped_data_;
    size_t file_size_;
    int fd_;  // File descriptor (Linux/macOS)
};

} // namespace io
} // namespace worker

#endif // WORKER_IO_MMAP_FILE_H

// ---
// Implemented by Llama-Beta 🦙
