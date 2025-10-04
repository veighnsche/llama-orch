# LT-003: Memory-Mapped I/O Implementation

**Team**: Llama-Beta  
**Sprint**: Sprint 1 - GGUF Foundation  
**Size**: M (2 days)  
**Days**: 20-21  
**Spec Ref**: M0-W-1221

---

## Story Description

Implement memory-mapped I/O for efficient GGUF file loading. Use mmap to map GGUF files directly into process address space, enabling zero-copy access to model weights and efficient chunked transfer to GPU VRAM.

---

## Acceptance Criteria

- [ ] Implement mmap-based file loading for GGUF files
- [ ] Map entire GGUF file into process address space
- [ ] Provide pointer access to tensor data via offsets
- [ ] Handle file mapping errors (file too large, permission denied)
- [ ] Implement proper cleanup (munmap) on destruction
- [ ] Support read-only mapping (MAP_PRIVATE or MAP_SHARED)
- [ ] Validate mapped region is accessible before use
- [ ] Unit tests validate mmap lifecycle (map, access, unmap)
- [ ] Integration tests validate mmap with real GGUF files
- [ ] Error handling for mmap failures (ENOMEM, EACCES)
- [ ] Log mmap operations at DEBUG level

---

## Dependencies

### Upstream (Blocks This Story)
- LT-001: GGUF Header Parser (needs header to know file structure)

### Downstream (This Story Blocks)
- LT-004: Chunked H2D Transfer (needs mmap pointers)
- LT-023: Qwen Weight Loading (needs mmap access)

---

## Technical Details

### Files to Create/Modify
- `bin/worker-orcd/cuda/src/io/mmap_file.cpp` - Memory-mapped file implementation
- `bin/worker-orcd/cuda/src/io/mmap_file.h` - MmapFile class interface
- `bin/worker-orcd/src/io/mmap.rs` - Rust mmap wrapper (optional)

### Key Interfaces
```cpp
class MmapFile {
public:
    // Open and map GGUF file
    static Result<MmapFile> open(const std::string& path);
    
    // Get pointer to tensor data at offset
    const void* get_tensor_data(size_t offset) const;
    
    // Get file size
    size_t size() const;
    
    // Get base pointer
    const void* data() const;
    
    // Destructor unmaps file
    ~MmapFile();
    
private:
    void* mapped_data_;
    size_t file_size_;
    int fd_;
};

// Error types
enum class MmapError {
    FileNotFound,
    PermissionDenied,
    FileTooLarge,
    MmapFailed,
    InvalidOffset,
};
```

```rust
pub struct MmapFile {
    mapped_data: *const u8,
    file_size: usize,
}

impl MmapFile {
    pub fn open(path: &Path) -> Result<Self, MmapError>;
    pub fn get_tensor_data(&self, offset: usize) -> Result<*const u8, MmapError>;
    pub fn size(&self) -> usize;
}
```

### Implementation Notes
- Use `mmap()` on Linux/macOS, `MapViewOfFile()` on Windows
- Map with `PROT_READ` (read-only access)
- Use `MAP_PRIVATE` to prevent modifications affecting file
- Validate offset + size <= file_size before returning pointers
- Log mmap operations: file path, size, mapping address
- Handle large files (>4GB) correctly on 64-bit systems
- Ensure proper alignment for CUDA transfers (page-aligned)

**Platform-Specific**:
```cpp
#ifdef __linux__
    void* addr = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (addr == MAP_FAILED) {
        return Err(MmapError::MmapFailed);
    }
#elif _WIN32
    HANDLE hMapFile = CreateFileMapping(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    void* addr = MapViewOfFile(hMapFile, FILE_MAP_READ, 0, 0, 0);
#endif
```

---

## Testing Strategy

### Unit Tests
- Test mmap lifecycle (open, access, close)
- Test error handling for non-existent file
- Test error handling for permission denied
- Test pointer access at various offsets
- Test file size reporting
- Test cleanup (munmap called on destruction)
- Test invalid offset access (beyond file size)

### Integration Tests
- Test mmap with real Qwen2.5-0.5B GGUF file
- Test mmap with large GGUF file (>1GB)
- Test concurrent mmap of same file (multiple readers)
- Test mmap + tensor data access

### Manual Verification
1. Load Qwen2.5-0.5B GGUF file via mmap
2. Verify file is mapped (check /proc/self/maps on Linux)
3. Access tensor data at various offsets
4. Verify no file I/O after initial mmap
5. Check munmap called on cleanup

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Code reviewed
- [ ] Unit tests passing (7+ tests)
- [ ] Integration tests passing
- [ ] Platform-specific code tested (Linux primary)
- [ ] Documentation updated
- [ ] Story marked complete in day-tracker.md

---

## References

- Spec: `bin/.specs/01_M0_worker_orcd.md` Section 6.3 (Model Loading)
- mmap man page: https://man7.org/linux/man-pages/man2/mmap.2.html
- Related Stories: LT-001, LT-004, LT-023

---

**Status**: Ready for execution  
**Owner**: Llama-Beta  
**Created**: 2025-10-04

---

Detailed by Project Management Team — ready to implement 📋
