# LT-004: Chunked Host-to-Device Transfer

**Team**: Llama-Beta  
**Sprint**: Sprint 1 - GGUF Foundation  
**Size**: M (2 days)  
**Days**: 22-23  
**Spec Ref**: M0-W-1222

---

## Story Description

Implement chunked host-to-device (H2D) transfer for loading large model weights from memory-mapped GGUF files to GPU VRAM. Transfer tensors in manageable chunks to avoid memory spikes and enable progress tracking for large models.

---

## Acceptance Criteria

- [ ] Implement chunked cudaMemcpy from host (mmap) to device (VRAM)
- [ ] Default chunk size: 256MB (configurable via env var)
- [ ] Transfer tensors in chunks with progress tracking
- [ ] Validate source pointer is within mmap bounds before transfer
- [ ] Validate destination pointer is within allocated VRAM before transfer
- [ ] Handle partial transfers (resume on error)
- [ ] Emit progress events for large transfers (>1GB)
- [ ] Unit tests validate chunked transfer logic
- [ ] Integration tests validate transfer with real GGUF tensors
- [ ] Error handling for cudaMemcpy failures
- [ ] Benchmark transfer throughput (GB/s)

---

## Dependencies

### Upstream (Blocks This Story)
- LT-003: Memory-Mapped I/O (needs mmap pointers)
- FT-013: Device Memory RAII (needs VRAM allocation)

### Downstream (This Story Blocks)
- LT-023: Qwen Weight Loading (needs chunked transfer)
- LT-030: Phi-3 Weight Loading (needs chunked transfer)

---

## Technical Details

### Files to Create/Modify
- `bin/worker-orcd/cuda/src/io/chunked_transfer.cpp` - Chunked transfer implementation
- `bin/worker-orcd/cuda/src/io/chunked_transfer.h` - Transfer interface
- `bin/worker-orcd/src/io/transfer.rs` - Rust transfer wrapper

### Key Interfaces
```cpp
struct TransferConfig {
    size_t chunk_size = 256 * 1024 * 1024;  // 256MB default
    bool enable_progress = true;
    cudaStream_t stream = nullptr;  // Use default stream if null
};

class ChunkedTransfer {
public:
    // Transfer tensor from host to device in chunks
    static Result<void> h2d_chunked(
        void* device_ptr,
        const void* host_ptr,
        size_t total_size,
        const TransferConfig& config
    );
    
    // Transfer with progress callback
    static Result<void> h2d_with_progress(
        void* device_ptr,
        const void* host_ptr,
        size_t total_size,
        const TransferConfig& config,
        std::function<void(size_t, size_t)> progress_callback
    );
};

// Error types
enum class TransferError {
    InvalidPointer,
    CudaMemcpyFailed,
    OutOfBounds,
    StreamSyncFailed,
};
```

```rust
pub struct TransferConfig {
    pub chunk_size: usize,
    pub enable_progress: bool,
}

impl ChunkedTransfer {
    pub fn h2d_chunked(
        device_ptr: *mut u8,
        host_ptr: *const u8,
        total_size: usize,
        config: &TransferConfig,
    ) -> Result<(), TransferError>;
}
```

### Implementation Notes
- Use `cudaMemcpy` with `cudaMemcpyHostToDevice`
- Transfer in chunks: `min(chunk_size, remaining_bytes)`
- Track progress: `bytes_transferred / total_size`
- Emit progress events every 10% for large transfers
- Validate pointers before each chunk transfer
- Use CUDA streams for async transfer (optional)
- Log transfer start/end with size and duration

**Chunked Transfer Logic**:
```cpp
size_t offset = 0;
while (offset < total_size) {
    size_t chunk = std::min(config.chunk_size, total_size - offset);
    
    cudaError_t err = cudaMemcpy(
        (char*)device_ptr + offset,
        (const char*)host_ptr + offset,
        chunk,
        cudaMemcpyHostToDevice
    );
    
    if (err != cudaSuccess) {
        return Err(TransferError::CudaMemcpyFailed);
    }
    
    offset += chunk;
    
    if (config.enable_progress && progress_callback) {
        progress_callback(offset, total_size);
    }
}
```

---

## Testing Strategy

### Unit Tests
- Test chunked transfer with various sizes (1MB, 100MB, 1GB)
- Test chunk size boundary conditions (exact multiple, partial chunk)
- Test progress callback invocation
- Test error handling for invalid pointers
- Test error handling for cudaMemcpy failures
- Test transfer with different chunk sizes (64MB, 256MB, 512MB)

### Integration Tests
- Test transfer of real GGUF tensor to VRAM
- Test transfer of large tensor (>1GB) with progress tracking
- Test concurrent transfers (multiple tensors)
- Test transfer + VRAM residency validation

### Performance Tests
- Benchmark transfer throughput (GB/s)
- Compare chunked vs. single cudaMemcpy (should be similar)
- Measure overhead of chunking (should be <1%)

### Manual Verification
1. Load Qwen2.5-0.5B tensor (e.g., 100MB weight matrix)
2. Transfer to VRAM in 256MB chunks
3. Verify progress events emitted
4. Check logs show transfer duration
5. Validate tensor data in VRAM (checksum)

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Code reviewed
- [ ] Unit tests passing (6+ tests)
- [ ] Integration tests passing
- [ ] Performance benchmarks recorded
- [ ] Documentation updated
- [ ] Story marked complete in day-tracker.md

---

## References

- Spec: `bin/.specs/01_M0_worker_orcd.md` Section 6.3 (Model Loading)
- CUDA cudaMemcpy docs: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html
- Related Stories: LT-003, LT-023, LT-030

---

**Status**: Ready for execution  
**Owner**: Llama-Beta  
**Created**: 2025-10-04

---

Detailed by Project Management Team — ready to implement 📋
