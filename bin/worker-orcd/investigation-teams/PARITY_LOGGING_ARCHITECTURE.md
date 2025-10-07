# Parity Logging Architecture - TEAM PICASSO

**Date:** 2025-10-07T17:36Z  
**Status:** Design Document  
**Purpose:** Proper async logging that doesn't break HTTP

---

## 🔴 What Went Wrong

### My Broken Implementation

**Attempt 1: Buffered with Mutex**
```cpp
void log_values(...) {
    std::lock_guard<std::mutex> lock(mutex_);  // BLOCKS
    entry.ts = get_timestamp();  // SYSTEM CALL while holding lock
    // ... string allocations, vector operations ...
    entries.push_back(entry);
}
```
**Problem:** Mutex contention + system calls on hot path = HTTP timeouts

**Attempt 2: Direct File I/O**
```cpp
void log_values(...) {
    FILE* f = fopen(log_file, "a");  // DISK I/O on every token!
    fprintf(f, ...);  // SLOW
    fclose(f);
}
```
**Problem:** Opening/closing file 100 times per inference = 5+ seconds overhead

**Result:** HTTP connections timeout with `hyper::Error(IncompleteMessage)`

### Why llama.cpp Works

**llama.cpp's logger:**
```cpp
void log_values(...) {
    if (!enabled) return;
    entries.push_back(entry);  // NO MUTEX (single-threaded)
}
```

**Key differences:**
1. ✅ **Single-threaded** - No mutex needed
2. ✅ **Buffers in memory** - No disk I/O until exit
3. ✅ **Flushes at exit** - Uses `atexit()`

**worker-orcd is different:**
- ❌ **Multi-threaded** - Tokio async runtime with thread pool
- ❌ **Long-running** - HTTP server stays alive, can't rely on `atexit()`
- ❌ **Concurrent requests** - Multiple inference jobs might run

---

## ✅ Proper Architecture

### Design Principles

1. **Zero overhead on hot path** - Logging must not block inference
2. **Lock-free or minimal locking** - No contention on CUDA thread
3. **Async I/O** - File writes happen on background thread
4. **Bounded memory** - Don't accumulate unbounded entries
5. **Graceful degradation** - If logging fails, inference continues

### Architecture: Lock-Free Queue + Background Thread

```
┌─────────────────────────────────────────────────────┐
│ CUDA Thread (Hot Path)                              │
│                                                      │
│  generate_token() {                                 │
│    // ... CUDA work ...                             │
│    ORCH_LOG_LOGITS(ptr, count, idx);  // ← Fast!   │
│  }                                                   │
│                                                      │
│  ORCH_LOG_LOGITS macro:                             │
│    1. Copy 10 floats to stack buffer (fast)         │
│    2. Push to lock-free queue (atomic op)           │
│    3. Return immediately (< 1 microsecond)          │
└──────────────────┬──────────────────────────────────┘
                   │ Lock-free queue
                   │ (bounded, fixed size)
                   ▼
┌─────────────────────────────────────────────────────┐
│ Background Thread (Async I/O)                       │
│                                                      │
│  while (true) {                                     │
│    entry = queue.pop_blocking();  // Wait for data │
│    batch.push(entry);                               │
│                                                      │
│    if (batch.size() >= 100 || timeout) {           │
│      FILE* f = fopen(log_file, "a");               │
│      for (entry in batch) {                         │
│        fprintf(f, ...);  // Batch write            │
│      }                                               │
│      fclose(f);                                     │
│      batch.clear();                                 │
│    }                                                 │
│  }                                                   │
└─────────────────────────────────────────────────────┘
```

### Implementation Plan

#### Phase 1: Lock-Free Queue (C++)

```cpp
// orch_log.hpp

#include <atomic>
#include <thread>
#include <condition_variable>

namespace worker_orch_log {

// Minimal log entry for queue (POD type)
struct QueueEntry {
    float values[10];  // Fixed size, no heap allocation
    int token_idx;
    const char* checkpoint;  // String literal, no allocation
    uint64_t timestamp_ns;   // Capture time, format later
};

// Lock-free bounded queue (SPSC - Single Producer Single Consumer)
class LockFreeQueue {
private:
    static constexpr size_t CAPACITY = 1024;  // Power of 2
    QueueEntry buffer_[CAPACITY];
    std::atomic<size_t> write_pos_{0};
    std::atomic<size_t> read_pos_{0};
    
public:
    // Producer (CUDA thread) - non-blocking
    bool try_push(const QueueEntry& entry) {
        size_t write = write_pos_.load(std::memory_order_relaxed);
        size_t next_write = (write + 1) % CAPACITY;
        size_t read = read_pos_.load(std::memory_order_acquire);
        
        if (next_write == read) {
            return false;  // Queue full, drop entry
        }
        
        buffer_[write] = entry;
        write_pos_.store(next_write, std::memory_order_release);
        return true;
    }
    
    // Consumer (background thread) - blocking with timeout
    bool pop(QueueEntry& entry, int timeout_ms) {
        size_t read = read_pos_.load(std::memory_order_relaxed);
        size_t write = write_pos_.load(std::memory_order_acquire);
        
        if (read == write) {
            return false;  // Queue empty
        }
        
        entry = buffer_[read];
        read_pos_.store((read + 1) % CAPACITY, std::memory_order_release);
        return true;
    }
};

class Logger {
private:
    LockFreeQueue queue_;
    std::thread worker_thread_;
    std::atomic<bool> shutdown_{false};
    const char* log_file_;
    bool enabled_;
    
    Logger() {
        log_file_ = std::getenv("ORCH_LOG_FILE");
        enabled_ = (log_file_ != nullptr);
        
        if (enabled_) {
            worker_thread_ = std::thread(&Logger::worker_loop, this);
        }
    }
    
    ~Logger() {
        if (enabled_) {
            shutdown_.store(true);
            worker_thread_.join();
        }
    }
    
    void worker_loop() {
        std::vector<QueueEntry> batch;
        batch.reserve(100);
        
        auto last_flush = std::chrono::steady_clock::now();
        
        while (!shutdown_.load()) {
            QueueEntry entry;
            
            // Try to pop with timeout
            if (queue_.pop(entry, 100)) {
                batch.push_back(entry);
            }
            
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_flush);
            
            // Flush if batch is full or timeout
            if (batch.size() >= 100 || elapsed.count() >= 1000) {
                flush_batch(batch);
                batch.clear();
                last_flush = now;
            }
        }
        
        // Final flush on shutdown
        if (!batch.empty()) {
            flush_batch(batch);
        }
    }
    
    void flush_batch(const std::vector<QueueEntry>& batch) {
        FILE* f = fopen(log_file_, "a");
        if (!f) return;
        
        for (const auto& entry : batch) {
            // Format timestamp
            time_t sec = entry.timestamp_ns / 1000000000;
            struct tm* tm_info = gmtime(&sec);
            char ts_buf[32];
            strftime(ts_buf, sizeof(ts_buf), "%Y-%m-%dT%H:%M:%SZ", tm_info);
            
            // Write JSONL
            fprintf(f, "{\"ts\":\"%s\",\"team\":\"worker-orcd\",\"checkpoint\":\"%s\",\"token_idx\":%d,\"dtype\":\"f32\",\"shape\":\"[1,151936]\",\"values\":[",
                    ts_buf, entry.checkpoint, entry.token_idx);
            
            for (int i = 0; i < 10; ++i) {
                if (i > 0) fprintf(f, ",");
                fprintf(f, "%.6f", entry.values[i]);
            }
            
            fprintf(f, "]}\n");
        }
        
        fclose(f);
    }
    
public:
    static Logger& get_instance() {
        static Logger instance;
        return instance;
    }
    
    // Called from CUDA thread (HOT PATH - must be fast!)
    void log_values(const char* checkpoint, const float* data, int count, int token_idx) {
        if (!enabled_) return;
        
        QueueEntry entry;
        entry.checkpoint = checkpoint;  // String literal, no copy
        entry.token_idx = token_idx;
        
        // Copy only 10 values (fast)
        int n = std::min(count, 10);
        for (int i = 0; i < n; ++i) {
            entry.values[i] = data[i];
        }
        for (int i = n; i < 10; ++i) {
            entry.values[i] = 0.0f;
        }
        
        // Capture timestamp (fast - just read CPU counter)
        auto now = std::chrono::high_resolution_clock::now();
        entry.timestamp_ns = now.time_since_epoch().count();
        
        // Push to queue (lock-free, atomic operation)
        if (!queue_.try_push(entry)) {
            // Queue full, drop entry (graceful degradation)
            // Could increment a dropped_count metric here
        }
    }
};

} // namespace worker_orch_log

// Macro for easy use
#define ORCH_LOG_LOGITS(ptr, count, token_idx) \
    worker_orch_log::Logger::get_instance().log_values("logits", ptr, count, token_idx)
```

#### Phase 2: Integration Points

**1. Startup (main.rs or cuda_backend.rs)**
```rust
// Logger starts background thread automatically on first use
// No explicit initialization needed
```

**2. Logging Call (ffi_inference.cpp)**
```cpp
// Existing call site - no changes needed
ORCH_LOG_LOGITS(ctx->logits_buffer, ctx->model->config.vocab_size, generation_token_idx);
```

**3. Shutdown (Drop impl)**
```rust
impl Drop for CudaInferenceBackend {
    fn drop(&mut self) {
        // Flush any remaining logs
        unsafe { orch_log_flush_now(); }
    }
}
```

#### Phase 3: Testing

**Test 1: Zero Overhead**
```bash
# Measure inference time WITHOUT logging
time cargo test --features cuda --release

# Measure inference time WITH logging
time cargo test --features cuda,orch_logging --release

# Difference should be < 1%
```

**Test 2: HTTP Stability**
```bash
# Run haiku test with logging enabled
ORCH_LOG_FILE=/tmp/test.jsonl \
cargo test --test haiku_generation_anti_cheat \
  --features cuda,orch_logging --release -- --ignored

# Should PASS (not timeout)
```

**Test 3: Log Correctness**
```bash
# Verify JSONL is valid
cat /tmp/test.jsonl | jq . > /dev/null

# Verify entry count matches token count
wc -l /tmp/test.jsonl  # Should be ~100 for haiku test
```

---

## 📋 Extensibility for Other Teams

### Adding New Checkpoints

**Example: Log attention outputs**

```cpp
// In attention kernel (cuda/src/kernels/attention.cu)
#ifdef ORCH_LOGGING
ORCH_LOG_ATTENTION(attention_output, hidden_dim, token_idx);
#endif
```

**Define new macro in orch_log.hpp:**
```cpp
#define ORCH_LOG_ATTENTION(ptr, count, token_idx) \
    worker_orch_log::Logger::get_instance().log_values("attention_output", ptr, count, token_idx)
```

### Adding New Data Types

**Example: Log FP16 values**

```cpp
void log_values_f16(const char* checkpoint, const half* data, int count, int token_idx) {
    if (!enabled_) return;
    
    QueueEntry entry;
    entry.checkpoint = checkpoint;
    entry.token_idx = token_idx;
    
    // Convert FP16 to FP32 for logging
    for (int i = 0; i < std::min(count, 10); ++i) {
        entry.values[i] = __half2float(data[i]);
    }
    
    queue_.try_push(entry);
}
```

### Configuration

**Environment Variables:**
- `ORCH_LOG_FILE` - Output file path (required)
- `ORCH_LOG_TEAM` - Team identifier (default: "worker-orcd")
- `ORCH_LOG_VALUES` - Number of values per entry (default: 10)
- `ORCH_LOG_BATCH_SIZE` - Batch size for flushing (default: 100)
- `ORCH_LOG_FLUSH_MS` - Flush interval in milliseconds (default: 1000)

---

## 🎯 Success Criteria

1. ✅ **Zero HTTP failures** - Test passes with logging enabled
2. ✅ **< 1% overhead** - Logging adds minimal latency
3. ✅ **Valid JSONL** - Output parses correctly
4. ✅ **Complete logs** - All tokens logged (no drops under normal load)
5. ✅ **Graceful degradation** - Drops entries if queue full (doesn't crash)
6. ✅ **Clean shutdown** - Flushes remaining entries on exit

---

## 📊 Performance Comparison

| Implementation | Hot Path Cost | HTTP Impact | Memory Usage |
|----------------|---------------|-------------|--------------|
| **Broken v1** (mutex + syscalls) | ~100 μs/token | ❌ Timeouts | Low |
| **Broken v2** (direct file I/O) | ~50 ms/token | ❌ Timeouts | Minimal |
| **llama.cpp** (single-threaded buffer) | ~1 μs/token | ✅ Works | Medium |
| **Proposed** (lock-free queue) | ~0.5 μs/token | ✅ Works | Fixed (8KB) |

---

## 🔧 Implementation Checklist

- [ ] Implement `LockFreeQueue` class
- [ ] Implement `Logger` with background thread
- [ ] Add worker loop with batching
- [ ] Add graceful shutdown
- [ ] Update `ORCH_LOG_LOGITS` macro
- [ ] Test with haiku test
- [ ] Verify zero HTTP failures
- [ ] Measure performance overhead
- [ ] Document usage for other teams
- [ ] Add to investigation-teams/README.md

---

## 📚 References

### Similar Implementations
- **spdlog** - Fast C++ logging library with async support
- **Rust tracing** - Async logging with subscribers
- **Linux perf** - Lock-free ring buffer for tracing

### Key Techniques
- **Lock-free SPSC queue** - Single producer, single consumer
- **Batch writes** - Amortize file I/O cost
- **Fixed-size entries** - No heap allocations on hot path
- **Graceful degradation** - Drop entries instead of blocking

---

**TEAM PICASSO**  
**Status:** Design complete, ready for implementation  
**Estimated effort:** 4-6 hours (implementation + testing)
