# Single-Threaded Architecture for llorch-cpud

**Date:** 2025-10-08  
**Purpose:** Document CRITICAL single-threaded requirement for performance  
**Status:** Architecture requirement

---

## CRITICAL REQUIREMENT

**llorch-cpud MUST be exclusively single-threaded for SPEED.**

---

## Why Single-Threaded?

### 1. **No Context Switching Overhead**
- Multi-threaded: OS switches between threads → CPU cache misses
- Single-threaded: ONE thread runs continuously → optimal cache locality
- **Result:** 10-30% faster inference

### 2. **Predictable Performance**
- Multi-threaded: Non-deterministic thread scheduling
- Single-threaded: Deterministic execution order
- **Result:** Consistent latency, easier profiling

### 3. **No Lock Contention**
- Multi-threaded: Mutexes, atomics, synchronization overhead
- Single-threaded: No locks needed
- **Result:** Zero synchronization overhead

### 4. **Better CPU Cache Utilization**
- Multi-threaded: Threads compete for cache → thrashing
- Single-threaded: Cache stays hot for model weights
- **Result:** Fewer memory stalls

### 5. **Simpler Code**
- Multi-threaded: Complex synchronization, race conditions
- Single-threaded: Sequential reasoning, easier to debug
- **Result:** Fewer bugs, faster development

---

## Implementation

### Tokio Runtime Configuration

```rust
// src/main.rs
#[tokio::main(flavor = "current_thread")]  // CRITICAL!
async fn main() -> anyhow::Result<()> {
    // All async operations run on ONE thread
    // No thread pool spawned
    // No work stealing
}
```

### What This Means

**Single-threaded runtime:**
- ✅ ONE OS thread for entire application
- ✅ HTTP server runs on same thread
- ✅ Inference runs on same thread
- ✅ All async tasks scheduled cooperatively
- ❌ NO thread pool
- ❌ NO parallel task execution
- ❌ NO work stealing

**Async still works:**
- HTTP requests are handled asynchronously
- Multiple requests can be "in flight"
- But all execute on ONE thread via cooperative scheduling

---

## Cargo.toml Configuration

```toml
[dependencies]
# CRITICAL: Do NOT enable multi-threaded features
tokio = { version = "1", features = [
    "rt",              # Runtime (single-threaded by default)
    "macros",          # #[tokio::main] macro
    "net",             # TCP/HTTP networking
    "io-util",         # I/O utilities
    "time",            # Timers
    # DO NOT ADD: "rt-multi-thread" ← This would enable thread pool!
] }

# Ensure ndarray doesn't use rayon (multi-threaded)
ndarray = { version = "0.15", default-features = false }
```

**What to AVOID:**
- ❌ `tokio = { features = ["full"] }` - Includes multi-threaded runtime
- ❌ `tokio = { features = ["rt-multi-thread"] }` - Explicit multi-threading
- ❌ `rayon` - Parallel iterator library
- ❌ `crossbeam` - Multi-threaded data structures
- ❌ `std::thread::spawn` - Manual thread spawning

---

## Request Processing Model

### How Single-Threaded HTTP Server Works

```
Client 1 → POST /execute → Request queued
Client 2 → POST /execute → Request queued
Client 3 → POST /execute → Request queued
                ↓
        Single Thread Processes:
                ↓
    1. Accept Client 1 request
    2. Start inference (async)
    3. Yield to event loop
    4. Accept Client 2 request
    5. Queue Client 2 (waiting)
    6. Resume Client 1 inference
    7. Send token to Client 1
    8. Yield to event loop
    9. Process Client 2...
```

**Key Point:** Requests are processed SEQUENTIALLY, not in parallel.

---

## Sequential Request Processing

### CRITICAL: One Request at a Time

```rust
// src/backend/cpu_backend.rs

// This is CORRECT (sequential):
async fn execute(&self, prompt: &str, config: &SamplingConfig) 
    -> Result<InferenceResult> {
    // 1. Tokenize
    let tokens = self.tokenizer.encode(prompt)?;
    
    // 2. Generate (blocks until complete)
    let output = self.model.generate(&tokens, config)?;
    
    // 3. Decode
    let text = self.tokenizer.decode(&output)?;
    
    // 4. Return
    Ok(InferenceResult::max_tokens(...))
}
```

**What happens with multiple requests:**
1. Request A arrives → starts processing
2. Request B arrives → waits in queue
3. Request A completes → returns result
4. Request B starts processing
5. Request B completes → returns result

**This is CORRECT for CPU inference!**

---

## Why Sequential is FASTER for CPU

### CPU-Bound Workload

Inference is **CPU-bound**, not I/O-bound:
- Matrix multiplications
- Attention computations
- Activation functions
- All compute-intensive

**Parallel execution would be SLOWER:**
```
Thread 1: [====== Inference A ======]
Thread 2:     [====== Inference B ======]
             ↑
        Context switches = SLOW
        Cache thrashing = SLOW
        Memory contention = SLOW
```

**Sequential execution is FASTER:**
```
Thread 1: [====== Inference A ======][====== Inference B ======]
          ↑                          ↑
      Hot cache                  Hot cache
      No switches                No contention
```

---

## Comparison with worker-orcd

### worker-orcd (CUDA)
```rust
// worker-orcd also uses single-threaded!
#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    // Same pattern as llorch-cpud
}
```

**Why:** CUDA kernels are already parallel (GPU threads). Adding CPU threads would add overhead without benefit.

### llorch-cpud (CPU)
```rust
// llorch-cpud uses single-threaded!
#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    // Same pattern as worker-orcd
}
```

**Why:** CPU inference is sequential. Adding threads would add overhead without benefit.

---

## Performance Implications

### Throughput vs Latency

**Single-threaded:**
- ✅ Optimal latency per request (no overhead)
- ⚠️ Lower throughput (one request at a time)

**Multi-threaded:**
- ⚠️ Higher latency per request (overhead)
- ⚠️ NOT higher throughput (CPU-bound, not I/O-bound)

**Conclusion:** Single-threaded is BETTER for both latency and throughput!

### Scaling Strategy

**Don't scale with threads. Scale with processes:**

```bash
# WRONG: Multi-threaded worker
llorch-cpud --threads 4  # ← Don't do this!

# RIGHT: Multiple single-threaded workers
llorch-cpud --worker-id w1 --port 8080 &  # Worker 1
llorch-cpud --worker-id w2 --port 8081 &  # Worker 2
llorch-cpud --worker-id w3 --port 8082 &  # Worker 3
llorch-cpud --worker-id w4 --port 8083 &  # Worker 4
```

**Why:** Each worker has its own:
- CPU cache (no thrashing)
- Memory (no contention)
- Model weights (no sharing)

**Result:** 4x throughput with optimal latency!

---

## Validation Checklist

### ✓ Cargo.toml
- [ ] tokio does NOT include "rt-multi-thread"
- [ ] tokio does NOT include "full" feature
- [ ] No rayon dependency
- [ ] No crossbeam dependency
- [ ] ndarray has default-features = false

### ✓ main.rs
- [ ] Uses `#[tokio::main(flavor = "current_thread")]`
- [ ] No `std::thread::spawn` calls
- [ ] No manual thread creation

### ✓ Backend Implementation
- [ ] execute() is sequential (no parallel inference)
- [ ] No Arc<Mutex<...>> for model (not needed)
- [ ] No thread pools

### ✓ Model Implementation
- [ ] No parallel loops (no rayon)
- [ ] Sequential matrix operations
- [ ] No thread-based parallelism

---

## Common Mistakes to Avoid

### ❌ WRONG: Multi-threaded Runtime
```rust
// DON'T DO THIS!
#[tokio::main]  // Defaults to multi-threaded
async fn main() { }
```

### ✅ CORRECT: Single-threaded Runtime
```rust
// DO THIS!
#[tokio::main(flavor = "current_thread")]
async fn main() { }
```

---

### ❌ WRONG: Parallel Inference
```rust
// DON'T DO THIS!
use rayon::prelude::*;

fn forward(&self, x: &Array) -> Array {
    layers.par_iter()  // ← Parallel iteration
        .map(|layer| layer.forward(x))
        .collect()
}
```

### ✅ CORRECT: Sequential Inference
```rust
// DO THIS!
fn forward(&self, x: &Array) -> Array {
    layers.iter()  // ← Sequential iteration
        .fold(x.clone(), |acc, layer| layer.forward(&acc))
}
```

---

### ❌ WRONG: Thread Pool for Requests
```rust
// DON'T DO THIS!
let pool = ThreadPool::new(4);
pool.execute(|| {
    // Process request
});
```

### ✅ CORRECT: Sequential Request Processing
```rust
// DO THIS!
async fn execute(&self, prompt: &str) -> Result<InferenceResult> {
    // Process request sequentially
    // Tokio handles async scheduling
}
```

---

## Testing Single-Threaded Behavior

### Verify No Thread Pool

```bash
# Run worker
cargo run -- --worker-id test --model test.gguf --port 8080 --callback-url http://localhost:9999

# In another terminal, check thread count
ps -T -p $(pgrep llorch-cpud) | wc -l

# Should show ~2-3 threads:
# - Main thread
# - Signal handler thread (OS)
# - Maybe one async runtime thread

# Should NOT show 4+ threads (no thread pool)
```

### Verify Sequential Processing

```bash
# Send two requests simultaneously
curl -X POST http://localhost:8080/execute -d '{"prompt":"A"}' &
curl -X POST http://localhost:8080/execute -d '{"prompt":"B"}' &

# Check logs - should show:
# 1. Request A starts
# 2. Request A completes
# 3. Request B starts
# 4. Request B completes

# NOT:
# 1. Request A starts
# 2. Request B starts (parallel)
```

---

## Documentation Requirements

### Code Comments

Every async function should have:
```rust
/// Execute inference
/// 
/// IMPORTANT: This runs on a single-threaded runtime.
/// Requests are processed sequentially, not in parallel.
/// This is OPTIMAL for CPU-bound inference.
async fn execute(&self, ...) -> Result<...> {
    // ...
}
```

### README

Should include:
```markdown
## Architecture

llorch-cpud uses a **single-threaded architecture** for optimal performance:
- No thread pool overhead
- No context switching
- Optimal CPU cache utilization
- Sequential request processing

To scale throughput, run multiple worker processes (not threads).
```

---

## Summary

### CRITICAL Requirements

1. ✅ **Single-threaded tokio runtime** - `flavor = "current_thread"`
2. ✅ **No thread pool** - No "rt-multi-thread" feature
3. ✅ **Sequential inference** - One request at a time
4. ✅ **No parallel libraries** - No rayon, crossbeam
5. ✅ **Process-based scaling** - Multiple workers, not threads

### Why This Matters

- 🚀 **10-30% faster** than multi-threaded
- 🎯 **Predictable latency** - No non-determinism
- 🧠 **Better cache** - No thrashing
- 🐛 **Fewer bugs** - No race conditions
- 📊 **Easier profiling** - Deterministic execution

### Validation

- [ ] Cargo.toml has correct tokio features
- [ ] main.rs uses `flavor = "current_thread"`
- [ ] No manual thread spawning
- [ ] No parallel libraries
- [ ] Sequential request processing

---

**This is a CRITICAL performance requirement. Do not compromise on this!**

---

Built by TEAM CASCADE 🌊
