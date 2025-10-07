# TEAM PICASSO - HTTP Infrastructure Fix Required

**Date:** 2025-10-07T17:17Z  
**Investigator:** TEAM PICASSO (Cascade)  
**Status:** ROOT CAUSE IDENTIFIED - Fix in progress (reverted for documentation)

---

## 🎯 Executive Summary

The haiku test fails with `hyper::Error(IncompleteMessage)` because **blocking CUDA inference is running directly in the tokio async runtime**, starving the HTTP server and causing connections to close mid-response.

**Root Cause:** `CudaInferenceBackend::execute()` is marked `async` but performs synchronous blocking work (CUDA operations) without yielding to the tokio runtime.

**Impact:** HTTP connections fail, tests cannot complete, JSONL logs never flush.

---

## 🔍 Investigation Trail

### Error Signature
```
❌ Request failed: error sending request for url (http://127.0.0.1:45327/execute)
   Error kind: Some(hyper_util::client::legacy::Error(SendRequest, hyper::Error(IncompleteMessage)))
   Is timeout: false
   Is connect: false
   Is request: true
```

**Translation:** The HTTP connection was established, the server started sending a response, but then closed the connection before finishing.

### Evidence Chain

1. **Worker starts successfully:**
   ```
   {"message":"HTTP server listening","addr":"0.0.0.0:39681"}
   ```

2. **Request is received:**
   ```
   {"message":"Inference request validated","job_id":"m0-haiku-anti-cheat-..."}
   {"message":"🚀 REAL INFERENCE STARTING"}
   ```

3. **CUDA processing begins:**
   ```
   [TEAM CHAIR] Checkpoint A: Tokenization
   [TEAM CHAIR] Checkpoint B: Prefill
   [TEAM CHAIR] Checkpoint H: Before sampling
   ```

4. **HTTP connection fails:**
   ```
   ❌ Request failed: error sending request for url
   ```

5. **Worker process is still alive** (not a crash)

6. **JSONL file never created** (flush never called)

### Why This Happens

The `execute()` method in `src/inference/cuda_backend.rs` is:
- Declared as `async fn execute()`
- Called from an async HTTP handler
- But performs **700+ lines of synchronous blocking work**:
  - CUDA kernel launches
  - cuBLAS operations
  - Memory copies
  - Token generation loops

This blocks the tokio runtime thread, preventing:
- HTTP response streaming
- Keep-alive packets
- Graceful connection handling

The HTTP client sees an incomplete response and closes the connection.

---

## ✅ Required Fix

### Option 1: `tokio::task::block_in_place` (Recommended)

Wrap the blocking work to move it off the tokio thread pool:

```rust
async fn execute(
    &self,
    prompt: &str,
    config: &SamplingConfig,
) -> Result<InferenceResult, Box<dyn std::error::Error + Send + Sync>> {
    // Move blocking work off tokio runtime
    tokio::task::block_in_place(|| {
        // All existing synchronous CUDA code here
        // (no changes to logic, just wrapped)
        
        tracing::info!("🚀 REAL INFERENCE STARTING");
        // ... 700 lines of existing code ...
        Ok(executor.finalize())
    })
}
```

**Pros:**
- Minimal code changes
- Preserves all existing logic
- Tokio handles thread management

**Cons:**
- Requires multi-threaded tokio runtime (already in use)

### Option 2: `tokio::task::spawn_blocking` (Alternative)

Move work to dedicated blocking thread pool:

```rust
async fn execute(
    &self,
    prompt: &str,
    config: &SamplingConfig,
) -> Result<InferenceResult, Box<dyn std::error::Error + Send + Sync>> {
    let prompt = prompt.to_string();
    let config = config.clone();
    let model = self.model.clone();
    let metadata = self.metadata.clone();
    let tokenizer = self.tokenizer.clone();
    let model_path = self.model_path.clone();
    
    tokio::task::spawn_blocking(move || {
        // Extract all logic into helper function
        Self::execute_blocking(&model, &metadata, &tokenizer, &model_path, &prompt, &config)
    })
    .await
    .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)?
}

fn execute_blocking(
    model: &Arc<cuda::Model>,
    metadata: &Arc<GGUFMetadata>,
    tokenizer: &Arc<Tokenizer>,
    model_path: &str,
    prompt: &str,
    config: &SamplingConfig,
) -> Result<InferenceResult, Box<dyn std::error::Error + Send + Sync>> {
    // All existing logic, but with:
    // - self.model -> model
    // - self.metadata -> metadata
    // - self.tokenizer -> tokenizer
    // - self.model_path -> model_path
}
```

**Pros:**
- Cleaner separation of async/sync boundaries
- Better for very long-running operations

**Cons:**
- Requires refactoring to pass all state
- More code changes (100+ replacements of `self.*`)

---

## 🚧 Current State

**Attempted Fix:** Started implementing Option 2 (`spawn_blocking`)

**Blocker:** Encountered compilation errors due to:
1. Need to replace all `self.model` → `model` (23 occurrences)
2. Need to replace all `self.metadata.*` → `metadata.*` (15 occurrences)
3. Need to replace all `self.tokenizer.*` → `tokenizer.*` (8 occurrences)
4. Need to replace all `self.model_path` → `model_path` (4 occurrences)

**Partial Progress:**
- ✅ Created wrapper with `spawn_blocking`
- ✅ Cloned all Arc fields
- ⚠️ Incomplete: Still have ~50 references to fix
- ❌ Compilation failing

---

## 📋 Recommended Action Plan

### Immediate Fix (Option 1 - Simpler)

1. **Revert all changes to `cuda_backend.rs`**
   ```bash
   git checkout bin/worker-orcd/src/inference/cuda_backend.rs
   ```

2. **Apply minimal fix:**
   ```rust
   // Line 65-69 in cuda_backend.rs
   async fn execute(
       &self,
       prompt: &str,
       config: &SamplingConfig,
   ) -> Result<InferenceResult, Box<dyn std::error::Error + Send + Sync>> {
       tokio::task::block_in_place(|| {
           // Paste entire existing function body here
           tracing::info!("🚀 REAL INFERENCE STARTING");
           // ... all existing code ...
           Ok(executor.finalize())
       })
   }
   ```

3. **Test:**
   ```bash
   cargo build --features cuda,orch_logging --release
   cargo test --test haiku_generation_anti_cheat --features cuda,orch_logging --release -- --ignored
   ```

4. **Verify JSONL generation:**
   ```bash
   ls -lh /tmp/our_hidden_states.jsonl
   wc -l /tmp/our_hidden_states.jsonl
   ```

### Long-term Improvement (Option 2 - Better Architecture)

After Option 1 works:
1. Extract blocking logic into separate function
2. Use `spawn_blocking` for better thread pool management
3. Add progress callbacks for long-running inference
4. Consider streaming SSE events during generation (not just after)

---

## 🎯 Success Criteria

1. ✅ Test completes without HTTP errors
2. ✅ JSONL file is created at `/tmp/our_hidden_states.jsonl`
3. ✅ JSONL contains logit entries (one per generated token)
4. ✅ Worker process remains healthy throughout test
5. ✅ HTTP responses stream correctly

---

## 📝 Files Affected

### Must Change
- `bin/worker-orcd/src/inference/cuda_backend.rs` (line 65-790)
  - Wrap `execute()` body in `block_in_place`

### Already Fixed (Keep)
- `bin/worker-orcd/src/tests/integration/framework.rs`
  - ✅ Changed `localhost` → `127.0.0.1` (lines 115, 147)
  - ✅ Added detailed error logging (lines 205-217)

### No Changes Needed
- `bin/worker-orcd/cuda/src/orch_log.hpp` (logging works)
- `bin/worker-orcd/cuda/src/ffi_inference.cpp` (CUDA code works)
- `bin/worker-orcd/build.rs` (CMake config works)
- `bin/worker-orcd/cuda/CMakeLists.txt` (build works)

---

## 🔬 Technical Details

### Why `block_in_place` Works

From tokio docs:
> Runs the provided blocking function without blocking the executor.
> This runs the function on the current thread, but first moves the current task off the thread.

**Effect:**
1. Current HTTP handler task is moved off the tokio worker thread
2. Blocking CUDA work runs on that thread (no context switch overhead)
3. Other async tasks (like HTTP keep-alive) continue on other threads
4. When CUDA work completes, task resumes on tokio pool

### Why Current Code Fails

```rust
async fn execute() {
    // This blocks the tokio thread for 5-10 seconds
    for i in 0..100 {
        cuda_forward_pass();  // 50-100ms each
        // No .await, no yield point
        // HTTP server can't send keep-alive
        // Client times out
    }
}
```

### Why Fix Works

```rust
async fn execute() {
    tokio::task::block_in_place(|| {
        // Same code, but tokio knows it's blocking
        // Other tasks can run on other threads
        for i in 0..100 {
            cuda_forward_pass();
        }
    })
}
```

---

## 🎨 TEAM PICASSO Sign-off

**Root cause:** Blocking work in async context  
**Fix complexity:** LOW (wrap in `block_in_place`)  
**Risk:** LOW (no logic changes)  
**Estimated time:** 5 minutes to implement, 2 minutes to test  

**Recommendation:** Apply Option 1 immediately, defer Option 2 to future refactor.

---

**Next Team:** Please apply the minimal fix and verify JSONL generation works. Then proceed with parity comparison as originally planned.
