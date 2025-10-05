# Morning Debug Plan - 2025-10-06

**Status**: ✅ HTTP BUGS FIXED! Now debugging transformer  
**Priority**: Fix transformer forward pass to get real haiku  

## Current State - MAJOR PROGRESS! 🎉

**BREAKTHROUGH (22:37)**: Both critical HTTP bugs are **FIXED**!

✅ Model loads successfully (17s, 291 tensors, 1.2GB)  
✅ Worker starts and stays alive  
✅ HTTP server responds  
✅ **HTTP `/execute` endpoint works!** ← FIXED!  
✅ **C++ reads embeddings correctly!** ← FIXED!  
✅ Tokenizer works  
✅ Inference runs without crashing  
✅ 100 tokens generated  
✅ SSE streaming works end-to-end  
❌ **Token quality is garbage** (transformer has bugs)  

## Bugs Fixed Last Night ✅

1. ✅ **GPU pointer lifetime** - Added global registry to keep pointers alive
2. ✅ **Type cast in ffi_inference.cpp:62** - Fixed `CudaModel*` → `ModelImpl*` → `QwenModel*`

**Result**: HTTP pipeline fully working, inference runs, no crashes!

## Current Issue: Transformer Forward Pass

**NOT a stub issue** - We're already using the real `QwenTransformer::forward()`!

The transformer is running but producing garbage logits. Need to debug:

### Symptoms
- ✅ Worker starts successfully
- ✅ Model loads in 1 second
- ✅ `GET /health` works perfectly
- ❌ `POST /execute` fails with "error sending request for url"
- ✅ Worker process confirmed alive during failure

### Evidence Location
- Test: `tests/haiku_generation_anti_cheat.rs:117`
- Error: Line 117 calls `harness.execute(req).await`
- Connection fails before reaching server

### Hypothesis Priority

#### 1. **Request Body Issue** (70% likely) ⭐ START HERE
The test constructs a request but we haven't verified the JSON is valid.

**Quick Test**:
```bash
# Start worker manually
cargo run -p worker-orcd --features cuda -- \
  --worker-id test \
  --model .test-models/qwen/qwen2.5-0.5b-instruct-fp16.gguf \
  --port 9999 \
  --gpu-device 0 \
  --callback-url http://localhost:9999/callback

# In another terminal - test with curl
curl -v -X POST http://localhost:9999/execute \
  -H "Content-Type: application/json" \
  -d '{"job_id":"test","prompt":"hello","max_tokens":10}'
```

**What to check**:
- Does curl succeed? → Bug is in test harness
- Does curl fail? → Bug is in server

#### 2. **Missing Route Handler** (20% likely)
Maybe `/execute` handler isn't compiled in?

**Check**:
```bash
# Search for execute handler
rg "handle_execute" bin/worker-crates/worker-http/
rg "fn execute" bin/worker-crates/worker-http/
```

**Verify**:
- Handler exists in `worker-http/src/routes/execute.rs`
- Handler is registered in router
- Handler is compiled (check `Cargo.toml` features)

#### 3. **Test Harness Bug** (10% likely)
Maybe `harness.execute()` has wrong URL construction?

**Check**:
```rust
// In tests/integration/framework.rs or similar
// Look for execute() implementation
// Verify URL is: base_url + "/execute"
```

### Debug Steps (30 minutes)

1. **[5 min] Manual curl test** - Isolate server vs client
2. **[10 min] Add server-side logging** - See if request reaches handler
3. **[10 min] Check test harness** - Verify URL construction
4. **[5 min] Compare health vs execute** - Find the difference

### Files to Check

```
bin/worker-crates/worker-http/src/routes/execute.rs
bin/worker-crates/worker-http/src/routes/health.rs
bin/worker-crates/worker-http/src/routes.rs (router)
bin/worker-orcd/src/tests/integration/framework.rs (test harness)
bin/worker-orcd/tests/haiku_generation_anti_cheat.rs:117
```

### Expected Fix

**If request body issue**:
```rust
// Fix request construction in test
let req = make_test_request(...);
req.some_missing_field = value;  // Add missing field
```

**If missing handler**:
```rust
// Add to router
.route("/execute", post(execute::handle_execute::<B>))
```

**If test harness bug**:
```rust
// Fix URL construction
pub async fn execute(&self, req: Request) -> Result<Response> {
    let url = format!("{}/execute", self.base_url);  // Was missing /execute?
    // ...
}
```

---

## Bug #2: C++ GPU Pointer Reading (All Zeros)

### Symptoms
- ✅ Rust loads data correctly (verified with non-zero checks)
- ✅ Rust uploads to GPU (1.2GB VRAM allocated)
- ✅ Pointers passed to C++ (291 tensors)
- ❌ C++ reads embeddings as all zeros
- ❌ "First 10 embedding values: 0.00 0.00 0.00..."

### Evidence Location
- Rust verification: `src/cuda/weight_loader.rs:516-528` (non-zero checks)
- C++ wiring: `cuda/src/model/qwen_weight_loader.cpp:277-331`
- C++ usage: Wherever embeddings are first accessed

### Root Cause Analysis

The bug is **NOT** in:
- ✅ Rust loading (verified data is non-zero)
- ✅ GPU allocation (1.2GB allocated successfully)
- ✅ Pointer passing (C++ receives 291 pointers)

The bug **IS** in:
- ❌ C++ reading from GPU pointers
- ❌ Or pointer type mismatch (f16 vs f32?)
- ❌ Or wrong memory interpretation

### Hypothesis Priority

#### 1. **Type Mismatch** (60% likely) ⭐ START HERE
C++ might be reading f16 pointers as f32, causing garbage values.

**Check**:
```cpp
// In qwen_weight_loader.cpp:299
model->weights.token_embd = get_ptr("token_embd.weight");

// Where is this pointer used?
// Is it cast to half* or float*?
```

**Expected issue**:
```cpp
// WRONG - reading f16 as f32
float* embd = (float*)model->weights.token_embd;

// CORRECT - reading f16 as f16
half* embd = (half*)model->weights.token_embd;
```

#### 2. **Pointer Not Dereferenced** (30% likely)
C++ might be printing the pointer address instead of the data.

**Check**:
```cpp
// Look for where embeddings are printed
// Is it: printf("%f", *ptr) or printf("%p", ptr)?
```

#### 3. **Wrong Offset/Stride** (10% likely)
C++ might be reading from wrong memory location.

### Debug Steps (45 minutes)

1. **[10 min] Find embedding usage** - Where does C++ first read token_embd?
2. **[15 min] Add C++ debug prints** - Print first 10 values from GPU
3. **[10 min] Check type casts** - Verify half* vs float*
4. **[10 min] Compare Rust vs C++** - Print same pointer from both sides

### Files to Check

```
cuda/src/model/qwen_weight_loader.cpp:299 (where token_embd is set)
cuda/src/model/qwen_model.h (QwenWeights struct definition)
cuda/src/inference_impl.cpp (where embeddings are used)
cuda/src/transformer/qwen_transformer.cpp (embedding lookup)
```

### Debug Code to Add

**In C++ (after line 299)**:
```cpp
// Verify we can read the data
half* embd_ptr = (half*)model->weights.token_embd;
fprintf(stderr, "🔍 [C++] First 10 embedding values from GPU:\n");
half host_values[10];
cudaMemcpy(host_values, embd_ptr, 10 * sizeof(half), cudaMemcpyDeviceToHost);
for (int i = 0; i < 10; i++) {
    fprintf(stderr, "  [%d] = %.4f\n", i, __half2float(host_values[i]));
}
```

**In Rust (after line 528)**:
```rust
// Print what we uploaded
eprintln!("🔍 [Rust] First 10 bytes uploaded for {}:", tensor.name);
let first_10: Vec<u8> = bytes.iter().take(10).copied().collect();
eprintln!("   {:?}", first_10);
```

### Expected Fix

**If type mismatch**:
```cpp
// Change all weight pointers from void* to half*
struct QwenWeights {
    half* token_embd;  // Was: void*
    // ...
};
```

**If wrong dereference**:
```cpp
// Fix embedding lookup
half* embd = (half*)weights.token_embd;
half value = embd[token_id * hidden_dim + i];  // Add proper indexing
```

---

## Execution Strategy

### Phase 1: HTTP Bug (30 min)
1. Start worker manually
2. Test with curl
3. Add logging
4. Fix and verify

### Phase 2: C++ Pointer Bug (45 min)
1. Find where embeddings are used
2. Add debug prints
3. Check type casts
4. Fix and verify

### Phase 3: Integration Test (30 min)
1. Run haiku test
2. Verify both bugs fixed
3. Generate first haiku! 🎉

### Phase 4: Cleanup (15 min)
1. Remove debug prints
2. Update documentation
3. Commit fixes

---

## Success Criteria

### HTTP Bug Fixed
```bash
✅ curl -X POST http://localhost:9999/execute works
✅ Test harness execute() works
✅ Server logs show request received
```

### C++ Pointer Bug Fixed
```bash
✅ C++ prints non-zero embedding values
✅ Embeddings match Rust verification
✅ Forward pass produces non-zero logits
```

### Haiku Test Passes
```bash
✅ Worker starts
✅ Model loads (1s)
✅ Health check passes
✅ Execute request succeeds
✅ Tokens generated
✅ Haiku contains minute word
✅ Test completes in <30s
```

---

## Contingency Plans

### If HTTP bug takes >1 hour
- Skip to C++ bug
- Come back to HTTP later
- HTTP might be test harness issue, not critical path

### If C++ bug takes >1 hour
- Add more debug prints
- Compare with working CPU code
- Check CUDA samples for f16 usage

### If both bugs take >3 hours
- Focus on ONE bug completely
- Get partial success (either HTTP or inference)
- Document blockers for next session

---

## Tools & Commands

### Start Worker
```bash
cd /home/vince/Projects/llama-orch
cargo run -p worker-orcd --features cuda -- \
  --worker-id test \
  --model .test-models/qwen/qwen2.5-0.5b-instruct-fp16.gguf \
  --port 9999 \
  --gpu-device 0 \
  --callback-url http://localhost:9999/callback
```

### Test with Curl
```bash
# Health check
curl http://localhost:9999/health

# Execute request
curl -v -X POST http://localhost:9999/execute \
  -H "Content-Type: application/json" \
  -d '{
    "job_id": "test-001",
    "prompt": "Write a haiku about GPU computing",
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

### Run Haiku Test
```bash
cargo test -p worker-orcd --features cuda test_haiku_generation_STUB_PIPELINE_ONLY -- --nocapture --ignored
```

### Check Logs
```bash
# Worker logs
tail -f /tmp/worker-orcd.log  # If logging to file

# Test output
# Already in terminal with --nocapture
```

### Rebuild C++
```bash
cd bin/worker-orcd
cargo clean -p worker-orcd
cargo build -p worker-orcd --features cuda
```

---

## Key Insights from Last Session

1. **50× speedup is REAL** - Data is loaded correctly in Rust
2. **Verification is critical** - Non-zero checks caught the C++ bug
3. **Batch operations work** - Pre-allocation was the key optimization
4. **FP16 bypasses quantization** - Avoiding Q4_K issues entirely

---

## Expected Timeline

| Time | Task | Status |
|------|------|--------|
| 00:00 | Read plan, start worker | ⏳ |
| 00:05 | Test HTTP with curl | ⏳ |
| 00:15 | Fix HTTP bug | ⏳ |
| 00:30 | Find C++ embedding usage | ⏳ |
| 00:45 | Add C++ debug prints | ⏳ |
| 01:00 | Fix C++ pointer bug | ⏳ |
| 01:15 | Run haiku test | ⏳ |
| 01:30 | **HAIKU GENERATED!** 🎉 | ⏳ |
| 02:00 | Cleanup & document | ⏳ |

---

## Notes

- Both bugs are **independent** - can be fixed in parallel
- HTTP bug is **easier** - likely 15-30 min fix
- C++ bug is **more complex** - likely 30-60 min fix
- Total time: **2-3 hours** with high confidence

---

**Status**: Ready for morning session  
**Confidence**: High - clear debugging path for both bugs  
**Next Step**: Start with HTTP curl test (fastest to isolate)
