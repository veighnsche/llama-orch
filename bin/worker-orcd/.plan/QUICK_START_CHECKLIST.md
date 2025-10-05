# Quick Start Checklist - Morning Session

**Goal**: Fix 2 bugs, generate haiku in 2-3 hours

---

## 🚀 Quick Start (5 minutes)

### 1. Terminal Setup
```bash
cd /home/vince/Projects/llama-orch

# Terminal 1: Worker
cargo run -p worker-orcd --features cuda -- \
  --worker-id test \
  --model .test-models/qwen/qwen2.5-0.5b-instruct-fp16.gguf \
  --port 9999 \
  --gpu-device 0 \
  --callback-url http://localhost:9999/callback

# Terminal 2: Testing
# (keep open for curl commands)
```

### 2. Verify Worker Started
```bash
# Should see:
# ✅ Model loaded in 1.0s
# ✅ HTTP server listening on 0.0.0.0:9999
```

---

## 🔍 Bug #1: HTTP `/execute` (30 min)

### Quick Test
```bash
# Terminal 2
curl http://localhost:9999/health
# Should return: {"status":"healthy"}

curl -v -X POST http://localhost:9999/execute \
  -H "Content-Type: application/json" \
  -d '{"job_id":"test","prompt":"hello","max_tokens":10}'
# Does this work?
```

### If curl WORKS → Bug is in test harness
- [ ] Check `src/tests/integration/framework.rs`
- [ ] Look for `execute()` method
- [ ] Verify URL construction

### If curl FAILS → Bug is in server
- [ ] Check `bin/worker-crates/worker-http/src/routes/execute.rs`
- [ ] Verify handler exists
- [ ] Add logging to handler
- [ ] Check router registration

### Fix Location
Likely in: `bin/worker-crates/worker-http/src/routes/`

---

## 🔍 Bug #2: C++ GPU Pointers (45 min)

### Find Embedding Usage
```bash
# Where are embeddings first used?
rg "token_embd" bin/worker-orcd/cuda/src/ -A 5
rg "embedding" bin/worker-orcd/cuda/src/ -A 5
```

### Add Debug Print (C++)
```cpp
// In cuda/src/model/qwen_weight_loader.cpp after line 299
half* embd_ptr = (half*)model->weights.token_embd;
half host_values[10];
cudaMemcpy(host_values, embd_ptr, 10 * sizeof(half), cudaMemcpyDeviceToHost);
fprintf(stderr, "🔍 [C++] First 10 embedding values:\n");
for (int i = 0; i < 10; i++) {
    fprintf(stderr, "  [%d] = %.4f\n", i, __half2float(host_values[i]));
}
```

### Rebuild & Test
```bash
cargo clean -p worker-orcd
cargo build -p worker-orcd --features cuda
# Run worker again, check output
```

### Check for Type Mismatch
- [ ] Are pointers `void*` or `half*`?
- [ ] Are casts using `(half*)` or `(float*)`?
- [ ] Is data read with correct stride?

### Fix Location
Likely in:
- `cuda/src/model/qwen_model.h` (struct definition)
- `cuda/src/inference_impl.cpp` (embedding lookup)
- `cuda/src/transformer/qwen_transformer.cpp` (usage)

---

## ✅ Verification

### After HTTP Fix
```bash
curl -X POST http://localhost:9999/execute \
  -H "Content-Type: application/json" \
  -d '{"job_id":"test","prompt":"hello","max_tokens":10}'
# Should return SSE stream
```

### After C++ Fix
```bash
# Run worker, check logs for:
# 🔍 [C++] First 10 embedding values:
#   [0] = 0.1234  # NON-ZERO!
#   [1] = -0.5678 # NON-ZERO!
```

### Final Test
```bash
cargo test -p worker-orcd --features cuda \
  test_haiku_generation_STUB_PIPELINE_ONLY \
  -- --nocapture --ignored

# Should see:
# ✅ Health check passed
# ✅ Execute request succeeded
# ✅ Tokens generated
# 🎨 M0 Haiku Anti-Cheat Test PASSED
```

---

## 📝 Files to Have Open

### For HTTP Bug
1. `bin/worker-crates/worker-http/src/routes/execute.rs`
2. `bin/worker-crates/worker-http/src/routes/health.rs`
3. `bin/worker-crates/worker-http/src/routes.rs`
4. `bin/worker-orcd/src/tests/integration/framework.rs`

### For C++ Bug
1. `bin/worker-orcd/cuda/src/model/qwen_weight_loader.cpp`
2. `bin/worker-orcd/cuda/src/model/qwen_model.h`
3. `bin/worker-orcd/cuda/src/inference_impl.cpp`
4. `bin/worker-orcd/src/cuda/weight_loader.rs` (for comparison)

---

## 🎯 Success = Haiku Generated

```
🎨 M0 Haiku Anti-Cheat Test PASSED
Minute: 27 ("twenty-seven")
Nonce: abc12345
Tokens: 42
Time: 5.2s

Haiku:
Silicon dreams flow
Through twenty-seven circuits
GPU awakens
```

---

## ⏱️ Time Tracking

- [ ] 00:00 - Start worker
- [ ] 00:05 - HTTP curl test
- [ ] 00:30 - HTTP bug fixed
- [ ] 01:00 - C++ debug prints added
- [ ] 01:30 - C++ bug fixed
- [ ] 02:00 - Haiku test passes ✅
- [ ] 02:30 - Cleanup complete

**Target**: Haiku by 02:00 (2 hours)

---

## 🆘 If Stuck

### HTTP Bug >30 min
- Skip to C++ bug
- Come back later
- Not critical path

### C++ Bug >1 hour
- Add MORE debug prints
- Print from Rust side too
- Compare pointer values
- Check CUDA error codes

### Both Bugs >3 hours
- Focus on ONE completely
- Document blocker
- Take break, fresh eyes

---

**Ready to start? Begin with HTTP curl test! 🚀**
