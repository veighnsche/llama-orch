# TEAM-006: Task 3.6 - Add Callbacks for KV Cache
**Part of:** Phase 3 - Implementation  
**Duration:** 5 minutes  
**Status:** ⏳ READY (NEEDS 2 CALLBACKS)  
**Depends on:** Task 3.5 (Checkpoint 2)  
**Updated by:** TEAM-006

---

## ⚠️ CALLBACKS NEEDED (TEAM-005 FINDING)

**Issue:** KV cache tensors retrieved via `get_k/get_v()` don't go through `cb()`  
**Solution:** Add 2 minimal callbacks after cache retrieval

See [COMPREHENSIVE_ANALYSIS.md](COMPREHENSIVE_ANALYSIS.md) for full analysis.

---

## Objective

Add callbacks for KV cache tensors so eval callback can extract them.

**Goal:** Add `cb(k, "cache_k", il)` and `cb(v, "cache_v", il)` after cache retrieval.

---

## Location (from Phase 2)

**File:** `reference/llama.cpp/src/llama-graph.cpp`  
**Function:** `llm_graph_context::build_attn` (KV cache version)  
**Lines:** ~1553-1554 (after `mctx_cur->get_k/v()`)  
**Marker:** `// TEAM-005: CHECKPOINT 3 - KV Cache State`

---

## Implementation

### Find the Location

```bash
cd /home/vince/Projects/llama-orch/reference/llama.cpp
grep -n "TEAM-005: CHECKPOINT 3" src/llama-graph.cpp
```

Should show line ~1550.

### Add Callbacks (2 lines)

**Find this code:**
```cpp
ggml_tensor * k = mctx_cur->get_k(ctx0, il);
ggml_tensor * v = mctx_cur->get_v(ctx0, il);
```

**Add after it:**
```cpp
ggml_tensor * k = mctx_cur->get_k(ctx0, il);
ggml_tensor * v = mctx_cur->get_v(ctx0, il);
// TEAM-006: Add callbacks for checkpoint extraction
cb(k, "cache_k", il);
cb(v, "cache_v", il);
```

### Key Points

1. **Cache includes history:** n_kv dimension includes all cached tokens
2. **Shape is 4D:** `[n_embd_head, n_head_kv, n_kv, n_batch]`
3. **After update:** Cache has been updated with current tokens
4. **Before attention:** Extracted before `build_attn_mha` call

---

## Expected Output

When running with LLORCH_VALIDATE=1:

```
✅ TEAM-005: checkpoint_03_cache_k [64 × 12 × N × 1] → /tmp/llama_cpp_checkpoints/checkpoint_03_cache_k.bin
✅ TEAM-005: checkpoint_03_cache_v [64 × 12 × N × 1] → /tmp/llama_cpp_checkpoints/checkpoint_03_cache_v.bin
```

Where N = history_length + current_batch_size.

---

## Verification Steps

1. **Compile:**
   ```bash
   cd build-validate
   make -j$(nproc)
   ```

2. **Run with cache:**
   ```bash
   export LLORCH_VALIDATE=1
   ./bin/llama-cli \
     --model /path/to/gpt2.gguf \
     --prompt "Hello world" \
     --n-predict 1
   
   ls -lh /tmp/llama_cpp_checkpoints/checkpoint_03_cache_*.bin
   ```

3. **Verify shapes:**
   ```python
   import struct
   import numpy as np
   
   for name in ['k', 'v']:
       path = f'/tmp/llama_cpp_checkpoints/checkpoint_03_cache_{name}.bin'
       with open(path, 'rb') as f:
           n_dims = struct.unpack('i', f.read(4))[0]
           shape = struct.unpack(f'{n_dims}q', f.read(8 * n_dims))
           data = np.frombuffer(f.read(), dtype=np.float32)
       
       print(f"Cache {name.upper()}: dims={n_dims}, shape={shape}")
       # Expected: dims=4, shape=(64, 12, n_kv, 1) where n_kv >= 2
   ```

4. **Compare with PyTorch:**
   ```python
   # Note: PyTorch cache may have different structure
   # llama.cpp: [n_embd_head, n_head_kv, n_kv, n_batch]
   # PyTorch: May be [n_batch, n_head, n_kv, n_embd_head]
   # Need to transpose for comparison
   ```

---

## Success Criteria

- [ ] Instrumentation code added at correct location
- [ ] Both K and V cache extracted
- [ ] Uses `il == 0` to filter first layer
- [ ] Correct tensor variables (`k`, `v` from `get_k/get_v`)
- [ ] Conditional compilation with `#ifdef LLORCH_VALIDATE`
- [ ] TEAM-005 comments present
- [ ] Code compiles without errors
- [ ] Two checkpoint files created
- [ ] Shapes include cache history dimension
- [ ] Documentation notes cache structure differences

---

## Troubleshooting

**Issue:** Cache files not created
- **Solution:** Verify cache is being used (not no-cache mode)
- **Solution:** Check that `get_k/get_v` are being called

**Issue:** Unexpected shape
- **Solution:** Cache shape depends on context length and batch size
- **Solution:** Verify n_kv dimension includes history

**Issue:** Data doesn't match PyTorch
- **Solution:** llama.cpp cache structure may differ from PyTorch
- **Solution:** May need to extract only current tokens, not full history
- **Solution:** Check cache update vs retrieval timing

---

## Notes

**Cache structure differences:**
- **llama.cpp:** Stores full history in contiguous memory
- **PyTorch:** May use different cache organization
- **Comparison:** May need to extract only the last N tokens

**Why this checkpoint matters:**
- Validates KV cache implementation
- Tests cache update logic
- Confirms attention has correct historical context
- Critical for multi-token generation

**Special considerations:**
- Cache size grows with sequence length
- First token has no cache history
- Subsequent tokens accumulate history
- May need separate validation for first vs. subsequent tokens

---

**Status:** ✅ COMPLETE (2 CALLBACKS ADDED)  
**Assigned to:** TEAM-006  
**Estimated time:** 5 minutes  
**Actual time:** 2 minutes

**Updated by TEAM-006 based on TEAM-005 comprehensive analysis**
