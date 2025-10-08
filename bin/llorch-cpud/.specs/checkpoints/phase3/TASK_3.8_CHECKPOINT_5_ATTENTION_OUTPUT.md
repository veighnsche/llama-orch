# TEAM-006: Task 3.8 - Add Callback for Attention Output
**Part of:** Phase 3 - Implementation  
**Duration:** 5 minutes  
**Status:** ⏳ READY (NEEDS 1 CALLBACK)  
**Depends on:** Task 3.7 (Attention Scores)  
**Updated by:** TEAM-006

---

## ⚠️ CALLBACK NEEDED (TEAM-005 FINDING)

**Issue:** No callback after attention output projection  
**Solution:** Add 1 minimal callback after wo_b

See [COMPREHENSIVE_ANALYSIS.md](COMPREHENSIVE_ANALYSIS.md) for full analysis.

---

## Objective

Add callback for attention output so eval callback can extract it.

**Goal:** Add `cb(cur, "attn_out_proj", il)` after attention output projection.

---

## Location (from Phase 2)

**File:** `reference/llama.cpp/src/llama-graph.cpp`  
**Function:** `llm_graph_context::build_attn` (KV cache version)  
**Line:** ~1574 (after wo_b, before return)  
**Marker:** `// TEAM-005: CHECKPOINT 5 - Attention Output`

---

## Implementation

### Find the Location

```bash
cd /home/vince/Projects/llama-orch/reference/llama.cpp
grep -n "TEAM-005: CHECKPOINT 5" src/llama-graph.cpp
```

Should show line ~1571.

### Add Callback (1 line)

**Find this code:**
```cpp
if (wo_b) {
    cur = ggml_add(ctx0, cur, wo_b);
}
return cur;
```

**Add callback before return:**
```cpp
if (wo_b) {
    cur = ggml_add(ctx0, cur, wo_b);
}
// TEAM-006: Add callback for checkpoint extraction
cb(cur, "attn_out_proj", il);
return cur;
```

### Key Points

1. **After projection:** Extracted after wo and wo_b applied
2. **Before residual:** This is before the residual connection in GPT-2 builder
3. **Shape is 3D:** `[n_embd, n_tokens, n_batch]`
4. **Optional callback:** Adding `cb(cur, "attn_out_proj", il)` helps debugging

---

## Expected Output

When running with LLORCH_VALIDATE=1:

```
✅ TEAM-005: checkpoint_05_output [768 × 2 × 1] → /tmp/llama_cpp_checkpoints/checkpoint_05_output.bin
```

---

## Verification Steps

1. **Compile:**
   ```bash
   cd build-validate
   make -j$(nproc)
   ```

2. **Run and check:**
   ```bash
   export LLORCH_VALIDATE=1
   ./bin/llama-cli \
     --model /path/to/gpt2.gguf \
     --prompt "Test" \
     --n-predict 1
   
   ls -lh /tmp/llama_cpp_checkpoints/checkpoint_05_output.bin
   ```

3. **Verify shape:**
   ```python
   import struct
   import numpy as np
   
   path = '/tmp/llama_cpp_checkpoints/checkpoint_05_output.bin'
   with open(path, 'rb') as f:
       n_dims = struct.unpack('i', f.read(4))[0]
       shape = struct.unpack(f'{n_dims}q', f.read(8 * n_dims))
       data = np.frombuffer(f.read(), dtype=np.float32)
   
   print(f"Dimensions: {n_dims}")
   print(f"Shape: {shape}")
   # Expected: dims=3, shape=(768, 2, 1)
   
   # Reshape for PyTorch comparison
   data_3d = data.reshape(shape)
   data_2d = data_3d.transpose(1, 0, 2).reshape(2, 768)
   print(f"Reshaped to 2D: {data_2d.shape}")  # Should be (2, 768)
   ```

4. **Verify placement:**
   ```bash
   # Check that extraction happens before residual add
   grep -A5 "checkpoint_05_output" src/llama-graph.cpp
   # Should see return statement after, not residual add
   ```

---

## Success Criteria

- [ ] Instrumentation code added at correct location
- [ ] After output projection (wo, wo_b)
- [ ] Before return from build_attn
- [ ] Uses `il == 0` to filter first layer
- [ ] Correct tensor variable (`cur`)
- [ ] Optional callback added for debugging
- [ ] Conditional compilation with `#ifdef LLORCH_VALIDATE`
- [ ] TEAM-005 comments present
- [ ] Code compiles without errors
- [ ] Checkpoint file created
- [ ] Shape is 3D as expected
- [ ] Reshape logic documented

---

## Troubleshooting

**Issue:** Checkpoint not created
- **Solution:** Verify extraction is in correct build_attn overload (KV cache version)
- **Solution:** Check that GPT-2 uses this version

**Issue:** Wrong shape
- **Solution:** Verify extraction is after wo_b, not before
- **Solution:** Check that you're in the right function (not build_attn_mha)

**Issue:** Data doesn't match PyTorch
- **Solution:** Remember to reshape from 3D to 2D
- **Solution:** llama.cpp: `[n_embd, n_tokens, n_batch]`
- **Solution:** PyTorch: `[n_tokens, n_embd]`

**Issue:** Multiple build_attn versions
- **Solution:** GPT-2 uses the KV cache version (line ~1515)
- **Solution:** Verify you're modifying the right overload

---

## Notes

**Why this checkpoint matters:**
- Validates full attention mechanism end-to-end
- Tests output projection weights
- Confirms attention output before residual
- Last checkpoint before residual connection

**Shape transformations:**
- **After attention:** `[n_embd, n_tokens, n_batch]` = `[768, 2, 1]`
- **For PyTorch:** Transpose to `[n_tokens, n_embd]` = `[2, 768]`

**Function overloads:**
llama-graph.cpp has multiple `build_attn` overloads:
- `build_attn(llm_graph_input_attn_no_cache *)` - No cache
- `build_attn(llm_graph_input_attn_kv *)` - **GPT-2 uses this**
- `build_attn(llm_graph_input_attn_kv_iswa *)` - Sliding window
- `build_attn(llm_graph_input_attn_cross *)` - Cross attention

Make sure to modify the KV cache version (line ~1515).

---

**Status:** ✅ COMPLETE (1 CALLBACK ADDED)  
**Assigned to:** TEAM-006  
**Estimated time:** 5 minutes  
**Actual time:** 2 minutes

**Updated by TEAM-006 based on TEAM-005 comprehensive analysis**
