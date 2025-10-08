# TEAM-005: Task 3.8 - Instrument Checkpoint 5 (Attention Output)
**Part of:** Phase 3 - Implementation  
**Duration:** 20 minutes  
**Status:** ⏳ PENDING  
**Depends on:** Task 3.7 (Checkpoint 4)

---

## Objective

Add checkpoint extraction for attention output after projection.

**Goal:** Extract attention output before residual connection to validate full attention mechanism.

---

## Location (from Phase 2)

**File:** `src/llama-graph.cpp`  
**Function:** `llm_graph_context::build_attn` (after build_attn_mha)  
**Line:** 1571 (after output projection, before return)  
**Marker:** `// TEAM-005: CHECKPOINT 5 - Attention Output`

---

## Implementation

### Find the Marker

```bash
cd /home/vince/Projects/llama-orch/reference/llama.cpp
grep -n "TEAM-005: CHECKPOINT 5" src/llama-graph.cpp
```

Should show line ~1571.

### Add Instrumentation Code

**Insert before the return statement:**

```cpp
    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    // TEAM-005: CHECKPOINT 5 - Attention Output
    // Instrumentation point for attention output after projection (before residual)
    // Expected shape: [n_embd, n_tokens, n_batch] → reshape to [n_tokens, n_embd]
    // Note: Add callback here: cb(cur, "attn_out_proj", il);

    // TEAM-005: Extract checkpoint for validation
    #ifdef LLORCH_VALIDATE
        #include "llama-checkpoint.h"
        if (llama_checkpoint::is_enabled() && il == 0) {
            llama_checkpoint::save_tensor("checkpoint_05_output", cur);
        }
    #endif
    
    return cur;
}
```

### Alternative: Add Callback First

For better integration with existing callback system, you can also add a callback:

```cpp
    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    // TEAM-005: CHECKPOINT 5 - Attention Output
    // Add callback for attention output after projection
    cb(cur, "attn_out_proj", il);

    // TEAM-005: Extract checkpoint for validation
    #ifdef LLORCH_VALIDATE
        #include "llama-checkpoint.h"
        if (llama_checkpoint::is_enabled() && il == 0) {
            llama_checkpoint::save_tensor("checkpoint_05_output", cur);
        }
    #endif
    
    return cur;
}
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

**Status:** ⏳ PENDING  
**Assigned to:** TEAM-005  
**Estimated time:** 20 minutes (includes verifying correct overload)  
**Actual time:** [fill after completion]
