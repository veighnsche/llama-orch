# TEAM-005: Task 3.9 - Instrument Checkpoint 6 (FFN Output)
**Part of:** Phase 3 - Implementation  
**Duration:** 15 minutes  
**Status:** ⏳ PENDING  
**Depends on:** Task 3.8 (Checkpoint 5)

---

## Objective

Add checkpoint extraction for FFN output after projection.

**Goal:** Extract feed-forward network output to validate FFN implementation.

---

## Location (from Phase 2)

**File:** `src/llama-model.cpp`  
**Function:** `llm_build_gpt2::llm_build_gpt2` (constructor)  
**Line:** 9951 (after `cb(cur, "ffn_out", il)`)  
**Marker:** `// TEAM-005: CHECKPOINT 6 - FFN Output`

---

## Implementation

### Find the Marker

```bash
cd /home/vince/Projects/llama-orch/reference/llama.cpp
grep -n "TEAM-005: CHECKPOINT 6" src/llama-model.cpp
```

Should show line ~9951.

### Add Instrumentation Code

**Insert after the FFN callback:**

```cpp
                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   NULL,
                        NULL,                      NULL,                        NULL,
                        model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                        NULL,
                        LLM_FFN_GELU, LLM_FFN_SEQ, il);
                // TEAM-005: CHECKPOINT 6 - FFN Output
                // Instrumentation point for FFN output (fc → gelu → proj)
                // Expected shape: [n_tokens, n_embd] = [2, 768] for GPT-2
                cb(cur, "ffn_out", il);

                // TEAM-005: Extract checkpoint for validation
                #ifdef LLORCH_VALIDATE
                    #include "llama-checkpoint.h"
                    if (llama_checkpoint::is_enabled() && il == 0) {
                        llama_checkpoint::save_tensor("checkpoint_06_ffn", cur);
                    }
                #endif
            }

            cur = ggml_add(ctx0, cur, ffn_inp);
```

### Key Points

1. **After full FFN:** Extracted after fc → gelu → proj sequence
2. **Before residual:** This is before the FFN residual connection
3. **Shape is 2D:** `[n_tokens, n_embd]` = `[2, 768]` for GPT-2
4. **Layer filter:** `il == 0` for first layer only

---

## Expected Output

When running with LLORCH_VALIDATE=1:

```
✅ TEAM-005: checkpoint_06_ffn [2 × 768] → /tmp/llama_cpp_checkpoints/checkpoint_06_ffn.bin
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
   
   ls -lh /tmp/llama_cpp_checkpoints/checkpoint_06_ffn.bin
   # Should be ~6KB (2 * 768 * 4 bytes)
   ```

3. **Verify shape:**
   ```python
   import struct
   import numpy as np
   
   path = '/tmp/llama_cpp_checkpoints/checkpoint_06_ffn.bin'
   with open(path, 'rb') as f:
       n_dims = struct.unpack('i', f.read(4))[0]
       shape = struct.unpack(f'{n_dims}q', f.read(8 * n_dims))
       data = np.frombuffer(f.read(), dtype=np.float32)
   
   print(f"Dimensions: {n_dims}")
   print(f"Shape: {shape}")
   print(f"Data size: {data.shape}")
   # Expected: dims=2, shape=(2, 768), size=(1536,)
   ```

4. **Verify all 6 checkpoints:**
   ```bash
   ls -lh /tmp/llama_cpp_checkpoints/
   # Should see all 6 checkpoint files:
   # - checkpoint_01_ln1_output.bin
   # - checkpoint_02_q.bin, checkpoint_02_k.bin, checkpoint_02_v.bin
   # - checkpoint_03_cache_k.bin, checkpoint_03_cache_v.bin
   # - checkpoint_04_scores.bin
   # - checkpoint_05_output.bin
   # - checkpoint_06_ffn.bin
   ```

---

## Success Criteria

- [ ] Instrumentation code added at correct location
- [ ] After full FFN (fc → gelu → proj)
- [ ] Before FFN residual connection
- [ ] Uses `il == 0` to filter first layer
- [ ] Correct tensor variable (`cur`)
- [ ] Conditional compilation with `#ifdef LLORCH_VALIDATE`
- [ ] TEAM-005 comments present
- [ ] Code compiles without errors
- [ ] Checkpoint file created
- [ ] Shape is 2D `[2, 768]` as expected
- [ ] All 6 checkpoints now present

---

## Troubleshooting

**Issue:** Checkpoint not created
- **Solution:** Verify `il == 0` filter is present
- **Solution:** Check LLORCH_VALIDATE=1 is set

**Issue:** Wrong shape
- **Solution:** Verify extraction is after `build_ffn`, not before
- **Solution:** Check that you're extracting `cur` not another variable

**Issue:** Data doesn't match PyTorch
- **Solution:** Verify FFN activation is GELU (not SiLU or other)
- **Solution:** Check FFN weights are loaded correctly

---

## Notes

**FFN structure in GPT-2:**
1. **Up projection:** `[n_embd] → [4 * n_embd]` = `[768] → [3072]`
2. **GELU activation:** Non-linear transformation
3. **Down projection:** `[4 * n_embd] → [n_embd]` = `[3072] → [768]`

**Why this checkpoint matters:**
- Validates FFN weights and biases
- Tests GELU activation implementation
- Confirms FFN output before residual
- Completes transformer block validation

**Comparison with other checkpoints:**
- Same shape as Checkpoint 1 (LayerNorm output)
- Both are `[n_tokens, n_embd]` = `[2, 768]`
- Easy to compare directly with PyTorch

---

## Final Verification

After completing this task, verify all 6 checkpoints:

```bash
export LLORCH_VALIDATE=1
./bin/llama-cli --model /path/to/gpt2.gguf --prompt "Test" --n-predict 1

# Should see 6 success messages:
# ✅ TEAM-005: checkpoint_01_ln1_output [2 × 768]
# ✅ TEAM-005: checkpoint_02_q [64 × 12 × 2]
# ✅ TEAM-005: checkpoint_02_k [64 × 12 × 2]
# ✅ TEAM-005: checkpoint_02_v [64 × 12 × 2]
# ✅ TEAM-005: checkpoint_03_cache_k [64 × 12 × N × 1]
# ✅ TEAM-005: checkpoint_03_cache_v [64 × 12 × N × 1]
# ✅ TEAM-005: checkpoint_04_scores [N × 2 × 12 × 1]
# ✅ TEAM-005: checkpoint_05_output [768 × 2 × 1]
# ✅ TEAM-005: checkpoint_06_ffn [2 × 768]
```

---

**Status:** ⏳ PENDING  
**Assigned to:** TEAM-005  
**Estimated time:** 15 minutes  
**Actual time:** [fill after completion]
