# TEAM-005: Task 3.7 - Instrument Checkpoint 4 (Attention Scores)
**Part of:** Phase 3 - Implementation  
**Duration:** 15 minutes  
**Status:** ⏳ PENDING  
**Depends on:** Task 3.6 (Checkpoint 3)

---

## Objective

Add checkpoint extraction for attention scores after softmax.

**Goal:** Extract attention weights to validate attention mechanism.

---

## Location (from Phase 2)

**File:** `src/llama-graph.cpp`  
**Function:** `llm_graph_context::build_attn_mha`  
**Line:** 1385 (after `cb(kq, "kq_soft_max", il)`)  
**Marker:** `// TEAM-005: CHECKPOINT 4 - Attention Scores`

---

## Implementation

### Find the Marker

```bash
cd /home/vince/Projects/llama-orch/reference/llama.cpp
grep -n "TEAM-005: CHECKPOINT 4" src/llama-graph.cpp
```

Should show line ~1385.

### Add Instrumentation Code

**Insert after the softmax callback:**

```cpp
        kq = ggml_soft_max_ext(ctx0, kq, kq_mask, kq_scale, hparams.f_max_alibi_bias);
        ggml_soft_max_add_sinks(kq, sinks);
        // TEAM-005: CHECKPOINT 4 - Attention Scores
        // Instrumentation point for attention weights after softmax
        // Expected shape: [n_kv, n_tokens, n_head, n_batch] (4D permuted)
        cb(kq, "kq_soft_max", il);

        // TEAM-005: Extract checkpoint for validation
        #ifdef LLORCH_VALIDATE
            #include "llama-checkpoint.h"
            if (llama_checkpoint::is_enabled() && il == 0) {
                llama_checkpoint::save_tensor("checkpoint_04_scores", kq);
            }
        #endif
```

### Key Points

1. **After softmax:** Values sum to 1 along appropriate dimension
2. **Shape is 4D:** `[n_kv, n_tokens, n_head, n_batch]` after permutations
3. **Before V multiply:** Extracted before attention output computation
4. **Values in [0, 1]:** Softmax ensures valid probability distribution

---

## Expected Output

When running with LLORCH_VALIDATE=1:

```
✅ TEAM-005: checkpoint_04_scores [N × 2 × 12 × 1] → /tmp/llama_cpp_checkpoints/checkpoint_04_scores.bin
```

Where N = n_kv (cache size + current tokens).

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
   
   ls -lh /tmp/llama_cpp_checkpoints/checkpoint_04_scores.bin
   ```

3. **Verify shape and values:**
   ```python
   import struct
   import numpy as np
   
   path = '/tmp/llama_cpp_checkpoints/checkpoint_04_scores.bin'
   with open(path, 'rb') as f:
       n_dims = struct.unpack('i', f.read(4))[0]
       shape = struct.unpack(f'{n_dims}q', f.read(8 * n_dims))
       data = np.frombuffer(f.read(), dtype=np.float32)
   
   print(f"Dimensions: {n_dims}")
   print(f"Shape: {shape}")
   print(f"Min: {data.min()}, Max: {data.max()}")
   print(f"Sum per head (should be ~1.0): {data.reshape(shape).sum(axis=0)}")
   
   # Expected:
   # - dims=4
   # - shape=(n_kv, 2, 12, 1)
   # - Min >= 0, Max <= 1
   # - Sum per attention head ≈ 1.0 (softmax property)
   ```

4. **Test reshape for PyTorch:**
   ```python
   # llama.cpp: [n_kv, n_tokens, n_head, n_batch]
   # PyTorch: [n_batch, n_head, n_tokens, n_kv]
   # Need to transpose for comparison
   data_4d = data.reshape(shape)
   data_pytorch = data_4d.transpose(3, 2, 1, 0)
   print(f"PyTorch shape: {data_pytorch.shape}")
   ```

---

## Success Criteria

- [ ] Instrumentation code added at correct location
- [ ] After softmax, before V multiply
- [ ] Uses `il == 0` to filter first layer
- [ ] Correct tensor variable (`kq`)
- [ ] Conditional compilation with `#ifdef LLORCH_VALIDATE`
- [ ] TEAM-005 comments present
- [ ] Code compiles without errors
- [ ] Checkpoint file created
- [ ] Shape is 4D as expected
- [ ] Values are in range [0, 1]
- [ ] Softmax property verified (sums to 1)

---

## Troubleshooting

**Issue:** Values not in [0, 1]
- **Solution:** Verify extraction is after softmax, not before
- **Solution:** Check that `ggml_soft_max_ext` completed

**Issue:** Sums don't equal 1
- **Solution:** Check which dimension should sum to 1
- **Solution:** Verify mask isn't corrupting values

**Issue:** Wrong shape
- **Solution:** Attention scores go through several permutations
- **Solution:** Verify extracting at correct point (after softmax callback)

---

## Notes

**Shape transformations:**
- **Q·K^T:** `[n_head, n_tokens, n_kv]` initially
- **After permute:** `[n_kv, n_tokens, n_head, n_batch]`
- **PyTorch:** Usually `[n_batch, n_head, n_tokens, n_kv]`

**Why this checkpoint matters:**
- Validates attention score computation
- Tests softmax implementation
- Confirms masking is correct
- Critical for understanding what model attends to

**Attention properties to verify:**
- Values in [0, 1] (probability distribution)
- Sum to 1 along key dimension
- Causal mask applied correctly (no future attention)
- Matches PyTorch attention pattern

---

**Status:** ⏳ PENDING  
**Assigned to:** TEAM-005  
**Estimated time:** 15 minutes  
**Actual time:** [fill after completion]
