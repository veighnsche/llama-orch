# TEAM-005: Task 3.4 - Instrument Checkpoint 1 (LayerNorm)
**Part of:** Phase 3 - Implementation  
**Duration:** 15 minutes  
**Status:** ⏳ PENDING  
**Depends on:** Task 3.3 (Initialization)

---

## Objective

Add checkpoint extraction for LayerNorm output in first transformer block.

**Goal:** Extract normalized activations after first LayerNorm, before attention.

---

## Location (from Phase 2)

**File:** `src/llama-model.cpp`  
**Function:** `llm_build_gpt2::llm_build_gpt2` (constructor)  
**Line:** 9898 (after `cb(cur, "attn_norm", il)`)  
**Marker:** `// TEAM-005: CHECKPOINT 1 - LayerNorm Output`

---

## Implementation

### Find the Marker

```bash
cd /home/vince/Projects/llama-orch/reference/llama.cpp
grep -n "TEAM-005: CHECKPOINT 1" src/llama-model.cpp
```

Should show line ~9898.

### Add Instrumentation Code

**Insert after the callback `cb(cur, "attn_norm", il);`:**

```cpp
            // TEAM-005: CHECKPOINT 1 - LayerNorm Output
            // Instrumentation point for first transformer block LayerNorm
            // Expected shape: [n_tokens, n_embd] = [2, 768] for GPT-2
            cb(cur, "attn_norm", il);

            // TEAM-005: Extract checkpoint for validation
            #ifdef LLORCH_VALIDATE
                #include "llama-checkpoint.h"
                if (llama_checkpoint::is_enabled() && il == 0) {
                    llama_checkpoint::save_tensor("checkpoint_01_ln1_output", cur);
                }
            #endif
```

### Key Points

1. **Layer filter:** `il == 0` ensures we only extract from first layer
2. **Tensor variable:** `cur` contains the LayerNorm output
3. **Checkpoint name:** `checkpoint_01_ln1_output` matches PyTorch reference
4. **Placement:** After LayerNorm, before attention

---

## Expected Output

When running with LLORCH_VALIDATE=1:

```
✅ TEAM-005: checkpoint_01_ln1_output [2 × 768] → /tmp/llama_cpp_checkpoints/checkpoint_01_ln1_output.bin
```

---

## Verification Steps

1. **Compile:**
   ```bash
   cd build-validate
   make -j$(nproc)
   ```

2. **Run with GPT-2 model:**
   ```bash
   export LLORCH_VALIDATE=1
   export LLORCH_CHECKPOINT_DIR=/tmp/llama_cpp_checkpoints
   mkdir -p $LLORCH_CHECKPOINT_DIR
   
   ./bin/llama-cli \
     --model /path/to/gpt2.gguf \
     --prompt "Hello world" \
     --n-predict 1
   ```

3. **Check checkpoint file:**
   ```bash
   ls -lh /tmp/llama_cpp_checkpoints/checkpoint_01_ln1_output.bin
   # Should exist and be ~6KB (2 * 768 * 4 bytes)
   ```

4. **Verify shape:**
   ```python
   import struct
   import numpy as np
   
   with open('/tmp/llama_cpp_checkpoints/checkpoint_01_ln1_output.bin', 'rb') as f:
       n_dims = struct.unpack('i', f.read(4))[0]
       shape = struct.unpack(f'{n_dims}q', f.read(8 * n_dims))
       data = np.frombuffer(f.read(), dtype=np.float32)
   
   print(f"Dimensions: {n_dims}")
   print(f"Shape: {shape}")
   print(f"Data size: {data.shape}")
   # Expected: Dimensions: 2, Shape: (2, 768), Data size: (1536,)
   ```

---

## Success Criteria

- [ ] Instrumentation code added at correct location
- [ ] Uses `il == 0` to filter first layer only
- [ ] Correct tensor variable (`cur`)
- [ ] Conditional compilation with `#ifdef LLORCH_VALIDATE`
- [ ] TEAM-005 comments present
- [ ] Code compiles without errors
- [ ] Checkpoint file created when running
- [ ] File size matches expected shape
- [ ] Shape verification passes

---

## Troubleshooting

**Issue:** Checkpoint not created
- **Solution:** Verify LLORCH_VALIDATE=1 is set (not just cmake flag)
- **Solution:** Check directory exists and is writable

**Issue:** Wrong shape
- **Solution:** Verify you're extracting `cur` not another variable
- **Solution:** Check that `il == 0` filter is working

**Issue:** Multiple files created
- **Solution:** Normal if multiple layers processed - verify `il == 0` is present

---

## Notes

**Why this checkpoint matters:**
- First point where we can compare with PyTorch
- Validates LayerNorm implementation
- Tests that tensor shapes match expectations
- Confirms data flow through first layer

---

**Status:** ⏳ PENDING  
**Assigned to:** TEAM-005  
**Estimated time:** 15 minutes  
**Actual time:** [fill after completion]
