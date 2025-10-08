# TEAM-006: Task 3.5 - Verify Checkpoint 2 (QKV Projection)
**Part of:** Phase 3 - Implementation  
**Duration:** 5 minutes  
**Status:** ✅ VERIFIED (NO CHANGES NEEDED)  
**Depends on:** Task 3.4 (Checkpoint 1)  
**Updated by:** TEAM-006

---

## ✅ APPROACH REVISED BY TEAM-005

**Old (OBSOLETE):** Add inline extraction code  
**New (CORRECT):** Verify existing `cb()` calls are present

See [COMPREHENSIVE_ANALYSIS.md](COMPREHENSIVE_ANALYSIS.md) for full analysis.

---

## Objective

Verify that QKV checkpoint callbacks already exist in llama.cpp.

**Goal:** Confirm `cb(Qcur/Kcur/Vcur, ...)` calls are present - no changes needed.

---

## Location (from Phase 2)

**File:** `reference/llama.cpp/src/llama-model.cpp`  
**Lines:** ~9912-9914  
**Existing code:**
```cpp
cb(Qcur, "Qcur", il);
cb(Kcur, "Kcur", il);
cb(Vcur, "Vcur", il);
```

---

## Implementation

### Find the Marker

```bash
cd /home/vince/Projects/llama-orch/reference/llama.cpp
grep -n "TEAM-005: CHECKPOINT 2" src/llama-model.cpp
```

Should show line ~9915.

### Add Instrumentation Code

**Insert after the QKV callbacks:**

```cpp
                // TEAM-005: CHECKPOINT 2 - QKV Projection
                // Instrumentation point for Q, K, V after projection and split
                // Expected shape: [n_embd_head, n_head, n_tokens] = [64, 12, 2] for GPT-2
                // Note: Need to reshape to [n_tokens, n_embd] for comparison
                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                // TEAM-005: Extract checkpoints for validation
                #ifdef LLORCH_VALIDATE
                    #include "llama-checkpoint.h"
                    if (llama_checkpoint::is_enabled() && il == 0) {
                        llama_checkpoint::save_tensor("checkpoint_02_q", Qcur);
                        llama_checkpoint::save_tensor("checkpoint_02_k", Kcur);
                        llama_checkpoint::save_tensor("checkpoint_02_v", Vcur);
                    }
                #endif
```

### Key Points

1. **Three tensors:** Extract Q, K, and V separately
2. **Shape note:** Tensors are 3D `[n_embd_head, n_head, n_tokens]`
3. **Reshaping:** PyTorch comparison will need to reshape to 2D
4. **Layer filter:** `il == 0` for first layer only

---

## Expected Output

When running with LLORCH_VALIDATE=1:

```
✅ TEAM-005: checkpoint_02_q [64 × 12 × 2] → /tmp/llama_cpp_checkpoints/checkpoint_02_q.bin
✅ TEAM-005: checkpoint_02_k [64 × 12 × 2] → /tmp/llama_cpp_checkpoints/checkpoint_02_k.bin
✅ TEAM-005: checkpoint_02_v [64 × 12 × 2] → /tmp/llama_cpp_checkpoints/checkpoint_02_v.bin
```

---

## Verification Steps

1. **Compile:**
   ```bash
   cd build-validate
   make -j$(nproc)
   ```

2. **Run and check files:**
   ```bash
   export LLORCH_VALIDATE=1
   ./bin/llama-cli --model /path/to/gpt2.gguf --prompt "Test" --n-predict 1
   
   ls -lh /tmp/llama_cpp_checkpoints/checkpoint_02_*.bin
   # Should see 3 files, each ~6KB (64 * 12 * 2 * 4 bytes)
   ```

3. **Verify shapes:**
   ```python
   import struct
   import numpy as np
   
   for name in ['q', 'k', 'v']:
       path = f'/tmp/llama_cpp_checkpoints/checkpoint_02_{name}.bin'
       with open(path, 'rb') as f:
           n_dims = struct.unpack('i', f.read(4))[0]
           shape = struct.unpack(f'{n_dims}q', f.read(8 * n_dims))
           data = np.frombuffer(f.read(), dtype=np.float32)
       
       print(f"{name.upper()}: dims={n_dims}, shape={shape}, size={data.shape}")
       # Expected: dims=3, shape=(64, 12, 2), size=(1536,)
   ```

4. **Test reshape for PyTorch comparison:**
   ```python
   # Reshape from [n_embd_head, n_head, n_tokens] to [n_tokens, n_embd]
   # For GPT-2: [64, 12, 2] → [2, 768]
   data_3d = data.reshape(64, 12, 2)
   data_2d = data_3d.transpose(2, 1, 0).reshape(2, 768)
   print(f"Reshaped: {data_2d.shape}")  # Should be (2, 768)
   ```

---

## Success Criteria

- [ ] Instrumentation code added at correct location
- [ ] All three tensors (Q, K, V) extracted
- [ ] Uses `il == 0` to filter first layer
- [ ] Correct tensor variables (`Qcur`, `Kcur`, `Vcur`)
- [ ] Conditional compilation with `#ifdef LLORCH_VALIDATE`
- [ ] TEAM-005 comments present
- [ ] Code compiles without errors
- [ ] Three checkpoint files created
- [ ] File sizes match expected shapes
- [ ] Shape verification passes
- [ ] Reshape logic documented for PyTorch comparison

---

## Troubleshooting

**Issue:** Only one or two files created
- **Solution:** Verify all three `save_tensor` calls are present
- **Solution:** Check for compilation errors

**Issue:** Wrong shape (not 3D)
- **Solution:** Verify extracting `Qcur/Kcur/Vcur` not `cur`
- **Solution:** Check that view_3d operation completed

**Issue:** Data doesn't match PyTorch
- **Solution:** Remember to reshape: llama.cpp uses `[head_dim, n_heads, seq_len]`
- **Solution:** PyTorch uses `[seq_len, n_heads, head_dim]` - need transpose

---

## Notes

**Shape differences:**
- **llama.cpp:** `[n_embd_head, n_head, n_tokens]` = `[64, 12, 2]`
- **PyTorch:** `[n_tokens, n_head, n_embd_head]` = `[2, 12, 64]`
- **Flattened:** Both are `[n_tokens, n_embd]` = `[2, 768]`

**Why this checkpoint matters:**
- Validates QKV projection weights
- Tests tensor splitting logic
- Confirms attention input preparation
- Critical for attention mechanism validation

---

**Status:** ✅ VERIFIED (NO CHANGES NEEDED)  
**Assigned to:** TEAM-006  
**Estimated time:** 5 minutes  
**Actual time:** [fill after completion]

**Updated by TEAM-006 based on TEAM-005 comprehensive analysis**
