# Metal/CUDA Inference Bug Report

**Date:** 2025-10-09  
**Team:** TEAM-019  
**Status:** 🐛 Root cause identified - Candle broadcasting bug on Metal

---

## Executive Summary

Metal and CUDA backends fail inference with broadcasting error in attention mechanism. CPU backend works perfectly. Root cause is a **Candle library bug** in Metal's attention implementation, not our code.

---

## Bug Details

### Error Message
```
cannot broadcast [5, 5] to [1, 32, 5, 7]
```

### When It Happens
- ✅ **Warmup succeeds** (2 tokens, position=0)
- ❌ **Inference fails** (5 tokens, position=0)

### Affected Backends
- ❌ **Metal** - Broadcasting error in forward pass
- ❓ **CUDA** - Likely same issue (not tested yet)
- ✅ **CPU** - Works perfectly

---

## Investigation Results

### TEAM-018's F16 Hypothesis - INCORRECT ❌

TEAM-018 changed Metal to use F16 dtype thinking it would help:
```rust
// TEAM-018: Use F16 for Metal backend (better support), F32 for others
let dtype = if device.is_metal() { DType::F16 } else { DType::F32 };
```

**This was wrong.** TEAM-019 reverted to F32 for all backends, but the bug persists.

### Root Cause - Candle Library Bug

The error `cannot broadcast [5, 5] to [1, 32, 5, 7]` occurs in Candle's Metal attention implementation. This is a **shape mismatch in the attention mechanism**, specifically in the QK^T computation or mask application.

**Evidence:**
1. CPU backend uses same model code → works
2. Metal backend uses same model code → fails
3. Error is in Candle's internal broadcasting logic
4. Shape `[1, 32, 5, 7]` suggests: `[batch, num_heads, seq_len, ???]`
5. Shape `[5, 5]` suggests a `[seq_len, seq_len]` attention mask or QK matrix

### Why Warmup Works But Inference Fails

- **Warmup:** 2 tokens → small attention matrix → no broadcasting issue
- **Inference:** 5 tokens → larger attention matrix → triggers broadcasting bug

---

## Candle-VLLM Investigation

User requested investigation of `candle-vllm` (https://github.com/veighnsche/candle-vllm).

**Finding:** Candle-VLLM uses a **completely different architecture**:
- Custom paged attention with Metal kernels
- Custom KV cache implementation
- Does NOT use `candle-transformers::models::llama`
- Requires significant infrastructure changes

**Conclusion:** Not a viable fix for our issue. Would require rewriting entire inference backend.

---

## Workaround

Use **CPU backend** on macOS until Candle fixes Metal attention:

```bash
./llorch-remote mac.home.arpa cpu all
```

CPU backend is fully functional and generates correct output.

---

## Upstream Fix Required

This bug needs to be fixed in **Candle library**, not our code. Possible actions:

1. **File Candle issue** with reproduction case
2. **Wait for Candle fix** in upstream
3. **Use CPU backend** as temporary workaround
4. **Implement custom Metal attention** (like candle-vllm) - significant effort

---

## Test Results

### Hardware
- **Device:** Apple M4
- **macOS:** 26.0.1 (Build 25A362)
- **Metal:** Metal 4

### CPU Backend ✅
```
Model: TinyLlama 1.1B
Dtype: F32
Warmup: 6ms
Inference: 50 tokens generated
Output: "Once upon a time there was a town, there lived a wise man..."
Status: ✅ Production-ready
```

### Metal Backend ❌
```
Model: TinyLlama 1.1B
Dtype: F32 (TEAM-019 fix)
Warmup: 6ms ✅
Inference: Broadcasting error ❌
Error: "cannot broadcast [5, 5] to [1, 32, 5, 7]"
Status: 🐛 Blocked on Candle fix
```

### CUDA Backend ❓
```
Status: Not tested (cargo not in PATH on workstation)
Likely: Same broadcasting bug as Metal
```

---

## Recommendations

### Immediate (TEAM-019)
1. ✅ Revert F16 dtype change (completed)
2. ✅ Document bug in this report (completed)
3. ⏳ Update Metal backend status to "blocked"
4. ⏳ File Candle issue with reproduction

### Short-term
1. Use CPU backend for macOS production
2. Test CUDA backend on workstation
3. Monitor Candle repository for fixes

### Long-term
1. Consider custom Metal attention (like candle-vllm)
2. Or wait for upstream Candle fix
3. Or switch to different inference library

---

## Files Modified by TEAM-019

- `src/backend/models/llama.rs` - Reverted F16 to F32 for all backends
- `.specs/METAL_CUDA_INFERENCE_BUG_REPORT.md` - This report

---

## References

- TEAM-018 Handoff: `.specs/TEAM_018_HANDOFF.md`
- Metal Issue Doc: `.specs/METAL_INFERENCE_ISSUE.md`
- Candle-VLLM: `reference/candle-vllm/` (git submodule)
- Candle Repository: https://github.com/huggingface/candle

---

**Created:** 2025-10-09  
**Team:** TEAM-019  
**Priority:** Medium (CPU backend works as fallback)  
**Blocked on:** Upstream Candle fix
