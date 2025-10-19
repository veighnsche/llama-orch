# Metal Backend Inference Issue

**Date:** 2025-10-09  
**Team:** TEAM-018  
**Status:** 🐛 Known Issue - Metal forward pass failing
**Updated:** 2025-10-09 by TEAM-019 - Root cause identified as Candle library bug

---

## Issue Summary

Metal backend successfully:
- ✅ Compiles on Apple Silicon (M4)
- ✅ Initializes Metal device
- ✅ Loads model (TinyLlama 1.1B SafeTensors)
- ✅ Completes GPU warmup (6ms)
- ✅ Starts HTTP server
- ❌ **FAILS on inference forward pass**

## Error

```
{"type":"error","code":"INFERENCE_FAILED","message":"Forward pass failed: Llama forward pass failed"}
```

## Comparison: CPU vs Metal

### CPU Backend (✅ Working)
```bash
./llorch-remote mac.home.arpa cpu inference
```

**Output:**
```
📖 Generated Story:
─────────────────────────────────────────────────────────
Once upon a timethere was a town, there lived a wise man. 
He was known for his wisdom and his wits, for he knew all 
the stories that had ever been told in the land.Once upon 
a time, there was a great and powerful
─────────────────────────────────────────────────────────
```

**Performance:**
- Model load: ~350ms
- Warmup: ~6ms
- Inference: ~50 tokens generated
- Status: ✅ Production-ready

### Metal Backend (❌ Failing)
```bash
./llorch-remote mac.home.arpa metal inference
```

**Output:**
```
❌ No tokens generated
Raw SSE response:
data: {"type":"error","code":"INFERENCE_FAILED","message":"Forward pass failed: Llama forward pass failed"}
```

**Performance:**
- Model load: ~350ms
- Warmup: ~6ms
- Inference: ❌ Forward pass fails
- Status: 🐛 Pre-release (broken)

---

## Investigation

### What Works
1. **Device initialization** - Metal device creates successfully
2. **Model loading** - SafeTensors loads to Metal GPU
3. **Warmup** - Single forward pass succeeds (6ms)
4. **HTTP server** - Accepts requests correctly

### What Fails
1. **Inference forward pass** - Fails during actual generation
2. **Error is generic** - No specific Metal error details

### Hypothesis

The warmup uses a simple "Hello" prompt (1 token), but inference uses "Once upon a time" (4 tokens). Possible issues:

1. **Batch dimension mismatch** - Metal may handle batch shapes differently
2. **KV cache issue** - Metal KV cache implementation may differ
3. **Tensor device mismatch** - Some tensors may not be on Metal device
4. **Metal kernel missing** - Some operation not implemented in Metal

---

## Reproduction Steps

1. Build Metal backend:
   ```bash
   ./llorch-remote mac.home.arpa metal build
   ```

2. Download model:
   ```bash
   ./llorch-remote mac.home.arpa metal download-model
   ```

3. Run inference:
   ```bash
   ./llorch-remote mac.home.arpa metal inference
   ```

4. Observe error in SSE response

---

## Debugging Recommendations

### 1. Enable Debug Logging

Modify `src/bin/metal.rs` to use debug tracing:
```rust
tracing_subscriber::fmt()
    .with_max_level(tracing::Level::DEBUG)
    .init();
```

### 2. Test with Single Token

Modify inference request to use 1 token:
```json
{"max_tokens": 1}
```

If this works, issue is in multi-token generation.

### 3. Check Candle Metal Support

Review Candle's Metal implementation:
- `reference/candle/candle-core/src/metal_backend.rs`
- `reference/candle/candle-metal-kernels/`

Check if Llama operations are fully supported on Metal.

### 4. Compare with CUDA

CUDA backend works perfectly. Compare:
- Device initialization
- Tensor creation
- Forward pass implementation

### 5. Test Warmup vs Inference

Warmup succeeds but inference fails. Key differences:
- Warmup: 1 token, pos=0
- Inference: Multiple tokens, pos>0

Issue may be in KV cache or position handling.

---

## Workaround

Use CPU backend on macOS until Metal issue is resolved:
```bash
./llorch-remote mac.home.arpa cpu all
```

CPU backend is fully functional and generates correct output.

---

## Next Steps

1. **Debug Metal forward pass** - Add detailed logging
2. **Test with minimal prompt** - Single token generation
3. **Review Candle Metal kernels** - Check operation support
4. **Compare with CUDA** - Identify differences
5. **File Candle issue** - If Metal kernels are incomplete

---

## Test Results

### Hardware
- **Device:** Apple M4
- **macOS:** 26.0.1 (Build 25A362)
- **Metal:** Metal 4

### Build Status
- **Compilation:** ✅ Success (12MB binary)
- **Device init:** ✅ Success
- **Model load:** ✅ Success (2.2GB SafeTensors)
- **Warmup:** ✅ Success (6ms)
- **Inference:** ❌ Forward pass fails

### Logs
```json
{"timestamp":"2025-10-09T08:36:42.836885Z","level":"INFO","fields":{"message":"GPU warmup complete","duration_ms":"6"}}
{"timestamp":"2025-10-09T08:36:42.836938Z","level":"INFO","fields":{"message":"Metal GPU warmup complete - ready for inference"}}
{"timestamp":"2025-10-09T08:36:42.837198Z","level":"INFO","fields":{"message":"HTTP server listening","addr":"0.0.0.0:9876"}}
```

**Inference request:**
```
POST /execute
{"job_id":"test-story","prompt":"Once upon a time","max_tokens":50,"temperature":0.7,"seed":42}
```

**Response:**
```
data: {"type":"error","code":"INFERENCE_FAILED","message":"Forward pass failed: Llama forward pass failed"}
```

---

## Conclusion

Metal backend infrastructure is complete but has a runtime issue in the forward pass. CPU backend works perfectly and should be used for production on macOS until Metal is debugged.

**Status:** Pre-release (blocked on Metal forward pass bug)

---

## TEAM-019 Update

**Root Cause Identified:** This is a **Candle library bug** in Metal's attention broadcasting, not our code.

**TEAM-018's F16 hypothesis was incorrect.** TEAM-019 reverted to F32 for all backends, but the bug persists because it's in Candle's Metal attention implementation.

**See:** `.specs/METAL_CUDA_INFERENCE_BUG_REPORT.md` for full analysis.

**Workaround:** Use CPU backend on macOS (fully functional).

---

**Created:** 2025-10-09  
**Team:** TEAM-018  
**Updated:** 2025-10-09 by TEAM-019  
**Priority:** Medium (CPU backend works as fallback)  
**Blocked on:** Upstream Candle fix
