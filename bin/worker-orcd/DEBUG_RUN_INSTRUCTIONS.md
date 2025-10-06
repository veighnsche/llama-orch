# Debug Run Instructions

**Date**: 2025-10-06  
**Status**: Comprehensive debug logging added  

---

## What Was Added

Full debug instrumentation across the entire inference pipeline:

### 🎯 Critical Components Instrumented

1. **Attention Mechanism** (`cuda/kernels/gqa_attention.cu`)
   - Q, K, V value inspection
   - Attention score computation
   - Softmax verification (sum should = 1.0)
   - Attention weight distribution
   - Output verification

2. **RoPE Positional Encoding** (`cuda/kernels/rope.cu`)
   - Position-dependent rotation angles (theta, cos, sin)
   - Before/after rotation values for Q and K
   - Verifies position encoding is applied correctly

3. **QKV Projections** (`cuda/src/transformer/qwen_transformer.cpp`)
   - Post-projection Q, K, V values
   - Post-RoPE Q, K values
   - Attention output values
   - Shows if weight matrices are working

4. **Embeddings & Hidden States**
   - Token embedding values
   - Layer-by-layer hidden state evolution
   - Final hidden state before logits

5. **Logit Generation** (existing)
   - Global max/min logit scan
   - Token probability distribution

---

## How to Build & Run

### 1. Rebuild CUDA Components

```bash
cd /home/vince/Projects/llama-orch/bin/worker-orcd

# Clean build to ensure CUDA kernels are recompiled
cargo clean -p worker-orcd

# Build with CUDA feature
cargo build --features cuda
```

### 2. Run Debug Test

```bash
# Run test with full debug output
cargo test --test haiku_generation_anti_cheat \
  --features cuda -- \
  --nocapture \
  --ignored \
  2>&1 | tee debug_full.log

# The test will generate verbose output to debug_full.log
```

### 3. Analyze Output

```bash
# View the full log
less debug_full.log

# Search for specific sections
grep -A 10 "FORWARD PASS" debug_full.log          # Forward pass summaries
grep -A 5 "ATTENTION DEBUG" debug_full.log        # Attention details
grep -A 5 "ROPE DEBUG" debug_full.log             # RoPE details
grep -A 5 "QKV DEBUG" debug_full.log              # QKV projection details
grep "GLOBAL LOGIT ANALYSIS" debug_full.log       # Logit statistics
```

---

## What to Look For

### ✅ Healthy Patterns

1. **Attention Weights**
   - Sum to exactly 1.0 (or very close like 0.9999)
   - Are diverse (not all same value)
   - Change between forward passes

2. **RoPE Application**
   - Q and K values change after RoPE
   - Rotation angles increase with position
   - cos and sin values look reasonable

3. **QKV Values**
   - Are in reasonable range (-10 to +10 typically)
   - Change between tokens
   - Are not all zeros or NaNs

4. **Hidden States**
   - Evolve through layers
   - Change between tokens
   - Non-zero values

### 🚩 Red Flags

1. **Broken Attention**
   - Attention weights don't sum to 1.0
   - All weights are identical (e.g., all 0.333)
   - Attention always attends to same position

2. **Broken RoPE**
   - Q, K values don't change after RoPE
   - All rotation angles are zero
   - No position-dependent behavior

3. **Weight Loading Issues**
   - Q, K, V are all zeros
   - NaN or Inf values appear
   - Values are extremely large (>1000)

4. **Cache Issues**
   - Attention doesn't see previous tokens
   - KV cache reads return zeros
   - Model ignores context

---

## Expected Debug Volume

The test will generate **~1000-2000 lines** of debug output for:
- First 5 forward passes (complete detail)
- First 2 layers (QKV detail)
- First 5 attention positions (kernel detail)

After this, logging automatically reduces to avoid overwhelming output.

---

## Common Issues & Solutions

### Issue: "CUDA out of memory"
**Solution**: The debug logging uses printf in CUDA kernels which uses limited buffer space. If you see truncated output, the buffer is full. This is normal and expected.

### Issue: "No debug output in log"
**Solution**: Make sure you're using `--nocapture` and redirecting stderr: `2>&1 | tee`

### Issue: "Test times out"
**Solution**: Debug logging adds overhead. The test may take 2-3x longer than normal.

---

## Next Steps After Running

1. **Check attention weights first** - This is the most likely culprit
2. **Verify RoPE is working** - Position encoding is critical
3. **Check if same logit is always max** - Indicates upstream issue
4. **Compare patterns between tokens** - Should see variation

If you find the issue, refer to:
- `INFERENCE_FIX_SUMMARY.md` - Current state
- `NEXT_INVESTIGATION_STEPS.md` - Debugging plan
- `reference/llama.cpp/` - Reference implementation

---

**Built with comprehensive instrumentation to identify the root cause** 🔍
