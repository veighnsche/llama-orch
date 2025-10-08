# Validation Checkpoint Usage Guide

**Date:** 2025-10-08  
**Purpose:** Guide for using validation checkpoints to verify GPT-2 implementation correctness

## Overview

Validation checkpoints have been added to both **tinygrad** and **Candle** reference implementations. These checkpoints allow you to compare intermediate tensor values between your implementation and the reference implementations to identify exactly where differences occur.

## Reference Implementations

### Tinygrad (Python)
- **File:** `/reference/tinygrad/examples/gpt2.py`
- **Branch:** `main` (checkpoints added to main file)
- **12 checkpoints** added as commented code

### Candle (Rust)
- **File:** `/reference/candle/candle-transformers/src/models/bigcode.rs`
- **Branch:** `orch_log` (dedicated branch for validation)
- **12 checkpoints** added as commented code

### Mistral.rs (Rust - Production)
- **Files:** Multiple core modules in `/reference/mistral.rs/mistralrs-core/src/`
  - `attention/mod.rs` - Checkpoint 4 (SDPA)
  - `kv_cache/mod.rs` - Checkpoint 3 (Cache management)
  - `layers.rs` - Checkpoint 1 (LayerNorm)
  - `sampler.rs` - Checkpoint 10 (Sampling)
- **Branch:** `orch_log` (dedicated branch for validation)
- **Key checkpoints** added to production-grade modules

## How to Use Checkpoints

### Step 1: Enable Validation Mode

**Tinygrad:**
```bash
cd /reference/tinygrad/examples
python gpt2.py --prompt "Hello." --temperature 0 --count 10 --model_size gpt2-medium --validate
```

**Candle:**
```bash
export VALIDATE=1
# Run your Candle application
```

**Mistral.rs:**
```bash
export VALIDATE=1
# Run mistralrs-server or your mistral.rs application
```

### Step 2: Uncomment Specific Checkpoints

Edit the source file and uncomment the checkpoint you want to inspect.

**Example (Checkpoint 1 in tinygrad):**
```python
# Before (commented out):
# if getenv("VALIDATE") and not hasattr(self, "_checkpoint1_printed"):
#   print(f"[CHECKPOINT 1] LayerNorm output shape: {ln1_out.shape}")
#   print(f"[CHECKPOINT 1] Output sample: {ln1_out[0, 0, :5].numpy()}")
#   self._checkpoint1_printed = True

# After (uncommented):
if getenv("VALIDATE") and not hasattr(self, "_checkpoint1_printed"):
  print(f"[CHECKPOINT 1] LayerNorm output shape: {ln1_out.shape}")
  print(f"[CHECKPOINT 1] Output sample: {ln1_out[0, 0, :5].numpy()}")
  self._checkpoint1_printed = True
```

**Example (Checkpoint 1 in Candle):**
```rust
// Before (commented out):
// if std::env::var("VALIDATE").is_ok() {
//     println!("[CHECKPOINT 1] LayerNorm output shape: {:?}", hidden_states.shape());
//     if let Ok(ln_sample) = hidden_states.i((0, 0, ..5))?.to_vec1::<f32>() {
//         println!("[CHECKPOINT 1] Output sample: {:?}", ln_sample);
//     }
// }

// After (uncommented):
if std::env::var("VALIDATE").is_ok() {
    println!("[CHECKPOINT 1] LayerNorm output shape: {:?}", hidden_states.shape());
    if let Ok(ln_sample) = hidden_states.i((0, 0, ..5))?.to_vec1::<f32>() {
        println!("[CHECKPOINT 1] Output sample: {:?}", ln_sample);
    }
}
```

### Step 3: Run and Compare

1. Run the reference implementation with checkpoint enabled
2. Run your implementation with same checkpoint
3. Compare the tensor values
4. If they match (within tolerance), move to next checkpoint
5. If they don't match, debug that specific component

## Checkpoint Reference

| # | Name | Tolerance | What It Validates |
|---|------|-----------|-------------------|
| 1 | Layer Normalization | 1e-5 | LayerNorm computation correctness |
| 2 | QKV Projection | 1e-4 | QKV split and weight transpose |
| 3 | KV Cache State | Exact | Cache management and indexing |
| 4 | Attention Scores | 1e-4 | Scaled dot-product before softmax |
| 5 | Attention Output | 1e-4 | Attention projection |
| 6 | FFN Output | 1e-4 | GELU and FFN layers |
| 7 | First Block Output | 1e-4 | Complete transformer block |
| 8 | Full Logits | 1e-3 | All 12 layers processed |
| 9 | Selected Logits | Exact | Last token selection |
| 10 | Argmax Sampling | Exact | Deterministic token selection |
| 11 | Softmax Probabilities | 1e-6 | Probability distribution |
| 12 | End-to-End | Exact | Complete generation pipeline |

## Debugging Strategy

### Sequential Validation
1. **Start with Checkpoint 1** - If this fails, fix LayerNorm first
2. **Progress in order** - Each checkpoint builds on previous ones
3. **Checkpoint 7** validates entire block architecture
4. **Checkpoint 12** is final proof of correctness

### When a Checkpoint Fails

**Checkpoint 1 (LayerNorm):**
- Check variance calculation (biased vs unbiased)
- Verify epsilon = 1e-5
- Check scale/bias parameter application

**Checkpoint 2 (QKV):**
- Verify weight transpose (Conv1D format)
- Check reshape dimensions
- Verify Q/K/V split indexing

**Checkpoint 3 (Cache):**
- Check cache indexing `[start_pos:start_pos+seq_len]`
- Verify contiguous memory layout
- Check cache retrieval logic

**Checkpoint 4 (Attention Scores):**
- Verify scale factor = sqrt(64) = 8.0
- Check K transpose
- Verify mask application

**Checkpoint 5 (Attention Output):**
- Check transpose order
- Verify reshape dimensions
- Check c_proj weight/bias

**Checkpoint 6 (FFN):**
- Verify GELU formula (exact vs tanh approx)
- Check c_fc/c_proj weights
- Verify 4x expansion (768 → 3072 → 768)

**Checkpoint 7 (Block Output):**
- Check residual connections
- Verify layer norm order (pre-norm)
- Check contiguous memory

**Checkpoint 8 (Full Logits):**
- Verify weight tying (lm_head uses wte.weight)
- Check no bias in lm_head
- Verify all 12 blocks processed

**Checkpoint 9 (Selected Logits):**
- Verify indexing `[:, -1, :]`
- Check edge case handling

**Checkpoint 10 (Argmax):**
- Verify temperature check `< 1e-6`
- Check argmax dimension (-1)

**Checkpoint 11 (Softmax):**
- Check division by temperature
- Verify softmax dimension
- Ensure probabilities sum to 1.0

**Checkpoint 12 (End-to-End):**
- If this fails, check all previous checkpoints
- Verify start_pos tracking
- Check cache management across iterations
- Ensure deterministic sampling (temp=0)

## Standard Test Case

Use this consistent test case for all checkpoints:

**Input:**
- Prompt: "Hello."
- Tokens: `[15496, 13]` (tiktoken GPT-2 encoding)
- Model: GPT-2 Medium (350M parameters)
- Temperature: 0 (deterministic)
- Max tokens: 10

**Expected Output:**
```
"Hello. I'm a little late to the party, but"
```

## Example Workflow

```bash
# 1. Test reference implementation
cd /reference/tinygrad/examples
python gpt2.py --prompt "Hello." --temperature 0 --count 10 --model_size gpt2-medium

# Expected output:
# [CHECKPOINT 12 PASSED] End-to-end validation successful!

# 2. Enable specific checkpoint (edit gpt2.py, uncomment checkpoint 1)
python gpt2.py --prompt "Hello." --temperature 0 --count 1 --model_size gpt2-medium --validate

# Output will show:
# [CHECKPOINT 1] LayerNorm output shape: (1, 2, 1024)
# [CHECKPOINT 1] Output sample: [0.123, -0.456, 0.789, ...]

# 3. Run your implementation with same checkpoint
# Compare the output values

# 4. If values match (within tolerance), move to checkpoint 2
# If values don't match, debug LayerNorm implementation
```

## Tips

1. **Enable one checkpoint at a time** - Easier to compare outputs
2. **Use same test input** - "Hello." with temp=0 for determinism
3. **Check shapes first** - If shapes don't match, there's a fundamental error
4. **Then check values** - Compare first 5-10 elements
5. **Use tolerances** - Floating point arithmetic has small differences
6. **Print to file** - For large tensors, redirect output to file for comparison

## Tolerance Explanation

- **1e-5:** Very tight tolerance for single operations
- **1e-4:** Standard tolerance for most operations
- **1e-3:** Looser tolerance for accumulated operations (12 layers)
- **1e-6:** Tight tolerance for probability distributions
- **Exact:** For deterministic operations (indexing, argmax with temp=0)

## Notes

- Checkpoints are **commented out by default** to avoid performance impact
- Only uncomment the specific checkpoint you're debugging
- Checkpoints print only once (first layer/first call) to avoid spam
- Use `VALIDATE=1` environment variable to enable checkpoint code paths
- Checkpoint 12 is always active for end-to-end validation

## See Also

- `01_GPT2_PIPELINE_COMPLETE_BEHAVIORS.md` - Full specification with checkpoint details
- `SPEC_COMPLETENESS_VERIFICATION.md` - Verification methodology
