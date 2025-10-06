# Bug Status - 2025-10-06 12:17

## Current State: CRITICAL BUG - Model Generates Same Token Repeatedly

**Status**: ❌ BROKEN - Model generates token ID=76156 ("suffice") 100 times in a row

---

## What's Fixed ✅

1. **Compilation Error** - Added missing `layer_call_count` variable declaration
2. **Matrix Layout** - Fixed GGUF row-major vs cuBLAS column-major mismatch
   - Q values now in correct range (0.01-0.26)
   - All 8 matrix multiplications corrected
3. **KV Cache** - Position tracking works correctly
   - Positions increment: 0→1→2→3...
   - Cache is being read and written properly
4. **Attention Mechanism** - Working correctly
   - Attention weights vary across positions
   - Weights sum to 1.0
   - Computes over all cached positions

---

## Current Bug ❌

### Symptom
Model generates the same token (ID=76156 "suffice") repeatedly, regardless of input.

### Evidence
```
Input tokens: 7985 → 264 → 6386 → 38242 (CHANGING)
Output tokens: 76156 → 76156 → 76156 → 76156 (STUCK)
Hidden states: -11.04, -2.41, 8.20... → -11.31, -2.75, 8.28... (CHANGING)
Logits: 0.19, 2.29, 3.11... → 0.33, 2.36, 3.32... (CHANGING)
Argmax finds: token_id=2966 with value 14.24 (BEYOND VOCAB!)
```

### Root Cause
The lm_head tensor in GGUF is [896, 151643] but vocab_size is 151936 (padded). The argmax kernel searches all 151936 positions and finds garbage values at positions beyond 151643.

**Specific Issue**: Token ID 2966 (and sometimes 85840) are beyond the actual vocab (151643) but within the padded buffer (151936). These positions contain uninitialized garbage with values ~14.0, higher than any real logit (~8.0).

### Attempted Fixes (All Failed)
1. ✗ Initialize logits buffer to -INFINITY at allocation
2. ✗ Initialize logits to -INFINITY before each projection
3. ✗ Change cuBLAS output stride from vocab_size to actual_vocab
4. ✗ Pass actual_vocab_size (151643) to sampling function

### Why Fixes Failed
The cuBLAS GEMM operation writes to the logits buffer with stride `actual_vocab` (151643), but the buffer is allocated for `vocab_size` (151936). The garbage values persist in positions 151643-151935.

---

## Reference: llama.cpp Behavior (Authoritative)

- **n_vocab source**: Derived from tokenizer/vocab: `vocab.n_tokens()`.
  - `reference/llama.cpp/src/llama-model.cpp` uses `const int64_t n_vocab = vocab.n_tokens();`.
- **lm_head dimensions**: `output.weight` is created as `{n_embd, n_vocab}`. No padding.
  - Multiple sites in `llama-model.cpp` create `output` with `{ n_embd, n_vocab }`.
- **Logits buffer sizing**: Always sized to `n_vocab`; copies exactly `n_tokens * n_vocab` floats.
  - `reference/llama.cpp/src/llama-context.cpp` uses `return logits + j*model.vocab.n_tokens();` and copies `n_tokens*n_vocab*sizeof(float)`.
- **Sampling range**: Iterates strictly `token_id in [0, n_vocab)`.
  - `reference/llama.cpp/src/llama-sampling.cpp` builds candidates for `0..n_vocab-1` only.

Implication: llama.cpp never exposes or scans padded slots. Our implementation must mirror this.

## Engineering Plan: Adopt Actual Vocab End-to-End (Fix)

- **Rust (determine actual vocab from GGUF)**
  - Parse tensors via `worker_gguf::GGUFMetadata::parse_tensors(model_path)` and get `actual_vocab = dims[1]` of `"output.weight"` (e.g., 151643 for Qwen2.5-0.5B).
  - Pass this `actual_vocab` as `vocab_size` to `ffi::cuda_inference_init()` instead of the padded config value.

- **C++ transformer (`cuda/src/transformer/qwen_transformer.cpp`)**
  - In `project_to_vocab()`: remove hardcoded `151643`. Use `config_.vocab_size` as `actual_vocab` for GEMM `m` and `ldc` and for any loops.
  - Do not rely on padding; produce exactly `actual_vocab` logits.

- **C++ FFI (`cuda/src/ffi_inference.cpp`)**
  - In `cuda_inference_generate_token`: remove `actual_vocab_size = 151643` constant. Call `cuda_sample_token(ctx->logits_buffer, /*vocab=*/config.vocab_size, ...)` (plumb if needed via `InferenceContext`).
  - Ensure `ctx->logits_buffer` was allocated with `vocab_size = actual_vocab` in `cuda_inference_init()` (already true if Rust passes actual).

- **Sampling (`cuda/kernels/sampling_wrapper.cu`)**
  - No functional change required if called with the correct `vocab_size`. `argmax_kernel` and softmax already iterate `i < vocab_size`.

- **Sanity checks**
  - At init, assert `lm_head` leading dimension equals `config_.vocab_size` and log mismatch if any.
  - Optional: assert token embedding table rows match tokenizer size.

## Execution Order (Today)

1. Implement Rust actual-vocab derivation and pass-through.
2. Remove hardcoded 151643 in C++ and use config-provided vocab.
3. Re-run haiku test (greedy) and verify that argmax stays within `[0, actual_vocab)` and output token varies.
4. If output still degrades, proceed to bias/quantization follow-ups already tracked.

## Files Modified

### Core Fixes
- `cuda/src/transformer/qwen_transformer.cpp` - Matrix layout fixes, position tracking
- `cuda/kernels/swiglu_ffn.cu` - FFN matrix operations
- `cuda/kernels/gqa_attention.cu` - Attention computation (already working)
- `cuda/src/ffi_inference.cpp` - Logits buffer initialization, vocab size handling
- `cuda/kernels/sampling_wrapper.cu` - Added debug output to argmax

### Debug Files
- Added `layer_call_count` tracking
- Added extensive debug logging for logits, attention, hidden states

---

## Next Steps to Fix

### Option 1: Fix Vocab Size Mismatch (RECOMMENDED)
1. Get actual lm_head tensor dimensions from GGUF metadata
2. Use actual dimensions for cuBLAS GEMM
3. Only search actual vocab range in argmax

### Option 2: Proper Buffer Initialization
1. Create a CUDA kernel to fill logits[151643:151935] with -INFINITY
2. Call kernel before each projection
3. Ensure argmax skips -INFINITY values

### Option 3: Limit Argmax Search Range
1. Modify argmax_kernel to only search first `actual_vocab` positions
2. Pass actual_vocab as parameter to cuda_sample_token
3. Ignore positions beyond actual_vocab

---

## Test Command

```bash
cd /home/vince/Projects/llama-orch/bin/worker-orcd
cargo test --release --test haiku_generation_anti_cheat test_haiku_generation_stub_pipeline_only \
  --features cuda -- --ignored --nocapture --test-threads=1
```

---

## Key Metrics

- **Compilation**: ✅ Success
- **Matrix Layout**: ✅ Fixed (Q values correct)
- **KV Cache**: ✅ Working (positions increment)
- **Attention**: ✅ Working (weights vary)
- **Logits**: ✅ Changing (values update each forward pass)
- **Sampling**: ❌ BROKEN (finds garbage beyond vocab)
- **Output Quality**: ❌ BROKEN (same token repeated)

---

## Debug Output Locations

- Logits: `qwen_transformer.cpp` line 520-540
- Argmax: `sampling_wrapper.cu` line 113-123
- Attention: `gqa_attention.cu` line 69-192
- Position tracking: `qwen_transformer.cpp` line 547-593

---

**Last Updated**: 2025-10-06 11:56
**Status**: Debugging in progress - root cause identified, fix in progress
