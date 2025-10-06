# llama.cpp Validation Results

**Date**: 2025-10-06  
**Status**: ✅ Model file is VALID - Bug is in our implementation

---

## Test Setup

- **Model**: `qwen2.5-0.5b-instruct-fp16.gguf` (1.17 GiB)
- **Tool**: llama.cpp reference implementation (commit c5fef0fc)
- **Prompt**: "Write a haiku about GPU computing that includes the word \"forty-six\""
- **Temperature**: 0.0 (greedy sampling, same as our test)
- **GPU**: CUDA enabled, 24 layers offloaded

---

## Result: SUCCESS ✅

llama.cpp generated a **coherent, valid haiku**:

```
Forty-six,  
CUDA's power,  
Compute's might.
```

### Performance Metrics
- **Prompt eval**: 21.49 tokens/sec (123 tokens)
- **Generation**: 195.30 tokens/sec (22 tokens)
- **Total time**: ~6 seconds

---

## What This Proves

### ✅ Model File is Valid
- GGUF file loads correctly
- All 291 tensors are intact
- Weights are not corrupted
- Model architecture is correct

### ✅ Qwen2.5 Can Generate Coherent Text
- Model responds appropriately to prompt
- Follows haiku format
- Includes requested word ("forty-six")
- Output is grammatically correct

### ❌ Our Implementation Has a Bug
Since the same model file works perfectly in llama.cpp but produces garbage in our implementation, **the bug is definitely in our code**, not the model.

---

## Comparison: llama.cpp vs Our Implementation

| Aspect | llama.cpp | Our Implementation |
|--------|-----------|-------------------|
| Model file | ✅ Same file | ✅ Same file |
| Output quality | ✅ Coherent haiku | ❌ Repetitive garbage |
| Token diversity | ✅ Varied tokens | ❌ Same token (78138) |
| Context awareness | ✅ Responds to prompt | ❌ Ignores context |
| Temperature | 0.0 (greedy) | 0.0 (greedy) |

---

## Root Cause Narrowed Down

Since llama.cpp works and we don't, the bug must be in one of these areas:

### 1. **Attention Implementation** 🔴 MOST LIKELY
- Our GQA attention kernel may have bugs
- KV cache read/write may be incorrect
- Attention mask may be wrong
- RoPE application may differ from llama.cpp

### 2. **Weight Loading/Mapping**
- Tensor names may be mapped incorrectly
- Dimensions may be transposed
- Some weights may be missing or duplicated

### 3. **Tokenizer Differences**
- Less likely since we're using same tokenizer
- But worth checking if BOS/EOS handling differs

---

## Next Steps

### Priority 1: Compare Attention Implementation
Compare our `gqa_attention.cu` with llama.cpp's attention:
- File: `reference/llama.cpp/ggml/src/ggml-cuda/template-instances/fattn-wmma-f16.cu`
- Check: Attention score calculation, softmax, KV cache indexing

### Priority 2: Add Host-Side Attention Debugging
Since CUDA printf isn't working, copy attention weights to CPU:
```cpp
// After attention kernel, copy scores to host and print
float* h_attn_scores = new float[cache_len];
cudaMemcpy(h_attn_scores, d_attn_scores, ...);
for (int i = 0; i < cache_len; i++) {
    fprintf(stderr, "attn[%d] = %.6f\n", i, h_attn_scores[i]);
}
```

### Priority 3: Verify Weight Tensor Mapping
Compare our weight loading with llama.cpp:
- Our file: `cuda/src/model/qwen_weight_loader.cpp`
- Reference: `reference/llama.cpp/src/llama-model.cpp`

---

## Conclusion

**The model is NOT the problem. Our implementation is.**

This is actually good news because:
1. We don't need to find a different model
2. The bug is in code we control
3. We have a working reference (llama.cpp) to compare against

The most likely culprit is the **attention mechanism**, specifically how we handle:
- Grouped-query attention (GQA) with 7 groups
- KV cache indexing
- Attention score computation

---

**Action**: Focus debugging on attention kernel and compare with llama.cpp implementation.


## Update (2025-10-06 12:18): Vocab Handling in llama.cpp and Action Plan

### What llama.cpp does

- **n_vocab source**: `n_vocab = vocab.n_tokens()` from tokenizer/gguf (no padding).
- **lm_head dimensions**: `output.weight` created as `{n_embd, n_vocab}`.
- **logits sizing**: All logits buffers sized to `n_vocab` and copied as `n_tokens * n_vocab` floats.
- **sampling bound**: Candidates built strictly for token IDs `[0, n_vocab)`.

References:
- `reference/llama.cpp/src/llama-model.cpp`: `const int64_t n_vocab = vocab.n_tokens();` and `{n_embd, n_vocab}` when creating `output`.
- `reference/llama.cpp/src/llama-context.cpp`: uses `model.vocab.n_tokens()` for logits stride/size.
- `reference/llama.cpp/src/llama-sampling.cpp`: iterates `token_id < n_vocab` when building candidates.

### Implication for our code

- We must propagate the **actual vocab size** from GGUF/tokenizer and avoid exposing padded columns in logits and sampling.

### Action Plan

- **Rust**: In `src/inference/cuda_backend.rs`, derive `actual_vocab` from tokenizer or `"output.weight"` dims; pass to `ffi::cuda_inference_init()`.
- **C++ transformer**: In `cuda/src/transformer/qwen_transformer.cpp::project_to_vocab()`, remove hardcoded `151643`; use `config_.vocab_size` for GEMM `m/ldc` and loops.
- **C++ FFI**: In `cuda/src/ffi_inference.cpp::cuda_inference_generate_token()`, remove `actual_vocab_size = 151643`; call `cuda_sample_token(logits, config.vocab_size, ...)`.
- **CUDA sampling**: `cuda/kernels/sampling_wrapper.cu` already honors the passed `vocab_size`.
- **Sanity**: At init, assert LM head leading dim equals `config_.vocab_size` and log mismatches.

Once aligned, argmax will not scan padded garbage and output should diversify under greedy sampling.
