# Next Investigation Steps

**Date**: 2025-10-06  
**Current Status**: Matrix transpose fixed, but model generates repetitive tokens  
**Test Status**: ✅ Pipeline works, ⚠️ Output quality poor

---

## What We Fixed ✅

1. **Matrix transpose bug** in `cuda/src/transformer/qwen_transformer.cpp:project_to_vocab()`
   - Changed `CUBLAS_OP_N, CUBLAS_OP_N` → `CUBLAS_OP_T, CUBLAS_OP_N`
   - Fixed leading dimension from `vocab_size` → `hidden_dim`
   - **Result**: Logits are now computed correctly

2. **Debug logging added**
   - Position tracking
   - Token IDs during prefill
   - Logits sampling
   - Embedding values

---

## Current Behavior

### Input
- Prompt: "Write a haiku about GPU computing that includes the word \"forty-six\" (nonce: x37jIGjZ)"
- Tokenizes to: 27 tokens
- Prefill: 26 tokens
- Starts generation from token ID=8

### Output
```
ĠcomponentWillMountĠcomponentWillMountĠcomponentWillMount...
```

**Token IDs**: 78138 (repeated), 118530, 80030, 14942

### Observations
1. Model generates same token repeatedly (78138 = "ĠcomponentWillMount")
2. Logits look reasonable: `[-3.60, 0.04, -0.75, 0.25, 0.76, -2.14, -2.46, 1.80, -2.16, -0.47]`
3. Position increments correctly (0, 1, 2, 3...)
4. KV cache is being updated (position advances)

---

## Hypotheses for Repetitive Output

### 1. **Attention is Broken** 🔴 HIGH PRIORITY
The model might be attending to the wrong positions or not using the KV cache properly.

**Evidence**:
- Repetitive output suggests model isn't "seeing" its previous generations
- Could be attention mask issue
- Could be KV cache read/write mismatch

**How to Test**:
```cpp
// In gqa_attention.cu, add debug logging:
- Print attention scores for first few positions
- Verify K/V are being written to cache
- Verify K/V are being read from cache correctly
```

### 2. **RoPE is Incorrect** 🟡 MEDIUM PRIORITY
Rotary Position Embeddings might not be applying correctly.

**Evidence**:
- RoPE is critical for position-aware attention
- If RoPE is wrong, model can't distinguish positions
- Would explain why it generates same token regardless of context

**How to Test**:
```cpp
// In rope.cu, add debug:
- Print RoPE frequencies
- Verify sin/cos values
- Check if position is being used correctly
```

### 3. **Weight Loading Issue** 🟡 MEDIUM PRIORITY
Some weights might be loaded incorrectly or with wrong dimensions.

**Evidence**:
- Token 78138 ("componentWillMount") is a React lifecycle method
- This is a very specific, rare token
- Suggests weights might be misaligned

**How to Test**:
```bash
# Compare tensor dimensions with GGUF metadata
# Verify all 291 tensors have correct shapes
# Check if any tensors are swapped
```

### 4. **Tokenizer/Prompt Format** 🟢 LOW PRIORITY
Qwen models might need specific chat template.

**Evidence**:
- We're using raw prompt without chat template
- Qwen typically uses `<|im_start|>` tokens

**How to Test**:
```rust
// Try with proper Qwen chat format:
let prompt = format!(
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
    user_prompt
);
```

---

## Recommended Investigation Order

### Phase 1: Verify Attention (1-2 hours)
1. Add debug logging to `gqa_attention.cu`
   - Print first 5 attention scores
   - Verify softmax sums to 1.0
   - Check if attending to recent positions

2. Verify KV cache writes
   - Print K/V values being written
   - Verify cache indices are correct

3. Verify KV cache reads
   - Print K/V values being read
   - Compare with what was written

### Phase 2: Check RoPE (30 min)
1. Add debug to `rope.cu`
   - Print RoPE frequencies for first few positions
   - Verify sin/cos calculations
   - Check position parameter

2. Compare with llama.cpp RoPE
   - Reference: `reference/llama.cpp/ggml/src/ggml-cuda/rope.cu`

### Phase 3: Weight Verification (1 hour)
1. Dump tensor dimensions from GGUF
   ```python
   # Use gguf-py to list all tensors and shapes
   ```

2. Compare with our weight loader
   - Verify `qwen_weight_loader.cpp` maps correctly
   - Check for any dimension mismatches

3. Spot-check a few weight values
   - Compare first/last values of `token_embd.weight`
   - Verify they match GGUF file

### Phase 4: Try Different Prompts (15 min)
1. Test with minimal prompt: "Hello"
2. Test with just BOS token
3. Test with Qwen chat template

---

## Quick Wins to Try First

### 1. Check if it's a Sampling Issue
```rust
// In cuda_backend.rs, try with temperature > 0
let next_token_id = inference.generate_token(
    current_token,
    0.7,  // ← Change from 0.0 to 0.7
    50,   // ← Enable top-k
    0.9,  // ← Enable top-p
    config.seed.wrapping_add(token_idx as u64),
)?;
```

If this produces different (even if still bad) output, the issue is in the logits distribution, not the sampling.

### 2. Check Global Logits Max
The debug code we added should show:
```
MAX logit: X.XXXX at token_id=78138
```

If token_id=78138 is consistently the max, then the logits are genuinely pointing to that token (which means the issue is upstream in the transformer).

### 3. Test with llama.cpp
```bash
cd reference/llama.cpp
./llama-cli -m /path/to/qwen2.5-0.5b-instruct-fp16.gguf \
    -p "Write a haiku about GPU computing" \
    -n 50
```

Compare output with ours. If llama.cpp also produces garbage, the model file might be corrupted.

---

## Files to Focus On

### High Priority
1. `cuda/kernels/gqa_attention.cu` - Attention mechanism
2. `cuda/kernels/rope.cu` - Position embeddings
3. `cuda/src/transformer/qwen_transformer.cpp` - Forward pass orchestration

### Medium Priority
4. `cuda/src/model/qwen_weight_loader.cpp` - Weight loading
5. `src/cuda/weight_loader.rs` - Rust-side weight loading
6. `src/inference/cuda_backend.rs` - Prefill logic

### Reference
7. `reference/llama.cpp/ggml/src/ggml-cuda/rope.cu` - RoPE reference
8. `reference/llama.cpp/src/llama-model.cpp` - Model building reference

---

## Debug Commands

### Run test with full output
```bash
cargo test --test haiku_generation_anti_cheat --features cuda -- --nocapture --ignored 2>&1 | tee debug.log
```

### Check specific debug output
```bash
grep -A 5 "Forward #" debug.log
grep "Sampled token" debug.log | head -20
grep "logits\[0:10\]" debug.log
```

### Compare with llama.cpp
```bash
cd reference/llama.cpp
./llama-cli -m ../../.test-models/qwen/qwen2.5-0.5b-instruct-fp16.gguf \
    -p "Test prompt" -n 20 --verbose-prompt
```

---

## Success Criteria

We'll know we've fixed it when:
1. ✅ Model generates diverse tokens (not repetitive)
2. ✅ Output is coherent English text
3. ✅ Model responds appropriately to prompt
4. ✅ Different prompts produce different outputs

---

## Resources

- **llama.cpp reference**: `/reference/llama.cpp/`
- **GGUF spec**: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- **Qwen2 model card**: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
- **Our docs**: `INFERENCE_FIX_SUMMARY.md`

---

## ✅ Debug Logging Complete - 2025-10-06

### What Was Added

Comprehensive debug logging has been added to all critical components:

1. **Attention kernel** (`gqa_attention.cu`) - Full pipeline visibility
2. **RoPE kernel** (`rope.cu`) - Position encoding verification
3. **QKV projections** (`qwen_transformer.cpp`) - Weight matrix outputs
4. **Embedding lookup** - Input token embeddings
5. **Final hidden states** - Pre-logit representations

### How to Run

```bash
cargo test --test haiku_generation_anti_cheat --features cuda -- --nocapture --ignored 2>&1 | tee debug_full.log
```

### What to Look For

The debug output will now show you **exactly** where the issue is:

- If **attention weights** are broken, you'll see it immediately
- If **RoPE** is wrong, you'll see Q/K not rotating properly
- If **weight loading** is wrong, you'll see zero or NaN values
- If **cache** is broken, you'll see attention not using past context

**Next step**: Run the test and analyze the debug output to identify the root cause.

---

## ✅ llama.cpp Validation Complete - 2025-10-06

### Critical Finding: Model File is VALID ✅

Tested the same GGUF model with llama.cpp reference implementation:

**Result**: llama.cpp generates **perfect, coherent output**:
```
Forty-six,  
CUDA's power,  
Compute's might.
```

### What This Proves

1. ✅ **Model file is NOT corrupted** - All weights are valid
2. ✅ **Qwen2.5 works correctly** - Can generate coherent text
3. ❌ **Bug is in OUR implementation** - Not the model

### Implication

Since the same model works in llama.cpp but fails in our code, **the bug is definitely in our implementation**. Most likely culprits:

1. **Attention mechanism** (GQA implementation)
2. **KV cache indexing**
3. **RoPE application**
4. **Weight tensor mapping**

**See**: `LLAMA_CPP_VALIDATION.md` for full details

**Next step**: Compare our attention implementation with llama.cpp's reference code.
