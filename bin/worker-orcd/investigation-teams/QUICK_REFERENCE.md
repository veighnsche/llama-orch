# Quick Reference - Garbage Output Bug Investigation

**Read this FIRST before investigating!**

---

## 🚨 Current Status

**Bug:** Model generates garbage output (code tokens, foreign language) instead of haiku

**What's VERIFIED CORRECT:**
- ✅ Tokenization (special tokens, IDs, sequence)
- ✅ Embeddings (all valid, not zeros)
- ✅ Prompt format (matches llama.cpp)
- ✅ Embedding lookup (CUDA kernel works)

**Where bug likely is:**
- ❓ Forward pass (attention/FFN/residual)
- ❓ KV cache (prefill/generation)
- ❓ Position encoding (RoPE)
- ❓ Sampling logic

---

## 📋 Before You Start

### 1. Read These Documents (in order):
1. `FALSE_LEADS_SUMMARY.md` ← **READ THIS FIRST!**
2. `TEAM_BLUE_FINAL_STATUS.md` (tokenization fix)
3. `TEAM_PURPLE_FINAL_STATUS.md` (verification)

### 2. Check Code Comments:
- `src/inference/cuda_backend.rs` lines 153-246
- `cuda/src/transformer/qwen_transformer.cpp` lines 1033-1110

### 3. Don't Re-Investigate:
- ❌ Special token IDs (151644, 151645 are correct!)
- ❌ Token embeddings (they're valid!)
- ❌ Chat template format (matches llama.cpp!)
- ❌ Tokenization approach (it's correct!)

---

## 🔍 Quick Facts

### Special Tokens (Qwen2.5)
```
151643 = BOS/PAD (endoftext)
151644 = <|im_start|>
151645 = <|im_end|>
```

### Token Sequence (Current)
```
[0] 151644 → <|im_start|>
[1] 872 → user
[2] 198 → \n
[...] → prompt text
[28] 151645 → <|im_end|>
[29] 198 → \n
[30] 151644 → <|im_start|>
[31] 77091 → assistant
```

### Embedding Values (Sample)
```
Token 151644: 0.0014 -0.0084 0.0073 ... (valid FP16)
```

---

## 🎯 Investigation Priorities

### Priority 1: Compare with llama.cpp
Run llama.cpp with SAME prompt and compare:
- Hidden states after each layer
- Attention weights
- KV cache contents
- Logits before sampling

### Priority 2: Check KV Cache
- Dump cache after prefill
- Verify positions are correct
- Check if values match expected range

### Priority 3: Verify Forward Pass
- Add logging at each layer
- Check for NaN/Inf values
- Verify residual connections

### Priority 4: Test Isolation
- Disable KV cache (force recompute)
- Test with single token generation
- Compare prefill vs generation phase

---

## 🧪 Useful Test Commands

### Run haiku test:
```bash
REQUIRE_REAL_LLAMA=1 cargo test --release --features cuda \
  --test haiku_generation_anti_cheat \
  test_haiku_generation_stub_pipeline_only \
  -- --ignored --nocapture --test-threads=1
```

### Check token sequence:
```bash
# Look for [TEAM_PURPLE] lines showing decoded tokens
... | grep "TEAM_PURPLE.*\["
```

### Check embeddings:
```bash
# Look for embedding values
... | grep "TEAM_PURPLE.*embedding"
```

---

## 📊 Symptom Analysis

### What We See:
```
Generated tokens:
[0] ID=131916 → "ãĤ¸ãĥ¥"
[1] ID=72696 → "Ġsupplementation"
[2] ID=13267 → "serve"
[3] ID=105030 → "åŁĭ"
```

### What This Means:
- **Code tokens** (psycopg, toHaveBeenCalledWith) → Wrong domain
- **Foreign language** (Chinese, Thai) → Random selection
- **No haiku words** → Model doesn't understand context

### What This Rules Out:
- ❌ NOT a tokenization issue (would see some correct words)
- ❌ NOT an embedding issue (would see semantic similarity)
- ❌ NOT a prompt format issue (llama.cpp works with same format)

### What This Suggests:
- ✅ Hidden states are corrupted
- ✅ Attention is focusing on wrong tokens
- ✅ KV cache contains wrong values
- ✅ Position encoding is incorrect

---

## 🚫 Common Mistakes

### Mistake #1: "Let me check the token IDs..."
**STOP!** Token IDs are verified correct. Read `FALSE_LEADS_SUMMARY.md`.

### Mistake #2: "Maybe embeddings are zeros..."
**STOP!** Embeddings are verified valid. Read Team Purple's findings.

### Mistake #3: "Let me try different chat template..."
**STOP!** Template matches llama.cpp exactly. Don't change it.

### Mistake #4: "Maybe we need BOS token..."
**STOP!** Qwen2.5 doesn't use BOS. llama.cpp confirms this.

---

## 💡 Debugging Tips

### Add Logging:
```cpp
// In qwen_transformer.cpp
fprintf(stderr, "[YOUR_TEAM] After layer %d: range=[%.4f, %.4f]\n", 
        layer_idx, min_val, max_val);
```

### Compare Values:
```bash
# Run llama.cpp with same prompt
./llama-cli -m model.gguf -p "Write a haiku..." -n 10

# Compare output with our test
```

### Check Ranges:
- Embeddings: ~0.01 to 0.04 (normal)
- Hidden states: should grow gradually, not explode
- Logits: -5 to +5 (normal), not -100 or +100

---

## 📞 Need Help?

1. **Read documents first** (especially `FALSE_LEADS_SUMMARY.md`)
2. **Check code comments** (they have detailed findings)
3. **Look at previous team's work** (don't repeat investigations)
4. **Document your findings** (add comments for next team)

---

## ✅ When You Find The Bug

1. **Document the root cause** clearly
2. **Add comments** explaining what was wrong
3. **Update this guide** with new findings
4. **Mark false leads** so others don't follow them

---

**Good luck! 🍀**

---
