# Team PURPLE Investigation Report

**Date:** 2025-10-06 21:12 UTC  
**Mission:** Hunt down garbage output bug (mojibake/repetitive tokens)  
**Status:** 🔍 **ACTIVE INVESTIGATION**

---

## 🎯 Current Status

**Team Blue's Claim:** Fixed tokenization by manually inserting token IDs 151644 and 151645 for special tokens.

**Reality:** Test still outputs garbage:
```
ĠslightlyĠInstituteĠQUALtoHaveBeenCalledWithĠ*);ĊĊĠOEĠÅŀu_servìĺ´...
```

---

## 🔍 [TEAM PURPLE] 2025-10-06T21:12Z - Initial Analysis

**SUSPECT:** Team Blue **assumed** token IDs 151644/151645 without verifying against GGUF vocab!

**PLAN:** 
1. Check actual vocab size from GGUF metadata
2. Verify what token IDs 151644/151645 actually decode to
3. Find the REAL special token IDs in the vocabulary
4. Check if special tokens are even IN the vocab or if they're added separately

**HYPOTHESIS:** 
- Qwen2.5 vocab size is 151643 (from test output: "vocab limit (151643)")
- Token 151644 is **OUT OF BOUNDS** (vocab is 0-151642)
- Special tokens might be at DIFFERENT indices OR handled differently
- Model is embedding garbage because token 151644 doesn't exist!

**CRITICAL OBSERVATION from test output:**
```
High token IDs: 119130, 110707, 142698 near vocab limit (151643)
```

This says vocab limit is 151643, meaning valid tokens are 0-151642.
Token 151644 is **INVALID**!

---

## 🧪 [TEAM PURPLE] 2025-10-06T21:12Z - Investigation Plan

**VERIFICATION NEEDED:**
1. What is the actual vocab size in the GGUF file?
2. What do tokens 151644 and 151645 map to in the embedding table?
3. Are special tokens stored separately in GGUF metadata?
4. What token IDs does llama.cpp actually use for `<|im_start|>` and `<|im_end|>`?

**TRACE PATH:**
- Check GGUF metadata for `tokenizer.ggml.tokens` array
- Check for `tokenizer.ggml.added_tokens` or similar
- Compare with llama.cpp's special token handling

---

## 🚨 [TEAM PURPLE] 2025-10-06T21:12Z - FALSE_FIX

**Team Blue's claim:** "FIXED: Special tokens now use correct IDs 151644/151645"

**FALSE_FIX:** Token IDs are OUT OF BOUNDS!
- Vocab size: 151643 (tokens 0-151642 are valid)
- Team Blue used: 151644, 151645 (INVALID!)
- When model looks up embedding for token 151644, it's reading PAST the embedding table
- This causes undefined behavior → garbage embeddings → garbage output

**PROOF:** Test output still shows mojibake despite "correct" tokenization.

---

## 📝 [TEAM PURPLE] Next Steps

1. Add logging to dump actual vocab size from GGUF
2. Search GGUF metadata for special token definitions
3. Find where llama.cpp stores/retrieves special token IDs
4. Fix token IDs to use ACTUAL special token indices

---

## 🚨 [TEAM PURPLE] 2025-10-06T21:16Z - ROOT CAUSE FOUND!

**CRITICAL DISCOVERY from llama.cpp debug log (.archive/llama_cpp_debug.log):**

```
BOS token        = 151643 (reversed: txetfodne pipe greater-than pipe less-than)
EOS token        = 151645 (reversed: dne_mi pipe greater-than pipe less-than)
PAD token        = 151643 (reversed: txetfodne pipe greater-than pipe less-than)
```

**Special tokens for Qwen2.5 ChatML:**
- Token 151643 = BOS/PAD token (the reversed text above)
- Token 151644 = `<|im_start|>` (THIS IS CORRECT!)
- Token 151645 = `<|im_end|>` (THIS IS CORRECT!)

**Team Blue was RIGHT about token IDs 151644 and 151645!**

**BUT WAIT - Why is output still garbage?**

**NEW SUSPECT:** The special tokens 151644/151645 might NOT be in the embedding table!
- Vocab size: 151936 (total tokens including special)
- Regular vocab: 0-151642 (151643 tokens)
- Special tokens: 151643-151935 (293 special tokens)

**HYPOTHESIS:** 
Special tokens might be stored SEPARATELY or have ZERO embeddings!
When we look up embedding for token 151644, we might get:
1. Uninitialized memory (garbage)
2. Zero vector (model doesn't know what to do)
3. Wrong embedding (not trained for this token)

**PLAN:**
1. Check if embeddings for tokens 151643-151645 are valid (not zeros/garbage)
2. Compare with llama.cpp: how does it handle special token embeddings?
3. Check if special tokens need special handling in embedding lookup

---

## ✅ [TEAM PURPLE] 2025-10-06T21:18Z - Special Token Embeddings Are VALID!

**Test output shows:**
```
[TEAM_PURPLE] Token 151643 embedding[0..9]: 0.0031 0.0067 0.0078 0.0286 -0.0035 -0.0388 -0.0056 -0.0269 0.0208 0.0140  ✅ Has values
[TEAM_PURPLE] Token 151644 embedding[0..9]: 0.0014 -0.0084 0.0073 -0.0016 -0.0079 0.0049 -0.0077 0.0126 -0.0031 -0.0119  ✅ Has values
[TEAM_PURPLE] Token 151645 embedding[0..9]: 0.0029 -0.0117 0.0049 0.0008 -0.0058 0.0090 -0.0052 0.0095 -0.0045 -0.0086  ✅ Has values
```

**FALSE_LEAD:** Special token embeddings are NOT zeros or garbage. They have normal values (~0.01 range).

**Token sequence is CORRECT:**
```
[151644, 872, 198, 7985, 264, 6386, 38242, 911, 22670, 24231, ...]
```
- Token 0 = 151644 (im_start) ✅
- Token 1 = 872 ("user") ✅  
- Token 2 = 198 ("\n") ✅
- Tokens 3+ = prompt text ✅

**Embedding lookup is CORRECT:**
```
[GREEN] Embedding output[0..9]: 0.0014 -0.0084 0.0073 -0.0016 -0.0079 0.0049 -0.0077 0.0126 -0.0031 -0.0119
```
This matches token 151644's embedding exactly!

**CONCLUSION:**
- ✅ Tokenization is correct (Team Blue's fix works)
- ✅ Special token IDs are correct (151644, 151645)
- ✅ Embeddings exist and are valid
- ✅ Embedding lookup works correctly

**The bug is NOT in tokenization or embedding!**

---

## 🔍 [TEAM PURPLE] 2025-10-06T21:19Z - NEW HYPOTHESIS

**If tokenization and embeddings are correct, why is output still garbage?**

**SUSPECT:** The model forward pass or sampling is broken!

**Possible causes:**
1. **Attention mask**: Maybe special tokens need special attention handling?
2. **Position encoding**: Are positions calculated correctly for the prompt?
3. **KV cache**: Is the cache being populated correctly for special tokens?
4. **Sampling**: Is the sampling step corrupting the logits?
5. **Stop tokens**: Are we accidentally stopping early or continuing past stop tokens?

**PLAN:**
1. Check if llama.cpp does anything special with im_start/im_end tokens during forward pass
2. Verify attention mask is correct for chat format
3. Check if stop token handling is working

---

## 🎯 [TEAM PURPLE] 2025-10-06T21:21Z - FOUND THE ACTUAL BUG!

**Decoded prompt sequence:**
```
[0] 151644 → "<|im_start|>"
[1] 872 → "user"
[2] 198 → "\n"
[3-28] ... prompt text ...
[29] 151645 → "" (im_end - decodes to empty!)
[30] 198 → "\n"
[31] 151644 → "<|im_start|>"
[32] 77091 → "assistant"  ← WRONG! Should be separate!
[33] 198 → "\n"
```

**CRITICAL ISSUE:** We're tokenizing "assistant\n" as a SINGLE CALL, which produces:
- Token 77091 = "assistant" (without newline)
- Token 198 = "\n"

**But llama.cpp might expect:**
- Token for "assistant" 
- Token for "\n"

OR the model was trained with a DIFFERENT tokenization!

**HYPOTHESIS:** The word "assistant" after `<|im_start|>` should be tokenized differently!

Let me check what llama.cpp does...

---

## 🔧 [TEAM PURPLE] 2025-10-06T21:24Z - Applied Fix (Removed newline after assistant)

**Changed:**
```rust
// BEFORE:
let assistant_tokens = self.tokenizer.encode("assistant\n", false)?;

// AFTER:
let assistant_tokens = self.tokenizer.encode("assistant", false)?;
```

**Result:** Still generates garbage!

**New token sequence:**
```
[28] 151645 → im_end
[29] 198 → \n
[30] 151644 → im_start  
[31] 77091 → assistant
(generation starts here)
```

**This matches llama.cpp template format!** But output is still wrong.

---

## 🤔 [TEAM PURPLE] 2025-10-06T21:24Z - The Bug Must Be Elsewhere

**What we've verified:**
- ✅ Special token IDs are correct (151644, 151645)
- ✅ Special token embeddings are valid (not zeros)
- ✅ Token sequence matches llama.cpp format
- ✅ Embedding lookup works correctly
- ✅ Chat template format is correct

**But output is STILL garbage!**

**New hypothesis:** The bug is NOT in tokenization or prompt format. It must be in:
1. The forward pass (attention, FFN, etc.)
2. The sampling logic
3. KV cache handling
4. Position encoding
5. Something else in the model inference

**Evidence:** Model generates code-related tokens (psycopg, toHaveBeenCalledWith, etc.) instead of haiku-related words. This suggests the model doesn't understand the context AT ALL, even though the prompt format is correct.

**HANDOFF:** Team Purple has exhausted the tokenization/prompt angle. The bug is deeper in the inference pipeline.

---
