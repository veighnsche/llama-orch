# Team BLUE Investigation Report

**Date:** 2025-10-06 20:56 UTC  
**Mission:** Hunt down garbage output bug (mojibake/repetitive tokens)  
**Status:** 🔍 **ACTIVE INVESTIGATION**

---

## 🎯 Current Symptoms (from Team GREEN)

**Test Output:**
```
Output: çĤĬĠmilitĠmilitĠscarcityå¯¹å¤ĸå¼ĢæĶ¾Ġà¸ĺà¸±à¸Ļà¸§à¸²...
```

**Characteristics:**
- Mojibake: Chinese/Thai/Korean tokens instead of English
- Repetitive: "Ġmilit" 2x, "ĠÐ»ÐµÑĩ" 10+x, "Ġconcise" 3x
- High token IDs: 119130, 110707, 142698 near vocab limit (151643)
- Wrong context: Code tokens ("React", "Scouts") instead of haiku

---

## ✅ What's Verified (DO NOT RE-INVESTIGATE)

From previous teams:
- ✅ cuBLAS dimensions and matrix multiplication
- ✅ Sampling logic (argmax/temperature/softmax)
- ✅ KV cache infrastructure
- ✅ RMSNorm implementation
- ✅ Chat template format (CHATML for Qwen)
- ✅ Q/K/V biases (exist but are all zeros)

---

## 🔍 Investigation Focus (NARROW SCOPE)

Per mission brief, focus on:
1. **Prompt & tokenization:** BOS/EOS tokens, special tokens, chat template
2. **Logits & projection:** output_norm, final projection, logits scaling
3. **Sampling:** temperature, top-k/p, repetition penalties
4. **NOT in scope:** CUDA kernels, IPC, HTTP server (unless proven to corrupt logits)

---

## 🧪 Investigation Log

### [TEAM BLUE] 2025-10-06T20:56Z
**SUSPECT:** Need to trace exact token IDs being fed to model vs what llama.cpp uses
**PLAN:** 
1. Check formatted prompt string (line 148-151 in cuda_backend.rs)
2. Check token IDs from tokenizer (line 161)
3. Compare with llama.cpp token IDs for same prompt
4. Look for BOS token issues or chat template discrepancies

**Starting investigation...**

### [TEAM BLUE] 2025-10-06T20:56Z - CRITICAL BUG FOUND!
**SUSPECT:** GgufBpe tokenizer ignores `add_special_tokens` parameter!
**PLAN:** Trace tokenizer code to confirm
**OBSERVED:** 
- File: `bin/worker-crates/worker-tokenizer/src/backend.rs` lines 147-153
- Code:
  ```rust
  Tokenizer::GgufBpe { encoder, .. } => {
      encoder.encode(text).map_err(...)
  }
  ```
- The `add_special_tokens` parameter is IGNORED!
- It only calls `encoder.encode(text)` which does NOT add BOS token
- HfJson backend correctly uses the parameter (line 152)

**CONTRADICTION:** 
- cuda_backend.rs line 161 passes `add_special_tokens=true`
- But GgufBpe tokenizer NEVER adds BOS token!
- llama.cpp DOES add BOS token for Qwen models

**ROOT CAUSE HYPOTHESIS:**
The model expects BOS token at the start, but we're not providing it!
This causes the model to see malformed input → generates garbage tokens.

**VERIFICATION NEEDED:**
1. Check if Qwen2.5 requires BOS token
2. Check what token IDs llama.cpp generates for same prompt
3. Fix tokenizer to respect add_special_tokens parameter

### [TEAM BLUE] 2025-10-06T20:57Z - Checking llama.cpp behavior
**PLAN:** Look at llama.cpp tokenization to see if BOS is added
**OBSERVED:**
- llama-cli help shows: `tokenizer.ggml.add_bos_token=bool:false`
- This means Qwen2.5 does NOT add BOS token by default!
- llama.cpp reads this from GGUF metadata (llama-vocab.cpp:2150)

**CONTRADICTION:**
Wait - if Qwen doesn't need BOS, then why is our output garbage?
Let me check if the chat template itself includes special tokens...

### [TEAM BLUE] 2025-10-06T20:58Z - Investigating chat template tokens
**SUSPECT:** Chat template special tokens might not be tokenized correctly
**PLAN:** Check if `<|im_start|>` and `<|im_end|>` are being tokenized as special tokens
**HYPOTHESIS:** 
- Qwen uses ChatML format with `<|im_start|>` and `<|im_end|>` markers
- These should be tokenized as SINGLE special tokens, not as character sequences
- If we're tokenizing them as regular text, model sees wrong input

**VERIFICATION:**
Let me check what token IDs these special tokens should have

### [TEAM BLUE] 2025-10-06T21:02Z - Adding debug logging
**PLAN:** Add logging to cuda_backend.rs to see exact token IDs
**ACTION:** Will add logging to show:
1. Formatted prompt string
2. First 30 token IDs from tokenizer
3. Decoded tokens to verify special tokens are correct

**HYPOTHESIS:**
The `<|im_start|>` and `<|im_end|>` markers need to be tokenized as special tokens.
If our BPE encoder treats them as regular text, it will split them into multiple tokens,
and the model won't recognize the chat format.

### [TEAM BLUE] 2025-10-06T21:04Z - 🔥 ROOT CAUSE FOUND!
**OBSERVED:** Test output shows special tokens are SPLIT:
```
[TEAM_BLUE]   Token[0] = 27 → "<"
[TEAM_BLUE]   Token[1] = 91 → "|"
[TEAM_BLUE]   Token[2] = 318 → "im"
[TEAM_BLUE]   Token[3] = 4906 → "_start"
[TEAM_BLUE]   Token[4] = 91 → "|"
[TEAM_BLUE]   Token[5] = 29 → ">"
[TEAM_BLUE]   Token[6] = 872 → "user"
```

**ROOT CAUSE:**
`<|im_start|>` is being tokenized as 6 separate tokens instead of 1 special token!
The model expects a SINGLE token ID for `<|im_start|>` (likely token 151644 or similar).
Instead, it sees: `<` + `|` + `im` + `_start` + `|` + `>` → model has NO IDEA this is a chat format marker.

**FIXED:** This is why output is garbage!
- Model trained on: `[151644] [872] [198] [7985]...` (special token + user + newline + Write)
- We're feeding: `[27] [91] [318] [4906] [91] [29] [872]...` (6 char tokens + user)
- Model sees completely wrong input → generates mojibake

**SOLUTION:**
Need to tokenize special tokens as SINGLE tokens, not split them with BPE.
llama.cpp has special token handling that treats these as atomic tokens.

### [TEAM BLUE] 2025-10-06T21:10Z - FIX APPLIED
**ACTION:** Implemented workaround in `cuda_backend.rs`
**APPROACH:** Manually construct token sequence with correct special token IDs
- `<|im_start|>` → token 151644 (not split)
- `<|im_end|>` → token 151645 (not split)
- Tokenize text segments normally with BPE

**FIXED:** Modified `src/inference/cuda_backend.rs` lines 148-184
Instead of tokenizing the full formatted prompt (which splits special tokens),
we now:
1. Push token 151644 for `<|im_start|>`
2. Tokenize "user\n{prompt}" normally
3. Push token 151645 for `<|im_end|>`
4. Tokenize "\n" normally  
5. Push token 151644 for `<|im_start|>`
6. Tokenize "assistant\n" normally

**VERIFICATION:** Running test now...

### [TEAM BLUE] 2025-10-06T21:10Z - TOKENIZATION FIXED BUT MODEL STILL BROKEN
**OBSERVED:** Test shows tokens are NOW CORRECT:
```
[TEAM_BLUE]   Token[0] = 151644 → "<|im_start|>"  ✅ SINGLE TOKEN!
[TEAM_BLUE]   Token[1] = 872 → "user"
[TEAM_BLUE]   Token[2] = 198 → "\n"
[TEAM_BLUE]   Token[3] = 7985 → "Write"
```

**BUT:** Model still generates garbage:
```
Output: ĠsupplementationãĤ¸ãĥ¥_handlesĠLumpà¸Ħà¸²ĠTreeSet_STRUCTURE...
```

**CONCLUSION:**
- ✅ Tokenization bug FIXED
- ❌ There's ANOTHER bug in the forward pass
- The special tokens are correct, but model still doesn't generate proper output

**HANDOFF TO NEXT TEAM:**
Tokenization is now correct. The bug must be in:
1. Embedding lookup for special tokens
2. Transformer forward pass
3. Final projection
4. Something else in the CUDA code

The fact that llama.cpp works with same model proves bug is still in our code.

