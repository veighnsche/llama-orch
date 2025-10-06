# Team FINNEY — Bug Hunt Findings

**Mission**: Find and fix defects causing Haiku test failure  
**Date**: 2025-10-06 19:32 UTC  
**Status**: 🔍 IN PROGRESS - Partial fix applied, still investigating

---

## 🎯 Executive Summary

**FOUND BUG #1**: Hardcoded system prompt injection (FIXED ✅)
- **Root Cause**: We always inject `<|im_start|>system\nYou are a helpful assistant<|im_end|>\n`
- **Evidence**: llama.cpp with `-p` flag does NOT add system prompt (see tools/main/main.cpp:281-284)
- **Fix Applied**: Removed system prompt block to match llama.cpp behavior
- **Result**: Prompt format now matches llama.cpp

**FOUND BUG #2**: Hardcoded temperature=0.0 ignores config (FIXED ✅)
- **Root Cause**: Generation loop hardcoded `temperature=0.0` (greedy sampling)
- **Evidence**: Test sets `temperature=0.7` but we override to 0.0
- **Fix Applied**: Use `config.temperature` instead of hardcoded 0.0
- **Result**: Tokens now vary with proper sampling (ï¦§ÇĻ×ª×¦çĦ¡æ³ķ...)

**STILL FAILING**: Haiku test produces diverse but nonsensical tokens
- Output: `ï¦§ÇĻ×ª×¦çĦ¡æ³ķĠà¸ĺà¸±à¸Ļà¸§à¸²Ġtiempoà¹Ģà¸Ńà¸ĩ...` (diverse garbage)
- Expected: Coherent haiku with minute word
- **Next Investigation**: Model is generating tokens but they're not forming coherent English

---

## 🔬 Verification Evidence

### Baseline: llama.cpp WORKS ✅

```bash
/home/vince/Projects/llama-orch/reference/llama.cpp/build/bin/llama-cli \
  -m /home/vince/Projects/llama-orch/.test-models/qwen/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  -p "Write a haiku about autumn:" -n 50 --temp 0.7
```

**Output**:
```
Autumn leaves fall,
Golden hues on the ground,
Silent whispers of fall.
```

**Perfect haiku! ✅**

### Before Fix: Our Code FAILS ❌

```bash
REQUIRE_REAL_LLAMA=1 cargo test --release --features cuda --test haiku_generation_anti_cheat \
  test_haiku_generation_stub_pipeline_only -- --ignored --nocapture --test-threads=1
```

**Prompt sent**:
```
<|im_start|>system
You are a helpful assistant<|im_end|>
<|im_start|>user
Write a haiku about GPU computing that includes the word "thirty-two" (nonce: FrikYRWx)<|im_end|>
<|im_start|>assistant
```

**Output**: `ĠstretchedĠstretchedĠstretchedĠstretchedĠstretched...` (repetitive garbage)

### After Fix: Our Code STILL FAILS ❌ (but improved)

**Prompt sent NOW**:
```
<|im_start|>user
Write a haiku about GPU computing that includes the word "thirty-four" (nonce: 44sQst3f)<|im_end|>
<|im_start|>assistant
```

**Output**: `ĠstretchedĠmilitĠskeà¹Ģà¸Ńà¸ĩçĥŃçĤ¹dance...` (still garbage, but tokens vary more)

**Progress**: Tokens are no longer stuck on single token, but still not coherent

---

## 🐛 Bug #1: System Prompt Injection (FIXED)

### Location
`bin/worker-orcd/src/inference/cuda_backend.rs` lines 131-143

### CONTRADICTION Found
- **llama.cpp behavior** (tools/main/main.cpp:281-284):
  ```cpp
  if (!params.system_prompt.empty()) {
      chat_add_and_format("system", params.system_prompt);
  }
  ```
  When using `-p` flag WITHOUT `-sys`, NO system message is added.

- **Our behavior** (BEFORE fix):
  ```rust
  let formatted_prompt = format!(
      "<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
      prompt
  );
  ```
  ALWAYS adds system message, even when not requested.

### Fix Applied
```rust
// FIXED: [TEAM_FINNEY] Remove hardcoded system prompt! (2025-10-06 19:32 UTC)
let formatted_prompt = format!(
    "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
    prompt
);
```

### Verification
- ✅ Prompt format now matches llama.cpp for `-p` flag usage
- ✅ Tokens now vary (not stuck on single token)
- ❌ Still produces garbage output (investigation continues)

---

## 🐛 Bug #2: Hardcoded Temperature (FIXED)

### Location
`bin/worker-orcd/src/inference/cuda_backend.rs` lines 290-301

### CONTRADICTION Found
- **Test configuration** (haiku_generation_anti_cheat.rs:125):
  ```rust
  req.temperature = 0.7;
  ```
  Test explicitly sets temperature=0.7 for diverse sampling.

- **Generation loop** (BEFORE fix):
  ```rust
  let next_token_id = inference.generate_token(
      current_token,
      0.0, // Greedy for debugging ← HARDCODED!
      0,   // Disable top-k
      1.0, // Disable top-p filtering
      config.seed.wrapping_add(token_idx as u64),
  )?;
  ```
  Hardcoded temperature=0.0 ignores config → always picks highest probability token.

### Fix Applied
```rust
// FIXED: [TEAM_FINNEY] Use config.temperature instead of hardcoded 0.0
let next_token_id = inference.generate_token(
    current_token,
    config.temperature, // Use configured temperature, not hardcoded 0.0!
    config.top_k, // 0 = disabled
    config.top_p, // 1.0 = disabled
    config.seed.wrapping_add(token_idx as u64),
)?;
```

### Verification
- ✅ Tokens now vary with each generation
- ✅ Sampling respects configured temperature
- ❌ Still produces nonsensical output (but diverse)

**Before fix**: `ĠstretchedĠstretchedĠstretchedĠstretched...` (repetitive)  
**After fix**: `ï¦§ÇĻ×ª×¦çĦ¡æ³ķĠà¸ĺà¸±à¸Ļà¸§à¸²Ġtiempo...` (diverse but garbage)

---

## 🔍 Remaining Investigation

### Hypothesis #1: Tokenization Mismatch
**SUSPECT**: Our tokenizer may encode the prompt differently than llama.cpp

**Evidence Needed**:
- Compare token IDs from our tokenizer vs llama.cpp
- Check if BOS token is added correctly
- Verify special tokens (<|im_start|>, <|im_end|>, etc.) are encoded correctly

**Next Step**: Add debug output to see token IDs (already added in cuda_backend.rs:155)

### Hypothesis #2: Model File Mismatch
**FALSE_LEAD** (per Team Charlie): Model file is NOT corrupted
- llama.cpp works perfectly with same model file
- GGUF weights are correct
- RMSNorm values are normal

### Hypothesis #3: CUDA Kernel Bug
**SUSPECT** (per previous teams): Attention mechanism may still have issues
- cuBLAS operations verified correct
- KV cache infrastructure verified correct
- But output quality still poor

**CONTRADICTION**: llama.cpp uses same CUDA kernels and works fine
- This suggests the bug is NOT in CUDA kernels
- More likely in prompt handling or tokenization

---

## 📝 Code Markers Added

### `/bin/worker-orcd/src/inference/cuda_backend.rs`

Lines 132-143:
```rust
// FIXED: [TEAM_FINNEY] Remove hardcoded system prompt! (2025-10-06 19:32 UTC)
//   ROOT CAUSE: We were ALWAYS injecting system prompt, llama.cpp does NOT when using -p flag
//   VERIFICATION: llama.cpp with -p "Write a haiku..." generates perfect haiku
//   VERIFICATION: Our code with system prompt generates garbage (ĠstretchedĠstretched...)
//   FIX: Remove system block to match llama.cpp behavior (tools/main/main.cpp:281-284)
//   Command that works: llama-cli -m model.gguf -p "Write a haiku about autumn:" -n 50 --temp 0.7
//   Rendered prompt (llama.cpp): <|im_start|>user\nWrite a haiku...<|im_end|>\n<|im_start|>assistant\n
//   Rendered prompt (ours NOW): <|im_start|>user\nWrite a haiku...<|im_end|>\n<|im_start|>assistant\n
```

Lines 146, 155:
```rust
eprintln!("[TEAM_FINNEY] Formatted prompt: {:?}", formatted_prompt);
// ... tokenization ...
eprintln!("[TEAM_FINNEY] Token IDs: {:?}", token_ids);
```

---

## ✅ Handoff Checklist

- [x] Baseline failure captured (garbage tokens)
- [x] llama.cpp verification (works perfectly)
- [x] Root cause identified for Bug #1 (system prompt injection)
- [x] Fix applied and tested
- [x] Code markers added with FIXED/SUSPECT annotations
- [x] Verification shows partial improvement (tokens vary now)
- [ ] Full fix pending (still produces garbage)

---

## 🚦 Next Steps for Continuation

1. **Run test with debug output** to see token IDs:
   ```bash
   REQUIRE_REAL_LLAMA=1 cargo test --release --features cuda --test haiku_generation_anti_cheat \
     test_haiku_generation_stub_pipeline_only -- --ignored --nocapture --test-threads=1
   ```

2. **Compare token IDs** with llama.cpp:
   - Use llama.cpp with `--log-disable` to see token IDs
   - Compare our token IDs vs llama.cpp token IDs
   - Look for mismatches in special tokens or BOS token

3. **If tokenization matches**, investigate:
   - Sampling parameters (temperature, top-k, top-p)
   - Logit processing
   - Stop token handling

4. **If tokenization differs**, fix:
   - BOS token addition
   - Special token encoding
   - Chat template markers

---

## 📚 Key Files

### Modified
- `bin/worker-orcd/src/inference/cuda_backend.rs` (lines 132-155)

### Referenced
- `reference/llama.cpp/tools/main/main.cpp` (lines 280-316)
- `bin/worker-orcd/investigation-teams/TEAM_PROMPT_INVESTIGATION.md`
- `bin/worker-orcd/investigation-teams/TEAM_CHARLIE_I_WAS_WRONG.md`

---

**Investigation by Team FINNEY**  
**Status**: Partial fix applied, investigation continues  
**Win Condition**: Haiku test passes with coherent output
