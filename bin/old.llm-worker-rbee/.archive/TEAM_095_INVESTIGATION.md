# TEAM-095: Question Mark Bug Investigation

**Status:** 🔴 NOT FIXED - Root cause unknown

## Problem

When running inference with prompts containing question marks (e.g., "Why is the sky blue?"), the model generates **ZERO tokens**.

```bash
./ASK_SKY_BLUE.sh  # Works with "Hello world"
# But with "Why is the sky blue?" -> 0 tokens generated
```

## What We Know

### ✅ Tokenization Works Correctly

Test: `cargo test -p llm-worker-rbee --test test_question_mark_tokenization -- --ignored`

Results:
- "Why?" → Token IDs: `[11008, 29973]` ✅
- "?" → Token ID: `29973` ✅
- Tokenizer has 32,000 tokens ✅
- Round-trip encoding/decoding works ✅

**Conclusion:** Tokenization is NOT the bug.

### ❌ Generation Loop Produces Zero Tokens

Log evidence (`/tmp/rbee-hive.log`):
```
Tokenized prompt (10 tokens) ✅
Reset KV cache ✅
Inference completed tokens_generated=0 ❌
```

The generation loop (lines 228-319 in `backend/inference.rs`) exits immediately without generating any tokens.

### 🔍 Possible Causes

1. **EOS detected on first iteration**
   - Model samples token on pos=0
   - EOS check returns `true` immediately
   - Loop breaks with 0 tokens

2. **Forward pass silent failure**
   - Model.forward() returns error
   - Error gets swallowed somewhere
   - Loop never executes

3. **Sampling always returns EOS**
   - LogitsProcessor broken
   - Always samples token ID 2 (EOS)
   - Immediate termination

## Debug Logging Added

File: `bin/llm-worker-rbee/src/backend/inference.rs` lines 284-316

Added logs to show:
- What token is sampled on each iteration
- EOS token IDs (tokenizer vs model)
- Why EOS check triggers

## Next Steps

1. **Run inference and check logs:**
   ```bash
   ./ASK_SKY_BLUE.sh
   grep "Sampled token" /tmp/rbee-hive.log
   grep "EOS check" /tmp/rbee-hive.log
   ```

2. **Look for:**
   - `Sampled token pos=0 next_token=XXXX`
   - `EOS check result is_eos=true` (if EOS triggers)
   - Any errors before "Inference completed"

3. **If EOS triggers:** Check why first token is EOS
4. **If no logs:** Forward pass failing silently
5. **If token=2 every time:** Sampling is broken

## Files Modified

- `src/backend/inference.rs` - Added debug logging (lines 284-316)
- `src/backend/mod.rs` - Made `gguf_tokenizer` public for testing
- `src/backend/gguf_tokenizer.rs` - Removed brittle unit tests
- `src/bin/cpu.rs` - Fixed `callback_ready()` signature
- `tests/test_question_mark_tokenization.rs` - Tokenization diagnostics

## What Doesn't Work (False Leads)

- ❌ TokenOutputStream `is_alphanumeric()` check - irrelevant if no tokens generated
- ❌ Question mark tokenization - works perfectly
- ❌ Round-trip encoding - works perfectly

The bug is in the **generation loop**, not tokenization.

---

**TEAM-095 | 2025-10-18 | Status: INVESTIGATING**
