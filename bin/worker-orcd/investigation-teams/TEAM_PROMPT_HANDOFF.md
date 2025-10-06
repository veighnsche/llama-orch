# Team PROMPT — Quick Handoff

**Date**: 2025-10-06  
**Status**: ✅ Audit Complete  
**Full Report**: `TEAM_PROMPT_INVESTIGATION.md`

---

## ❌ Before (llama.cpp vs ours)

### Case A: User-only prompt `-p "Write a haiku about autumn:"`

**llama.cpp**:
```
<|im_start|>user
Write a haiku about autumn:<|im_end|>
<|im_start|>assistant

```

**Ours**:
```
<|im_start|>system
You are a helpful assistant<|im_end|>
<|im_start|>user
Write a haiku about autumn:<|im_end|>
<|im_start|>assistant

```

**Difference**: We inject unwanted system block ❌

---

### Case B: System + user `--sys "You are a helpful assistant" -p "Write a haiku..."`

**llama.cpp**:
```
<|im_start|>system
You are a helpful assistant<|im_end|>
<|im_start|>user
Write a haiku about autumn:<|im_end|>
<|im_start|>assistant

```

**Ours**:
```
<|im_start|>system
You are a helpful assistant<|im_end|>
<|im_start|>user
Write a haiku about autumn:<|im_end|>
<|im_start|>assistant

```

**Difference**: Byte-for-byte identical ✅

---

## 🛠️ Change Made

**File**: `bin/worker-orcd/src/inference/cuda_backend.rs` lines 95-130

**What**: Added comprehensive code markers documenting:
- How llama.cpp reads template from GGUF (`tokenizer.chat_template` metadata key)
- How llama.cpp constructs messages (separate system/user roles)
- How we hardcode both template and system prompt
- The "pre-start conversation" warning meaning
- Two fix paths: minimal (v0.1.0) vs full (v0.2.0)

**Markers Used**:
- `CONTRADICTION`: llama.cpp vs our approach
- `SUSPECT`: System prompt always injected
- `RESOLVED`: Template format correct, needs flexibility

---

## ✅ After

**Code Documentation**: ✅ Complete with file/line references  
**Investigation Report**: ✅ `TEAM_PROMPT_INVESTIGATION.md`  
**Rendered Strings**: ✅ Captured for Cases A & B  
**Root Cause**: ✅ Identified (hardcoded template, no GGUF reading)  
**Fix Proposals**: ✅ Two options documented  

---

## 💡 Key Insight (2025-10-06 19:29 UTC)

**llama.cpp has TWO modes**:
1. **Conversation mode** (default for instruct models): Applies chat template
2. **Raw text mode** (`-no-cnv`): No template, prompt as-is

**We only implement conversation mode** (always apply template).

This explains why:
- Instruct models work (we apply template correctly)
- But we can't support raw completion mode
- And we can't make system prompt optional

---

## 🚦 Recommendations

### For v0.1.0 (Minimal Fix)
```rust
// Add to InferenceBackend::execute() signature
system_prompt: Option<&str>

// Replace hardcoded format with:
let formatted_prompt = if let Some(sys) = system_prompt {
    format!(
        "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
        sys, prompt
    )
} else {
    format!(
        "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
        prompt
    )
};
```

**Effort**: ~30 minutes  
**Impact**: Fixes Case A mismatch  
**Limitation**: Still Qwen-only

### For v0.2.0+ (Full Solution)
1. Add `minijinja` crate to `worker-gguf`
2. Parse `tokenizer.chat_template` from GGUF metadata
3. Add `ChatMessage { role, content }` struct
4. Render via template engine with message array
5. Test with Llama-3, Phi-3, Mistral

**Effort**: ~2-3 days  
**Impact**: Full multi-model support  
**Benefit**: Matches llama.cpp exactly

---

## 📚 Key Files

- **Investigation**: `investigation-teams/TEAM_PROMPT_INVESTIGATION.md`
- **Our Code**: `src/inference/cuda_backend.rs` lines 73-134
- **llama.cpp CLI**: `reference/llama.cpp/tools/main/main.cpp` lines 280-316
- **llama.cpp Templates**: `reference/llama.cpp/common/chat.cpp` lines 539-621

---

**Next team can now implement either fix with full context.**
