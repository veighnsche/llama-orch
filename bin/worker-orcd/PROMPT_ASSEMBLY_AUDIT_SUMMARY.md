# Prompt Assembly Audit — One-Page Summary

**Team**: PROMPT  
**Date**: 2025-10-06  
**Status**: ✅ Complete  

---

## 🎯 Mission

Compare how llama.cpp and our pipeline construct the final prompt string before tokenization.

---

## 📊 Key Finding

**We hardcode both the chat template and system prompt. llama.cpp reads them from GGUF metadata.**

---

## ❌ Before (the difference)

| Aspect | llama.cpp | Our Pipeline |
|--------|-----------|--------------|
| **Template Source** | Read from GGUF `tokenizer.chat_template` | Hardcoded CHATML string |
| **System Prompt** | Optional via `--system-prompt` | Always "You are a helpful assistant" |
| **Multi-model** | Supports Llama, Qwen, Phi, Mistral | Only Qwen CHATML format |
| **API Flexibility** | Message array `[{role, content}]` | Single prompt string |

---

## 🔬 Rendered String Comparison

### Case A: User-only `-p "Write a haiku"`

**llama.cpp** (no system):
```
<|im_start|>user
Write a haiku<|im_end|>
<|im_start|>assistant

```

**Ours** (always system):
```
<|im_start|>system
You are a helpful assistant<|im_end|>
<|im_start|>user
Write a haiku<|im_end|>
<|im_start|>assistant

```

**Difference**: ❌ Extra system block

---

### Case B: System + user `--sys "..." -p "Write a haiku"`

**Both produce**:
```
<|im_start|>system
You are a helpful assistant<|im_end|>
<|im_start|>user
Write a haiku<|im_end|>
<|im_start|>assistant

```

**Difference**: ✅ Identical

---

## 🛠️ Proposed Fixes

### Option 1: Minimal (v0.1.0) — ~30 minutes

Make system prompt conditional:

```rust
let formatted_prompt = if let Some(sys) = system_prompt {
    format!("<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n", sys, prompt)
} else {
    format!("<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n", prompt)
};
```

**Pros**: Quick fix, resolves Case A mismatch  
**Cons**: Still Qwen-only

---

### Option 2: Full (v0.2.0+) — ~2-3 days

1. Add `minijinja` crate for template rendering
2. Parse `tokenizer.chat_template` from GGUF metadata (extend `worker-gguf`)
3. Build message-based API: `Vec<ChatMessage { role, content }>`
4. Render via template engine
5. Test with Llama-3, Phi-3, Mistral

**Pros**: Full multi-model support, matches llama.cpp exactly  
**Cons**: Larger implementation effort

---

## 📁 Documentation

- **Full Investigation**: `investigation-teams/TEAM_PROMPT_INVESTIGATION.md`
- **Quick Handoff**: `investigation-teams/TEAM_PROMPT_HANDOFF.md`
- **Code Markers**: `src/inference/cuda_backend.rs` lines 95-130

---

## 📚 Reference Files

### llama.cpp
- `tools/main/main.cpp` lines 280-316 (message construction)
- `common/chat.cpp` lines 539-621 (template init)
- `common/chat.cpp` lines 790-825 (template render)

### Our Pipeline
- `src/inference/cuda_backend.rs` lines 131-134 (hardcoded format)
- `bin/worker-crates/worker-tokenizer/` (no template support)
- `bin/worker-crates/worker-gguf/` (no template parsing)

---

## ✅ Acceptance Criteria Met

- [x] Rendered strings captured for Cases A & B
- [x] Differences documented (Case A: extra system; Case B: identical)
- [x] Root cause identified (hardcoded template, no GGUF reading)
- [x] Code markers added with file/line references
- [x] Two fix options proposed with effort estimates

---

**Investigation complete. No tests run per mission brief.**  
**Ready for implementation team handoff.**
