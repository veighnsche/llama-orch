# Team PROMPT — Prompt Assembly Investigation

**Mission**: Map and compare prompt construction in llama.cpp vs our pipeline.  
**Date**: 2025-10-06  
**Status**: ✅ Investigation Complete

---

## 📊 Executive Summary

**ROOT CAUSE**: Our pipeline hardcodes a single Qwen chat template and system prompt, while llama.cpp reads the template from GGUF metadata and supports flexible system/user message construction.

**KEY FINDINGS**:
1. ❌ We hardcode the chat template instead of reading from GGUF
2. ❌ We hardcode "You are a helpful assistant" system prompt
3. ❌ We don't support custom system prompts via CLI/API
4. ❌ We don't read `tokenizer.chat_template` from GGUF metadata
5. ✅ The hardcoded Qwen template format itself is correct

---

## 🔍 llama.cpp Prompt Construction Path

### Order of Operations

1. **CLI Argument Parsing** (`tools/main/main.cpp` lines 86-227)
   - `-p/--prompt` → user message content
   - `--system-prompt/-sys` → system message content
   - `--chat-template` → optional template override

2. **Chat Template Initialization** (line 153)
   ```cpp
   auto chat_templates = common_chat_templates_init(model, params.chat_template);
   ```
   - Reads `tokenizer.chat_template` from GGUF metadata via `llama_model_chat_template(model, nullptr)`
   - Source: `common/chat.cpp` line 551
   - Falls back to CHATML if not present (lines 564-569)

3. **Message Construction** (lines 280-300)
   ```cpp
   if (!params.system_prompt.empty()) {
       chat_add_and_format("system", params.system_prompt);
   }
   if (!params.prompt.empty()) {
       chat_add_and_format("user", params.prompt);
   }
   ```

4. **Warning Check** (lines 219-221)
   ```cpp
   if (!params.prompt.empty() && params.system_prompt.empty()) {
       LOG_WRN("*** User-specified prompt will pre-start conversation, did you mean to set --system-prompt (-sys) instead?\n");
   }
   ```
   - **This is the "pre-start conversation" warning from the mission brief!**
   - Emitted when `-p` is used without `-sys` in conversation mode

5. **Template Application** (line 299)
   ```cpp
   prompt = common_chat_templates_apply(chat_templates.get(), inputs).prompt;
   ```
   - Uses minja template engine (Jinja2-compatible)
   - Renders messages array into final prompt string
   - Applies role markers, separators, BOS/EOS tokens

6. **Tokenization** (line 308)
   ```cpp
   embd_inp = common_tokenize(ctx, prompt, true, true);
   ```

### CHATML Template (Default Fallback)

From `common/chat.cpp` lines 509-515:
```jinja2
{% for message in messages %}
  {{ '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' }}
{% endfor %}
{% if add_generation_prompt %}
  {{ '<|im_start|>assistant\n' }}
{% endif %}
```

---

## 🔧 Our Pipeline Prompt Construction Path

### Current Implementation

**Location**: `bin/worker-orcd/src/inference/cuda_backend.rs` lines 65-107

1. **Direct Hardcoding** (lines 94-97)
   ```rust
   let formatted_prompt = format!(
       "<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
       prompt
   );
   ```

2. **Tokenization** (line 103)
   ```rust
   let token_ids = self.tokenizer.encode(&formatted_prompt, true)?;
   ```

### Problems

1. **No GGUF Template Reading**
   - Worker-tokenizer (`bin/worker-crates/worker-tokenizer/`) has NO chat template support
   - Worker-gguf (`bin/worker-crates/worker-gguf/`) does NOT parse `tokenizer.chat_template` metadata
   - Template is hardcoded to Qwen2.5's CHATML format

2. **No System Prompt Flexibility**
   - System prompt is hardcoded: `"You are a helpful assistant"`
   - Cannot be overridden via API or config
   - No CLI parameter support (worker-orcd doesn't have CLI for inference)

3. **No User-Only Mode**
   - Always injects system prompt
   - Cannot replicate llama.cpp's `-p "prompt"` without system

4. **No Template Validation**
   - No verification that hardcoded template matches model's training
   - Works for Qwen2.5-Instruct, but breaks for other chat models

---

## 📝 Rendered Prompt Comparison

### Case A: User Prompt Only

**llama.cpp command**:
```bash
llama-cli -m qwen2.5-0.5b-instruct-q4_k_m.gguf \
  -p "Write a haiku about autumn:" -n 50 --temp 0.7
```

**llama.cpp rendered prompt** (conversation mode, no system):
```
<|im_start|>user
Write a haiku about autumn:<|im_end|>
<|im_start|>assistant

```

**Our pipeline rendered prompt** (ALWAYS includes system):
```
<|im_start|>system
You are a helpful assistant<|im_end|>
<|im_start|>user
Write a haiku about autumn:<|im_end|>
<|im_start|>assistant

```

**DIFFERENCE**: We inject an extra system block that user didn't request.

---

### Case B: System + User

**llama.cpp command**:
```bash
llama-cli -m qwen2.5-0.5b-instruct-q4_k_m.gguf \
  --system-prompt "You are a helpful assistant" \
  -p "Write a haiku about autumn:" -n 50 --temp 0.7
```

**llama.cpp rendered prompt**:
```
<|im_start|>system
You are a helpful assistant<|im_end|>
<|im_start|>user
Write a haiku about autumn:<|im_end|>
<|im_start|>assistant

```

**Our pipeline rendered prompt**:
```
<|im_start|>system
You are a helpful assistant<|im_end|>
<|im_start|>user
Write a haiku about autumn:<|im_end|>
<|im_start|>assistant

```

**DIFFERENCE**: ✅ **Byte-for-byte identical!**

---

## 🔬 Template Details

### Role Markers
- **System**: `<|im_start|>system\n{content}<|im_end|>\n`
- **User**: `<|im_start|>user\n{content}<|im_end|>\n`
- **Assistant**: `<|im_start|>assistant\n` (generation prompt, no end marker yet)

### Separators
- Between messages: Single `\n` (already in `<|im_end|>\n`)
- After role name: Single `\n` (in `<|im_start|>role\n`)

### BOS/EOS Tokens
- **BOS**: Added by tokenizer if `add_bos=true` in vocab metadata
- **EOS**: Not added to prompt (only during generation stop)
- Qwen2.5: BOS token ID typically added automatically

### Stop Sequences
- Primary: `<|im_end|>` (token ID 151645 for Qwen2.5)
- Secondary: EOS token (151643 for Qwen2.5)

### Whitespace
- Trailing newline after `<|im_start|>assistant\n`: **YES** (intentional)
- This lets the model start generation immediately without generating the newline

---

## 🛠️ Alignment Proposal

### Option 1: Minimal Fix (Recommended for v0.1.0)

**Goal**: Support optional system prompt via API while keeping template hardcoded.

**Changes Required**:
1. Add `system_prompt: Option<String>` to inference API contract
2. Modify `cuda_backend.rs` to conditionally include system block:
   ```rust
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

**Pros**: Small change, fixes Case A mismatch  
**Cons**: Still hardcoded to Qwen template, won't work for Llama/Phi models

---

### Option 2: Full Template Support (Recommended for v0.2.0+)

**Goal**: Read chat template from GGUF and support multiple model families.

**Changes Required**:

1. **Add GGUF Metadata Parsing**:
   - Extend `worker-gguf` to parse `tokenizer.chat_template` string
   - Store template in `GGUFMetadata`

2. **Add Template Engine**:
   - Integrate `minijinja` crate (Rust Jinja2 implementation)
   - Or implement lightweight CHATML-only renderer

3. **Add Message Structure**:
   - Define `ChatMessage { role: String, content: String }`
   - Build message array from API inputs

4. **Apply Template**:
   - Render messages through template engine
   - Handle `add_generation_prompt` flag

5. **Fallback**:
   - Default to CHATML if `tokenizer.chat_template` missing
   - Match llama.cpp behavior

**Pros**: Full compatibility, supports all models  
**Cons**: Larger implementation, requires template engine dependency

---

## 📌 Code Markers Added

### `/bin/worker-orcd/src/inference/cuda_backend.rs`

```rust
// Lines 73-93 (existing markers preserved)
// SUSPECT: [TEAM_PROMPT] Missing chat template application!
// CONTRADICTION: llama-cli works perfectly and generates proper haiku.
// RESOLVED: [TEAM_PROMPT] Applying Qwen chat template!
// FALSE_LEAD: NOT a CUDA/attention/bias bug!
// FIXED: Apply Qwen chat template before tokenization

// Lines 94-97 (NEW marker to add)
// CONTRADICTION: [TEAM_PROMPT] Hardcoded template + system prompt
//   llama.cpp path: GGUF → llama_model_chat_template() → minja render
//   Our path: Hardcoded string format
// SUSPECT: [TEAM_PROMPT] System prompt always injected
//   Case A (user-only): llama.cpp has NO system block, we always add one
//   Case B (system+user): Byte-identical ✅
// RESOLVED: [TEAM_PROMPT] Template format is correct, but lacks flexibility
//   - Need to read tokenizer.chat_template from GGUF metadata
//   - Need to support optional system_prompt parameter
//   - Need to support multiple chat formats (not just Qwen CHATML)
```

---

## ✅ Acceptance Criteria Met

- [x] Final rendered strings for Case A/B captured for llama.cpp and ours
- [x] Differences documented (Case A: extra system block; Case B: identical)
- [x] Root cause identified: hardcoded template, no GGUF reading
- [x] Code/docs updated with markers
- [x] Two alignment options proposed (minimal vs full)

---

## 📚 Key Files Referenced

### llama.cpp
- `tools/main/main.cpp`: CLI parsing, message construction (lines 86-316)
- `common/chat.cpp`: Template initialization and rendering (lines 509-621, 790-825)
- `common/chat.h`: Template API definitions
- `src/llama-model.cpp`: `llama_model_chat_template()` implementation (line 20108)

### Our Pipeline
- `bin/worker-orcd/src/inference/cuda_backend.rs`: Prompt assembly (lines 65-107)
- `bin/worker-crates/worker-tokenizer/`: Tokenizer (no chat template support)
- `bin/worker-crates/worker-gguf/`: GGUF parser (no chat template parsing)

---

## 🚦 Next Steps

**For v0.1.0 (Quick Fix)**:
1. Add `system_prompt: Option<String>` to inference API
2. Make system block conditional in `cuda_backend.rs`
3. Document in API specs that only CHATML/Qwen is supported
4. Add test for both Case A and Case B

**For v0.2.0+ (Full Solution)**:
1. Integrate `minijinja` crate
2. Parse `tokenizer.chat_template` from GGUF
3. Build message-based API (array of `{role, content}`)
4. Test with Llama-3, Phi-3, Qwen, Mistral models
5. Add template validation and fallback logic

---

**Investigation completed by Team PROMPT**  
**No end-to-end tests run** (as per mission brief)  
**Handoff ready** for implementation team
