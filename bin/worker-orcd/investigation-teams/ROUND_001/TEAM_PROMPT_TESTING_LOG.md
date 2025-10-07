# Team PROMPT — Testing Log

**Date**: 2025-10-06 19:29 UTC  
**Tester**: Team PROMPT  
**Goal**: Verify llama.cpp behavior with and without conversation mode

---

## Test 1: Default Conversation Mode

**Command**:
```bash
./llama-cli -m qwen2.5-0.5b-instruct-fp16.gguf \
  -p "Write a haiku about autumn:" -n 50 --temp 0.7
```

**Result**:
- ✅ Chat template applied automatically
- ✅ Generates proper haiku
- Token count: ~27 tokens (with template)
- Output quality: Excellent

---

## Test 2: No Conversation Mode (`-no-cnv`)

**Command**:
```bash
./llama-cli -m qwen2.5-0.5b-instruct-fp16.gguf \
  -p "Write a haiku about autumn:" -n 50 --temp 0.7 -no-cnv
```

**Result**:
- ✅ Chat template **NOT** applied
- ✅ Prompt used as raw text
- Token count: 7 tokens (no template)
- Output: Still coherent! "Autumn leaves falling, Nature's golden reward..."

**Key Finding**: Model can generate reasonable output even without chat template formatting!

---

## Test 3: System Prompt with `-no-cnv`

**Command**:
```bash
./llama-cli -m qwen2.5-0.5b-instruct-fp16.gguf \
  --system-prompt "You are a helpful assistant" \
  -p "Write a haiku about autumn:" -n 50 --temp 0.7 -no-cnv
```

**Result**:
- ✅ `--system-prompt` flag **IGNORED** in `-no-cnv` mode
- Token count: Still 7 tokens (same as Test 2)
- Behavior: Identical to Test 2

**Key Finding**: `-no-cnv` completely disables conversation mode features, including system prompts.

---

## Implications for Our Pipeline

### What We Learned

1. **llama.cpp has two distinct modes**:
   - **Conversation mode** (default): Applies chat template from GGUF
   - **Raw text mode** (`-no-cnv`): No template, direct tokenization

2. **Our pipeline only implements conversation mode**:
   - Always applies hardcoded CHATML template
   - Always includes system prompt
   - No way to disable template application

3. **The model is flexible**:
   - Works with chat template (instruct mode)
   - Works without chat template (completion mode)
   - Quality depends on how it was trained

### What This Means

**For Qwen2.5-0.5B-Instruct**:
- Trained primarily with chat template
- Best results with template applied
- Can still generate without template (but not optimal)

**For Our Implementation**:
- Hardcoded template is correct for Qwen instruct models
- But we lack flexibility for:
  - Raw completion mode
  - Optional system prompts
  - Other model families (Llama, Phi, Mistral)

---

## Recommendations

### Short Term (v0.1.0)
Keep hardcoded template, but add:
- Optional system prompt parameter
- Conditional system block inclusion

### Long Term (v0.2.0+)
Add full conversation mode support:
- Read `tokenizer.chat_template` from GGUF
- Support raw text mode (no template)
- Multi-model template support

---

## Test Evidence

All tests run on:
- **Model**: qwen2.5-0.5b-instruct-fp16.gguf
- **Hardware**: RTX 3090 + RTX 3060
- **llama.cpp**: Latest build from reference/llama.cpp
- **Date**: 2025-10-06 19:29 UTC

Test outputs saved in terminal history.

---

**Conclusion**: llama.cpp's flexibility comes from supporting both conversation and raw modes. We currently only support conversation mode with a hardcoded template.
