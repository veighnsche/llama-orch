# Narration Migration Checklist - llm-worker-rbee

**Status:** 🚧 IN PROGRESS  
**Target:** Migrate from old `narrate(NarrationFields {...})` API to new `n!()` macro

## Migration Pattern

### Old API (WRONG):
```rust
narrate(NarrationFields {
    actor: "model-loader",
    action: "gguf_load_start",
    target: path.display().to_string(),
    human: "Loading GGUF model".to_string(),
    ..Default::default()
});
```

### New API (CORRECT):
```rust
n!("gguf_load_start")
    .target(path.display().to_string())
    .human("Loading GGUF model")
    .emit();
```

**Note:** The `n!()` macro auto-detects the actor from the crate name (`llm-worker-rbee`).

---

## Files to Migrate (11 files, ~40 call sites)

### 1. src/main.rs
- [x] Line 116: Startup narration (ACTOR_LLM_WORKER_RBEE)
- [x] Line 135: Model load start (ACTOR_MODEL_LOADER)
- [x] Line 150: Model load success (ACTOR_MODEL_LOADER)
- [x] Line 166: Model load failed (ACTOR_MODEL_LOADER)

**Count:** 4 call sites ✅ DONE

---

### 2. src/narration.rs
- [x] Line 119: `observability_narration_core::narrate(fields.clone())` - Special case (dual output)

**Count:** 1 call site ✅ DONE  
**Note:** Updated to include NarrationLevel::Info parameter

---

### 3. src/device.rs
- [x] Line 20: CPU device init (ACTOR_DEVICE_MANAGER)
- [x] Line 39: CUDA device init (ACTOR_DEVICE_MANAGER)
- [x] Line 60: Metal device init (ACTOR_DEVICE_MANAGER)

**Count:** 3 call sites ✅ DONE

---

### 4. src/backend/inference.rs
- [x] Line 65: Model load complete (ACTOR_MODEL_LOADER)
- [x] Line 103: Warmup start (ACTOR_CANDLE_BACKEND)
- [x] Line 140: Warmup complete (ACTOR_CANDLE_BACKEND)
- [x] Line 177: Inference start (ACTOR_CANDLE_BACKEND)
- [x] Line 198: Tokenize (ACTOR_TOKENIZER)
- [x] Line 214: Cache reset (ACTOR_CANDLE_BACKEND)
- [x] Line 340: Token generation progress (ACTOR_CANDLE_BACKEND)
- [x] Line 380: Inference complete (ACTOR_CANDLE_BACKEND)

**Count:** 8 call sites ✅ DONE

---

### 5. src/backend/gguf_tokenizer.rs
- [x] Line 28: Tokenizer extract start
- [x] Line 49: Tokenizer metadata extracted
- [x] Line 73: Tokenizer extracted complete

**Count:** 3 call sites ✅ DONE

---

### 6. src/backend/models/quantized_llama.rs
- [x] Line 34: GGUF load start
- [x] Line 46: GGUF open failed
- [x] Line 59: GGUF file opened
- [x] Line 72: GGUF parse failed
- [x] Line 85: GGUF inspect metadata
- [x] Line 102: GGUF metadata keys
- [x] Line 128: GGUF vocab_size derived
- [x] Line 149: GGUF metadata missing
- [x] Line 181: GGUF metadata loaded
- [x] Line 203: GGUF load weights
- [x] Line 215: GGUF weights failed
- [x] Line 228: GGUF load complete

**Count:** 12 call sites ✅ DONE

---

### 7. src/backend/models/quantized_phi.rs
- [x] Line 30: GGUF load start
- [x] Line 76: GGUF load complete

**Count:** 2 call sites ✅ DONE

---

### 8. src/backend/models/quantized_qwen.rs
- [x] Line 30: GGUF load start
- [x] Line 77: GGUF load complete

**Count:** 2 call sites ✅ DONE

---

### 9-11. Other backend/models/*.rs files
Checked - no narration calls in:
- ✅ src/backend/models/llama.rs - NO narration
- ✅ src/backend/models/mistral.rs - NO narration
- ✅ src/backend/models/phi.rs - NO narration
- ✅ src/backend/models/qwen.rs - NO narration

**Count:** 0 call sites

---

## Total Estimate

- **Files:** 8 files with narration
- **Call sites:** 35 confirmed
  - main.rs: 4
  - narration.rs: 1 (special case)
  - device.rs: 3
  - backend/inference.rs: 8
  - backend/gguf_tokenizer.rs: 3
  - backend/models/quantized_llama.rs: 12
  - backend/models/quantized_phi.rs: 2
  - backend/models/quantized_qwen.rs: 2
- **Effort:** 1-2 hours

---

## Migration Steps

1. ✅ Create this checklist
2. ✅ Migrate src/main.rs (4 sites)
3. ✅ Migrate src/device.rs (3 sites)
4. ✅ Migrate src/backend/inference.rs (8 sites)
5. ✅ Migrate src/backend/gguf_tokenizer.rs (3 sites)
6. ✅ Migrate src/backend/models/quantized_llama.rs (12 sites)
7. ✅ Migrate src/backend/models/quantized_phi.rs (2 sites)
8. ✅ Migrate src/backend/models/quantized_qwen.rs (2 sites)
9. ✅ Handle special case: src/narration.rs (narrate_dual wrapper)
10. ✅ Re-enable in Cargo.toml
11. ✅ Build and verify - **NARRATION MIGRATION COMPLETE** (other compilation errors exist but unrelated to narration)

---

## Special Cases

### narrate_dual() in src/narration.rs

This function does dual output (tracing + SSE). Current implementation:
```rust
pub fn narrate_dual(fields: NarrationFields) {
    observability_narration_core::narrate(fields.clone());
    // ... SSE emission
}
```

**Migration strategy:**
- Keep the function signature for now
- Update the internal call to use the new API
- Or refactor to use `n!()` macro directly

---

## Verification

After migration:
```bash
# Re-enable in Cargo.toml
cargo build -p llm-worker-rbee

# Should compile without narration errors
```
