# Narration Migration Examples

Real examples from `llm-worker-rbee` showing before/after for each migration pattern.

---

## Pattern 1: Simple Narration (No Context)

### Before:
```rust
// src/device.rs:20
narrate(NarrationFields {
    actor: ACTOR_DEVICE_MANAGER,
    action: ACTION_DEVICE_INIT,
    target: "cpu".to_string(),
    human: "CPU device initialized".to_string(),
    ..Default::default()
});
```

### After:
```rust
n!(ACTION_DEVICE_INIT)
    .target("cpu")
    .human("CPU device initialized")
    .emit();
```

---

## Pattern 2: Narration with Dynamic Target

### Before:
```rust
// src/backend/models/quantized_llama.rs:34
narrate(NarrationFields {
    actor: "model-loader",
    action: "gguf_load_start",
    target: path.display().to_string(),
    human: format!("Loading GGUF model from {}", path.display()),
    ..Default::default()
});
```

### After:
```rust
n!("gguf_load_start")
    .target(path.display().to_string())
    .human(format!("Loading GGUF model from {}", path.display()))
    .emit();
```

---

## Pattern 3: Narration with Context

### Before:
```rust
// src/backend/inference.rs:340
narrate(NarrationFields {
    actor: ACTOR_CANDLE_BACKEND,
    action: ACTION_TOKEN_GENERATE,
    target: format!("token-{}", pos + 1),
    human: format!("Generated {} tokens", pos + 1),
    context: Some(json!({
        "tokens_generated": pos + 1,
        "max_tokens": max_tokens
    })),
    ..Default::default()
});
```

### After:
```rust
n!(ACTION_TOKEN_GENERATE)
    .target(format!("token-{}", pos + 1))
    .human(format!("Generated {} tokens", pos + 1))
    .context("tokens_generated", pos + 1)
    .context("max_tokens", max_tokens)
    .emit();
```

---

## Pattern 4: Error Narration

### Before:
```rust
// src/backend/models/quantized_llama.rs:46
narrate(NarrationFields {
    actor: "model-loader",
    action: "gguf_open_failed",
    target: path.display().to_string(),
    human: format!("Failed to open GGUF file: {}", path.display()),
    error_kind: Some("io_error".to_string()),
    ..Default::default()
});
```

### After:
```rust
n!("gguf_open_failed")
    .target(path.display().to_string())
    .human(format!("Failed to open GGUF file: {}", path.display()))
    .error_kind("io_error")
    .emit();
```

---

## Pattern 5: Narration Inside Error Context

### Before:
```rust
// src/backend/models/quantized_llama.rs:72
let content = candle_core::quantized::gguf_file::Content::read(&mut file)
    .with_context(|| {
        narrate(NarrationFields {
            actor: "model-loader",
            action: "gguf_parse_failed",
            target: path.display().to_string(),
            human: "Failed to parse GGUF file".to_string(),
            ..Default::default()
        });
        "Failed to parse GGUF content"
    })?;
```

### After:
```rust
let content = candle_core::quantized::gguf_file::Content::read(&mut file)
    .with_context(|| {
        n!("gguf_parse_failed")
            .target(path.display().to_string())
            .human("Failed to parse GGUF file")
            .emit();
        "Failed to parse GGUF content"
    })?;
```

---

## Pattern 6: Startup Narration

### Before:
```rust
// src/main.rs:116
narrate(NarrationFields {
    actor: ACTOR_LLM_WORKER_RBEE,
    action: ACTION_STARTUP,
    target: args.worker_id.clone(),
    human: format!(
        "Candle worker starting (worker_id={}, model={}, port={})",
        args.worker_id, args.model, args.port
    ),
    context: Some(json!({
        "worker_id": args.worker_id,
        "model": args.model,
        "port": args.port,
        "device": device_type
    })),
    ..Default::default()
});
```

### After:
```rust
n!(ACTION_STARTUP)
    .target(&args.worker_id)
    .human(format!(
        "Candle worker starting (worker_id={}, model={}, port={})",
        args.worker_id, args.model, args.port
    ))
    .context("worker_id", &args.worker_id)
    .context("model", &args.model)
    .context("port", args.port)
    .context("device", device_type)
    .emit();
```

---

## Pattern 7: Inference Complete with Metrics

### Before:
```rust
// src/backend/inference.rs:380
narrate(NarrationFields {
    actor: ACTOR_CANDLE_BACKEND,
    action: ACTION_INFERENCE_COMPLETE,
    target: format!("{}-tokens", generated_tokens.len()),
    human: format!(
        "Inference complete: {} tokens in {}ms (answer: {})",
        generated_tokens.len(),
        elapsed_ms,
        answer
    ),
    context: Some(json!({
        "tokens_generated": generated_tokens.len(),
        "elapsed_ms": elapsed_ms,
        "tokens_per_second": tokens_per_sec,
        "answer": answer
    })),
    ..Default::default()
});
```

### After:
```rust
n!(ACTION_INFERENCE_COMPLETE)
    .target(format!("{}-tokens", generated_tokens.len()))
    .human(format!(
        "Inference complete: {} tokens in {}ms (answer: {})",
        generated_tokens.len(),
        elapsed_ms,
        answer
    ))
    .context("tokens_generated", generated_tokens.len())
    .context("elapsed_ms", elapsed_ms)
    .context("tokens_per_second", tokens_per_sec)
    .context("answer", &answer)
    .emit();
```

---

## Special Case: narrate_dual() Wrapper

### Current Implementation:
```rust
// src/narration.rs:117
pub fn narrate_dual(fields: NarrationFields) {
    // 1. ALWAYS emit to tracing (for operators/developers)
    observability_narration_core::narrate(fields.clone());
    
    // 2. IF in HTTP request context, ALSO emit to SSE (for users)
    let sse_event = InferenceEvent::Narration { /* ... */ };
    // ... SSE emission logic
}
```

### Migration Strategy:

**Option A:** Keep wrapper, update internal call:
```rust
pub fn narrate_dual(fields: NarrationFields) {
    // 1. Emit to tracing using new API
    observability_narration_core::narrate(fields.clone(), NarrationLevel::Info);
    
    // 2. SSE emission (unchanged)
    // ...
}
```

**Option B:** Deprecate wrapper, use n!() directly:
```rust
// Callers would use:
n!("action")
    .target("target")
    .human("message")
    .emit();  // This handles both tracing + SSE automatically
```

**Recommendation:** Option A (keep wrapper for now, update later)

---

## Quick Reference

| Old API | New API |
|---------|---------|
| `narrate(NarrationFields { ... })` | `n!("action").emit()` |
| `actor: "model-loader"` | Auto-detected from crate |
| `action: "load"` | `n!("load")` |
| `target: "x".to_string()` | `.target("x")` or `.target(x)` |
| `human: "msg".to_string()` | `.human("msg")` |
| `context: Some(json!({...}))` | `.context("key", value)` (multiple) |
| `error_kind: Some("err")` | `.error_kind("err")` |
| `..Default::default()` | (not needed) |

---

## Verification

After each file migration:
```bash
cargo check -p llm-worker-rbee
```

After all migrations:
```bash
# Re-enable in Cargo.toml, then:
cargo build --workspace
```
