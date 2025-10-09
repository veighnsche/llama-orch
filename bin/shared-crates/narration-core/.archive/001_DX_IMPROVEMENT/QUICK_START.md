# Narration Core - Quick Start Guide 🎀

**Version**: 0.0.0 (Week 1-2 Complete)  
**Status**: Foundation ready, optimization pending

---

## 🚀 Getting Started

### Add to Your Crate

```toml
[dependencies]
observability-narration-core = { path = "../narration-core" }

# Optional: For proc macros (Week 3+)
observability-narration-macros = { path = "../narration-macros" }
```

### Basic Usage

```rust
use observability_narration_core::{narrate, NarrationFields};

fn main() {
    narrate(NarrationFields {
        actor: "my-service",
        action: "startup",
        target: "main".to_string(),
        human: "Service started successfully".to_string(),
        ..Default::default()
    });
}
```

---

## 📊 Logging Levels

### INFO (Default)
```rust
use observability_narration_core::{narrate, NarrationFields};

narrate(NarrationFields {
    actor: "rbees-orcd",
    action: "dispatch",
    target: "job-123".to_string(),
    human: "Dispatched job to worker-gpu0-r1".to_string(),
    ..Default::default()
});
```

### WARN
```rust
use observability_narration_core::{narrate_warn, NarrationFields};

narrate_warn(NarrationFields {
    actor: "pool-managerd",
    action: "capacity_check",
    target: "GPU0".to_string(),
    human: "GPU0 capacity low: 512MB available".to_string(),
    ..Default::default()
});
```

### ERROR
```rust
use observability_narration_core::{narrate_error, NarrationFields};

narrate_error(NarrationFields {
    actor: "worker-orcd",
    action: "inference",
    target: "job-456".to_string(),
    human: "Inference failed: CUDA out of memory".to_string(),
    error_kind: Some("CudaOOM".to_string()),
    ..Default::default()
});
```

### FATAL
```rust
use observability_narration_core::{narrate_fatal, NarrationFields};

narrate_fatal(NarrationFields {
    actor: "pool-managerd",
    action: "startup",
    target: "GPU0".to_string(),
    human: "CRITICAL: GPU0 initialization failed - cannot continue".to_string(),
    error_kind: Some("GPUInitFailure".to_string()),
    ..Default::default()
});
```

---

## 🔗 Correlation IDs

### Generate
```rust
use observability_narration_core::generate_correlation_id;

let correlation_id = generate_correlation_id();
// Returns: "550e8400-e29b-41d4-a716-446655440000"
```

### Validate
```rust
use observability_narration_core::validate_correlation_id;

if let Some(valid_id) = validate_correlation_id(&user_provided_id) {
    // Use valid_id (zero-copy, <100ns)
} else {
    // Invalid format
}
```

### Extract from HTTP Headers
```rust
use observability_narration_core::correlation_from_header;

let header_value = req.headers().get("X-Correlation-Id")?;
if let Some(correlation_id) = correlation_from_header(header_value.to_str()?) {
    // Use correlation_id
}
```

### Propagate to Downstream
```rust
use observability_narration_core::correlation_propagate;

let downstream_header = correlation_propagate(&correlation_id);
req.headers_mut().insert("X-Correlation-Id", downstream_header.parse()?);
```

---

## 🔒 Secret Redaction

### Automatic (Default)
```rust
use observability_narration_core::{narrate, NarrationFields};

// Secrets are automatically redacted
narrate(NarrationFields {
    actor: "auth-service",
    action: "login",
    target: "user-123".to_string(),
    human: "Authorization: Bearer abc123xyz".to_string(), // → "Authorization: [REDACTED]"
    ..Default::default()
});
```

### Patterns Redacted
1. **Bearer tokens** - `Bearer abc123` → `[REDACTED]`
2. **API keys** - `api_key=secret` → `[REDACTED]`
3. **JWT tokens** - `eyJ...` → `[REDACTED]`
4. **Private keys** - `-----BEGIN PRIVATE KEY-----...` → `[REDACTED]`
5. **URL passwords** - `://user:pass@host` → `[REDACTED]`
6. **UUIDs** - Optional (off by default)

### Custom Redaction Policy
```rust
use observability_narration_core::{redact_secrets, RedactionPolicy};

let mut policy = RedactionPolicy::default();
policy.mask_uuids = true; // Enable UUID redaction
policy.replacement = "***".to_string(); // Custom replacement

let redacted = redact_secrets("session_id: 550e8400-e29b-41d4-a716-446655440000", policy);
// → "session_id: ***"
```

---

## 🎀 Cute & Story Modes

### Cute Mode (Whimsical)
```rust
narrate(NarrationFields {
    actor: "vram-residency",
    action: "seal",
    target: "llama-7b".to_string(),
    human: "Sealed model shard 'llama-7b' in 2048 MB VRAM on GPU 0 (5 ms)".to_string(),
    cute: Some("Tucked llama-7b safely into GPU0's warm 2GB nest! Sweet dreams! 🛏️✨".to_string()),
    ..Default::default()
});
```

### Story Mode (Dialogue)
```rust
narrate(NarrationFields {
    actor: "rbees-orcd",
    action: "vram_request",
    target: "pool-managerd-3".to_string(),
    human: "Requesting 2048 MB VRAM on GPU 0 for model 'llama-7b'".to_string(),
    cute: Some("Orchestratord politely asks pool-managerd-3 for a cozy 2GB spot! 🏠".to_string()),
    story: Some("\"Do you have 2GB VRAM on GPU0?\" asked rbees-orcd. \"Yes!\" replied pool-managerd-3, \"Allocating now.\"".to_string()),
    ..Default::default()
});
```

---

## 🏃 Trace Macros (Dev Builds Only)

### Enable in Cargo.toml
```toml
[dependencies]
observability-narration-core = { path = "../narration-core", features = ["trace-enabled"] }
```

### Function Entry/Exit
```rust
use observability_narration_core::{trace_enter, trace_exit};

fn process_request(job_id: &str) -> Result<()> {
    trace_enter!("rbees-orcd", "process_request", format!("job_id={}", job_id));
    
    // ... processing logic ...
    
    trace_exit!("rbees-orcd", "process_request", "→ Ok (5ms)");
    Ok(())
}
```

### Minimal Trace
```rust
use observability_narration_core::trace_tiny;

for (i, token) in tokens.iter().enumerate() {
    trace_tiny!("tokenizer", "decode", format!("token_{}", i), 
                format!("Decoding token {} of {}", i, tokens.len()));
}
```

### Loop Iteration
```rust
use observability_narration_core::trace_loop;

for (i, worker) in workers.iter().enumerate() {
    trace_loop!("rbees-orcd", "select_worker", i, workers.len(),
                format!("worker={}, load={}/8", worker.id, worker.load));
}
```

---

## 🧪 Testing

### Install Capture Adapter
```rust
use observability_narration_core::CaptureAdapter;

#[test]
fn test_narration() {
    let capture = CaptureAdapter::install();
    
    narrate(NarrationFields {
        actor: "test",
        action: "test",
        target: "test".to_string(),
        human: "Test message".to_string(),
        ..Default::default()
    });
    
    let captured = capture.captured();
    assert_eq!(captured.len(), 1);
    assert_eq!(captured[0].human, "Test message");
}
```

### Assertions
```rust
capture.assert_includes("Test message");
capture.assert_field("actor", "test");
capture.assert_correlation_id_present();
capture.assert_cute_present();
capture.assert_story_present();
```

---

## 🎯 Field Taxonomy

### Required Fields
- `actor` - Who performed the action (e.g., "rbees-orcd")
- `action` - What action was performed (e.g., "dispatch")
- `target` - What was acted upon (e.g., "job-123")
- `human` - Human-readable description (≤100 chars)

### Optional Fields

**Correlation & Identity**:
- `correlation_id` - Request tracking ID
- `session_id` - User session ID
- `job_id`, `task_id`, `pool_id`, `replica_id`, `worker_id`

**Contextual**:
- `error_kind` - Error classification
- `retry_after_ms`, `backoff_ms`, `duration_ms`
- `queue_position`, `predicted_start_ms`

**Engine/Model**:
- `engine`, `engine_version`, `model_ref`, `device`

**Performance**:
- `tokens_in`, `tokens_out`, `decode_time_ms`

**Provenance** (auto-injected with `narrate_auto`):
- `emitted_by` - Service identity
- `emitted_at_ms` - Timestamp
- `trace_id`, `span_id`, `parent_span_id`

---

## 🚀 Auto-Injection

```rust
use observability_narration_core::{narrate_auto, NarrationFields};

// Automatically injects service identity and timestamp
narrate_auto(NarrationFields {
    actor: "pool-managerd",
    action: "spawn",
    target: "GPU0".to_string(),
    human: "Spawning engine llamacpp-v1".to_string(),
    ..Default::default()
});
// emitted_by and emitted_at_ms are automatically added
```

---

## 📝 Best Practices

### ✅ DO
- Use present tense for in-progress actions ("Spawning engine")
- Use past tense for completed actions ("Spawned engine")
- Include specific numbers and IDs
- Keep `human` field under 100 characters
- Use correlation IDs for request tracking
- Let auto-redaction handle secrets

### ❌ DON'T
- Use cryptic abbreviations ("Alloc fail" → "VRAM allocation failed")
- Log secrets directly (use auto-redaction)
- Use passive voice ("Request was received" → "Received request")
- Use error codes without context ("ERR_5023" → "Insufficient VRAM")

---

## 🔧 Feature Flags

```toml
[features]
trace-enabled = []  # Enable trace macros (dev/debug builds only)
debug-enabled = []  # Enable debug-level narration
cute-mode = []      # Enable cute narration fields
otel = ["opentelemetry"]  # OpenTelemetry integration
test-support = []   # Test capture adapter in non-test builds
production = []     # Production profile (all tracing disabled)
```

---

## 📚 More Resources

- **Implementation Plan**: `IMPLEMENTATION_PLAN.md`
- **Team Responsibilities**: `TEAM_RESPONSIBILITY.md`
- **Implementation Status**: `IMPLEMENTATION_STATUS.md`
- **Week 1-2 Summary**: `WEEK_1_2_SUMMARY.md`
- **Testing Notes**: `TESTING_NOTES.md`

---

*Happy narrating! May your logs be readable and your correlation IDs present! 🎀*

*— The Narration Core Team 💝*
