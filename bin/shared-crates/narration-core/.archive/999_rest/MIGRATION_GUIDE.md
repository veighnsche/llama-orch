# Service Migration Guide 🎀

**Guide for migrating services to the new narration system**

**Target Services**: rbees-orcd, pool-managerd, worker-orcd  
**Timeline**: Week 4  
**Effort**: ~8 hours per service

---

## 📋 Pre-Migration Checklist

Before migrating a service:

- [ ] Review current logging/tracing usage
- [ ] Identify all narration points (INFO, WARN, ERROR)
- [ ] Map existing log messages to narration fields
- [ ] Identify correlation ID usage
- [ ] Plan cute/story mode additions (optional)

---

## 🚀 Migration Steps

### Step 1: Add Dependencies

**Update `Cargo.toml`**:

```toml
[dependencies]
observability-narration-core = { path = "../shared-crates/narration-core" }

# Optional: For proc macros (if using #[trace_fn])
observability-narration-macros = { path = "../shared-crates/narration-macros" }
```

**For development builds with trace macros**:

```toml
[features]
default = []
trace-enabled = ["observability-narration-core/trace-enabled"]
```

### Step 2: Replace Existing Logging

#### Before (Old Style)
```rust
tracing::info!("Dispatching job {} to worker {}", job_id, worker_id);
```

#### After (New Narration)
```rust
use observability_narration_core::{narrate, NarrationFields};

narrate(NarrationFields {
    actor: "rbees-orcd",
    action: "dispatch",
    target: job_id.to_string(),
    human: format!("Dispatching job {} to worker {}", job_id, worker_id),
    correlation_id: Some(correlation_id.clone()),
    job_id: Some(job_id.clone()),
    worker_id: Some(worker_id.clone()),
    ..Default::default()
});
```

### Step 3: Add Correlation ID Tracking

**Generate correlation ID** (at request entry):

```rust
use observability_narration_core::generate_correlation_id;

let correlation_id = generate_correlation_id();
```

**Extract from HTTP headers**:

```rust
use observability_narration_core::correlation_from_header;

let correlation_id = req.headers()
    .get("X-Correlation-Id")
    .and_then(|h| h.to_str().ok())
    .and_then(|s| correlation_from_header(s));
```

**Propagate to downstream services**:

```rust
use observability_narration_core::correlation_propagate;

req.headers_mut().insert(
    "X-Correlation-Id",
    correlation_propagate(&correlation_id).parse()?
);
```

### Step 4: Use Appropriate Levels

#### INFO (Default)
```rust
use observability_narration_core::narrate;

narrate(NarrationFields {
    actor: "rbees-orcd",
    action: "enqueue",
    target: job_id.to_string(),
    human: "Enqueued job successfully".to_string(),
    ..Default::default()
});
```

#### WARN
```rust
use observability_narration_core::narrate_warn;

narrate_warn(NarrationFields {
    actor: "pool-managerd",
    action: "capacity_check",
    target: "GPU0".to_string(),
    human: "GPU0 capacity low: 512MB available (2GB requested)".to_string(),
    ..Default::default()
});
```

#### ERROR
```rust
use observability_narration_core::narrate_error;

narrate_error(NarrationFields {
    actor: "worker-orcd",
    action: "inference",
    target: job_id.to_string(),
    human: "Inference failed: CUDA out of memory".to_string(),
    error_kind: Some("CudaOOM".to_string()),
    ..Default::default()
});
```

#### FATAL
```rust
use observability_narration_core::narrate_fatal;

narrate_fatal(NarrationFields {
    actor: "pool-managerd",
    action: "startup",
    target: "GPU0".to_string(),
    human: "CRITICAL: GPU0 initialization failed - cannot continue".to_string(),
    error_kind: Some("GPUInitFailure".to_string()),
    ..Default::default()
});
```

### Step 5: Add Cute Mode (Optional)

**For user-facing operations**:

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

**Guidelines**:
- Use whimsical metaphors (VRAM = "cozy blanket", GPU = "friendly helper")
- Include emoji (🛏️, ✨, 😟, 🏠, 👋, 🔍, 💪, 🎉)
- Keep under 150 characters
- NO dialogue (save that for story mode)

### Step 6: Add Story Mode (Optional)

**For multi-service interactions**:

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

**When to use story mode**:
- Request/response flows
- Worker callbacks
- Heartbeat checks
- Multi-service negotiations
- Error reporting with context

**When NOT to use**:
- Single-component internal operations
- Pure metrics emission
- Silent background tasks

### Step 7: Add Trace Macros (Dev Builds Only)

**For hot paths** (only in dev builds):

```rust
#[cfg(feature = "trace-enabled")]
use observability_narration_core::{trace_enter, trace_exit, trace_tiny};

fn process_request(job_id: &str) -> Result<()> {
    #[cfg(feature = "trace-enabled")]
    trace_enter!("rbees-orcd", "process_request", format!("job_id={}", job_id));
    
    // ... processing logic ...
    
    #[cfg(feature = "trace-enabled")]
    trace_exit!("rbees-orcd", "process_request", "→ Ok (5ms)");
    
    Ok(())
}
```

**For loops**:

```rust
#[cfg(feature = "trace-enabled")]
for (i, worker) in workers.iter().enumerate() {
    trace_loop!("rbees-orcd", "select_worker", i, workers.len(),
                format!("worker={}, load={}/8", worker.id, worker.load));
    // ... evaluation logic ...
}
```

---

## 📊 Service-Specific Migration

### rbees-orcd

**Key narration points**:
1. **Admission** - Request accepted, queued
2. **Dispatch** - Job dispatched to worker
3. **Completion** - Job completed successfully
4. **Errors** - Dispatch failures, timeouts

**Example**:

```rust
// Admission
narrate(NarrationFields {
    actor: "rbees-orcd",
    action: "admission",
    target: session_id.to_string(),
    human: format!("Accepted request; queued at position {} (ETA {} ms) on pool '{}'", 
                   position, eta_ms, pool_id),
    correlation_id: Some(correlation_id.clone()),
    session_id: Some(session_id.clone()),
    pool_id: Some(pool_id.clone()),
    queue_position: Some(position),
    predicted_start_ms: Some(eta_ms),
    ..Default::default()
});

// Dispatch
narrate(NarrationFields {
    actor: "rbees-orcd",
    action: "dispatch",
    target: job_id.to_string(),
    human: format!("Dispatching job {} to worker {}", job_id, worker_id),
    correlation_id: Some(correlation_id.clone()),
    job_id: Some(job_id.clone()),
    worker_id: Some(worker_id.clone()),
    ..Default::default()
});
```

### pool-managerd

**Key narration points**:
1. **Worker spawn** - Engine spawning
2. **Ready callback** - Worker ready
3. **Heartbeat** - Worker health check
4. **Capacity** - VRAM/slot availability

**Example**:

```rust
// Worker spawn
narrate(NarrationFields {
    actor: "pool-managerd",
    action: "spawn",
    target: "GPU0".to_string(),
    human: format!("Spawning engine {} for pool '{}' on GPU0", engine, pool_id),
    pool_id: Some(pool_id.clone()),
    replica_id: Some(replica_id.clone()),
    engine: Some(engine.clone()),
    device: Some("GPU0".to_string()),
    ..Default::default()
});

// Ready callback
narrate(NarrationFields {
    actor: "pool-managerd",
    action: "ready_callback",
    target: worker_id.to_string(),
    human: format!("Worker {} ready with engine {}, {} slots available", 
                   worker_id, engine, slots),
    worker_id: Some(worker_id.clone()),
    engine: Some(engine.clone()),
    ..Default::default()
});
```

### worker-orcd

**Key narration points**:
1. **Startup** - Worker initialization
2. **Model load** - Model loading
3. **Inference** - Inference execution
4. **Errors** - CUDA OOM, model errors

**Example**:

```rust
// Startup
narrate(NarrationFields {
    actor: "worker-orcd",
    action: "startup",
    target: "main".to_string(),
    human: format!("Worker starting with engine {} on {}", engine, device),
    engine: Some(engine.clone()),
    device: Some(device.clone()),
    ..Default::default()
});

// Inference
narrate(NarrationFields {
    actor: "worker-orcd",
    action: "inference",
    target: job_id.to_string(),
    human: format!("Completed inference: {} tokens in {} ms", tokens_out, duration_ms),
    job_id: Some(job_id.clone()),
    tokens_in: Some(tokens_in),
    tokens_out: Some(tokens_out),
    decode_time_ms: Some(duration_ms),
    ..Default::default()
});
```

---

## 🧪 Testing Migration

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use observability_narration_core::CaptureAdapter;

    #[test]
    fn test_dispatch_narration() {
        let capture = CaptureAdapter::install();
        
        // Your dispatch logic that emits narration
        dispatch_job("job-123", "worker-gpu0-r1").unwrap();
        
        let captured = capture.captured();
        assert_eq!(captured.len(), 1);
        assert_eq!(captured[0].actor, "rbees-orcd");
        assert_eq!(captured[0].action, "dispatch");
        assert!(captured[0].human.contains("job-123"));
    }
}
```

### BDD Tests

See `bdd/features/` for comprehensive BDD scenarios.

---

## 📝 Migration Checklist (Per Service)

### Pre-Migration
- [ ] Review current logging
- [ ] Identify narration points
- [ ] Plan correlation ID flow
- [ ] Design cute/story mode (optional)

### Implementation
- [ ] Add dependencies
- [ ] Replace INFO logs with `narrate()`
- [ ] Replace WARN logs with `narrate_warn()`
- [ ] Replace ERROR logs with `narrate_error()`
- [ ] Add correlation ID tracking
- [ ] Add cute mode (optional)
- [ ] Add story mode (optional)
- [ ] Add trace macros for hot paths (dev only)

### Testing
- [ ] Unit tests pass
- [ ] BDD tests pass
- [ ] Correlation IDs propagate correctly
- [ ] Secrets are redacted
- [ ] Performance acceptable

### Verification
- [ ] Run with `RUST_LOG=info` - verify output
- [ ] Run with `RUST_LOG=trace` - verify trace macros (dev build)
- [ ] Check production build has zero overhead
- [ ] Verify correlation ID tracking works end-to-end

---

## 🎯 Best Practices

### ✅ DO
- Use correlation IDs for all request flows
- Include specific numbers and IDs in `human` field
- Keep `human` field under 100 characters
- Use present tense for in-progress ("Spawning engine")
- Use past tense for completed ("Spawned engine")
- Let auto-redaction handle secrets
- Add cute mode for user-facing operations
- Add story mode for multi-service interactions

### ❌ DON'T
- Log secrets directly (use auto-redaction)
- Use cryptic abbreviations
- Use passive voice
- Use error codes without context
- Force dialogue where there's no conversation
- Exceed length guidelines (human: 100, cute: 150, story: 200)

---

## 🚨 Common Pitfalls

### Pitfall 1: Forgetting Correlation IDs
**Problem**: Requests can't be tracked across services  
**Solution**: Always propagate correlation IDs in HTTP headers

### Pitfall 2: Logging Secrets
**Problem**: Secrets leak into logs  
**Solution**: Trust auto-redaction, or use `RedactionPolicy`

### Pitfall 3: Overusing Story Mode
**Problem**: Dialogue where there's no conversation  
**Solution**: Only use story mode for actual multi-service interactions

### Pitfall 4: Trace Macros in Production
**Problem**: Performance overhead in production  
**Solution**: Always guard with `#[cfg(feature = "trace-enabled")]`

---

## 📊 Migration Timeline

### rbees-orcd (Day 19)
- **Effort**: 8 hours
- **Priority**: High (most narration points)
- **Focus**: Admission, dispatch, completion flows

### pool-managerd (Day 19-20)
- **Effort**: 8 hours
- **Priority**: High (worker lifecycle)
- **Focus**: Spawn, ready callbacks, heartbeats

### worker-orcd (Day 20)
- **Effort**: 8 hours
- **Priority**: Medium (inference execution)
- **Focus**: Startup, model load, inference

---

## 🎀 Final Notes

**Remember**: This is not just logging. This is **narration** — telling the story of your distributed system.

Make it:
- **Clear** - No jargon, no cryptic messages
- **Concise** - Under 100 characters for `human`
- **Cute** - Add whimsy where appropriate 🎀
- **Conversational** - Use story mode for interactions 🎭
- **Correlated** - Track requests end-to-end
- **Secure** - Let auto-redaction protect secrets

---

*Happy migrating! May your logs be readable and your correlation IDs present!* 🎀

*— The Narration Core Team 💝*
