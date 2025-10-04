# Worker-Orcd Narration Integration Guide

**Audience**: worker-orcd Foundation-Alpha team  
**Purpose**: Step-by-step guide to integrate narration-core  
**Status**: Ready for implementation

---

## Overview

This guide shows how to integrate `observability-narration-core` into worker-orcd for structured, human-readable logging with correlation ID propagation.

---

## Step 1: Add Dependency

Edit `bin/worker-orcd/Cargo.toml`:

```toml
[dependencies]
observability-narration-core = { path = "../shared-crates/narration-core" }
```

---

## Step 2: Import Taxonomy

In your Rust files:

```rust
use observability_narration_core::{
    narrate_auto,
    NarrationFields,
    ACTOR_WORKER_ORCD,
    ACTION_INFERENCE_START,
    ACTION_INFERENCE_COMPLETE,
    ACTION_HEARTBEAT_SEND,
    ACTION_READY_CALLBACK,
    ACTION_CANCEL,
};
```

---

## Step 3: Extract Correlation IDs

Extract correlation IDs from HTTP headers (orchestratord → worker):

```rust
use axum::{
    extract::Request,
    http::HeaderMap,
};

fn extract_correlation_id(headers: &HeaderMap) -> Option<String> {
    headers
        .get("X-Correlation-Id")
        .and_then(|v| v.to_str().ok())
        .map(String::from)
}

// In your handler:
async fn inference_handler(
    headers: HeaderMap,
    // ... other extractors
) -> Result<Response, Error> {
    let correlation_id = extract_correlation_id(&headers);
    
    // Use in narration
    narrate_auto(NarrationFields {
        actor: ACTOR_WORKER_ORCD,
        action: ACTION_INFERENCE_START,
        target: job_id.clone(),
        correlation_id,
        human: format!("Starting inference for job {}", job_id),
        ..Default::default()
    });
    
    // ... rest of handler
}
```

---

## Step 4: Critical Path Narrations

### Inference Start

```rust
narrate_auto(NarrationFields {
    actor: ACTOR_WORKER_ORCD,
    action: ACTION_INFERENCE_START,
    target: job_id.clone(),
    correlation_id: Some(correlation_id.clone()),
    model_ref: Some("llama-7b".to_string()),
    tokens_in: Some(prompt_tokens),
    human: format!("Starting inference for job {} with model llama-7b", job_id),
    ..Default::default()
});
```

### Inference Complete

```rust
let elapsed = start_time.elapsed();

narrate_auto(NarrationFields {
    actor: ACTOR_WORKER_ORCD,
    action: ACTION_INFERENCE_COMPLETE,
    target: job_id.clone(),
    correlation_id: Some(correlation_id.clone()),
    duration_ms: Some(elapsed.as_millis() as u64),
    tokens_out: Some(generated_tokens),
    decode_time_ms: Some(decode_ms),
    human: format!("Completed inference: {} tokens in {} ms", generated_tokens, elapsed.as_millis()),
    ..Default::default()
});
```

### Heartbeat

```rust
narrate_auto(NarrationFields {
    actor: ACTOR_WORKER_ORCD,
    action: ACTION_HEARTBEAT_SEND,
    target: "pool-managerd".to_string(),
    worker_id: Some(worker_id.clone()),
    human: "Sending heartbeat to pool-managerd".to_string(),
    ..Default::default()
});
```

### Ready Callback

```rust
narrate_auto(NarrationFields {
    actor: ACTOR_WORKER_ORCD,
    action: ACTION_READY_CALLBACK,
    target: "pool-managerd".to_string(),
    worker_id: Some(worker_id.clone()),
    engine: Some("llamacpp-v1".to_string()),
    engine_version: Some("b1234".to_string()),
    model_ref: Some("llama-7b".to_string()),
    human: format!("Worker ready with engine llamacpp-v1, model llama-7b"),
    ..Default::default()
});
```

### Error Handling

```rust
narrate_auto(NarrationFields {
    actor: ACTOR_WORKER_ORCD,
    action: ACTION_INFERENCE_START,
    target: job_id.clone(),
    correlation_id: Some(correlation_id.clone()),
    error_kind: Some("cuda_oom".to_string()),
    human: format!("Inference failed: CUDA out of memory (requested 4GB, only 2GB available)"),
    ..Default::default()
});
```

---

## Step 5: Correlation ID Propagation

### Outgoing HTTP Requests (Worker → Pool-Manager)

When calling pool-managerd, propagate correlation IDs:

```rust
use reqwest::Client;

async fn send_heartbeat(
    client: &Client,
    pool_manager_url: &str,
    correlation_id: Option<&str>,
) -> Result<(), Error> {
    let mut req = client.post(format!("{}/heartbeat", pool_manager_url));
    
    // Propagate correlation ID
    if let Some(cid) = correlation_id {
        req = req.header("X-Correlation-Id", cid);
    }
    
    req.send().await?;
    Ok(())
}
```

---

## Step 6: Editorial Guidelines

Follow narration-core's editorial standards:

### ✅ Good Narration

```rust
// Specific, clear, under 100 chars
human: "Starting inference for job job-123 with model llama-7b"
human: "Completed inference: 150 tokens in 2500 ms"
human: "Inference failed: CUDA out of memory (requested 4GB, only 2GB available)"
```

### ❌ Bad Narration

```rust
// Too vague
human: "Starting inference"
human: "Done"
human: "Error occurred"

// Too long (>100 chars)
human: "Starting inference for job job-123 with model llama-7b on GPU 0 with 8 slots available and 4096 context length"
```

### Editorial Checklist

- [ ] **Clarity**: Can a developer understand what happened?
- [ ] **Specificity**: Are all relevant IDs/numbers included?
- [ ] **Brevity**: Is it under 100 characters?
- [ ] **Present tense**: "Starting inference" (not "Started")
- [ ] **Active voice**: "Worker sends heartbeat" (not "Heartbeat was sent")
- [ ] **Context**: Does it answer "why" not just "what"?
- [ ] **No secrets**: No bearer tokens, API keys, passwords
- [ ] **Correlation ID**: Included when available

---

## Step 7: Testing

### Unit Test Example

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use observability_narration_core::CaptureAdapter;

    #[tokio::test]
    async fn test_inference_emits_narration() {
        let capture = CaptureAdapter::install();
        
        // Run inference
        let result = handle_inference(job_id, correlation_id).await;
        
        // Assert narration was emitted
        capture.assert_includes("Starting inference");
        capture.assert_field("actor", "worker-orcd");
        capture.assert_field("action", "inference_start");
        capture.assert_correlation_id_present();
    }
}
```

---

## Step 8: Verification

After integration, verify:

1. **Dependency added**: `cargo build -p worker-orcd` succeeds
2. **Narration emitted**: Run worker and check logs for structured events
3. **Correlation IDs present**: All events include `correlation_id` field
4. **Performance metrics**: Completion events include `duration_ms`, `tokens_out`
5. **No secrets**: Grep logs for "Bearer", "api_key" — should be `[REDACTED]`

### Verification Commands

```bash
# Build worker-orcd
cargo build -p worker-orcd

# Run worker and capture logs
RUST_LOG=info cargo run -p worker-orcd 2>&1 | tee worker.log

# Check for narration events
grep "actor=worker-orcd" worker.log

# Check for correlation IDs
grep "correlation_id=" worker.log

# Check for secrets (should be redacted)
grep -i "bearer\|api_key\|password" worker.log
```

---

## Step 9: Submit for Editorial Review

Once integrated, notify narration-core team for editorial review. We'll check:

- ✅ Correlation ID discipline
- ✅ Human-readable narration quality
- ✅ Performance metrics completeness
- ✅ Error context specificity
- ✅ Secret redaction effectiveness

---

## Example: Complete Inference Flow

```rust
use observability_narration_core::{
    narrate_auto,
    NarrationFields,
    ACTOR_WORKER_ORCD,
    ACTION_INFERENCE_START,
    ACTION_INFERENCE_COMPLETE,
};
use std::time::Instant;

async fn handle_inference(
    job_id: String,
    correlation_id: Option<String>,
    prompt: String,
) -> Result<String, Error> {
    // Start narration
    narrate_auto(NarrationFields {
        actor: ACTOR_WORKER_ORCD,
        action: ACTION_INFERENCE_START,
        target: job_id.clone(),
        correlation_id: correlation_id.clone(),
        model_ref: Some("llama-7b".to_string()),
        tokens_in: Some(prompt.len() as u64),
        human: format!("Starting inference for job {}", job_id),
        ..Default::default()
    });
    
    let start = Instant::now();
    
    // Run inference
    let result = run_inference(&prompt).await?;
    
    let elapsed = start.elapsed();
    
    // Complete narration
    narrate_auto(NarrationFields {
        actor: ACTOR_WORKER_ORCD,
        action: ACTION_INFERENCE_COMPLETE,
        target: job_id.clone(),
        correlation_id: correlation_id.clone(),
        duration_ms: Some(elapsed.as_millis() as u64),
        tokens_out: Some(result.tokens.len() as u64),
        human: format!("Completed inference: {} tokens in {} ms", result.tokens.len(), elapsed.as_millis()),
        ..Default::default()
    });
    
    Ok(result.text)
}
```

---

## Support

Questions? Contact the narration-core team:
- **Location**: `bin/shared-crates/narration-core`
- **Docs**: `TEAM_RESPONSIBILITY.md`, `README.md`
- **BDD Tests**: `bdd/features/`

We have ultimate editorial authority over narration and we're here to help! 🎀

---

*Integration guide prepared by the Narration Core Team — may your correlation IDs be present and your logs be readable! 🎀*
