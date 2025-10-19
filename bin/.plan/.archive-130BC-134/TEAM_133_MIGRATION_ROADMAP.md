# TEAM-133: Migration Roadmap

**Step-by-step implementation guide for TEAM-137 (Preparation Phase)**

---

## PHASE OVERVIEW

**Phase 1: Investigation** (TEAM-133) ✅ COMPLETE  
**Phase 2: Preparation** (TEAM-137) ← YOU ARE HERE  
**Phase 3: Implementation** (TEAM-141) → Next

---

## MIGRATION ORDER (CRITICAL!)

**Start with easiest, end with hardest:**

1. ✅ **worker-rbee-error** (0 days) - DONE BY TEAM-130!
2. **worker-rbee-health** (1 day) - Simple, no dependencies
3. **worker-rbee-startup** (2 days) - Depends on error
4. **worker-rbee-sse-streaming** (4 days) - Refactor for generics
5. **worker-rbee-http-server** (6 days) - Large, complex
6. **worker-rbee-inference-base** (9 days) - Most complex

**Total:** 22 days (4.5 weeks) + 1 week buffer = **5 weeks**

---

## STEP-BY-STEP MIGRATION

### CRATE 1: worker-rbee-error ✅ DONE

**Status:** ✅ Complete (TEAM-130)

**Verification:**
```bash
cd libs/worker-rbee-crates/error
cargo test
```

---

### CRATE 2: worker-rbee-health (Day 1)

#### Step 1: Create crate structure
```bash
cd libs/worker-rbee-crates/
cargo new --lib health
cd health
```

#### Step 2: Add dependencies to Cargo.toml
```toml
[dependencies]
reqwest = { version = "0.12", features = ["json"] }
tokio = { version = "1", features = ["time", "sync"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
chrono = "0.4"
tracing = "0.1"
```

#### Step 3: Move files
```bash
# From bin/llm-worker-rbee/src/
cp heartbeat.rs libs/worker-rbee-crates/health/src/lib.rs

# Update module exports
# Remove internal tests (move to tests/)
```

#### Step 4: Update imports
```rust
// Old: (none - standalone file)
// New: pub mod config; pub mod heartbeat; pub mod types;
```

#### Step 5: Test
```bash
cargo test
cargo clippy
cargo check
```

#### Step 6: Update binary to use crate
```toml
# bin/llm-worker-rbee/Cargo.toml
[dependencies]
worker-rbee-health = { path = "../../libs/worker-rbee-crates/health" }
```

```rust
// bin/llm-worker-rbee/src/main.rs
use worker_rbee_health::{HeartbeatConfig, start_heartbeat_task};

// Rest stays the same!
```

#### Step 7: Integration test
```bash
cd bin/llm-worker-rbee
cargo test
```

**Success Criteria:**
- [ ] Crate compiles
- [ ] All unit tests pass
- [ ] Binary still compiles
- [ ] Integration tests pass
- [ ] No performance regression

---

### CRATE 3: worker-rbee-startup (Days 2-3)

#### Step 1: Create crate structure
```bash
cd libs/worker-rbee-crates/
cargo new --lib startup
cd startup
```

#### Step 2: Add dependencies
```toml
[dependencies]
worker-rbee-error = { path = "../error" }
observability-narration-core = { path = "../../shared-crates/narration-core" }
reqwest = { version = "0.12", features = ["json"] }
tokio = { version = "1", features = ["macros"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
anyhow = "1.0"
tracing = "0.1"

[dev-dependencies]
wiremock = "0.6"
```

#### Step 3: Move files
```bash
cp ../../../bin/llm-worker-rbee/src/common/startup.rs src/lib.rs
```

#### Step 4: Update imports
```rust
// Add to top of file:
use worker_rbee_error::WorkerError; // if needed
use observability_narration_core::{narrate, NarrationFields};
```

#### Step 5: Add retry logic (NEW FEATURE!)
```rust
pub async fn callback_ready_with_retry(
    callback_url: &str,
    worker_id: &str,
    model_ref: &str,
    backend: &str,
    device: u32,
    vram_bytes: u64,
    port: u16,
    max_retries: u32,
) -> Result<()> {
    for attempt in 0..=max_retries {
        match callback_ready(callback_url, worker_id, model_ref, backend, device, vram_bytes, port).await {
            Ok(()) => return Ok(()),
            Err(e) if attempt < max_retries => {
                let backoff = 2u64.pow(attempt);
                tracing::warn!(
                    attempt = attempt + 1,
                    max_retries,
                    backoff_secs = backoff,
                    error = %e,
                    "Callback failed, retrying..."
                );
                tokio::time::sleep(tokio::time::Duration::from_secs(backoff)).await;
            }
            Err(e) => return Err(e),
        }
    }
    unreachable!()
}
```

#### Step 6: Test
```bash
cargo test --lib
cargo test --test '*'
```

#### Step 7: Update binary
```toml
# bin/llm-worker-rbee/Cargo.toml
worker-rbee-startup = { path = "../../libs/worker-rbee-crates/startup" }
```

```rust
// bin/llm-worker-rbee/src/main.rs
use worker_rbee_startup::{callback_ready, callback_ready_with_retry};

// In main():
callback_ready_with_retry(
    &args.callback_url,
    &args.worker_id,
    &args.model_ref,
    &args.backend,
    args.device,
    backend.memory_bytes(),
    args.port,
    3, // max_retries
).await?;
```

**Success Criteria:**
- [ ] Crate compiles
- [ ] All 10 TEAM-130 tests pass
- [ ] New retry tests pass
- [ ] Binary compiles
- [ ] Integration test with mock rbee-hive passes

---

### CRATE 4: worker-rbee-sse-streaming (Days 4-7)

**This is the CRITICAL crate - must get generics right!**

#### Step 1: Design generic events
```rust
// src/event.rs
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InferenceEvent<T> {
    Started {
        job_id: String,
        model: String,
        started_at: String,
    },
    Output(T),
    Metrics {
        tokens_per_sec: f32,
        vram_bytes: u64,
    },
    Narration {
        actor: String,
        action: String,
        target: String,
        human: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        cute: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        story: Option<String>,
    },
    End {
        items_out: u32,
        decode_time_ms: u64,
        stop_reason: StopReason,
    },
    Error {
        code: String,
        message: String,
    },
}

// LLM-specific output
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
pub struct TokenOutput {
    pub t: String,
    pub i: u32,
}

// Type alias for LLM events
pub type LlmEvent = InferenceEvent<TokenOutput>;

// Embedding-specific output (future)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingOutput {
    pub embedding: Vec<f32>,
    pub dim: usize,
}
pub type EmbeddingEvent = InferenceEvent<EmbeddingOutput>;
```

#### Step 2: Verify JSON serialization
```rust
#[test]
fn test_llm_event_serialization() {
    let event = LlmEvent::Output(TokenOutput {
        t: "hello".into(),
        i: 0,
    });
    
    let json = serde_json::to_string(&event).unwrap();
    
    // MUST match old format!
    assert_eq!(json, r#"{"type":"output","t":"hello","i":0}"#);
}
```

#### Step 3: Create golden files
```bash
mkdir tests/golden/
# Save current JSON outputs to golden files
# Compare after migration
```

#### Step 4: Move files
```bash
cp ../../../bin/llm-worker-rbee/src/http/sse.rs src/event.rs
cp ../../../bin/llm-worker-rbee/src/common/inference_result.rs src/result.rs
```

#### Step 5: Test backward compatibility
```bash
cargo test
# Check golden files match
```

**Success Criteria:**
- [ ] Generic events compile
- [ ] LlmEvent produces same JSON as before
- [ ] Golden file tests pass
- [ ] Binary compiles
- [ ] SSE integration test passes

---

### CRATE 5: worker-rbee-http-server (Days 8-13)

**LARGEST CRATE - Break into sub-steps**

#### Step 1: Create structure
```bash
cargo new --lib http-server
cd http-server
mkdir src/{endpoints,middleware,validation}
```

#### Step 2: Move files (one at a time!)
```bash
# Day 8: Server infrastructure
cp server.rs routes.rs backend.rs src/

# Day 9: Endpoints (part 1)
cp health.rs ready.rs src/endpoints/

# Day 10: Endpoints (part 2)
cp execute.rs loading.rs narration_channel.rs src/endpoints/

# Day 11: Middleware
cp middleware/auth.rs src/middleware/

# Day 12: Validation
cp validation.rs src/validation/

# Day 13: Integration testing
```

#### Step 3: Test after each day
```bash
cargo test --lib
cargo test --test '*'
cd ../../../bin/llm-worker-rbee && cargo test
```

**Success Criteria:**
- [ ] Crate compiles
- [ ] All endpoint tests pass
- [ ] Middleware tests pass
- [ ] Binary compiles
- [ ] Full HTTP integration test passes

---

### CRATE 6: worker-rbee-inference-base (Days 14-22)

**MOST COMPLEX - Requires careful testing**

#### Step 1: Split LLM-specific code
```bash
cargo new --lib inference-base
cargo new --lib ../../bin/llm-worker-rbee-crates/llm-inference
```

#### Step 2: Move generic code to inference-base
```bash
# Device management
cp device.rs src/

# Model loading (SafeTensors, GGUF)
cp backend/models/mod.rs src/model_loader.rs

# VRAM tracking
# (extract from inference.rs)
```

#### Step 3: Move LLM-specific code to llm-inference
```bash
# Tokenizer loading
cp backend/tokenizer_loader.rs ../llm-worker-rbee-crates/llm-inference/src/

# Generation loop
cp backend/inference.rs ../llm-worker-rbee-crates/llm-inference/src/

# Sampling
cp backend/sampling.rs ../llm-worker-rbee-crates/llm-inference/src/
cp common/sampling_config.rs ../llm-worker-rbee-crates/llm-inference/src/

# Models
cp -r backend/models/ ../llm-worker-rbee-crates/llm-inference/src/
```

#### Step 4: Test all model architectures
```bash
# Test each model separately
cargo test --lib test_llama
cargo test --lib test_mistral
cargo test --lib test_phi
cargo test --lib test_qwen
cargo test --lib test_quantized_llama
cargo test --lib test_quantized_phi
cargo test --lib test_quantized_qwen
```

#### Step 5: Performance benchmarks
```bash
cargo bench --bench inference_speed
# Ensure <5% regression
```

**Success Criteria:**
- [ ] inference-base compiles
- [ ] llm-inference compiles
- [ ] All model tests pass
- [ ] Binary compiles
- [ ] Integration tests pass
- [ ] Performance acceptable (<5% regression)

---

## VERIFICATION CHECKLIST

After each crate:
- [ ] `cargo build --release` succeeds
- [ ] `cargo test` all pass
- [ ] `cargo clippy` no warnings
- [ ] `cargo fmt --check` passes
- [ ] Integration tests pass
- [ ] Performance benchmarks acceptable
- [ ] Git commit with clear message

After all crates:
- [ ] Binary runs
- [ ] Can load models
- [ ] Can execute inference
- [ ] SSE streaming works
- [ ] Health checks work
- [ ] Heartbeat works
- [ ] Callbacks work
- [ ] Authentication works
- [ ] All integration tests pass
- [ ] Performance acceptable

---

## ROLLBACK TRIGGERS

**Abort migration if:**
- More than 2 crates fail
- >10% performance regression
- >1 week behind schedule
- Critical bugs discovered
- Integration tests fail consistently

**Rollback procedure:**
```bash
git log --oneline | grep "TEAM-133"  # Find migration commits
git revert <commit>...                # Revert all
cargo test                            # Verify rollback
```

---

## SUCCESS METRICS

**Definition of Done:**
- [ ] All 6 crates created
- [ ] Binary still works
- [ ] All tests pass (220+ tests)
- [ ] No performance regression
- [ ] Documentation complete
- [ ] Peer review approved
- [ ] Ready for future workers

**Timeline:** 5 weeks (22 working days + buffer)

**Next Phase:** TEAM-141 can build embedding-worker-rbee using these crates!
