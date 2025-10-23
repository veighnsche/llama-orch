# TEAM-258: Implementation Guide

## Overview

This guide explains how the hive-forwarding consolidation works and how to extend it.

---

## How It Works

### Step 1: Operation Arrives at queen-rbee

```
POST /v1/jobs
{
  "operation": "worker_spawn",
  "hive_id": "localhost",
  "model": "llama2",
  "worker": "gpu-0",
  "device": 0
}
```

### Step 2: Parse into Operation Enum

```rust
let operation: Operation = serde_json::from_value(payload)?;
// operation = Operation::WorkerSpawn { 
//     hive_id: "localhost",
//     model: "llama2",
//     worker: "gpu-0",
//     device: 0
// }
```

### Step 3: Route Operation

```rust
match operation {
    Operation::HiveInstall { alias } => {
        // Handled directly by queen-rbee
        execute_hive_install(...).await?
    }
    Operation::HiveStart { alias } => {
        // Handled directly by queen-rbee
        execute_hive_start(...).await?
    }
    // ... other hive operations ...
    
    op if op.should_forward_to_hive() => {
        // Forwarded to rbee-hive
        hive_forwarder::forward_to_hive(&job_id, op, state.config.clone()).await?
    }
}
```

### Step 4: Check if Operation Should Forward

```rust
// In rbee-operations/src/lib.rs
impl Operation {
    pub fn should_forward_to_hive(&self) -> bool {
        matches!(
            self,
            Operation::WorkerSpawn { .. }
                | Operation::WorkerList { .. }
                | Operation::WorkerGet { .. }
                | Operation::WorkerDelete { .. }
                | Operation::ModelDownload { .. }
                | Operation::ModelList { .. }
                | Operation::ModelGet { .. }
                | Operation::ModelDelete { .. }
                | Operation::Infer { .. }
        )
    }
}
```

**Result:** `true` → Forward to hive

### Step 5: Forward to Hive

```rust
// In queen-rbee/src/hive_forwarder.rs
pub async fn forward_to_hive(
    job_id: &str,
    operation: Operation,
    config: Arc<RbeeConfig>,
) -> Result<()> {
    // 1. Extract hive_id
    let hive_id = operation.hive_id()?;
    
    // 2. Look up hive config
    let hive_config = config.hives.get(hive_id)?;
    
    // 3. POST operation to hive
    let client = reqwest::Client::new();
    let hive_url = format!("http://{}:{}", hive_config.hostname, hive_config.hive_port);
    
    let job_response: serde_json::Value = client
        .post(format!("{}/v1/jobs", hive_url))
        .json(&operation)
        .send()
        .await?
        .json()
        .await?;
    
    let hive_job_id = job_response["job_id"].as_str()?;
    
    // 4. Stream responses from hive
    let stream = client
        .get(format!("{}/v1/jobs/{}/stream", hive_url, hive_job_id))
        .send()
        .await?
        .bytes_stream();
    
    // 5. Forward each line to client
    while let Some(chunk) = stream.next().await {
        let text = String::from_utf8(chunk?.to_vec())?;
        for line in text.lines() {
            NARRATE
                .action("forward_data")
                .job_id(job_id)
                .context(line)
                .human("{}")
                .emit();
        }
    }
    
    Ok(())
}
```

### Step 6: Hive Processes Operation

```
POST http://localhost:8081/v1/jobs
{
  "operation": "worker_spawn",
  "hive_id": "localhost",
  "model": "llama2",
  "worker": "gpu-0",
  "device": 0
}

↓

rbee-hive/src/job_router.rs matches:
Operation::WorkerSpawn { hive_id, model, worker, device } => {
    execute_worker_spawn(...).await?
}

↓

GET http://localhost:8081/v1/jobs/{hive_job_id}/stream

↓

SSE stream with narration events
```

### Step 7: Stream Back to Client

```
GET /v1/jobs/{queen_job_id}/stream

↓

Queen-rbee's SSE sink receives narration from hive_forwarder

↓

Client receives events in real-time
```

---

## Adding a New Forwarded Operation

### Example: Add `WorkerRestart` Operation

#### 1. Add to Operation Enum

**File:** `bin/99_shared_crates/rbee-operations/src/lib.rs`

```rust
pub enum Operation {
    // ... existing operations ...
    
    WorkerRestart {
        hive_id: String,
        id: String,
    },
}
```

#### 2. Add to Operation::name()

```rust
impl Operation {
    pub fn name(&self) -> &'static str {
        match self {
            // ... existing cases ...
            Operation::WorkerRestart { .. } => "worker_restart",
        }
    }
}
```

#### 3. Add to Operation::hive_id()

```rust
impl Operation {
    pub fn hive_id(&self) -> Option<&str> {
        match self {
            // ... existing cases ...
            Operation::WorkerRestart { hive_id, .. } => Some(hive_id),
        }
    }
}
```

#### 4. Add to Operation::should_forward_to_hive()

```rust
impl Operation {
    pub fn should_forward_to_hive(&self) -> bool {
        matches!(
            self,
            // ... existing cases ...
            | Operation::WorkerRestart { .. }  // ← ADD THIS LINE
        )
    }
}
```

**That's it!** No changes needed to queen-rbee/src/job_router.rs

#### 5. Add to rbee-keeper CLI

**File:** `bin/00_rbee_keeper/src/main.rs`

```rust
Commands::Worker(WorkerAction::Restart { hive_id, id }) => {
    let operation = Operation::WorkerRestart { hive_id, id };
    // Send to queen-rbee
}
```

#### 6. Implement in rbee-hive

**File:** `bin/01_rbee_hive/src/job_router.rs`

```rust
match operation {
    // ... existing cases ...
    Operation::WorkerRestart { hive_id, id } => {
        execute_worker_restart(hive_id, id, &job_id).await?
    }
}
```

---

## Key Design Patterns

### 1. Guard Clause Pattern

```rust
op if op.should_forward_to_hive() => {
    hive_forwarder::forward_to_hive(&job_id, op, state.config.clone()).await?
}
```

**Why this pattern:**
- Explicit: Intent is clear
- Safe: Guard prevents accidental matches
- Maintainable: Single place to update

### 2. Extraction Pattern

```rust
let hive_id = operation.hive_id()?;
```

**Why this pattern:**
- Generic: Works for any operation with hive_id
- Safe: Returns error if operation doesn't have hive_id
- Reusable: Used in multiple places

### 3. Narration Pattern

```rust
NARRATE
    .action("forward_start")
    .job_id(job_id)
    .context(operation.name())
    .context(hive_id)
    .human("Forwarding {} operation to hive '{}'")
    .emit();
```

**Why this pattern:**
- Observable: Users see what's happening
- Traceable: job_id links events together
- Debuggable: Context helps diagnose issues

---

## Testing Strategy

### Unit Tests (rbee-operations)

```rust
#[test]
fn test_worker_spawn_should_forward() {
    let op = Operation::WorkerSpawn {
        hive_id: "localhost".to_string(),
        model: "llama2".to_string(),
        worker: "gpu-0".to_string(),
        device: 0,
    };
    assert!(op.should_forward_to_hive());
}

#[test]
fn test_hive_install_should_not_forward() {
    let op = Operation::HiveInstall { alias: "localhost".to_string() };
    assert!(!op.should_forward_to_hive());
}

#[test]
fn test_worker_spawn_has_hive_id() {
    let op = Operation::WorkerSpawn {
        hive_id: "localhost".to_string(),
        model: "llama2".to_string(),
        worker: "gpu-0".to_string(),
        device: 0,
    };
    assert_eq!(op.hive_id(), Some("localhost"));
}
```

### Integration Tests (queen-rbee)

```rust
#[tokio::test]
async fn test_forward_worker_spawn_to_localhost() {
    // 1. Start rbee-hive on localhost:8081
    // 2. Send WorkerSpawn to queen-rbee
    // 3. Verify operation reaches hive
    // 4. Verify response streams back to client
}

#[tokio::test]
async fn test_forward_to_missing_hive_returns_error() {
    // 1. Send WorkerSpawn with non-existent hive_id
    // 2. Verify error: "Hive 'xyz' not found in configuration"
}

#[tokio::test]
async fn test_forward_sse_stream_propagation() {
    // 1. Start rbee-hive
    // 2. Send WorkerSpawn to queen-rbee
    // 3. Connect to queen-rbee's SSE stream
    // 4. Verify narration events from hive appear in stream
}
```

---

## Error Handling

### Hive Not Found

```rust
let hive_config = config
    .hives
    .get(hive_id)
    .ok_or_else(|| anyhow::anyhow!("Hive '{}' not found in configuration", hive_id))?;
```

**User sees:**
```
❌ Error: Hive 'xyz' not found in configuration
```

### Connection Failed

```rust
let response = client
    .post(format!("{}/v1/jobs", hive_url))
    .json(&payload)
    .send()
    .await
    .map_err(|e| anyhow::anyhow!("Failed to connect to hive: {}", e))?;
```

**User sees:**
```
❌ Error: Failed to connect to hive: connection refused
```

### Invalid Response

```rust
let hive_job_id = job_response
    .get("job_id")
    .and_then(|v| v.as_str())
    .ok_or_else(|| anyhow::anyhow!("Hive did not return job_id"))?;
```

**User sees:**
```
❌ Error: Hive did not return job_id
```

---

## Performance Considerations

### Connection Pooling

Currently, each operation creates a new `reqwest::Client`. For better performance:

```rust
// TODO: Use connection pooling
let client = SHARED_HTTP_CLIENT.clone();
```

### Timeout Handling

Currently, no timeout on hive operations. Should add:

```rust
let response = client
    .post(format!("{}/v1/jobs", hive_url))
    .json(&payload)
    .timeout(Duration::from_secs(30))
    .send()
    .await?;
```

### Stream Buffering

Currently, forwards each chunk individually. Could batch:

```rust
let mut buffer = Vec::new();
while let Some(chunk) = stream.next().await {
    buffer.extend(chunk?);
    if buffer.len() > 4096 {
        // Forward batch
        buffer.clear();
    }
}
```

---

## Summary

**Key Points:**
1. `should_forward_to_hive()` determines which operations forward
2. `forward_to_hive()` handles the actual forwarding
3. Adding new operations only requires updating rbee-operations
4. No changes to queen-rbee when adding new operations
5. Hive operations (install, start, stop) stay in queen-rbee

**Files:**
- `bin/99_shared_crates/rbee-operations/src/lib.rs` - Operation enum
- `bin/10_queen_rbee/src/hive_forwarder.rs` - Forwarding logic
- `bin/10_queen_rbee/src/job_router.rs` - Routing logic

**Next Steps:**
- Implement actual forwarding in hive_forwarder.rs
- Add unit tests for should_forward_to_hive()
- Add integration tests for forwarding behavior
- Add timeout handling
- Add connection pooling
