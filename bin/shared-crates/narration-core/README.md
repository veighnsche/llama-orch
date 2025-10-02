# narration-core

**Structured observability with human-readable narration**

`libs/observability/narration-core` — Emits structured logs with actor/action/target taxonomy and plain English descriptions.

---

## What This Library Does

narration-core provides **structured observability** for llama-orch:

- **Narration events** — Actor/action/target with human-readable descriptions
- **Cute mode** — Optional whimsical children's book narration! 🎀✨
- **Correlation IDs** — Track requests across service boundaries
- **Secret redaction** — Automatic masking of bearer tokens and API keys
- **Zero-cost abstractions** — Built on `tracing` for minimal overhead
- **Test capture** — Assertion helpers for BDD tests
- **JSON logs** — Structured output for production

**Used by**: All services (orchestratord, pool-managerd, provisioners)

---

## Key Concepts

### Narration Event

Every event includes:

- **actor** — Who performed the action (orchestratord, pool-managerd, etc.)
- **action** — What was done (enqueue, provision, register, etc.)
- **target** — What was acted upon (job_id, pool_id, model_id, etc.)
- **human** — Plain English description for humans
- **cute** — Whimsical children's book narration (optional) 🎀

Optional fields:
- **correlation_id** — Request tracking across services
- **session_id** — Session identifier
- **pool_id** — Pool identifier
- **replica_id** — Replica identifier

---

## Usage

### Emit Narration

```rust
use narration_core::{narrate, Actor, Action};

narrate!(
    actor = Actor::Orchestratord,
    action = Action::Enqueue,
    target = job_id,
    correlation_id = req_id,
    human = "Enqueued job {job_id} for pool {pool_id}"
);
```

### With Correlation ID

```rust
use narration_core::{narrate_with_correlation, Actor, Action};

narrate_with_correlation!(
    correlation_id = req_id,
    actor = Actor::PoolManagerd,
    action = Action::Provision,
    target = pool_id,
    human = "Provisioning engine for pool {pool_id}"
);
```

### Secret Redaction

```rust
use narration_core::{narrate, Actor, Action};

// Automatically redacts bearer tokens
narrate!(
    actor = Actor::Orchestratord,
    action = Action::Authenticate,
    target = "api",
    authorization = format!("Bearer {}", token), // Will be redacted
    human = "Authenticated API request"
);
```

### Cute Mode (Children's Book Narration)

```rust
use narration_core::{narrate, NarrationFields};

narrate(NarrationFields {
    actor: "vram-residency",
    action: "seal",
    target: "llama-7b".to_string(),
    human: "Sealed model shard 'llama-7b' in 2048 MB VRAM on GPU 0 (5 ms)".to_string(),
    cute: Some("Tucked llama-7b safely into GPU0's warm 2GB nest! Sweet dreams! 🛏️✨".to_string()),
    ..Default::default()
});
```

**Output**:
```json
{
  "actor": "vram-residency",
  "action": "seal",
  "target": "llama-7b",
  "human": "Sealed model shard 'llama-7b' in 2048 MB VRAM on GPU 0 (5 ms)",
  "cute": "Tucked llama-7b safely into GPU0's warm 2GB nest! Sweet dreams! 🛏️✨"
}
```

---

## Event Taxonomy

### Actors

- **Orchestratord** — Main orchestrator service
- **PoolManagerd** — GPU node pool manager
- **EngineProvisioner** — Engine provisioning service
- **ModelProvisioner** — Model provisioning service
- **Adapter** — Worker adapter

### Actions

- **Enqueue** — Add job to queue
- **Dispatch** — Send job to worker
- **Provision** — Provision engine or model
- **Register** — Register node or pool
- **Heartbeat** — Send heartbeat
- **Deregister** — Remove node or pool
- **Complete** — Job completed
- **Error** — Error occurred

---

## JSON Output

### Production Format

```json
{
  "timestamp": "2025-10-01T00:00:00Z",
  "level": "INFO",
  "actor": "orchestratord",
  "action": "enqueue",
  "target": "job-123",
  "correlation_id": "req-abc",
  "pool_id": "default",
  "human": "Enqueued job job-123 for pool default"
}
```

### Console Format (Development)

```
2025-10-01T00:00:00Z INFO orchestratord enqueue job-123 [req-abc] Enqueued job job-123 for pool default
```

---

## Testing

### Capture Adapter

```rust
use narration_core::testing::CaptureAdapter;

#[tokio::test]
async fn test_narration() {
    let capture = CaptureAdapter::install();
    
    // Perform actions that emit narration
    orchestrator.enqueue(job).await?;
    
    // Assert narration was emitted
    capture.assert_includes("enqueue");
    capture.assert_field("actor", "orchestratord");
    capture.assert_correlation_id_present();
}
```

### Assertion Helpers

```rust
// Assert event contains text
capture.assert_includes("Enqueued job");

// Assert field value
capture.assert_field("action", "enqueue");
capture.assert_field("target", "job-123");

// Assert correlation ID present
capture.assert_correlation_id_present();

// Get all events
let events = capture.events();
assert_eq!(events.len(), 3);
```

---

## Correlation ID Propagation

### Across Services

```rust
// orchestratord generates correlation ID
let correlation_id = CorrelationId::new();

narrate!(
    correlation_id = correlation_id.clone(),
    actor = Actor::Orchestratord,
    action = Action::Dispatch,
    target = job_id,
    human = "Dispatching job to pool-managerd"
);

// Pass to pool-managerd via HTTP header
let response = client
    .post("/provision")
    .header("X-Correlation-ID", correlation_id.to_string())
    .send()
    .await?;

// pool-managerd extracts and uses correlation ID
let correlation_id = extract_correlation_id(&request)?;

narrate!(
    correlation_id = correlation_id,
    actor = Actor::PoolManagerd,
    action = Action::Provision,
    target = pool_id,
    human = "Received provision request"
);
```

---

## Secret Redaction

### Automatic Redaction

The following fields are automatically redacted:

- `authorization` — Bearer tokens
- `api_key` — API keys
- `token` — Generic tokens
- `password` — Passwords
- `secret` — Generic secrets

### Example

```rust
narrate!(
    actor = Actor::Orchestratord,
    action = Action::Authenticate,
    authorization = "Bearer abc123", // Becomes "[REDACTED]"
    human = "Authenticated request"
);
```

Output:

```json
{
  "actor": "orchestratord",
  "action": "authenticate",
  "authorization": "[REDACTED]",
  "human": "Authenticated request"
}
```

---

## Debugging

### Grep by Correlation ID

```bash
# Find all events for a request
grep "correlation_id=req-abc" logs/*.log

# Extract human descriptions
grep "correlation_id=req-abc" logs/*.log | jq -r '.human'
```

### Filter by Actor

```bash
# All pool-managerd events
grep "actor=pool-managerd" logs/*.log

# All provision actions
grep "action=provision" logs/*.log
```

### Read the Story

```bash
# Get human-readable story
grep "human=" logs/*.log | jq -r '.human'
```

---

## Testing

### Unit Tests

```bash
# Run all tests
cargo test -p observability-narration-core -- --nocapture

# Run specific test
cargo test -p observability-narration-core -- test_narration --nocapture
```

---

## Dependencies

### Internal

- None (foundational library)

### External

- `tracing` — Structured logging
- `serde` — Serialization
- `serde_json` — JSON output

---

## Specifications

Implements requirements from `.specs/00_llama-orch.md`:
- Structured observability
- Correlation ID propagation
- Secret redaction
- Human-readable narration

---

## Status

- **Version**: 0.0.0 (early development)
- **License**: GPL-3.0-or-later
- **Stability**: Alpha
- **Maintainers**: @llama-orch-maintainers
