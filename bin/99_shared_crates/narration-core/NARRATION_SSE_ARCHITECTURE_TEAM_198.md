# Narration SSE Architecture - Complete Investigation

**TEAM-198** | **Date:** 2025-10-22 | **Status:** ARCHITECTURE ANALYSIS + SOLUTION

---

## Mission Statement

Design a **simple and ergonomic** narration system where:
- ✅ **All queen narration** goes through SSE jobs channel (web-UI proof)
- ✅ **Only bee-keeper** can use stdout/stderr (CLI tool)
- ✅ **Hive and worker** narration flows back to queen → keeper → stdout
- ✅ **No manual formatting** in consumers (centralized, consistent)
- ✅ **Simple API** for developers (no thinking about output paths)

---

## Current Architecture (TEAM-197 State)

### The Complete Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ NARRATION EMISSION POINTS                                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. rbee-keeper (CLI)                                          │
│     ├─ NARRATE.action().emit()                                 │
│     └─ → narration-core::narrate() → stderr (DIRECT STDOUT)    │
│                                                                 │
│  2. queen-rbee (daemon)                                        │
│     ├─ NARRATE.action().emit()                                 │
│     └─ → narration-core::narrate() → stderr + SSE broadcaster  │
│                                                                 │
│  3. rbee-hive (daemon)                                         │
│     ├─ Currently: println!() only (NO NARRATION YET)          │
│     └─ FUTURE: Must send narration to queen via API           │
│                                                                 │
│  4. llm-worker (daemon)                                        │
│     ├─ narrate_dual() → stderr + thread-local SSE channel     │
│     └─ Thread-local channel merges into inference SSE stream   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ SSE TRANSPORT PATHS                                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Path 1: Queen Job Router → Keeper (CURRENT)                   │
│  ─────────────────────────────────────────────                 │
│  job-registry::execute_and_stream()                            │
│    ↓                                                            │
│  NARRATE.action().emit()                                       │
│    ↓                                                            │
│  narration-core::narrate_at_level()                            │
│    ├─ stderr (for daemon logs)                                 │
│    └─ sse_sink::send() → global SSE broadcaster               │
│         ↓                                                       │
│  queen-rbee /jobs.rs handle_stream_job()                       │
│    ├─ Subscribes to sse_sink::subscribe()                     │
│    ├─ RE-FORMATS: format!("[{:<10}] {:<15}: {}") ⚠️          │
│    └─ Sends formatted string as SSE events                     │
│         ↓                                                       │
│  keeper job_client.rs                                          │
│    ├─ Receives SSE stream                                      │
│    └─ println!() to stdout (just prints data)                  │
│                                                                 │
│  Path 2: Worker Inference → Queen → Keeper (FUTURE)            │
│  ────────────────────────────────────────────                  │
│  worker narrate_dual()                                         │
│    ├─ stderr (for daemon logs)                                 │
│    └─ thread-local channel → inference SSE stream              │
│         ↓                                                       │
│  llm-worker /execute.rs endpoint                               │
│    ├─ Merges narration + tokens into single SSE stream        │
│    └─ Streams to queen (via orchestrator)                      │
│         ↓                                                       │
│  queen-rbee (future: forward to keeper)                        │
│    └─ Must handle inference narration forwarding               │
│                                                                 │
│  Path 3: Hive Lifecycle → Queen → Keeper (NOT YET BUILT)       │
│  ───────────────────────────────────────────────────           │
│  rbee-hive (currently: println! only)                          │
│    ↓ [NEEDS TO BE BUILT]                                       │
│  Send narration events to queen via API                        │
│    ↓                                                            │
│  queen-rbee receives and broadcasts                            │
│    ↓                                                            │
│  keeper displays via job SSE stream                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Problem: Decentralized Formatting

### The Core Issue (TEAM-197 Discovered This)

**narration-core sends TWO outputs:**
1. **stderr:** Pre-formatted with `eprintln!("[{:<10}] {:<15}: {}", actor, action, human)`
2. **SSE:** Raw `NarrationEvent` struct (no formatting)

**The Bug:** Each SSE consumer must format events themselves, leading to:
- ❌ Inconsistent formats across consumers
- ❌ Manual duplication of formatting logic
- ❌ Breaking changes when format updates (must update all consumers)

**Example of the Problem:**
```rust
// narration-core: Formatted stderr
eprintln!("[{:<10}] {:<15}: {}", actor, action, human);
// → "[job-exec  ] execute        : Executing job..."

// narration-core: Raw SSE event
sse_sink::send(NarrationEvent { actor, action, human, ... });

// queen-rbee consumer: Must format manually (TEAM-197 fixed this)
let formatted = format!("[{:<10}] {:<15}: {}", event.actor, event.action, event.human);
```

---

## Architecture Analysis: Where Can Daemons Output?

### CLI Tools (rbee-keeper)
✅ **Can use stdout/stderr directly**
- User is running the tool interactively
- Output goes to their terminal
- This is the ONLY component that can println!() directly

### Daemons (queen, hive, worker)
❌ **CANNOT use stdout/stderr for user-facing output**
- Running in background (systemd, tmux, etc.)
- User cannot see stdout/stderr
- Logs go to files, not terminals

✅ **MUST use SSE for user-facing narration**
- Web UI connects via SSE
- Keeper CLI connects via SSE → then prints to stdout
- This is the ONLY way to make narration "web-UI proof"

---

## Requirements from User

> "the narration is going to need to be web-ui proof. meaning that ALL the queens narration output MUST go through the jobs sse channel."

> "if the beehive would do stdout. or the worker does. and the hive and worker are not in the same system as the bee keeper. then we don't see the narration."

### Interpretation:

1. **Queen:** ALL narration → jobs SSE channel ✅ (already doing this)
2. **Hive:** ALL narration → send to queen → jobs SSE channel ❌ (not yet built)
3. **Worker:** ALL narration → send to queen → jobs SSE channel ⚠️ (partial, for inference only)
4. **Keeper:** Can stdout ✅ (CLI tool, local to user)

---

## Solution: Centralized Formatting with Pre-Formatted SSE

### Design Principle

**SSE events should carry pre-formatted text, not raw structs.**

### Why This Works

1. **Single formatting location:** Only narration-core formats
2. **No consumer duplication:** Consumers just print what they receive
3. **Consistent everywhere:** Format changes update all consumers automatically
4. **Simple API:** Consumers don't need to know formatting rules

### Implementation Strategy

#### Option A: Add `formatted` field to NarrationEvent (RECOMMENDED)

```rust
// narration-core/src/sse_sink.rs
#[derive(Debug, Clone, serde::Serialize)]
pub struct NarrationEvent {
    pub actor: String,
    pub action: String,
    pub target: String,
    pub human: String,
    
    // NEW: Pre-formatted text (matches stderr output)
    pub formatted: String,  // "[job-exec  ] execute        : Executing job..."
    
    // Optional fields for programmatic access
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cute: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub story: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub correlation_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub job_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub emitted_by: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub emitted_at_ms: Option<u64>,
}

impl From<NarrationFields> for NarrationEvent {
    fn from(fields: NarrationFields) -> Self {
        // Apply redaction
        let human = redact_secrets(&fields.human, RedactionPolicy::default());
        
        // PRE-FORMAT the text (same as stderr)
        let formatted = format!("[{:<10}] {:<15}: {}", fields.actor, fields.action, human);
        
        Self {
            actor: fields.actor.to_string(),
            action: fields.action.to_string(),
            target: fields.target,
            human: human.to_string(),
            formatted,  // ← Pre-formatted text
            cute: fields.cute,
            story: fields.story,
            correlation_id: fields.correlation_id,
            job_id: fields.job_id,
            emitted_by: fields.emitted_by,
            emitted_at_ms: fields.emitted_at_ms,
        }
    }
}
```

#### Consumers: Just Use the Formatted Field

```rust
// queen-rbee /http/jobs.rs
match result {
    Ok(event) => {
        // TEAM-198: Just use pre-formatted text (NO manual formatting!)
        yield Ok(Event::default().data(&event.formatted));
    }
    // ...
}
```

```rust
// keeper /job_client.rs
if let Some(data) = line.strip_prefix("data: ") {
    // TEAM-198: Just print the pre-formatted text
    println!("{}", data);
}
```

#### Benefits

- ✅ **Zero duplication:** Formatting happens once in narration-core
- ✅ **Consistent:** stderr and SSE always match
- ✅ **Forward compatible:** Format changes don't break consumers
- ✅ **Backward compatible:** Old consumers can still use actor/action/human fields
- ✅ **Simple:** Consumers don't need to know formatting rules

#### Option B: Send only formatted string (MORE RADICAL)

```rust
// narration-core/src/sse_sink.rs
#[derive(Debug, Clone, serde::Serialize)]
pub struct NarrationEvent {
    // Only send the formatted text
    pub formatted: String,
    
    // Optional metadata for filtering/routing
    #[serde(skip_serializing_if = "Option::is_none")]
    pub correlation_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub job_id: Option<String>,
}
```

**Pros:**
- Even simpler
- Forces everyone to use the same format

**Cons:**
- Less flexible (can't programmatically filter by actor/action)
- Breaks any existing code that uses actor/action fields

**Verdict:** Use Option A (add `formatted` field but keep existing fields)

---

## Hive/Worker Narration: The Missing Piece

### Current State

**Hive:** Uses `println!()` for output (not narration at all)
```rust
// bin/20_rbee_hive/src/main.rs
println!("🐝 rbee-hive starting on port {}", args.port);
println!("💓 Heartbeat task started (5s interval)");
```

**Worker:** Has dual-output narration (stderr + thread-local SSE)
```rust
// bin/30_llm_worker_rbee/src/narration.rs
pub fn narrate_dual(fields: NarrationFields) {
    // 1. stderr (for daemon logs)
    observability_narration_core::narrate(fields.clone());
    
    // 2. Thread-local SSE channel (for inference requests)
    narration_channel::send_narration(sse_event);
}
```

### The Problem

When hive/worker run on **remote machines:**
```
┌──────────────┐         ┌──────────────┐         ┌──────────────┐
│   keeper     │ ─SSH─→  │     hive     │ ───→    │    worker    │
│  (local)     │         │  (remote-1)  │         │  (remote-1)  │
└──────────────┘         └──────────────┘         └──────────────┘
      ↑                         │                        │
      │                    println!() ✗              println!() ✗
      │                    (not visible)            (not visible)
      │
      └─── NEEDS SSE ←─────────────────────────────────────────┘
```

**User cannot see hive/worker stdout** because they're on remote machines!

### The Solution: All Narration Flows to Queen

```
┌─────────────────────────────────────────────────────────────────┐
│ COMPLETE FLOW (TEAM-198 SOLUTION)                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. HIVE narrates                                              │
│     ├─ NARRATE.action().emit()                                 │
│     ├─ → narration-core (stderr for daemon logs)              │
│     └─ → POST to queen: /v1/narration (new endpoint)          │
│          {                                                      │
│            "formatted": "[hive      ] spawn          : ..."    │
│            "job_id": "job-xyz",                                │
│            "correlation_id": "..."                             │
│          }                                                      │
│                                                                 │
│  2. WORKER narrates                                            │
│     ├─ narrate_dual() → stderr + thread-local channel         │
│     └─ → Worker's SSE stream → Queen (for inference)          │
│          OR                                                     │
│          → POST to queen: /v1/narration (for lifecycle)       │
│                                                                 │
│  3. QUEEN receives and broadcasts                              │
│     ├─ Receives from hive/worker via POST /v1/narration       │
│     ├─ OR receives from own job execution                      │
│     ├─ Broadcasts via global SSE broadcaster                   │
│     └─ → Sent to keeper via job SSE stream                     │
│                                                                 │
│  4. KEEPER receives and displays                               │
│     ├─ Subscribes to job SSE stream                            │
│     ├─ Receives pre-formatted text                             │
│     └─ println!() to stdout (user's terminal)                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Implementation Plan

### Phase 1: Fix narration-core (Centralized Formatting)

**File:** `bin/99_shared_crates/narration-core/src/sse_sink.rs`

1. Add `formatted: String` field to `NarrationEvent`
2. Update `From<NarrationFields>` to pre-format text
3. Apply redaction before formatting

**File:** `bin/99_shared_crates/narration-core/src/lib.rs`

1. Extract format logic into helper function:
   ```rust
   pub fn format_narration(actor: &str, action: &str, human: &str) -> String {
       format!("[{:<10}] {:<15}: {}", actor, action, human)
   }
   ```
2. Use in both `narrate_at_level()` (stderr) and `NarrationEvent::from()` (SSE)

### Phase 2: Update queen-rbee SSE consumer

**File:** `bin/10_queen_rbee/src/http/jobs.rs`

1. Remove manual formatting (line 107-109)
2. Use `event.formatted` instead:
   ```rust
   Ok(event) => {
       // TEAM-198: Use pre-formatted text from narration-core
       yield Ok(Event::default().data(&event.formatted));
   }
   ```

### Phase 3: Add queen narration ingestion endpoint (NEW)

**File:** `bin/10_queen_rbee/src/http/narration.rs` (NEW)

```rust
//! Narration ingestion endpoint
//! 
//! TEAM-198: Allows hive/worker to send narration events to queen
//! for broadcast via job SSE streams

use axum::{extract::State, http::StatusCode, Json};
use observability_narration_core::sse_sink;

#[derive(serde::Deserialize)]
pub struct NarrationIngestionRequest {
    pub formatted: String,
    pub job_id: Option<String>,
    pub correlation_id: Option<String>,
}

/// POST /v1/narration - Ingest narration from hive/worker
pub async fn handle_ingest_narration(
    Json(req): Json<NarrationIngestionRequest>,
) -> Result<StatusCode, (StatusCode, String)> {
    // Broadcast to global SSE sink
    // (This is a simplified event - could expand if needed)
    sse_sink::send_formatted(&req.formatted, req.job_id, req.correlation_id);
    
    Ok(StatusCode::OK)
}
```

**File:** `bin/99_shared_crates/narration-core/src/sse_sink.rs`

Add helper for ingesting pre-formatted narration:
```rust
/// Send a pre-formatted narration string to SSE subscribers
/// 
/// TEAM-198: Used by queen ingestion endpoint to broadcast
/// narration from hive/worker
pub fn send_formatted(formatted: &str, job_id: Option<String>, correlation_id: Option<String>) {
    SSE_BROADCASTER.send(NarrationEvent {
        formatted: formatted.to_string(),
        job_id,
        correlation_id,
        // Other fields can be empty for ingested events
        actor: String::new(),
        action: String::new(),
        target: String::new(),
        human: String::new(),
        cute: None,
        story: None,
        emitted_by: None,
        emitted_at_ms: None,
    });
}
```

### Phase 4: Update hive to send narration to queen

**File:** `bin/20_rbee_hive/src/narration.rs` (NEW)

```rust
//! Hive narration with queen forwarding
//!
//! TEAM-198: Dual-output narration for hive
//! - stderr for daemon logs
//! - HTTP POST to queen for user visibility

use observability_narration_core::{NarrationFields, NarrationFactory};
use reqwest::Client;

pub const NARRATE: NarrationFactory = NarrationFactory::new("hive");

/// Send narration to queen for broadcast
/// 
/// TEAM-198: Allows keeper to see hive narration even on remote machines
pub async fn narrate_to_queen(
    client: &Client,
    queen_url: &str,
    fields: NarrationFields,
) {
    // 1. Always emit to stderr (for daemon logs)
    observability_narration_core::narrate(fields.clone());
    
    // 2. Format and send to queen
    let formatted = observability_narration_core::format_narration(
        fields.actor,
        fields.action,
        &fields.human,
    );
    
    let payload = serde_json::json!({
        "formatted": formatted,
        "job_id": fields.job_id,
        "correlation_id": fields.correlation_id,
    });
    
    // Fire and forget (don't block on queen response)
    let _ = client
        .post(format!("{}/v1/narration", queen_url))
        .json(&payload)
        .send()
        .await;
}
```

**File:** `bin/20_rbee_hive/src/main.rs`

Replace `println!()` with narration:
```rust
// BEFORE:
println!("🐝 rbee-hive starting on port {}", args.port);

// AFTER:
NARRATE
    .action("startup")
    .context(&args.port.to_string())
    .human("🐝 Starting on port {}")
    .emit();
```

### Phase 5: Update worker lifecycle narration

**File:** `bin/30_llm_worker_rbee/src/narration.rs`

Add queen forwarding option:
```rust
/// Narrate with optional queen forwarding
/// 
/// TEAM-198: For lifecycle events (not inference), send to queen
pub async fn narrate_with_queen(
    fields: NarrationFields,
    queen_url: Option<&str>,
) {
    // 1. Always emit to stderr (for daemon logs)
    observability_narration_core::narrate(fields.clone());
    
    // 2. If queen URL provided, send to queen
    if let Some(url) = queen_url {
        let formatted = observability_narration_core::format_narration(
            fields.actor,
            fields.action,
            &fields.human,
        );
        
        let client = reqwest::Client::new();
        let payload = serde_json::json!({
            "formatted": formatted,
            "job_id": fields.job_id,
            "correlation_id": fields.correlation_id,
        });
        
        // Fire and forget
        let _ = client
            .post(format!("{}/v1/narration", url))
            .json(&payload)
            .send()
            .await;
    }
}
```

**Note:** Inference narration (via `narrate_dual()`) already flows through SSE streams. Only lifecycle events need queen forwarding.

---

## API Design: Simple and Ergonomic

### For Keeper (CLI)

```rust
// No change needed - already simple!
NARRATE
    .action("job_submit")
    .context(job_id)
    .human("📋 Job {} submitted")
    .emit();
```

### For Queen (Daemon)

```rust
// No change needed - SSE broadcaster handles it automatically
NARRATE
    .action("job_create")
    .context(&job_id)
    .human("Job {} created")
    .emit();
```

### For Hive (Daemon - Remote)

```rust
// Need queen URL for forwarding
let client = reqwest::Client::new();
let queen_url = "http://queen:8500";

// Lifecycle narration
narrate_to_queen(
    &client,
    queen_url,
    NARRATE
        .action("spawn")
        .context("worker-1")
        .human("Spawning worker {}")
        .build()
).await;
```

### For Worker (Daemon - Remote)

```rust
// Inference narration: Use existing narrate_dual() (thread-local SSE)
narrate_dual(NarrationFields {
    actor: ACTOR_CANDLE_BACKEND,
    action: ACTION_INFERENCE_START,
    // ...
});

// Lifecycle narration: Send to queen
narrate_with_queen(
    NarrationFields {
        actor: ACTOR_LLM_WORKER_RBEE,
        action: ACTION_STARTUP,
        // ...
    },
    Some(queen_url)
).await;
```

---

## Verification Checklist

### Phase 1: Centralized Formatting
- [ ] Add `formatted` field to `NarrationEvent`
- [ ] Extract `format_narration()` helper
- [ ] Pre-format in `NarrationEvent::from()`
- [ ] Verify stderr and SSE match exactly

### Phase 2: Update Queen Consumer
- [ ] Remove manual formatting in `jobs.rs`
- [ ] Use `event.formatted`
- [ ] Test with keeper: `./rbee hive status`
- [ ] Verify output format is correct

### Phase 3: Queen Ingestion Endpoint
- [ ] Add `POST /v1/narration` endpoint
- [ ] Add route to main router
- [ ] Test with curl:
  ```bash
  curl -X POST http://localhost:8500/v1/narration \
    -H "Content-Type: application/json" \
    -d '{"formatted": "[test      ] action         : Test message"}'
  ```
- [ ] Verify message appears in keeper SSE stream

### Phase 4: Hive Narration
- [ ] Add `narration.rs` module to hive
- [ ] Replace all `println!()` with `NARRATE.action().emit()`
- [ ] Add `narrate_to_queen()` wrapper
- [ ] Test with remote hive
- [ ] Verify keeper sees hive narration

### Phase 5: Worker Narration
- [ ] Add `narrate_with_queen()` for lifecycle events
- [ ] Keep `narrate_dual()` for inference (already works)
- [ ] Test with remote worker
- [ ] Verify keeper sees worker lifecycle narration
- [ ] Verify inference narration still works

### Integration Test
- [ ] Start queen on localhost
- [ ] Start hive on "remote" (different port/container)
- [ ] Start worker via hive
- [ ] Run keeper command: `./rbee hive status`
- [ ] Verify ALL narration visible in keeper:
  - Keeper's own narration
  - Queen's job routing narration
  - Hive's lifecycle narration
  - Worker's lifecycle narration
- [ ] Run inference: `./rbee infer "hello"`
- [ ] Verify inference narration visible in keeper

---

## Benefits of This Architecture

### 1. Web-UI Proof ✅
All daemon narration flows through queen's SSE, accessible to web UI

### 2. Simple API ✅
Developers call `.emit()`, system handles routing automatically

### 3. Consistent Formatting ✅
One place formats, everyone else just displays

### 4. No Manual Work ✅
Consumers don't format, don't parse, just print

### 5. Forward Compatible ✅
Format changes in narration-core update everyone automatically

### 6. Backward Compatible ✅
Existing `actor`/`action`/`human` fields still available for filtering

### 7. Remote-Friendly ✅
Hive/worker on remote machines send narration to queen → keeper sees it

---

## Files to Create/Modify

### Create New Files
- `bin/10_queen_rbee/src/http/narration.rs` - Ingestion endpoint
- `bin/20_rbee_hive/src/narration.rs` - Hive narration helpers

### Modify Existing Files
- `bin/99_shared_crates/narration-core/src/sse_sink.rs` - Add `formatted` field
- `bin/99_shared_crates/narration-core/src/lib.rs` - Extract format helper
- `bin/10_queen_rbee/src/http/jobs.rs` - Use `event.formatted`
- `bin/10_queen_rbee/src/main.rs` - Add narration route
- `bin/20_rbee_hive/src/main.rs` - Use narration instead of println!
- `bin/30_llm_worker_rbee/src/narration.rs` - Add queen forwarding

---

## Code Size Impact

**Additions:**
- narration-core: +15 lines (format helper + formatted field)
- queen ingestion endpoint: +30 lines
- hive narration module: +40 lines
- worker queen forwarding: +30 lines

**Removals:**
- queen manual formatting: -3 lines

**Net Change:** ~+112 lines for complete web-UI proof narration

---

## Alternative Considered: Narration Service

**Rejected Approach:** Dedicated narration aggregation service

**Why Rejected:**
- ❌ Over-engineering for current scale
- ❌ Extra complexity (another service to manage)
- ❌ More network hops
- ❌ Not needed when queen already aggregates

**Current Approach:** Queen as natural aggregation point
- ✅ Queen already manages jobs
- ✅ Queen already has SSE infrastructure
- ✅ Queen is already central hub
- ✅ Simple POST endpoint, no new service

---

## Summary for Next Team

### What Works Now (TEAM-197)
- ✅ Keeper can see queen's own narration via SSE
- ✅ Queen broadcasts narration via global SSE broadcaster
- ✅ Format is consistent (fixed by TEAM-197)

### What's Missing (TEAM-198 Must Build)
- ❌ Centralized formatting in narration-core
- ❌ Hive narration doesn't go through SSE
- ❌ Worker lifecycle narration doesn't go through queen
- ❌ Queen has no ingestion endpoint for remote narration

### Implementation Order
1. **Phase 1** (narration-core) - Foundation, no breaking changes
2. **Phase 2** (queen consumer) - Simplifies existing code
3. **Phase 3** (queen endpoint) - New capability, doesn't break anything
4. **Phase 4** (hive) - Makes hive narration visible
5. **Phase 5** (worker) - Makes worker lifecycle narration visible

### Testing Strategy
- Unit test: `format_narration()` helper
- Integration test: Curl POST to `/v1/narration`
- E2E test: Remote hive/worker narration visible in keeper
- Regression test: Existing job SSE streams still work

---

**TEAM-198** | **Status:** READY FOR IMPLEMENTATION

**Key Insight:** Centralized formatting + queen as aggregation point = simple, ergonomic, web-UI proof narration!
