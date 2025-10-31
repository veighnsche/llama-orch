# TEAM-371 Phase 1: Create Hive SSE Stream

**Status:** 📋 READY FOR IMPLEMENTATION  
**Duration:** 1 day  
**Team:** TEAM-372 (or whoever implements this)

---

## Mission

Create SSE heartbeat stream on Hive that broadcasts worker telemetry every 1 second.
This stream will be consumed by Queen (after handshake) and potentially Hive SDK.

**IMPORTANT:** Handshake/discovery logic is PRESERVED. Only the continuous telemetry changes from POST to SSE.

---

## Current State Analysis

**Telemetry collection (KEEP THIS):**
- `bin/20_rbee_hive/src/heartbeat.rs:62` - `rbee_hive_monitor::collect_all_workers()`
- Returns `Vec<ProcessStats>` with worker telemetry
- Collection logic is perfect, reuse it

**Continuous push (CHANGE THIS):**
- `bin/20_rbee_hive/src/heartbeat.rs:196-252` - `start_normal_telemetry_task()`
- Sends POST every 1 second
- This becomes SSE broadcaster instead

**Discovery/handshake (PRESERVE THIS):**
- `bin/20_rbee_hive/src/heartbeat.rs:158-189` - Exponential backoff discovery
- `bin/20_rbee_hive/src/main.rs:280-358` - GET /capabilities endpoint
- **DO NOT TOUCH** - Discovery stays exactly as-is

---

## Implementation

### Step 1: Create SSE Stream Module

**NEW FILE:** `bin/20_rbee_hive/src/http/heartbeat_stream.rs`

```rust
// TEAM-372: Created by TEAM-372
//! Hive SSE heartbeat stream
//!
//! Exposes GET /v1/heartbeats/stream for Queen and Hive SDK to subscribe.
//! Broadcasts worker telemetry every 1 second after discovery completes.

use axum::{
    extract::State,
    response::sse::{Event, KeepAlive, Sse},
};
use futures::stream::Stream;
use hive_contract::HiveInfo;
use rbee_hive_monitor::ProcessStats;
use serde::Serialize;
use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::broadcast;
use tokio::time::interval;

/// Heartbeat event for SSE stream
#[derive(Clone, Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum HiveHeartbeatEvent {
    /// Worker telemetry (sent every 1s)
    Telemetry {
        hive_id: String,
        hive_info: HiveInfo,
        timestamp: String,
        workers: Vec<ProcessStats>,
    },
}

/// State for heartbeat stream
#[derive(Clone)]
pub struct HeartbeatStreamState {
    pub hive_info: HiveInfo,
    pub event_tx: broadcast::Sender<HiveHeartbeatEvent>,
}

/// GET /v1/heartbeats/stream - SSE endpoint for hive telemetry
///
/// TEAM-372: Queen subscribes to this after discovery handshake completes.
/// Broadcasts worker telemetry every 1 second.
pub async fn handle_heartbeat_stream(
    State(state): State<Arc<HeartbeatStreamState>>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    tracing::info!("New SSE client connected to hive heartbeat stream");
    
    let mut event_rx = state.event_tx.subscribe();
    
    let stream = async_stream::stream! {
        loop {
            match event_rx.recv().await {
                Ok(event) => {
                    let json = serde_json::to_string(&event)
                        .unwrap_or_else(|_| "{}".to_string());
                    yield Ok(Event::default().event("heartbeat").data(json));
                }
                Err(e) => {
                    tracing::warn!("SSE broadcast error: {}", e);
                    break;
                }
            }
        }
    };
    
    Sse::new(stream).keep_alive(KeepAlive::default())
}

/// Background task to collect and broadcast telemetry
///
/// TEAM-372: Replaces `start_normal_telemetry_task()` POST loop.
/// Collects worker telemetry every 1s and broadcasts to SSE subscribers.
pub fn start_telemetry_broadcaster(
    hive_info: HiveInfo,
    event_tx: broadcast::Sender<HiveHeartbeatEvent>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut interval = interval(Duration::from_secs(1));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        
        tracing::info!("Telemetry broadcaster started (1s interval)");
        
        loop {
            interval.tick().await;
            
            // TEAM-361: Collect worker telemetry (same as before)
            let workers = rbee_hive_monitor::collect_all_workers()
                .await
                .unwrap_or_else(|e| {
                    tracing::warn!("Failed to collect worker telemetry: {}", e);
                    Vec::new()
                });
            
            tracing::trace!("Collected telemetry for {} workers", workers.len());
            
            // Broadcast to SSE subscribers
            let event = HiveHeartbeatEvent::Telemetry {
                hive_id: hive_info.id.clone(),
                hive_info: hive_info.clone(),
                timestamp: chrono::Utc::now().to_rfc3339(),
                workers,
            };
            
            // Send to all subscribers (if any)
            // Errors are fine - means no subscribers
            let _ = event_tx.send(event);
        }
    })
}
```

### Step 2: Export Module

**MODIFY:** `bin/20_rbee_hive/src/http/mod.rs`

```rust
mod jobs;
pub mod heartbeat_stream; // TEAM-372: SSE heartbeat stream

pub use jobs::{handle_shutdown, jobs}; // Existing exports
pub use heartbeat_stream::{handle_heartbeat_stream, HeartbeatStreamState}; // TEAM-372
```

### Step 3: Register Route and Start Broadcaster

**MODIFY:** `bin/20_rbee_hive/src/main.rs`

**Add imports (near top):**
```rust
use tokio::sync::broadcast; // TEAM-372: For SSE broadcast channel
```

**After HiveState creation (around line 103), add:**

```rust
// TEAM-372: Create broadcast channel for SSE heartbeat stream
// Capacity of 100 events - if clients are slow, old events are dropped
let (heartbeat_tx, _) = broadcast::channel::<http::heartbeat_stream::HiveHeartbeatEvent>(100);

let heartbeat_stream_state = Arc::new(http::heartbeat_stream::HeartbeatStreamState {
    hive_info: hive_info.clone(),
    event_tx: heartbeat_tx.clone(),
});

// TEAM-372: Start telemetry broadcaster (replaces POST loop)
let _broadcaster_handle = http::heartbeat_stream::start_telemetry_broadcaster(
    hive_info.clone(),
    heartbeat_tx,
);

tracing::info!("SSE telemetry broadcaster started");
```

**Register route (in Router::new() block, around line 140):**

```rust
let app = Router::new()
    .route("/health", get(health_check))
    .route("/capabilities", get(get_capabilities))
    .with_state(hive_state.clone())
    // TEAM-372: SSE heartbeat stream (for Queen and Hive SDK)
    .route("/v1/heartbeats/stream", get(http::handle_heartbeat_stream))
    .with_state(heartbeat_stream_state)
    .route("/v1/shutdown", post(http::handle_shutdown))
    .route("/v1/jobs", post(http::jobs::handle_create_job))
    .route("/v1/jobs/{job_id}/stream", get(http::jobs::handle_stream_job))
    .route("/v1/jobs/{job_id}", delete(http::jobs::handle_cancel_job))
    .with_state(job_state);
```

---

## Testing

### Manual Test

```bash
# Terminal 1: Start hive
cargo run --bin rbee-hive -- --port 7835 --queen-url http://localhost:7833

# Terminal 2: Subscribe to SSE stream
curl -N http://localhost:7835/v1/heartbeats/stream

# Expected output (every 1 second):
# event: heartbeat
# data: {"type":"telemetry","hive_id":"localhost","workers":[...]}
```

### Verification Checklist

- [ ] `cargo check --bin rbee-hive` passes
- [ ] SSE endpoint responds on `GET /v1/heartbeats/stream`
- [ ] Events arrive every 1 second
- [ ] Worker telemetry included in events
- [ ] Multiple clients can subscribe simultaneously
- [ ] Disconnecting client doesn't affect other clients

---

## What's NOT Changing (Yet)

**Discovery/handshake:** Still works exactly as before
- Exponential backoff: `bin/20_rbee_hive/src/heartbeat.rs:158-189`
- Capabilities endpoint: `bin/20_rbee_hive/src/main.rs:280-358`
- Queen discovery: `bin/10_queen_rbee/src/discovery.rs`

**POST heartbeat:** Still sends to Queen (both systems run in parallel)
- `start_normal_telemetry_task()` keeps running
- Phase 2 will update Queen to subscribe instead of receive POST
- Phase 3 will delete POST logic

---

## Dependencies

**Add to `bin/20_rbee_hive/Cargo.toml`:**

```toml
[dependencies]
# ... existing deps ...
async-stream = "0.3" # TEAM-372: For SSE stream
```

---

## Rule ZERO Compliance

✅ **No deprecation** - Creating new endpoint alongside existing POST  
✅ **Clean implementation** - SSE stream is self-contained module  
✅ **Breaking later** - Phase 3 will DELETE POST logic (no backwards compat wrapper)  

**Why both systems run in Phase 1:**
- Allows testing SSE stream without breaking Queen
- Queen still expects POST in Phase 1
- Phase 2 updates Queen to use SSE
- Phase 3 deletes POST (compiler will find all call sites)

---

## Success Criteria

1. ✅ Hive exposes `GET /v1/heartbeats/stream`
2. ✅ Stream sends events every 1 second
3. ✅ Events contain worker telemetry
4. ✅ Existing POST heartbeat still works (unchanged)
5. ✅ No compilation errors

---

## Handoff to Phase 2

**Next team receives:**
- Working SSE stream on Hive
- POST heartbeat still active (parallel systems)
- Discovery/handshake unchanged

**Next team will:**
- Update Queen to subscribe to SSE instead of receiving POST
- Modify discovery callback to trigger SSE subscription
- Test both systems working together

---

**TEAM-372: Implement this phase. Do NOT modify handshake/discovery logic.**
