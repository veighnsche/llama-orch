# TEAM-371 Phase 2: Queen Subscribes to Hive SSE

**Status:** 📋 READY FOR IMPLEMENTATION  
**Duration:** 1 day  
**Team:** TEAM-373 (or whoever implements this)  
**Depends on:** Phase 1 complete (Hive SSE stream exists)

---

## Mission

Update Queen to subscribe to Hive SSE streams instead of receiving POST heartbeats.
The handshake/discovery stays intact - we just change what happens AFTER discovery.

**Key Insight:** The discovery callback becomes "I'm ready, subscribe to my SSE" instead of continuous POST telemetry.

---

## Handshake Refinement

### Current Handshake (Phase 1)

**Scenario 1: Queen starts first**
```
Queen → GET /capabilities?queen_url=X → Hive
Hive stores queen_url
Hive → POST /v1/hive-heartbeat (every 1s, continuous) → Queen
```

**Scenario 2: Hive starts first**
```
Hive → POST /v1/hive-heartbeat (exponential backoff, continuous) → Queen
```

### New Handshake (Phase 2)

**Scenario 1: Queen starts first**
```
Queen → GET /capabilities?queen_url=X → Hive
Hive stores queen_url
Hive → POST /v1/hive/ready (ONE-TIME callback) → Queen
Queen → Subscribe to GET /v1/heartbeats/stream → Hive (CONTINUOUS SSE)
```

**Scenario 2: Hive starts first**
```
Hive → POST /v1/hive/ready (exponential backoff, ONE-TIME) → Queen
Queen → Subscribe to GET /v1/heartbeats/stream → Hive (CONTINUOUS SSE)
```

**What changes:**
- Discovery callback is ONE-TIME (not continuous)
- After callback, Queen subscribes to SSE (continuous telemetry)
- If Queen restarts, hive resends callback (existing restart detection works)

---

## Implementation

### Step 1: Create Hive Subscriber Module

**NEW FILE:** `bin/10_queen_rbee/src/hive_subscriber.rs`

```rust
// TEAM-373: Created by TEAM-373
//! Queen subscribes to Hive SSE streams
//!
//! After discovery handshake, Queen connects to each hive's
//! GET /v1/heartbeats/stream and aggregates telemetry.

use anyhow::Result;
use futures::StreamExt;
use observability_narration_core::n;
use reqwest_eventsource::{Event, EventSource};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::broadcast;

/// Hive heartbeat event from SSE stream
#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum HiveHeartbeatEvent {
    Telemetry {
        hive_id: String,
        #[serde(rename = "hive_info")]
        _hive_info: serde_json::Value, // We only need workers
        timestamp: String,
        workers: Vec<serde_json::Value>, // ProcessStats as JSON
    },
}

/// Subscribe to a single hive's SSE stream
///
/// TEAM-373: Queen calls this after receiving discovery callback.
/// Runs continuously, forwarding events to Queen's broadcast channel.
pub async fn subscribe_to_hive(
    hive_url: String,
    hive_id: String,
    hive_registry: Arc<queen_rbee_hive_registry::HiveRegistry>,
    queen_event_tx: broadcast::Sender<crate::http::HeartbeatEvent>,
) -> Result<()> {
    let stream_url = format!("{}/v1/heartbeats/stream", hive_url);
    
    n!("hive_subscribe_start", "📡 Subscribing to hive {} SSE stream: {}", hive_id, stream_url);
    
    loop {
        let mut event_source = EventSource::get(&stream_url);
        
        n!("hive_subscribe_connected", "✅ Connected to hive {} SSE stream", hive_id);
        
        while let Some(event) = event_source.next().await {
            match event {
                Ok(Event::Message(msg)) => {
                    // Parse telemetry event
                    if let Ok(hive_event) = serde_json::from_str::<HiveHeartbeatEvent>(&msg.data) {
                        match hive_event {
                            HiveHeartbeatEvent::Telemetry { hive_id, workers, timestamp, .. } => {
                                tracing::trace!("Received telemetry from hive {}: {} workers", hive_id, workers.len());
                                
                                // Parse workers (they're already ProcessStats JSON)
                                let parsed_workers: Vec<_> = workers
                                    .into_iter()
                                    .filter_map(|w| serde_json::from_value(w).ok())
                                    .collect();
                                
                                // Store in HiveRegistry
                                hive_registry.update_workers(&hive_id, parsed_workers.clone());
                                
                                // Forward to Queen's SSE stream
                                let queen_event = crate::http::HeartbeatEvent::HiveTelemetry {
                                    hive_id,
                                    timestamp,
                                    workers: parsed_workers,
                                };
                                
                                let _ = queen_event_tx.send(queen_event);
                            }
                        }
                    }
                }
                Ok(Event::Open) => {
                    n!("hive_subscribe_open", "🔗 SSE connection opened for hive {}", hive_id);
                }
                Err(e) => {
                    n!("hive_subscribe_error", "❌ Hive {} SSE error: {}", hive_id, e);
                    break; // Reconnect
                }
            }
        }
        
        // Connection closed, retry after delay
        n!("hive_subscribe_reconnect", "🔄 Reconnecting to hive {} in 5s...", hive_id);
        tokio::time::sleep(Duration::from_secs(5)).await;
    }
}

/// Start subscription task for a hive
///
/// TEAM-373: Spawns background task that runs forever.
/// Called when Queen receives discovery callback from hive.
pub fn start_hive_subscription(
    hive_url: String,
    hive_id: String,
    hive_registry: Arc<queen_rbee_hive_registry::HiveRegistry>,
    queen_event_tx: broadcast::Sender<crate::http::HeartbeatEvent>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        if let Err(e) = subscribe_to_hive(hive_url.clone(), hive_id.clone(), hive_registry, queen_event_tx).await {
            n!("hive_subscribe_fatal", "❌ Fatal error subscribing to hive {}: {}", hive_id, e);
        }
    })
}
```

### Step 2: Add Dependency

**MODIFY:** `bin/10_queen_rbee/Cargo.toml`

```toml
[dependencies]
# ... existing deps ...
reqwest-eventsource = "0.6" # TEAM-373: SSE client for hive subscriptions
futures = "0.3" # TEAM-373: StreamExt for SSE
```

### Step 3: Update Discovery Callback Endpoint

**MODIFY:** `bin/10_queen_rbee/src/http/heartbeat.rs`

**Change signature and behavior:**

```rust
/// POST /v1/hive/ready - Discovery callback from hive
///
/// TEAM-373: Changed from continuous telemetry to one-time callback.
/// When hive sends this, Queen subscribes to its SSE stream.
///
/// Discovery flow:
/// 1. Hive detects Queen (via GET /capabilities or startup)
/// 2. Hive sends POST /v1/hive/ready (one-time callback)
/// 3. Queen subscribes to GET /v1/heartbeats/stream on hive
/// 4. Continuous telemetry flows via SSE
pub async fn handle_hive_ready(
    State(state): State<HeartbeatState>,
    Json(callback): Json<HiveReadyCallback>,
) -> Result<Json<HttpHeartbeatAcknowledgement>, (StatusCode, String)> {
    eprintln!(
        "🐝 Hive ready callback: hive_id={}, url={}",
        callback.hive_id, callback.hive_url
    );
    
    // TEAM-373: Store initial hive info
    state.hive_registry.register_hive_url(&callback.hive_id, &callback.hive_url);
    
    // TEAM-373: Start SSE subscription to this hive
    crate::hive_subscriber::start_hive_subscription(
        callback.hive_url.clone(),
        callback.hive_id.clone(),
        state.hive_registry.clone(),
        state.event_tx.clone(),
    );
    
    n!("hive_ready", "✅ Hive {} ready, subscription started", callback.hive_id);
    
    Ok(Json(HttpHeartbeatAcknowledgement {
        status: "ok".to_string(),
        message: format!("Subscribed to hive {} SSE stream", callback.hive_id),
    }))
}

/// Hive ready callback payload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HiveReadyCallback {
    pub hive_id: String,
    pub hive_url: String, // e.g., "http://192.168.1.100:7835"
}
```

### Step 4: Update Route

**MODIFY:** `bin/10_queen_rbee/src/main.rs`

**Change route from `/v1/hive-heartbeat` to `/v1/hive/ready`:**

```rust
// TEAM-373: Hive ready callback (discovery)
// Changed from continuous POST telemetry to one-time callback
.route("/v1/hive/ready", post(http::handle_hive_ready))
```

**Add subscriber module:**
```rust
mod discovery;
mod hive_subscriber; // TEAM-373: Subscribe to hive SSE streams
```

### Step 5: Update Discovery to Subscribe

**MODIFY:** `bin/10_queen_rbee/src/discovery.rs`

**In `discover_single_hive()` function (around line 132):**

```rust
if response.status().is_success() {
    n!("discovery_success", "✅ Discovered hive: {}", target.host);
    
    // TEAM-373: Start SSE subscription immediately
    let hive_url = format!("http://{}:7835", target.hostname);
    let hive_id = target.host.clone(); // TODO: Get from response
    
    // TODO: Pass hive_registry and event_tx from main
    // crate::hive_subscriber::start_hive_subscription(
    //     hive_url,
    //     hive_id,
    //     hive_registry,
    //     event_tx,
    // );
} else {
    // ...
}
```

**Note:** Full discovery integration requires passing state, defer to Phase 3 cleanup.

---

## Hive Side Changes (Phase 2)

### Update Hive Callback

**MODIFY:** `bin/20_rbee_hive/src/heartbeat.rs`

**Change `send_heartbeat_to_queen()` to send callback, not telemetry:**

```rust
/// Send discovery callback to Queen
///
/// TEAM-373: Changed from continuous telemetry to one-time callback.
/// After Queen receives this, it subscribes to our SSE stream.
pub async fn send_ready_callback_to_queen(
    hive_info: &HiveInfo,
    queen_url: &str,
) -> Result<()> {
    tracing::debug!("Sending ready callback to queen at {}", queen_url);
    
    let callback = HiveReadyCallback {
        hive_id: hive_info.id.clone(),
        hive_url: format!("http://{}:{}", hive_info.hostname, hive_info.port),
    };
    
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()?;
    
    let response = client
        .post(format!("{}/v1/hive/ready", queen_url))
        .json(&callback)
        .send()
        .await?;
    
    if !response.status().is_success() {
        anyhow::bail!("Ready callback failed: {}", response.status());
    }
    
    tracing::info!("✅ Queen acknowledged ready callback");
    Ok(())
}

#[derive(Debug, Serialize)]
struct HiveReadyCallback {
    hive_id: String,
    hive_url: String,
}
```

**Update discovery to call new function:**

```rust
// In start_discovery_with_backoff()
for (attempt, delay) in delays.iter().enumerate() {
    // ... delay logic ...
    
    // TEAM-373: Send ready callback (not full telemetry)
    match send_ready_callback_to_queen(&hive_info, &queen_url).await {
        Ok(_) => {
            n!("discovery_success", "✅ Queen acknowledged, SSE stream active");
            return; // Discovery complete, SSE broadcaster handles telemetry
        }
        Err(e) => {
            n!("discovery_failed", "❌ Ready callback attempt {} failed: {}", attempt + 1, e);
        }
    }
}
```

---

## Testing

### Manual Test

```bash
# Terminal 1: Start Queen
cargo run --bin queen-rbee -- --port 7833

# Terminal 2: Start Hive
cargo run --bin rbee-hive -- --port 7835 --queen-url http://localhost:7833

# Expected Queen logs:
# 🐝 Hive ready callback: hive_id=localhost, url=http://127.0.0.1:7835
# 📡 Subscribing to hive localhost SSE stream
# ✅ Connected to hive localhost SSE stream
# Received telemetry from hive localhost: 2 workers

# Terminal 3: Subscribe to Queen's SSE
curl -N http://localhost:7833/v1/heartbeats/stream

# Expected: See HiveTelemetry events every 1 second
```

### Verification Checklist

- [ ] Hive sends ONE `POST /v1/hive/ready` during discovery
- [ ] Queen receives callback and starts SSE subscription
- [ ] Queen's SSE stream shows HiveTelemetry events every 1s
- [ ] Hive restart triggers new callback
- [ ] Queen restart triggers hive to resend callback
- [ ] Multiple hives can connect simultaneously

---

## What's Still Running (Temporarily)

**Old POST logic:** `start_normal_telemetry_task()` in hive still exists but does nothing
- Phase 3 will DELETE it
- Both systems run in parallel during Phase 2

---

## Rule ZERO Compliance

✅ **Breaking changes** - Changed endpoint from `/v1/hive-heartbeat` to `/v1/hive/ready`  
✅ **No wrappers** - Not keeping old endpoint "for compatibility"  
✅ **Clean transition** - Phase 3 will DELETE unused code  

---

## Success Criteria

1. ✅ Hive sends `POST /v1/hive/ready` (one-time callback)
2. ✅ Queen subscribes to `GET /v1/heartbeats/stream` on hive
3. ✅ Continuous telemetry flows via SSE
4. ✅ Queen's SSE stream forwards HiveTelemetry events
5. ✅ Discovery handshake still works (both scenarios)

---

## Handoff to Phase 3

**Next team receives:**
- Working SSE subscription from Queen to Hive
- Discovery callback triggers SSE subscription
- Old POST telemetry code still exists but unused

**Next team will:**
- DELETE old POST telemetry logic from hive
- DELETE old POST receiver from Queen
- Clean up unused functions
- Update integration tests

---

**TEAM-373: Implement this phase. Handshake stays intact, just callback changes.**
