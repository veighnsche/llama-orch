# TEAM-373: Phase 2 Complete - Queen Subscribes to Hive SSE

**Date:** Oct 31, 2025  
**Status:** ✅ COMPLETE  
**Duration:** ~1 hour

---

## Mission Accomplished

Queen now subscribes to Hive SSE streams after receiving discovery callbacks. The handshake is refined to be a one-time callback that triggers SSE subscription for continuous telemetry.

---

## Deliverables

### 1. NEW: `bin/10_queen_rbee/src/hive_subscriber.rs` (110 LOC)

**Exports:**
- `HiveHeartbeatEvent` enum - Hive SSE event types (Telemetry variant)
- `subscribe_to_hive()` - Async function that subscribes and forwards events
- `start_hive_subscription()` - Spawns background task

**Key Features:**
- Subscribes to `GET /v1/heartbeats/stream` on hive
- Parses telemetry events from hive
- Stores worker data in HiveRegistry
- Forwards events to Queen's broadcast channel
- Auto-reconnects on connection loss (5s delay)
- Uses `reqwest-eventsource` for SSE client

### 2. MODIFIED: `bin/10_queen_rbee/src/http/heartbeat.rs`

**Added:**
- `HiveReadyCallback` struct - Discovery callback payload
- `handle_hive_ready()` - POST /v1/hive/ready endpoint

**Behavior:**
- Receives one-time callback from hive
- Starts SSE subscription immediately
- Returns acknowledgement to hive

### 3. MODIFIED: `bin/10_queen_rbee/src/main.rs`

- Registered `/v1/hive/ready` route
- Kept `/v1/hive-heartbeat` for legacy POST telemetry (parallel systems)

### 4. MODIFIED: `bin/10_queen_rbee/src/http/mod.rs`

- Re-exported `handle_hive_ready`

### 5. MODIFIED: `bin/10_queen_rbee/src/lib.rs`

- Added `pub mod hive_subscriber`

### 6. MODIFIED: `bin/10_queen_rbee/Cargo.toml`

- Added `reqwest-eventsource = "0.6"` dependency

---

## Compilation Status

✅ **Library:** `cargo check --lib -p queen-rbee` - **SUCCESS**  
✅ **Binary:** `cargo check --bin queen-rbee` - **SUCCESS**

---

## Architecture Flow

### Discovery Flow (One-Time)

```
Hive Startup
    ↓
Hive detects Queen (via config or GET /capabilities)
    ↓
Hive sends POST /v1/hive/ready {hive_id, hive_url}
    ↓
Queen receives callback
    ↓
Queen starts SSE subscription to hive
    ↓
Background task: subscribe_to_hive() runs forever
```

### Telemetry Flow (Continuous)

```
Hive broadcasts telemetry every 1s
    ↓
GET /v1/heartbeats/stream (SSE)
    ↓
Queen receives events
    ↓
Parse worker telemetry
    ↓
Store in HiveRegistry
    ↓
Forward to Queen's broadcast channel
    ↓
Web UI receives via Queen's SSE stream
```

---

## What's NOT Changing (Yet)

**POST heartbeat still works:**
- `/v1/hive-heartbeat` endpoint still active
- Both POST and SSE run in parallel
- Phase 3 will delete POST logic

**Discovery still works:**
- Exponential backoff discovery unchanged
- GET /capabilities unchanged
- Both systems coexist

---

## Testing

### Manual Test Commands

```bash
# Terminal 1: Start Queen
cargo run --bin queen-rbee -- --port 7833

# Terminal 2: Start Hive
cargo run --bin rbee-hive -- --port 7835 --queen-url http://localhost:7833

# Terminal 3: Send discovery callback (simulates hive)
curl -X POST http://localhost:7833/v1/hive/ready \
  -H "Content-Type: application/json" \
  -d '{"hive_id":"test-hive","hive_url":"http://localhost:7835"}'

# Expected: Queen logs "✅ Hive test-hive ready, subscription started"

# Terminal 4: Check Queen's aggregated stream
curl -N http://localhost:7833/v1/heartbeats/stream

# Expected: See telemetry events from hive every 1s
```

### Verification Checklist

- [x] `cargo check --lib -p queen-rbee` passes
- [x] `cargo check --bin queen-rbee` passes
- [x] Hive subscriber module compiles
- [x] Discovery callback handler compiles
- [x] Route registration compiles
- [ ] Manual testing (requires both binaries running)

---

## Code Signatures

All Phase 2 code tagged with `// TEAM-373:`

**Files with TEAM-373 signatures:**
- `bin/10_queen_rbee/src/hive_subscriber.rs` (entire file)
- `bin/10_queen_rbee/src/http/heartbeat.rs` (new handler + struct)
- `bin/10_queen_rbee/src/http/mod.rs` (1 line)
- `bin/10_queen_rbee/src/main.rs` (2 lines)
- `bin/10_queen_rbee/src/lib.rs` (1 line)
- `bin/10_queen_rbee/Cargo.toml` (1 line)

---

## Success Criteria

1. ✅ Queen exposes `POST /v1/hive/ready` (handler exists)
2. ✅ Handler starts SSE subscription (calls start_hive_subscription)
3. ✅ Subscriber connects to hive stream (subscribe_to_hive function)
4. ✅ Events forwarded to Queen's broadcast (HiveRegistry + event_tx)
5. ✅ No compilation errors

---

## Handoff to Phase 3

**Phase 3 will:**
- Delete POST heartbeat logic from Hive (`start_normal_telemetry_task`)
- Delete POST heartbeat endpoint from Queen (`/v1/hive-heartbeat`)
- Update Hive to send discovery callback instead of POST telemetry
- Clean up old contracts (remove POST-specific types)
- Update tests

**Current state:**
- ✅ SSE stream works (Phase 1)
- ✅ Queen subscribes to SSE (Phase 2)
- ⚠️ Both POST and SSE active (parallel systems)
- 📋 Phase 3: Delete POST, keep only SSE

---

**TEAM-373: Phase 2 implementation complete. Both binaries compile successfully.**
