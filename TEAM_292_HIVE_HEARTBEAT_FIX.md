# TEAM-292: Hive Heartbeat Fix

**Date:** Oct 25, 2025  
**Status:** ✅ COMPLETE (Backend) | ⚠️ Needs Frontend Restart

## Problem

Web UI at http://localhost:3002/ showed **0 hives** even though rbee-hive daemon was running on port 9000.

## Root Cause

**TEAM-261 removed hive heartbeat functionality** from `rbee-hive/src/main.rs` (line 51):
```rust
// TEAM-261: Simplified - no hive heartbeat (workers send to queen directly)
```

The `heartbeat` module existed but was never imported or called, so hives never sent heartbeats to queen.

## Solution

### Backend Changes (✅ COMPLETE)

**File:** `bin/20_rbee_hive/src/main.rs`

1. **Added heartbeat module import:**
```rust
mod heartbeat; // TEAM-292: Re-enabled hive heartbeat
```

2. **Added CLI parameters:**
```rust
/// Queen URL for heartbeat reporting
/// TEAM-292: Added to enable hive heartbeat
#[arg(long, default_value = "http://localhost:8500")]
queen_url: String,

/// Hive ID (alias)
/// TEAM-292: Added to identify this hive
#[arg(long, default_value = "localhost")]
hive_id: String,
```

3. **Started heartbeat task after server ready:**
```rust
// TEAM-292: Start heartbeat task to send status to queen
let hive_info = hive_contract::HiveInfo {
    id: args.hive_id.clone(),
    hostname: "127.0.0.1".to_string(),
    port: args.port,
    operational_status: hive_contract::OperationalStatus::Ready,
    health_status: hive_contract::HealthStatus::Healthy,
    version: env!("CARGO_PKG_VERSION").to_string(),
};

let _heartbeat_handle = heartbeat::start_heartbeat_task(hive_info, args.queen_url.clone());
```

### Verification

**Backend heartbeat stream shows correct data:**
```json
{
  "type": "queen",
  "workers_online": 0,
  "workers_available": 0,
  "hives_online": 1,
  "hives_available": 1,
  "worker_ids": [],
  "hive_ids": ["localhost"],
  "timestamp": "2025-10-24T22:09:32.804765189+00:00"
}
```

✅ **Backend is working correctly!**

### Frontend Status (⚠️ Needs Restart)

The WASM SDK was rebuilt successfully:
```bash
cd frontend/packages/rbee-sdk && pnpm build
# ✅ Built successfully
```

However, the Next.js dev server (`turbo dev`) needs to be **restarted** to pick up the new WASM binary.

**Current state:**
- Backend: ✅ Sending heartbeats correctly
- Frontend: ⚠️ Shows "0 hives" (stale WASM)

**To fix:**
```bash
# Stop turbo dev (Ctrl+C)
# Restart it
turbo dev
```

After restart, the web UI should show **1 hive online**.

## Files Changed

- `bin/20_rbee_hive/src/main.rs` (+27 LOC, TEAM-292 signatures)
- `frontend/packages/rbee-sdk/pkg/bundler/*` (rebuilt WASM)

## Running the Fixed Hive

```bash
# Build the updated hive
cargo build --bin rbee-hive

# Stop old hive
pkill -f rbee-hive

# Start with heartbeat enabled
./target/debug/rbee-hive \
  --port 9000 \
  --queen-url http://localhost:8500 \
  --hive-id localhost
```

## Heartbeat Flow

```
rbee-hive (port 9000)
    ↓ Every 30 seconds
    POST /v1/hive-heartbeat
    ↓
queen-rbee (port 8500)
    ↓ Updates HiveRegistry
    ↓ Broadcasts to SSE
    GET /v1/heartbeats/stream
    ↓
Web UI (port 3002)
    ↓ WASM SDK receives
    ↓ Updates React state
    Shows "1 hive online"
```

## Testing

```bash
# Test heartbeat endpoint directly
curl -s http://localhost:8500/v1/heartbeats/stream

# Should see events like:
# event: heartbeat
# data: {"type":"queen","hives_online":1,"hive_ids":["localhost"],...}
```

## Next Steps

1. **Restart `turbo dev`** to load new WASM
2. **Verify web UI shows 1 hive**
3. **Test with multiple hives** (different ports/IDs)

---

**TEAM-292 Signature:** Re-enabled hive heartbeat functionality that was removed by TEAM-261.
