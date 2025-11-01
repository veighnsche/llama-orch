# TEAM-371: Heartbeat Architecture Refactor - Complete Summary

**Date:** Oct 31, 2025  
**Author:** TEAM-371  
**Status:** 📋 READY FOR IMPLEMENTATION

---

## Why SSE? Why Keep Handshake?

### The User's Question

> "Why do we need to get rid of the handshake? What is the replacement?"

**Answer:** We DON'T get rid of the handshake. We KEEP it.

---

## What Actually Changes

### Before (Push-Based Continuous Telemetry)

```
Hive Discovery:
  Hive → POST /v1/hive-heartbeat (exponential backoff) → Queen
  
Continuous Telemetry:
  Hive → POST /v1/hive-heartbeat (EVERY 1 SECOND) → Queen
  Queen → Stores → Broadcasts to Queen SSE
```

**Problem:** Discovery callback and continuous telemetry use the SAME endpoint.

### After (SSE-Based Continuous Telemetry)

```
Hive Discovery:
  Hive → POST /v1/hive/ready (exponential backoff) → Queen [ONE-TIME]
  
Continuous Telemetry:
  Hive → Broadcasts to local SSE stream (EVERY 1 SECOND)
  Queen → Subscribes to hive SSE → Forwards to Queen SSE
```

**Solution:** Separate discovery (callback) from telemetry (SSE).

---

## Handshake Details (PRESERVED)

### Discovery is Essential Because:

1. **Remote hives** - Queen needs to discover which machines have hives
2. **Dynamic topology** - Hives can start/stop independently
3. **Bidirectional startup** - Either Queen or Hive can start first
4. **Network awareness** - SSH config tells Queen where to look

### Handshake Scenarios (UNCHANGED)

**Scenario 1: Queen Starts First**
```
1. Queen wakes up
2. Queen waits 5 seconds (services stabilize)
3. Queen reads ~/.ssh/config
4. Queen finds: hive-gpu-0, hive-cpu-1, hive-workstation
5. Queen sends GET /capabilities?queen_url=http://queen:7833 to each
6. Each hive receives queen_url, stores it
7. Each hive sends POST /v1/hive/ready → Queen [NEW: ONE-TIME]
8. Queen subscribes to GET /v1/heartbeats/stream on each hive [NEW: SSE]
9. Telemetry flows continuously via SSE [NEW]
```

**Scenario 2: Hive Starts First**
```
1. Hive wakes up with --queen-url http://queen:7833
2. Hive tries POST /v1/hive/ready → Queen [NEW: ONE-TIME]
3. Attempt 1: 0s delay → 404 (Queen not ready)
4. Attempt 2: 2s delay → 404
5. Attempt 3: 4s delay → 200 OK (Queen is up!)
6. Queen subscribes to GET /v1/heartbeats/stream on hive [NEW: SSE]
7. Telemetry flows continuously via SSE [NEW]
```

**Scenario 3: Queen Restarts**
```
1. Queen crashes
2. Hive detects 404/connection refused on POST /v1/hive/ready
3. Hive restarts exponential backoff (same as Scenario 2)
4. Queen comes back, receives callback
5. Queen subscribes to hive SSE again
6. Telemetry resumes
```

### What's UNCHANGED in Handshake

✅ Exponential backoff: 0s, 2s, 4s, 8s, 16s  
✅ Queen reads SSH config  
✅ GET /capabilities endpoint (Queen pull discovery)  
✅ POST callback (Hive push discovery)  
✅ Bidirectional startup (either can start first)  
✅ Queen restart detection  

### What CHANGES in Handshake

❌ **Before:** POST callback sends FULL telemetry + workers  
✅ **After:** POST callback sends hive_id + hive_url ONLY

❌ **Before:** POST callback repeats EVERY 1 SECOND  
✅ **After:** POST callback happens ONCE (discovery complete)

❌ **Before:** Continuous telemetry via POST  
✅ **After:** Continuous telemetry via SSE subscription

---

## Why SSE is Better

### Problem with Push (Current)

**Hive POSTs every 1 second:**
- Hive needs to know Queen's URL
- Hive needs to handle Queen restarts
- Hive needs exponential backoff logic
- Queen needs to receive and store
- Extra network hop for every telemetry update

**What if Hive SDK wants hive-specific data?**
- Can't connect directly to hive (hive only POSTs to Queen)
- Must go through Queen (extra hop)
- Queen becomes single point of failure

### Solution with SSE (Proposed)

**Hive broadcasts locally:**
- Hive doesn't care WHO subscribes
- Queen subscribes (aggregates all hives)
- Hive SDK can subscribe (direct connection)
- Standard SSE pattern (browsers understand it)

**Discovery callback is separate:**
- Hive sends ONE POST "I'm ready" → Queen
- Queen subscribes to hive SSE
- Continuous telemetry flows via SSE

**Benefits:**
1. **Direct access** - Hive SDK → hive SSE (no Queen needed)
2. **Decoupled** - Hive doesn't need to know subscribers
3. **Resilient** - SSE auto-reconnects on failure
4. **Standard** - EventSource is browser-native
5. **Cleaner** - Discovery separate from telemetry

---

## Data Flow Comparison

### Current (Push)

```
┌─────────────┐
│   Hive A    │───POST (1s)───┐
└─────────────┘                │
                              ▼
┌─────────────┐         ┌─────────┐         ┌────────┐
│   Hive B    │───POST──▶│  Queen  │───SSE───▶│   UI   │
└─────────────┘  (1s)   └─────────┘         └────────┘
                              ▲
┌─────────────┐               │
│   Hive C    │───POST (1s)───┘
└─────────────┘

Problems:
- Hive SDK can't access hive data directly
- Queen is central bottleneck
- Extra POST overhead every second
```

### Proposed (SSE)

```
┌─────────────┐
│   Hive A    │─SSE─┬─────────────────────┐
└─────────────┘     │                     │
                    │                     ▼
                    │              ┌────────────┐
                    │              │  Hive SDK  │
                    │              │ (direct!)  │
                    │              └────────────┘
                    │
                    ▼
              ┌─────────┐         ┌────────┐
              │  Queen  │───SSE───▶│   UI   │
              │(subscribe)│       └────────┘
              └─────────┘
                    ▲
┌─────────────┐     │
│   Hive B    │─SSE─┤
└─────────────┘     │
                    │
┌─────────────┐     │
│   Hive C    │─SSE─┘
└─────────────┘

Benefits:
+ Hive SDK can connect directly to hive
+ Queen subscribes (passive, not active receiver)
+ Standard SSE pattern throughout
+ Hives don't need to know about subscribers
```

---

## Implementation Phases

### Phase 1: Create Hive SSE Stream (1 day)
**Team:** TEAM-372  
**Output:** Hive exposes `GET /v1/heartbeats/stream`

**What's Created:**
- `bin/20_rbee_hive/src/http/heartbeat_stream.rs` - SSE endpoint
- Background broadcaster (collects workers every 1s, broadcasts to SSE)
- Route registration

**What Stays Unchanged:**
- Discovery/handshake logic (all of it)
- POST telemetry to Queen (runs in parallel)

### Phase 2: Queen Subscribes to SSE (1 day)
**Team:** TEAM-373  
**Output:** Queen subscribes to hive SSE instead of receiving POST

**What's Created:**
- `bin/10_queen_rbee/src/hive_subscriber.rs` - SSE subscription client
- `handle_hive_ready()` - Discovery callback triggers subscription
- Update discovery to subscribe after handshake

**What Changes:**
- POST callback becomes ONE-TIME (not continuous)
- Continuous telemetry flows via SSE
- Route changes from `/v1/hive-heartbeat` to `/v1/hive/ready`

### Phase 3: Delete Old POST Logic (0.5 day)
**Team:** TEAM-374  
**Output:** Clean codebase, SSE-only

**What's Deleted:**
- `send_heartbeat_to_queen()` function (continuous POST)
- `start_normal_telemetry_task()` function
- `handle_hive_heartbeat()` receiver
- Unused contract types

**Rule ZERO:** Break cleanly, let compiler find call sites, fix errors.

---

## Contract Changes

### Current HiveHeartbeat (Used for Everything)

```rust
pub struct HiveHeartbeat {
    pub hive: HiveInfo,
    pub timestamp: HeartbeatTimestamp,
    pub workers: Vec<ProcessStats>,
    pub capabilities: Option<Vec<HiveDevice>>, // During discovery
}
```

**Problem:** Same struct for discovery AND continuous telemetry.

### New Contracts (Separated Concerns)

```rust
// Discovery callback (ONE-TIME)
pub struct HiveReadyCallback {
    pub hive_id: String,
    pub hive_url: String, // e.g., "http://192.168.1.100:7835"
}

// SSE telemetry event (CONTINUOUS)
pub enum HiveHeartbeatEvent {
    Telemetry {
        hive_id: String,
        hive_info: HiveInfo,
        timestamp: String,
        workers: Vec<ProcessStats>,
    },
}
```

**Benefit:** Discovery and telemetry are clearly separate.

---

## Why This Preserves the Handshake

### Discovery is NOT Eliminated

**Discovery callback is REQUIRED for:**
1. Queen to learn hive exists
2. Queen to learn hive URL (for SSE subscription)
3. Bidirectional startup coordination
4. Queen restart recovery

**What changes:** The callback is ONE-TIME (not continuous).

### SSE Subscription is NOT Discovery

**SSE subscription happens AFTER discovery:**
1. Discovery: Hive → POST /v1/hive/ready → Queen
2. Queen learns: "Hive exists at http://192.168.1.100:7835"
3. Queen subscribes: Connect to http://192.168.1.100:7835/v1/heartbeats/stream
4. Telemetry: Flows via SSE

**SSE is for continuous telemetry, not discovery.**

---

## Frequently Asked Questions

### Q: Why not remove the callback and just have Queen try SSE connections?

**A:** Because Queen doesn't know which hives exist until discovery. SSH config gives hostnames, but:
- Hive might not be running
- Hive might be on different port
- Hive might not have rbee-hive installed

The callback is the hive saying "I'm here and ready, connect to me."

### Q: Why not keep POST telemetry if discovery uses POST anyway?

**A:** Because discovery is ONE-TIME (cheap) but telemetry is CONTINUOUS (expensive).

- Discovery: 5 attempts max, then done
- Telemetry: Every 1 second, forever

SSE is much more efficient for continuous streams.

### Q: What if hive crashes and restarts?

**A:** Same as before:
1. Hive restarts
2. Hive sends POST /v1/hive/ready (discovery)
3. Queen subscribes to SSE
4. Telemetry resumes

**SSE connection closes when hive dies, Queen detects it and waits for new callback.**

### Q: What if Queen crashes and restarts?

**A:** Same as before:
1. Queen restarts
2. Hive detects POST /v1/hive/ready fails (404)
3. Hive restarts exponential backoff
4. Queen receives callback
5. Queen subscribes to SSE
6. Telemetry resumes

**Existing restart detection works perfectly with this model.**

---

## Success Criteria (All Phases Complete)

1. ✅ Hive exposes `GET /v1/heartbeats/stream`
2. ✅ Hive sends `POST /v1/hive/ready` (one-time callback)
3. ✅ Queen subscribes to hive SSE after callback
4. ✅ Continuous telemetry flows via SSE
5. ✅ Discovery handshake still works (both scenarios)
6. ✅ Queen restart triggers rediscovery
7. ✅ Hive restart triggers rediscovery
8. ✅ Multiple hives can connect simultaneously
9. ✅ Hive SDK can connect directly to hive SSE
10. ✅ Queen SDK gets aggregated view of all hives

---

## Key Takeaway

**Handshake = Discovery (ONE-TIME)**  
**SSE = Continuous Telemetry (ONGOING)**

We're NOT replacing the handshake. We're separating discovery from telemetry.

**Before:** Discovery callback = continuous telemetry (mixed)  
**After:** Discovery callback ≠ continuous telemetry (separated)

**The handshake is essential and stays intact. We just change what happens after it completes.**

---

**TEAM-371 Investigation Complete. Implementation documents ready.**
