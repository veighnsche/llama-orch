# TEAM-374: Integration Tests for SSE-Only Telemetry

**Status:** 📋 PLAN  
**Date:** Oct 31, 2025

---

## Test Scenarios

### Test 1: Hive Ready Callback
**Verify:** Hive sends POST /v1/hive/ready to Queen

**Steps:**
1. Start Queen on port 7833
2. Start Hive on port 7835 with queen_url
3. Wait for discovery callback
4. Verify Queen received callback
5. Verify Queen started SSE subscription

**Expected:**
- Hive logs: "✅ Discovery successful! Queen will subscribe to our SSE stream"
- Queen logs: "✅ Hive {id} ready, subscription started"

### Test 2: SSE Stream Subscription
**Verify:** Queen subscribes to Hive SSE stream after callback

**Steps:**
1. Start Queen
2. Start Hive
3. Wait for callback
4. Verify SSE connection established
5. Verify telemetry events flowing

**Expected:**
- Queen connects to GET /v1/heartbeats/stream on Hive
- Telemetry events received every 1s
- Worker data stored in TelemetryRegistry

### Test 3: Discovery Exponential Backoff
**Verify:** Hive retries with exponential backoff if Queen not ready

**Steps:**
1. Start Hive (no Queen)
2. Verify attempts: 0s, 2s, 4s, 8s, 16s
3. Start Queen after 3rd attempt
4. Verify 4th attempt succeeds

**Expected:**
- Hive logs show 5 attempts with correct delays
- Discovery succeeds on first attempt after Queen starts

### Test 4: Queen Restart Detection
**Verify:** Hive reconnects SSE stream when Queen restarts

**Steps:**
1. Start Queen + Hive
2. Verify SSE connection
3. Stop Queen
4. Start Queen again
5. Verify Hive reconnects

**Expected:**
- Hive detects connection loss
- Hive reconnects to SSE stream
- Telemetry resumes

---

## Manual Testing Commands

```bash
# Terminal 1: Start Queen
cargo run --bin queen-rbee -- --port 7833

# Terminal 2: Start Hive
cargo run --bin rbee-hive -- --port 7835 --queen-url http://localhost:7833

# Terminal 3: Monitor Queen's SSE stream
curl -N http://localhost:7833/v1/heartbeats/stream

# Terminal 4: Check Hive's SSE stream directly
curl -N http://localhost:7835/v1/heartbeats/stream
```

---

## Expected Logs

### Hive Startup
```
🔍 Starting discovery with exponential backoff
🔍 Discovery attempt 1 (delay: 0s)
✅ Discovery successful! Queen will subscribe to our SSE stream
```

### Queen Startup
```
🐝 Hive ready callback: hive_id=localhost, url=http://localhost:7835
📡 Subscribing to hive localhost SSE stream: http://localhost:7835/v1/heartbeats/stream
✅ Connected to hive localhost SSE stream
✅ Hive localhost ready, subscription started
```

### Telemetry Flow
```
# Hive broadcasts (every 1s)
Broadcasting telemetry: 2 workers

# Queen receives (via SSE)
Received telemetry from hive localhost: 2 workers
```

---

## Verification Checklist

- [ ] Hive sends ready callback successfully
- [ ] Queen receives callback and responds with 200 OK
- [ ] Queen subscribes to Hive SSE stream
- [ ] Telemetry events flow via SSE (1s interval)
- [ ] Worker data stored in TelemetryRegistry
- [ ] Discovery exponential backoff works (0s, 2s, 4s, 8s, 16s)
- [ ] Hive reconnects SSE on Queen restart
- [ ] No POST /v1/hive-heartbeat calls (old endpoint deleted)

---

## Test Implementation

Will create automated tests in `xtask/src/integration/sse_telemetry.rs`
