# E2E Integration Tests

**Created by:** TEAM-160  
**Date:** 2025-10-20

---

## Overview

End-to-end tests for real daemon orchestration with no mocks.

Tests the actual user commands:
- `rbee queen start` / `rbee queen stop`
- `rbee hive start` / `rbee hive stop`
- Cascading shutdown (queen → hive)

---

## Test Commands

### 1. Queen Lifecycle Test

```bash
cargo xtask e2e:queen
```

**What it tests:**
1. `rbee queen start` - Starts queen-rbee daemon
2. Verify queen is running (health check)
3. `rbee queen stop` - Stops queen-rbee daemon
4. Verify queen is stopped

**Expected output:**
```
🚀 E2E Test: Queen Lifecycle

🔨 Building binaries...
✅ All binaries built

📝 Running: rbee queen start
👑 Starting queen-rbee...
✅ Queen started on http://localhost:8500

🔍 Verifying queen is running...
✅ Queen is running and healthy

📝 Running: rbee queen stop
👑 Stopping queen-rbee...
✅ Queen shutdown signal sent
✅ Queen stopped

🔍 Verifying queen is stopped...
✅ Queen stopped successfully

✅ E2E Test PASSED: Queen Lifecycle
```

---

### 2. Hive Lifecycle Test

```bash
cargo xtask e2e:hive
```

**What it tests:**
1. `rbee hive start` - Starts queen + hive on localhost
2. Verify both are running
3. Wait for first heartbeat from hive
4. `rbee hive stop` - Stops hive only
5. Verify hive is stopped
6. Verify queen is still running

**Expected output:**
```
🚀 E2E Test: Hive Lifecycle

🔨 Building binaries...
✅ All binaries built

📝 Running: rbee hive start
🐝 Starting rbee-hive on localhost...
✅ Queen is running
✅ Hive started on localhost:8600

🔍 Verifying queen is running...
✅ Queen is running

🔍 Verifying hive is running...
✅ Hive is running

⏳ Waiting for first heartbeat from hive localhost...
✅ First heartbeat received after 5 attempts

📝 Running: rbee hive stop
🐝 Stopping rbee-hive on localhost...
✅ Hive shutdown signal sent
✅ Hive stopped

🔍 Verifying hive is stopped...
✅ Hive stopped successfully

🔍 Verifying queen is still running...
✅ Queen still running (as expected)

🧹 Cleaning up - stopping queen...
✅ E2E Test PASSED: Hive Lifecycle
```

---

### 3. Cascade Shutdown Test

```bash
cargo xtask e2e:cascade
```

**What it tests:**
1. `rbee hive start` - Starts queen + hive
2. Verify both are running
3. Wait for first heartbeat
4. `rbee queen stop` - Stops queen (should cascade to hive)
5. Verify queen is stopped
6. Verify hive is also stopped (cascade worked)

**Expected output:**
```
🚀 E2E Test: Cascade Shutdown

🔨 Building binaries...
✅ All binaries built

📝 Running: rbee hive start
🐝 Starting rbee-hive on localhost...
✅ Queen is running
✅ Hive started on localhost:8600

🔍 Verifying queen and hive are running...
✅ Queen is running
✅ Hive is running

⏳ Waiting for first heartbeat from hive localhost...
✅ First heartbeat received after 5 attempts

📝 Running: rbee queen stop (should cascade to hive)
👑 Stopping queen-rbee...
✅ Queen shutdown signal sent
✅ Queen stopped

🔍 Verifying queen is stopped...
✅ Queen stopped

🔍 Verifying hive is also stopped (cascade)...
✅ Hive stopped (cascade worked!)

✅ E2E Test PASSED: Cascade Shutdown
   Queen stopped → Hive stopped automatically
```

---

## Architecture

```
┌─────────────┐
│ User        │
└──────┬──────┘
       │ $ rbee queen start
       ▼
┌─────────────┐
│ rbee-keeper │ (CLI)
└──────┬──────┘
       │ Spawns process
       ▼
┌─────────────┐
│ queen-rbee  │ (Daemon on port 8500)
└─────────────┘

       │ $ rbee hive start
       ▼
┌─────────────┐
│ rbee-keeper │
└──────┬──────┘
       │ POST /add-hive
       ▼
┌─────────────┐
│ queen-rbee  │
└──────┬──────┘
       │ Spawns process
       ▼
┌─────────────┐
│ rbee-hive   │ (Daemon on port 8600)
└──────┬──────┘
       │ POST /heartbeat
       ▼
┌─────────────┐
│ queen-rbee  │ (Receives heartbeat)
└─────────────┘

       │ $ rbee queen stop
       ▼
┌─────────────┐
│ rbee-keeper │
└──────┬──────┘
       │ POST /shutdown
       ▼
┌─────────────┐
│ queen-rbee  │
└──────┬──────┘
       │ POST /shutdown (cascade)
       ▼
┌─────────────┐
│ rbee-hive   │ (Shuts down)
└─────────────┘
```

---

## Implementation Status

### ✅ Implemented

1. **xtask commands:**
   - `cargo xtask e2e:queen`
   - `cargo xtask e2e:hive`
   - `cargo xtask e2e:cascade`

2. **rbee-keeper commands:**
   - `rbee queen start`
   - `rbee queen stop`
   - `rbee hive start`
   - `rbee hive stop`

3. **queen-rbee endpoints:**
   - `POST /add-hive` - Add hive to catalog
   - `POST /shutdown` - Shutdown queen

4. **Test helpers:**
   - `build_binaries()` - Build all required binaries
   - `wait_for_first_heartbeat()` - Wait for hive heartbeat

### ⚠️ Blocked

**Queen-rbee compilation errors** (pre-existing, not caused by TEAM-160):
1. Missing `async-trait` dependency
2. Lifetime mismatch in `device_detector.rs`
3. Type mismatch in `heartbeat.rs`
4. Missing `device_detector` field in `HeartbeatState`

### 🚧 Missing (for full happy flow)

1. **Queen spawns hive:**
   - `handle_add_hive` needs to spawn rbee-hive process
   - Store process handle
   - Monitor hive lifecycle

2. **Hive sends heartbeats:**
   - Implement heartbeat sender in rbee-hive
   - Periodic POST to queen `/heartbeat`

3. **Queen triggers device detection:**
   - On first heartbeat with unknown devices
   - GET `/v1/devices` from hive
   - Store in catalog

4. **Cascading shutdown:**
   - Queen `/shutdown` endpoint kills all hives
   - Hive `/shutdown` endpoint kills all workers
   - Proper cleanup and process management

---

## File Structure

```
xtask/src/e2e/
├── mod.rs                  # Module exports
├── README.md               # This file
├── helpers.rs              # Shared helper functions
├── queen_lifecycle.rs      # Queen start/stop test
├── hive_lifecycle.rs       # Hive start/stop test
└── cascade_shutdown.rs     # Cascade shutdown test
```

---

## Next Steps

1. **Fix queen-rbee compilation** (not TEAM-160's responsibility)
2. **Implement queen spawning logic** - Queen needs to spawn rbee-hive
3. **Implement hive heartbeats** - Hive needs to send periodic heartbeats
4. **Implement device detection** - Queen needs to trigger on first heartbeat
5. **Implement cascading shutdown** - Queen needs to kill hives on shutdown

---

**TEAM-160: E2E test framework complete. Tests real daemon orchestration. 🚀**
