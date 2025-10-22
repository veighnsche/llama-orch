# rbee-hive HTTP Changes Summary

**Status:** 🟡 60% Complete (6/10 endpoints implemented)

---

## Quick Comparison

| Endpoint | Current | Required | Status | Priority |
|----------|---------|----------|--------|----------|
| **POST /v1/workers/spawn** | ✅ Implemented | ✅ Required | ✅ DONE | - |
| **POST /v1/workers/ready** | ✅ Implemented | ✅ Required | ✅ DONE | - |
| **GET /v1/workers/list** | ✅ Implemented | ✅ Required | ✅ DONE | - |
| **POST /v1/models/download** | ✅ Implemented | ✅ Required | ✅ DONE | - |
| **GET /v1/models/download/progress** | ✅ Implemented | ✅ Required | ✅ DONE | - |
| **POST /v1/heartbeat** | ⚠️ Partial | ✅ Required | ⚠️ NEEDS FIX | HIGH |
| **GET /v1/devices** | ❌ Missing | ✅ Required | 🚧 TODO | HIGH |
| **GET /v1/capacity** | ❌ Missing | ✅ Required | 🚧 TODO | HIGH |
| **POST /v1/shutdown** | ❌ Missing | ✅ Required | 🚧 TODO | HIGH |
| **POST /v1/workers/provision** | ❌ Missing | 🟦 Optional | 🟦 OPTIONAL | LOW |

---

## Critical Issues

### 1. Heartbeat Relay Missing ⚠️

**Current behavior:**
```
Worker → Hive: POST /v1/heartbeat
Hive updates registry
[STOPS HERE]
```

**Required behavior:**
```
Worker → Hive: POST /v1/heartbeat
Hive updates registry
Hive → Queen: POST /heartbeat (with nested worker data)
```

**Fix:** Update `/src/http/heartbeat.rs` to spawn async task that relays to queen

**Architecture ref:** Phase 10 (a_Claude_Sonnet_4_5_refined_this.md lines 300-313)

---

### 2. Device Detection Missing 🚧

**Required:** Queen asks hive for device capabilities

**Endpoint:** `GET /v1/devices`

**Returns:**
```json
{
  "cpu": {"cores": 8, "ram_gb": 32},
  "gpus": [
    {"id": "gpu0", "name": "RTX 3060", "vram_gb": 12},
    {"id": "gpu1", "name": "RTX 3090", "vram_gb": 24}
  ],
  "models": 0,
  "workers": 0
}
```

**Implementation:**
- Create `/src/http/devices.rs`
- Use `rbee-hive-device-detection` crate (already exists)
- Count models from `model_catalog`
- Count workers from `registry`

**Architecture ref:** Phase 4 (a_Claude_Sonnet_4_5_refined_this.md lines 136-164)

---

### 3. VRAM Capacity Check Missing 🚧

**Required:** Queen checks if device has enough VRAM before spawning

**Endpoint:** `GET /v1/capacity?device=gpu1&model=HF:author/minillama`

**Returns:**
- `204 No Content` → OK, enough VRAM
- `409 Conflict` → Insufficient VRAM

**Implementation:**
- Create `/src/http/capacity.rs`
- Use `rbee-hive-vram-checker` crate (already exists)
- Logic: Total VRAM - loaded models - estimated model size

**Architecture ref:** Phase 6 (a_Claude_Sonnet_4_5_refined_this.md lines 181-189)

---

### 4. Shutdown Endpoint Missing 🚧

**Required:** Queen triggers cascading shutdown

**Endpoint:** `POST /v1/shutdown`

**Action:**
1. Shutdown all workers in registry
2. Signal HTTP server to shutdown gracefully
3. Return acknowledgment

**Implementation:**
- Create `/src/http/shutdown.rs`
- Use existing `WorkerRegistry` to get worker URLs
- POST `/v1/shutdown` to each worker
- Call `server.shutdown()` (already exists in `server.rs`)

**Architecture ref:** Phase 12 (a_Claude_Sonnet_4_5_refined_this.md lines 368-387)

---

## Worker Spawning (Already Correct ✅)

**Current implementation** in `/src/http/workers.rs`:

```rust
// Line 236: Worker is spawned with --hive-url (correct!)
.arg("--hive-url")
.arg(format!("http://{}:{}", hostname, hive_port))
```

**Worker callback pattern:**
```
Worker → Hive: POST http://127.0.0.1:8600/v1/workers/ready
```

**This is CORRECT per architecture!** ✅

The worker sends heartbeats to hive, NOT to queen. Queen only needs to know worker exists (via hive's heartbeat relay).

---

## File Changes Required

### New Files
1. `/src/http/devices.rs` - Device detection endpoint
2. `/src/http/capacity.rs` - VRAM capacity check endpoint
3. `/src/http/shutdown.rs` - Graceful shutdown endpoint

### Modified Files
1. `/src/http/heartbeat.rs` - Add relay to queen
2. `/src/http/routes.rs` - Add new endpoints to router
3. `/src/http/mod.rs` - Export new modules
4. `/src/http/workers.rs` - (Optional) Add provision endpoint

---

## Implementation Order

### Phase 1: Critical Endpoints (2 days)
1. ✅ Create `devices.rs` → `GET /v1/devices`
2. ✅ Create `capacity.rs` → `GET /v1/capacity`
3. ✅ Create `shutdown.rs` → `POST /v1/shutdown`
4. ✅ Update `routes.rs` to wire endpoints
5. ✅ Update `mod.rs` to export modules

### Phase 2: Heartbeat Enhancement (1 day)
6. ✅ Update `heartbeat.rs` to relay to queen
7. ✅ Add queen callback URL to `AppState`
8. ✅ Test heartbeat chain: worker → hive → queen

### Phase 3: Optional Features (0.5 days)
9. 🟦 Add worker provisioning to `workers.rs` (optional)

### Phase 4: Testing (1 day)
10. ✅ Unit tests for each new module
11. ✅ Integration test: full happy path
12. ✅ Update README.md

**Total:** 3-4 days

---

## Mock Hive Server (xtask)

**Location:** `/xtask/src/tasks/worker.rs`

**Current implementation:**
- ✅ Receives worker heartbeats on `POST /v1/heartbeat`
- ✅ Prints heartbeat count and details

**Does NOT implement:**
- Device detection
- Capacity check
- Worker spawning
- Model provisioning
- Shutdown

**This is CORRECT for its purpose** (testing worker heartbeat in isolation).

The mock hive is a minimal test harness, not a full hive implementation.

---

## Questions to Resolve

### 1. Device Detection: Public or Protected?
**Recommendation:** Protected (queen-only)

### 2. Heartbeat Relay: Periodic or Event-Driven?
**Options:**
- **Event-driven:** Relay only when worker heartbeat received
- **Periodic:** Hive sends heartbeat every 30s with all worker states

**Recommendation:** Periodic (more robust, aligns with architecture Phase 3)

### 3. Shutdown Timeout?
**Recommendation:** 30s graceful shutdown, then force-kill workers

### 4. Worker Provisioning: Dev Mode Only?
**Current:** Dev uses hardcoded target paths  
**Recommendation:** Implement basic version, stub prod artifact download

---

## Next Steps

1. **Review this plan** with team/user
2. **Confirm priorities** (all 4 critical endpoints needed?)
3. **Start with devices.rs** (simplest, no side effects)
4. **Test incrementally** (add endpoint → test → next endpoint)
5. **Update BDD scenarios** after implementation

---

**Full details:** See `HTTP_UPDATE_PLAN.md` (comprehensive implementation guide)
