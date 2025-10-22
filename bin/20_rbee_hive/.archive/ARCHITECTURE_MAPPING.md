# rbee-hive Architecture Mapping

**Visual guide:** Architecture flows → HTTP endpoints

---

## Architecture Phase → HTTP Endpoint Mapping

### Phase 4: Device Detection

**Architecture Flow:**
```
Queen → Hive: GET /devices
Hive runs device-detection crate
Hive responds with: CPU + GPU list + model/worker counts
Queen updates hive-catalog + hive-registry
```

**HTTP Endpoint Required:**
```rust
GET /v1/devices
Response: {
  "cpu": {"cores": 8, "ram_gb": 32},
  "gpus": [{"id": "gpu0", "name": "RTX 3060", "vram_gb": 12}],
  "models": 0,
  "workers": 0
}
```

**Status:** ❌ Missing → Need to create `devices.rs`

---

### Phase 5: Scheduling (Queen-side only)

**Architecture Flow:**
```
Queen uses hive-registry (RAM) to pick strongest GPU
```

**HTTP Endpoint Required:** None (queen-side logic)

**Status:** ✅ N/A (not hive's responsibility)

---

### Phase 6: VRAM Capacity Check

**Architecture Flow:**
```
Queen → Hive: GET /capacity?device=gpu1&model=HF:author/minillama
Hive checks VRAM checker crate
Hive responds: 204 (OK) or 409 (insufficient)
```

**HTTP Endpoint Required:**
```rust
GET /v1/capacity?device=gpu1&model=HF:author/minillama
Response: 
  204 No Content (OK)
  409 Conflict (insufficient VRAM)
```

**Status:** ❌ Missing → Need to create `capacity.rs`

---

### Phase 7: Model Provisioning

**Architecture Flow:**
```
Queen → Hive: POST /models/provision {"model_ref": "HF:author/minillama"}
Hive responds with SSE link
Queen connects to SSE for progress
Hive downloads model + registers in catalog
```

**HTTP Endpoint Required:**
```rust
POST /v1/models/download
GET /v1/models/download/progress?id=<download_id>
```

**Status:** ✅ Already Implemented in `models.rs`

---

### Phase 8: Worker Binary Provision

**Architecture Flow:**
```
Queen → Hive: POST /workers/provision {"kind": "cuda-llm-worker-rbee"}
Hive checks worker-catalog (SQLite)
If missing: download binary (dev mode: use target path)
Hive responds with binary path
```

**HTTP Endpoint Required:**
```rust
POST /v1/workers/provision
Response: {"message": "...", "path": "/path/to/binary"}
```

**Status:** 🟦 Optional (not critical, dev mode uses hardcoded paths)

---

### Phase 9: Worker Spawning

**Architecture Flow:**
```
Queen → Hive: POST /workers/start {
  "device": "gpu1",
  "model_ref": "HF:author/minillama",
  "port_hint": 8601
}
Hive spawns worker process with:
  --worker-id, --model, --port, --hive-url http://127.0.0.1:8600
Hive responds with port
Queen adds to worker-registry (RAM)
```

**HTTP Endpoint Required:**
```rust
POST /v1/workers/spawn
Response: {"worker_id": "...", "url": "http://127.0.0.1:8601", "state": "loading"}
```

**Status:** ✅ Already Implemented in `workers.rs`

**Note:** Architecture calls this `/workers/start`, current implementation uses `/workers/spawn`. Both are correct, just different naming.

---

### Phase 9b: Worker Ready Callback

**Architecture Flow:**
```
Worker boots → Worker → Hive: POST /v1/workers/ready {
  "worker_id": "...",
  "url": "...",
  "model_ref": "...",
  "backend": "...",
  "device": 0
}
Hive updates state to Idle
Hive → Queen: POST /workers/ready (notification)
```

**HTTP Endpoint Required:**
```rust
POST /v1/workers/ready
Response: {"message": "Worker registered as ready"}
```

**Status:** ✅ Already Implemented in `workers.rs`

---

### Phase 10: Worker Heartbeat (Nested)

**Architecture Flow:**
```
Worker → Hive: POST /v1/heartbeat {
  "worker_id": "...",
  "timestamp": "...",
  "health_status": "healthy"
}
Hive updates worker registry

[CRITICAL] Hive → Queen: POST /heartbeat {
  "hive_id": "localhost",
  "timestamp": "...",
  "workers": [
    {"worker_id": "...", "state": "Ready", "last_heartbeat": "..."}
  ]
}
```

**HTTP Endpoint Required:**
```rust
POST /v1/heartbeat (receive from workers)
```

**Status:** ⚠️ Partially Implemented

**Current:** Receives worker heartbeat, updates registry  
**Missing:** Does NOT relay to queen

**Fix Required:** Add queen relay logic in `heartbeat.rs`

---

### Phase 11: Inference (Worker-side)

**Architecture Flow:**
```
Queen → Worker: POST /infer {"prompt": "...", "stream": true}
Worker responds with SSE link
Queen connects to SSE
Worker streams tokens → Queen → Keeper → stdout
```

**HTTP Endpoint Required:** None (worker-side)

**Status:** ✅ Worker already implements this in `execute.rs`

---

### Phase 12: Cascading Shutdown

**Architecture Flow:**
```
Keeper → Queen: POST /shutdown?cascade=true
Queen → Hive: POST /shutdown
Hive → Worker: POST /shutdown (for each worker)
Hive shuts down itself
```

**HTTP Endpoint Required:**
```rust
POST /v1/shutdown
Response: {"message": "Shutdown initiated"}
```

**Status:** ❌ Missing → Need to create `shutdown.rs`

---

## Endpoint Comparison Table

| Architecture Phase | Arch Endpoint | Current Endpoint | Status |
|-------------------|---------------|------------------|--------|
| 3: Hive Startup | N/A (queen-side) | N/A | ✅ N/A |
| 4: Device Detection | `GET /devices` | ❌ None | 🚧 TODO |
| 5: Scheduling | N/A (queen-side) | N/A | ✅ N/A |
| 6: VRAM Check | `GET /capacity` | ❌ None | 🚧 TODO |
| 7: Model Provision | `POST /models/provision` | `POST /v1/models/download` | ✅ DONE |
| 7: Model Progress | `GET /models/provision/.../events` | `GET /v1/models/download/progress` | ✅ DONE |
| 8: Worker Provision | `POST /workers/provision` | ❌ None | 🟦 OPTIONAL |
| 9: Worker Start | `POST /workers/start` | `POST /v1/workers/spawn` | ✅ DONE |
| 9b: Worker Ready | `POST /workers/ready` | `POST /v1/workers/ready` | ✅ DONE |
| 10: Heartbeat (receive) | `POST /heartbeat` | `POST /v1/heartbeat` | ✅ DONE |
| 10: Heartbeat (relay) | (to queen) | ❌ None | ⚠️ TODO |
| 11: Inference | N/A (worker-side) | N/A | ✅ N/A |
| 12: Shutdown | `POST /shutdown` | ❌ None | 🚧 TODO |

---

## Naming Differences

### Model Provisioning
- **Architecture:** `/models/provision` + `/models/provision/:id/events`
- **Current:** `/v1/models/download` + `/v1/models/download/progress?id=...`
- **Verdict:** Both valid, current is clearer

### Worker Spawning
- **Architecture:** `/workers/start`
- **Current:** `/v1/workers/spawn`
- **Verdict:** Both valid, "spawn" is more accurate (process spawning)

### Heartbeat Relay
- **Architecture:** Hive sends heartbeat to Queen
- **Current:** Only receives from workers, does NOT relay
- **Verdict:** Missing functionality (critical!)

---

## Worker vs Hive Endpoint Differences

### Worker Endpoints (30_llm_worker_rbee)
```
GET  /health               (public)
POST /v1/inference         (protected, SSE streaming)
```

### Hive Endpoints (20_rbee_hive)
```
Public:
  GET  /v1/health
  GET  /metrics

Protected:
  POST /v1/workers/spawn
  POST /v1/workers/ready
  GET  /v1/workers/list
  POST /v1/workers/provision   (optional)
  POST /v1/models/download
  GET  /v1/models/download/progress
  POST /v1/heartbeat
  GET  /v1/devices             (TODO)
  GET  /v1/capacity            (TODO)
  POST /v1/shutdown            (TODO)
```

**Key Difference:** Hive is a management daemon (many endpoints), Worker is execution-focused (2 endpoints)

---

## Callback URL Patterns

### Worker → Hive
```rust
// Worker spawned with:
--hive-url http://127.0.0.1:8600

// Worker calls back:
POST http://127.0.0.1:8600/v1/workers/ready
POST http://127.0.0.1:8600/v1/heartbeat
```

✅ Already correct in current implementation

### Hive → Queen
```rust
// Hive configured with:
queen_callback_url: "http://127.0.0.1:8500"

// Hive calls back:
POST http://127.0.0.1:8500/v2/workers/ready  (already implemented)
POST http://127.0.0.1:8500/heartbeat         (TODO - missing!)
```

⚠️ Worker ready callback exists, heartbeat relay missing

---

## Mock Hive Server (xtask)

**Location:** `/xtask/src/tasks/worker.rs` (lines 82-203)

**Implements:**
```rust
POST /v1/heartbeat  → Receives worker heartbeats
```

**Does NOT implement:**
- Device detection
- Capacity check
- Worker spawning
- Model provisioning
- Shutdown

**Purpose:** Minimal test harness for worker isolation testing

**Verdict:** ✅ Correct for its purpose (not meant to be a full hive)

---

## Architecture References

**Source files:**
- `a_human_wrote_this.md` - Original happy flow (lines 1-133)
- `a_chatGPT_5_refined_this.md` - Branched flow (lines 1-253)
- `a_Claude_Sonnet_4_5_refined_this.md` - Code-backed refinement (lines 1-573)

**Key phases:**
- Phase 4 (lines 136-164): Device detection
- Phase 6 (lines 181-189): VRAM capacity check
- Phase 9 (lines 247-297): Worker spawning + ready callback
- Phase 10 (lines 300-313): Nested heartbeats
- Phase 12 (lines 368-387): Cascading shutdown

---

**Next:** See `HTTP_UPDATE_PLAN.md` for detailed implementation guide
