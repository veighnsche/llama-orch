# Happy Flow: Code-Backed Refinement

**Generated:** 2025-10-19  
**Based on:** Actual codebase at `/home/vince/Projects/llama-orch/bin/`  
**Command:** `rbee-keeper infer --model HF:author/minillama "hello"`  
**Context:** Fresh install, all services off → cascading shutdown after inference

This refinement maps every step to **actual crates that exist in the workspace**, correcting ChatGPT's assumptions with real implementation details.

---

## 🏗️ Verified Architecture

**4 Binaries:**
- `bin/00_rbee_keeper` - CLI in binary (`src/main.rs`)
- `bin/10_queen_rbee` - HTTP server in binary (`src/http_server.rs`)
- `bin/20_rbee_hive` - HTTP server in binary (`src/http_server.rs`)
- `bin/30_llm_worker_rbee` - HTTP server + backend in binary (`src/backend/`)

**21 Binary-Specific Crates:**
- `bin/05_rbee_keeper_crates/` → 4 crates
- `bin/15_queen_rbee_crates/` → 8 crates
- `bin/25_rbee_hive_crates/` → 9 crates
- `bin/39_worker_rbee_crates/` → REMOVED (all logic in binary)

**15 Shared Crates:** `bin/99_shared_crates/`

---

## 📝 Phase-by-Phase Flow with Actual Crates

### Phase 0: CLI Entry

**Command:** `rbee-keeper infer --model HF:author/minillama "hello"`

**Binary:** `bin/00_rbee_keeper/src/main.rs` (clap parsing)  
**→ Dispatches to:** `rbee-keeper-commands::infer`  
**Crate:** `bin/05_rbee_keeper_crates/commands/`

```
bee-keeper: starting inference request for HF:author/minillama
```

---

### Phase 1: Start Queen (Cold Start)

**Step 1.1 - Health Check Fails**

**Crate:** `rbee-http-client` (shared: `bin/99_shared_crates/rbee-http-client/`)  
**Action:** `GET http://localhost:8500/health` → Connection refused

```
bee-keeper: queen is asleep, waking queen...
```

**Step 1.2 - Spawn Queen Process**

**Crate:** `rbee-keeper-queen-lifecycle` (`bin/05_rbee_keeper_crates/queen-lifecycle/`)  
**Uses:** `daemon-lifecycle` (shared: `bin/99_shared_crates/daemon-lifecycle/`)  
**Spawns:** `queen-rbee --port 8500` (dev: hardcoded target path)

```
bee-keeper: launching queen on :8500 and waiting for health...
```

**Step 1.3 - Poll Until Healthy**

**Crate:** `rbee-keeper-polling` (`bin/05_rbee_keeper_crates/polling/`)  
**Retry loop:** `GET http://localhost:8500/health` (exponential backoff)  
**Target endpoint:** `queen-rbee-health` (`bin/15_queen_rbee_crates/health/`)

```
bee-keeper: queen is awake and healthy
```

---

### Phase 2: Job Submission

**Step 2.1 - Open SSE**

**Uses:** `sse-relay` (shared: `bin/99_shared_crates/sse-relay/`)  
**Action:** `GET http://localhost:8500/events`

```
bee-keeper: connected to queen via SSE
```

**Step 2.2 - Submit Job**

**Crate:** `rbee-keeper-commands` (from Phase 0)  
**Action:** `POST http://localhost:8500/jobs {"model_ref": "HF:author/minillama", "prompt": "hello"}`  
**Response:** `{"job_id": "uuid", "events_url": "/jobs/uuid/events"}`

---

### Phase 3: Hive Discovery & Startup

**Step 3.1 - Check Catalog (Empty)**

**Crate:** `queen-rbee-hive-catalog` (`bin/15_queen_rbee_crates/hive-catalog/`)  
**Storage:** SQLite (persistent hive storage)  
**Query:** `SELECT * FROM hives WHERE status='online'` → 0 rows

```
queen: no hives found; adding local machine
```

**Step 3.2 - Start Local Hive**

**Crate:** `queen-rbee-hive-lifecycle` (`bin/15_queen_rbee_crates/hive-lifecycle/`)  
**Uses:** `daemon-lifecycle` (shared)  
**Spawns:** `rbee-hive --port 8600`

**Note:** For remote hives, uses `queen-rbee-ssh-client` (`bin/15_queen_rbee_crates/ssh-client/`)  
⚠️ **ONLY queen-rbee uses SSH!**

```
queen: waking the beehive at localhost:8600
```

**Step 3.3 - Wait for Heartbeat**

**Binary:** `bin/10_queen_rbee/src/http_server.rs` (receives heartbeat)  
**Uses:** `heartbeat` (shared: `bin/99_shared_crates/heartbeat/`)  
**Hive sends:** `POST http://localhost:8500/heartbeat`

```
queen: first heartbeat from hive localhost; checking capabilities...
```

---

### Phase 4: Device Detection

**Step 4.1 - Request Devices**

**Action:** `GET http://localhost:8600/devices`  
**Target:** `bin/20_rbee_hive/src/http_server.rs`  
**Delegates to:** `rbee-hive-device-detection` (`bin/25_rbee_hive_crates/device-detection/`)

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

**Step 4.2 - Update Catalog & Registry**

**Catalog (SQLite):** `queen-rbee-hive-catalog` (persistent)  
**Registry (RAM):** `queen-rbee-hive-registry` (`bin/15_queen_rbee_crates/hive-registry/`)

```
queen: hive localhost has CPU, GPU0 RTX 3060 (12GB), GPU1 RTX 3090 (24GB); models: 0, workers: 0
```

---

### Phase 5: Scheduling

**Crate:** `queen-rbee-scheduler` (`bin/15_queen_rbee_crates/scheduler/`)  
**Uses:** `queen-rbee-hive-registry` (cluster view)  
**Algorithm:** Pick strongest GPU (highest VRAM) → GPU1

```
queen: basic scheduler picked GPU1 on hive localhost (for advanced scheduling, see docs)
```

---

### Phase 6: VRAM Check

**Action:** `GET http://localhost:8600/capacity?device=gpu1&model=HF:author/minillama`  
**Delegates to:** `rbee-hive-vram-checker` (`bin/25_rbee_hive_crates/vram-checker/`)  
**Logic:** Total VRAM - loaded models - estimated model size  
**Response:** `204 No Content` (OK) or `409 Conflict` (insufficient)

```
queen: asking hive if there's room on GPU1...
queen: there is room in GPU1 for model HF:author/minillama
```

---

### Phase 7: Model Provisioning (Concurrent with Phase 8)

**Step 7.1 - Request Provision**

**Action:** `POST http://localhost:8600/models/provision {"model_ref": "HF:author/minillama"}`  
**Response:** SSE link: `GET /models/provision/uuid/events`

```
queen: asked hive to download model HF:author/minillama
```

**Step 7.2 - Download Model**

**Crate:** `rbee-hive-model-provisioner` (`bin/25_rbee_hive_crates/model-provisioner/`)  
**Uses:** `rbee-hive-download-tracker` (`bin/25_rbee_hive_crates/download-tracker/`)  
**Flow:** Fetch HF metadata → Download files → Track progress → Write to disk

**SSE Stream (relayed via `sse-relay`):**
```
hive: model size = 4.2 GB; starting download
hive: 12% @ 45 MB/s
hive: 35% @ 42 MB/s
hive: model downloaded in 95 seconds
```

**Step 7.3 - Update Catalog**

**Crate:** `rbee-hive-model-catalog` (`bin/25_rbee_hive_crates/model-catalog/`)  
**Storage:** SQLite (hive-local, NOT shared!)  
**Action:** `INSERT INTO models (ref, path, size_bytes, status) VALUES (...)`

```
queen: model is downloaded by the hive
```

---

### Phase 8: Worker Binary Provision (Concurrent with Phase 7)

**Action:** `POST http://localhost:8600/workers/provision {"kind": "cuda-llm-worker-rbee"}`  
**Dev Mode:** Use hardcoded path: `target/debug/llm-worker-rbee` (built with `--features cuda`)

**Crate:** `rbee-hive-worker-catalog` (`bin/25_rbee_hive_crates/worker-catalog/`)  
**Storage:** SQLite (hive-local)  
**Action:** `INSERT INTO workers (kind, path, version) VALUES (...)`

```
queen: asked hive to prepare worker cuda-llm-worker-rbee
queen: worker prepared and ready to deploy
```

---

### Phase 9: Start Worker Process

**Step 9.1 - Request Start**

**Action:** `POST http://localhost:8600/workers/start {"device": "gpu1", "model_ref": "HF:author/minillama", "port_hint": 8601}`

**Step 9.2 - Spawn Worker**

**Crate:** `rbee-hive-worker-lifecycle` (`bin/25_rbee_hive_crates/worker-lifecycle/`)  
**Uses:** `daemon-lifecycle` (shared)  
**Queries:**
1. `rbee-hive-model-catalog` (SQLite) → model path
2. `rbee-hive-worker-catalog` (SQLite) → worker binary path

**Spawns:**
```bash
/path/to/llm-worker-rbee \
  --model-path /path/to/model \
  --device gpu1 \
  --port 8601 \
  --hive-url http://localhost:8600
```

**Registers in:** `rbee-hive-worker-registry` (`bin/25_rbee_hive_crates/worker-registry/`)  
**Stores:** PID, port, model_ref, state=Loading, restart_count=0

```
hive: waking worker on :8601 with model=/path/to/model, device=gpu1
```

**Step 9.3 - Queen Registry Update**

**Crate:** `queen-rbee-worker-registry` (`bin/15_queen_rbee_crates/worker-registry/`)  
**Purpose:** Cluster-wide routing context (different from hive's lifecycle context!)

**Stores:**
```rust
WorkerInfo {
    id, url, model_ref,
    state: WorkerState::Loading, // from rbee-types (shared enum)
    node_name: "localhost",
    slots_available: 1,
    vram_bytes: Some(24_000_000_000),
}
```

⚠️ **WorkerInfo NOT shared!** Different contexts (routing vs lifecycle) need different fields.

```
queen: worker registered, waiting for heartbeat...
```

---

### Phase 10: Worker Heartbeat (Nested)

**Worker → Hive:** `POST http://localhost:8600/heartbeat/workers {"worker_id": "...", "state": "Ready"}`  
**Updates:** `rbee-hive-worker-registry` (state: Loading → Ready)

**Hive → Queen:** `POST http://localhost:8500/heartbeat {"hive_id": "...", "workers": [...]}`  
**Updates:** `queen-rbee-worker-registry` (state: Loading → Ready)

**Uses:** `heartbeat` (shared: `bin/99_shared_crates/heartbeat/`)

```
queen: CUDA worker is awake and ready
```

---

### Phase 11: Inference (SSE Streaming)

**Step 11.1 - Request Inference**

**Action:** `POST http://localhost:8601/infer {"prompt": "hello", "stream": true}`  
**Response:** SSE link: `GET /infer/uuid/events`

```
queen: connection to worker established; starting inference
```

**Step 11.2 - Worker Inference**

**Binary:** `bin/30_llm_worker_rbee/src/backend/inference.rs`  
**Uses:**
- `src/backend/tokenizer_loader.rs`
- `src/backend/gguf_tokenizer.rs`
- `src/backend/models/llama.rs`
- `src/backend/sampling.rs`

⚠️ **Exception:** `src/backend/` stays in binary (tightly coupled to Candle framework)

**Flow:** Tokenize → Load model → Forward pass → Sample → Decode → Send SSE → Repeat

**Step 11.3 - SSE Relay (3-hop)**

**Uses:** `sse-relay` (shared)  
**Chain:** Worker → Queen → Keeper → stdout

```
he... l... lo... ! ... How... can... I... help... you... today... ?
```

**Completion:**
```
[DONE]
queen: worker finished inference
```

---

### Phase 12: Cascading Shutdown

**Step 12.1 - Shutdown Request**

**Action:** `POST http://localhost:8500/shutdown?cascade=true`

```
bee-keeper: asking queen to cascade shutdown
```

**Step 12.2 - Shutdown Workers**

**Uses:** `queen-rbee-worker-registry` (get worker URLs)  
**Action:** `POST http://localhost:8601/shutdown` (per worker)

```
queen: shutting down workers...
```

**Step 12.3 - Shutdown Hives**

**Uses:**
- `queen-rbee-hive-registry` (get hive URLs)
- `queen-rbee-hive-lifecycle` (shutdown logic)

**Action:** `POST http://localhost:8600/shutdown` (per hive)  
**Fallback (remote):** SSH kill via `queen-rbee-ssh-client`

```
queen: shutting down hives...
queen: goodbye
```

---

## 📊 Crate Reference Matrix

### rbee-keeper Layer

| Component | Crate | Path |
|-----------|-------|------|
| CLI parsing | Binary | `bin/00_rbee_keeper/src/main.rs` |
| Commands | `rbee-keeper-commands` | `bin/05_rbee_keeper_crates/commands/` |
| Config | `rbee-keeper-config` | `bin/05_rbee_keeper_crates/config/` |
| Queen lifecycle | `rbee-keeper-queen-lifecycle` | `bin/05_rbee_keeper_crates/queen-lifecycle/` |
| Health polling | `rbee-keeper-polling` | `bin/05_rbee_keeper_crates/polling/` |

### queen-rbee Layer

| Component | Crate | Path |
|-----------|-------|------|
| HTTP server | Binary | `bin/10_queen_rbee/src/http_server.rs` |
| Health endpoint | `queen-rbee-health` | `bin/15_queen_rbee_crates/health/` |
| Hive catalog (SQLite) | `queen-rbee-hive-catalog` | `bin/15_queen_rbee_crates/hive-catalog/` |
| Hive registry (RAM) | `queen-rbee-hive-registry` | `bin/15_queen_rbee_crates/hive-registry/` |
| Worker registry (RAM) | `queen-rbee-worker-registry` | `bin/15_queen_rbee_crates/worker-registry/` |
| Hive lifecycle | `queen-rbee-hive-lifecycle` | `bin/15_queen_rbee_crates/hive-lifecycle/` |
| SSH client | `queen-rbee-ssh-client` | `bin/15_queen_rbee_crates/ssh-client/` |
| Scheduler | `queen-rbee-scheduler` | `bin/15_queen_rbee_crates/scheduler/` |
| Preflight | `queen-rbee-preflight` | `bin/15_queen_rbee_crates/preflight/` |

### rbee-hive Layer

| Component | Crate | Path |
|-----------|-------|------|
| HTTP server | Binary | `bin/20_rbee_hive/src/http_server.rs` |
| Device detection | `rbee-hive-device-detection` | `bin/25_rbee_hive_crates/device-detection/` |
| VRAM checker | `rbee-hive-vram-checker` | `bin/25_rbee_hive_crates/vram-checker/` |
| Model catalog (SQLite) | `rbee-hive-model-catalog` | `bin/25_rbee_hive_crates/model-catalog/` |
| Model provisioner | `rbee-hive-model-provisioner` | `bin/25_rbee_hive_crates/model-provisioner/` |
| Download tracker | `rbee-hive-download-tracker` | `bin/25_rbee_hive_crates/download-tracker/` |
| Worker catalog (SQLite) | `rbee-hive-worker-catalog` | `bin/25_rbee_hive_crates/worker-catalog/` |
| Worker lifecycle | `rbee-hive-worker-lifecycle` | `bin/25_rbee_hive_crates/worker-lifecycle/` |
| Worker registry (RAM) | `rbee-hive-worker-registry` | `bin/25_rbee_hive_crates/worker-registry/` |
| Monitor | `rbee-hive-monitor` | `bin/25_rbee_hive_crates/monitor/` |

### llm-worker-rbee (All in Binary)

| Component | Location |
|-----------|----------|
| HTTP server | `bin/30_llm_worker_rbee/src/http_server.rs` |
| Inference | `bin/30_llm_worker_rbee/src/backend/inference.rs` |
| Tokenizer | `bin/30_llm_worker_rbee/src/backend/tokenizer_loader.rs` |
| GGUF tokenizer | `bin/30_llm_worker_rbee/src/backend/gguf_tokenizer.rs` |
| Sampling | `bin/30_llm_worker_rbee/src/backend/sampling.rs` |
| Models | `bin/30_llm_worker_rbee/src/backend/models/*.rs` |

### Shared Crates

| Purpose | Crate | Path |
|---------|-------|------|
| HTTP client | `rbee-http-client` | `bin/99_shared_crates/rbee-http-client/` |
| Shared types | `rbee-types` | `bin/99_shared_crates/rbee-types/` |
| Daemon spawning | `daemon-lifecycle` | `bin/99_shared_crates/daemon-lifecycle/` |
| Heartbeat | `heartbeat` | `bin/99_shared_crates/heartbeat/` |
| SSE relay | `sse-relay` | `bin/99_shared_crates/sse-relay/` |
| Logging | `narration-core` | `bin/99_shared_crates/narration-core/` |

---

## 🔑 Key Architecture Corrections

### 1. Entry Points in Binaries (Not Separate Crates)

**ChatGPT assumed:** CLI and HTTP servers are separate crates  
**Actual:** Entry points implemented directly in binaries

- ❌ `bin/05_rbee_keeper_crates/cli/` → REMOVED
- ✅ `bin/00_rbee_keeper/src/main.rs` → CLI parsing here
- ❌ `bin/15_queen_rbee_crates/http-server/` → REMOVED
- ✅ `bin/10_queen_rbee/src/http_server.rs` → HTTP server here

### 2. Catalog vs Registry Distinction

**ChatGPT missed:** Critical difference between persistent (SQLite) and in-memory (RAM)

**Actual:**
- **Catalog** = SQLite (persistent, survives restarts)
  - `queen-rbee-hive-catalog` (hives)
  - `rbee-hive-model-catalog` (models)
  - `rbee-hive-worker-catalog` (worker binaries)
- **Registry** = RAM (volatile, cluster state)
  - `queen-rbee-hive-registry` (cluster view)
  - `queen-rbee-worker-registry` (routing context)
  - `rbee-hive-worker-registry` (lifecycle context)

### 3. WorkerInfo NOT Shared

**ChatGPT assumed:** Single `WorkerInfo` struct across crates  
**Actual:** Different contexts need different fields

**queen-rbee** (routing):
```rust
WorkerInfo { id, url, model_ref, state, node_name, slots_available, vram_bytes }
```

**rbee-hive** (lifecycle):
```rust
WorkerInfo { id, url, model_ref, state, pid, restart_count, failed_health_checks, last_heartbeat }
```

**Only `WorkerState` enum shared** (from `rbee-types`)

### 4. SSH Only in Queen

**ChatGPT vaguely mentioned:** SSH for remote operations  
**Actual:** ⚠️ ONLY `queen-rbee` has SSH access

- Used ONLY for remote hive startup/force-kill
- After startup, all communication via HTTP
- Single SSH entry point = better security

### 5. No worker-rbee Crates

**ChatGPT listed:** `bin/39_worker_rbee_crates/http-server/`  
**Actual:** `bin/39_worker_rbee_crates/` REMOVED entirely

- HTTP server in `bin/30_llm_worker_rbee/src/http_server.rs`
- Backend inference in `bin/30_llm_worker_rbee/src/backend/`
- Exception to minimal binary rule (justified for LLM-specific logic)

### 6. Three-Hop SSE Relay

**ChatGPT simplified:** SSE connections  
**Actual:** Multi-hop relay via `sse-relay` (shared crate)

**Chain:**
1. Worker → Queen SSE connection
2. Queen → Keeper SSE connection
3. Keeper → stdout

All narration flows through this chain.

---

## 🎯 README Wiring Guide

Each crate's README should reference its position in this happy flow:

**Example (`rbee-keeper-polling/README.md`):**
```markdown
## Happy Flow Step
**Phase:** 1.3 - Poll Until Healthy
**Trigger:** After queen-rbee process spawned
**Action:** Retry `GET /health` until 200 OK
**Next:** Job submission (Phase 2)
```

**Example (`queen-rbee-scheduler/README.md`):**
```markdown
## Happy Flow Step
**Phase:** 5 - Scheduling
**Inputs:** Hive registry (cluster view)
**Algorithm:** Pick strongest GPU (highest VRAM)
**Output:** Device + hive selection
**Next:** VRAM capacity check (Phase 6)
```

---

## 📚 Implementation Status

**Verified from workspace:**
- ✅ All 4 binaries exist with minimal `main.rs`
- ✅ 21 binary-specific crates created
- ✅ 15 shared crates available
- ✅ Entry points moved to binaries (no separate CLI/HTTP server crates)
- ✅ New crates from happy flow: polling, health, hive-catalog, scheduler, vram-checker, worker-catalog, sse-relay

**Next steps:** Implement READMEs + BDD tests per crate, wire happy flow end-to-end

---

**END OF CODE-BACKED REFINEMENT**  
**Generated:** 2025-10-19  
**Workspace:** `/home/vince/Projects/llama-orch/bin/`
