# Test-001: Cross-Node Inference Request Flow (CORRECTED)

**Updated by:** TEAM-038 (aligned with queen-rbee orchestration and narration architecture)  
**Date:** 2025-10-10

---

## Topology

- **blep** = blep.home.arpa (with rbee-keeper and queen-rbee, can run workers on cpu)
- **workstation** = workstation.home.arpa (only rbee-hive and llm-worker-rbee, can run workers on cuda device 0, 1 and cpu)
- **mac** = mac.home.arpa (only rbee-hive and llm-worker-rbee, can only run workers on metal)

---

## Test Objective

On **blep**, I want to run inference on **mac**:
- **Model:** hf:TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF
- **Prompt:** "write a short story"
- **Max tokens:** 20
- **Temperature:** 0.7
- **Backend:** metal, device: 0

**Command:**
```bash
rbee-keeper infer --node mac --model hf:TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
  --prompt "write a short story" --max-tokens 20 --temperature 0.7
```

---

## Complete Flow with Narration Paths

### Phase 1: rbee-keeper → queen-rbee

**rbee-keeper** (on blep) sends task to **queen-rbee** (on blep):
```
POST http://localhost:8080/v2/tasks
{
  "node": "mac",
  "model": "hf:TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
  "prompt": "write a short story",
  "max_tokens": 20,
  "temperature": 0.7
}
```

**Narration:** None yet (just HTTP request)

---

### Phase 2: queen-rbee → rbee-hive (SSH)

**queen-rbee** starts **rbee-hive** on mac via SSH:
```bash
ssh mac.home.arpa "rbee-hive daemon --port 9200"
```

**rbee-hive startup narration:**
```
narrate("rbee-hive starting on port 9200")
  → stdout → SSH tunnel → queen-rbee
  → queen-rbee → stdout → rbee-keeper shell
  → USER SEES: [rbee-hive] 🌅 Starting pool manager on port 9200
```

**rbee-hive HTTP server ready:**
```
narrate("HTTP server listening on 0.0.0.0:9200")
  → stdout → SSH tunnel → queen-rbee
  → queen-rbee → stdout → rbee-keeper shell
  → USER SEES: [http-server] 🚀 HTTP server ready on port 9200
```

---

### Phase 3: queen-rbee checks worker registry

**queen-rbee** queries **rbee-hive** worker registry:
```
GET http://mac.home.arpa:9200/v1/workers/list
```

**Response:** Empty (no workers yet)

**Narration:** None (just HTTP query)

---

### Phase 4: queen-rbee → rbee-hive: Spawn worker

**queen-rbee** sends task to **rbee-hive**:
```
POST http://mac.home.arpa:9200/v1/workers/spawn
{
  "model": "hf:TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
  "backend": "metal",
  "device": 0
}
```

---

### Phase 5: rbee-hive checks model catalog (SQLite)

**rbee-hive** checks model catalog (SQLite at ~/.rbee/models.db):
```sql
SELECT local_path FROM models 
WHERE reference = 'TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF' AND provider = 'hf';
```

**Result:** Not found

**Narration:** None (internal SQLite query)

---

### Phase 6: rbee-hive downloads model

**rbee-hive** downloads model from Hugging Face:

**rbee-hive narration:**
```
narrate("Downloading model from Hugging Face")
  → stdout → SSH tunnel → queen-rbee
  → queen-rbee → SSE → rbee-keeper
  → USER SEES: [model-provisioner] 📦 Downloading model from Hugging Face
```

**Progress updates:**
```
narrate("Downloaded 1 MB / 5 MB (20%)")
  → stdout → SSH tunnel → queen-rbee
  → queen-rbee → SSE → rbee-keeper
  → USER SEES: [model-provisioner] Downloading... [████----] 20% (1 MB / 5 MB)
```

**Download complete:**
```
narrate("Model downloaded to /models/tinyllama-q4.gguf")
  → stdout → SSH tunnel → queen-rbee
  → queen-rbee → SSE → rbee-keeper
  → USER SEES: [model-provisioner] ✅ Model downloaded to /models/tinyllama-q4.gguf
```

---

### Phase 7: rbee-hive registers model in catalog

**rbee-hive** registers model in SQLite:
```sql
INSERT INTO models (id, provider, reference, local_path, size_bytes, downloaded_at_unix)
VALUES ('tinyllama-q4', 'hf', 'TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF', 
        '/models/tinyllama-q4.gguf', 5242880, 1728508603);
```

**Narration:** None (internal SQLite operation)

---

### Phase 8: rbee-hive worker preflight

**rbee-hive** checks RAM:
```rust
let available_ram_mb = get_available_ram();  // 8000 MB
let required_ram_mb = model_size_mb * 1.2;   // 6000 MB

if available_ram_mb < required_ram_mb {
    return Err("Insufficient RAM");
}
```

**Narration:** None (internal check, only narrates if error)

**rbee-hive** checks Metal backend:
```rust
if !metal_available() {
    return Err("Metal backend not available");
}
```

**Narration:** None (internal check, only narrates if error)

---

### Phase 9: rbee-hive spawns worker

**rbee-hive** spawns **llm-worker-rbee**:
```bash
llm-worker-rbee \
  --model /models/tinyllama-q4.gguf \
  --backend metal \
  --device 0 \
  --port 8001 \
  --api-key <worker_api_key>
```

**Worker startup narration (HTTP server NOT ready yet):**
```
worker narrate("Worker starting on port 8001")
  → stdout → rbee-hive captures
  → rbee-hive → SSE → queen-rbee
  → queen-rbee → stdout → rbee-keeper shell
  → USER SEES: [llm-worker-rbee] 🌅 Worker starting on port 8001
```

**Device initialization:**
```
worker narrate("Initialized Metal device 0")
  → stdout → rbee-hive captures
  → rbee-hive → SSE → queen-rbee
  → queen-rbee → stdout → rbee-keeper shell
  → USER SEES: [device-manager] 🖥️ Initialized Metal device 0
```

**Model loading:**
```
worker narrate("Loading model from /models/tinyllama-q4.gguf")
  → stdout → rbee-hive captures
  → rbee-hive → SSE → queen-rbee
  → queen-rbee → stdout → rbee-keeper shell
  → USER SEES: [model-loader] 📦 Loading model from /models/tinyllama-q4.gguf
```

**Model loaded:**
```
worker narrate("Model loaded! 669 MB in VRAM")
  → stdout → rbee-hive captures
  → rbee-hive → SSE → queen-rbee
  → queen-rbee → stdout → rbee-keeper shell
  → USER SEES: [model-loader] 🛏️ Model loaded! 669 MB cozy in VRAM!
```

**HTTP server starts:**
```
worker narrate("HTTP server listening on 0.0.0.0:8001")
  → stdout → rbee-hive captures
  → rbee-hive → SSE → queen-rbee
  → queen-rbee → stdout → rbee-keeper shell
  → USER SEES: [http-server] 🚀 HTTP server ready on port 8001
```

**Worker ready callback:**
```
worker → POST http://mac.home.arpa:9200/v1/workers/ready
{
  "worker_id": "worker-abc123",
  "url": "http://mac.home.arpa:8001",
  "model_ref": "hf:TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
  "backend": "metal",
  "device": 0
}
```

**Narration:**
```
worker narrate("Calling rbee-hive ready callback")
  → stdout → rbee-hive captures
  → rbee-hive → SSE → queen-rbee
  → queen-rbee → stdout → rbee-keeper shell
  → USER SEES: [llm-worker-rbee] 👋 Reporting ready to rbee-hive
```

---

### Phase 10: rbee-hive registers worker

**rbee-hive** updates in-memory registry:
```rust
registry.register(WorkerInfo {
    id: "worker-abc123",
    url: "http://mac.home.arpa:8001",
    model_ref: "hf:TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
    backend: "metal",
    device: 0,
    state: WorkerState::Idle,
    last_activity: SystemTime::now(),
});
```

**Narration:** None (internal registry update)

---

### Phase 11: rbee-hive returns worker URL to queen-rbee

**rbee-hive** responds to queen-rbee:
```json
{
  "worker_id": "worker-abc123",
  "url": "http://mac.home.arpa:8001",
  "state": "idle"
}
```

**Narration:** None (HTTP response)

---

### Phase 12: queen-rbee returns worker URL to rbee-keeper

**queen-rbee** responds to rbee-keeper:
```json
{
  "worker_url": "http://mac.home.arpa:8001",
  "worker_id": "worker-abc123"
}
```

**Narration:** None (HTTP response)

---

### Phase 13: rbee-keeper → worker: Execute inference

**rbee-keeper** sends inference request to **worker** (DIRECT, bypassing rbee-hive):
```
POST http://mac.home.arpa:8001/execute
{
  "job_id": "job-123",
  "prompt": "write a short story",
  "max_tokens": 20,
  "temperature": 0.7
}
```

**Worker inference narration (HTTP server ACTIVE - uses SSE):**

**Inference start:**
```
worker narrate("Starting inference (prompt: 18 chars, max_tokens: 20)")
  → SSE → queen-rbee
  → queen-rbee → stdout → rbee-keeper shell
  → USER SEES: [candle-backend] 🚀 Starting inference (prompt: 18 chars, max_tokens: 20)
```

**Tokenization:**
```
worker narrate("Tokenized prompt (4 tokens)")
  → SSE → queen-rbee
  → queen-rbee → stdout → rbee-keeper shell
  → USER SEES: [tokenizer] 🍰 Tokenized prompt (4 tokens)
```

**Cache reset:**
```
worker narrate("Reset KV cache for fresh start")
  → SSE → queen-rbee
  → queen-rbee → stdout → rbee-keeper shell
  → USER SEES: [candle-backend] 🧹 Reset KV cache for fresh start
```

**Token generation (interleaved with tokens):**
```
SSE stream:
  event: token
  data: {"t":"Once","i":0}
  
  event: token
  data: {"t":" upon","i":1}
  
  event: narration
  data: {"actor":"candle-backend","action":"token_generate","human":"Generated 10 tokens","cute":"🎯"}
  
  event: token
  data: {"t":" a","i":2}
  
  ...
```

**rbee-keeper displays:**
- **Tokens → stdout:** `Once upon a time...`
- **Narration → stderr:** `[candle-backend] 🎯 Generated 10 tokens`

**Inference complete:**
```
worker narrate("Inference complete! 20 tokens in 150ms (133 tok/s)")
  → SSE → queen-rbee
  → queen-rbee → stdout → rbee-keeper shell
  → USER SEES: [candle-backend] 🎉 Inference complete! 20 tokens in 150ms (133 tok/s)
```

---

### Phase 14: Cascading Shutdown

**Worker** transitions to idle state.

**rbee-keeper** exits (user got their result).

**Cascading shutdown sequence:**

**1. rbee-keeper exits:**
```
rbee-keeper completes inference
rbee-keeper displays final result
rbee-keeper sends SIGTERM to queen-rbee (if it spawned it)
rbee-keeper exits
```

**2. queen-rbee shuts down:**
```
queen-rbee receives SIGTERM
queen-rbee sends shutdown to all rbee-hive instances via SSH
queen-rbee exits
```

**3. rbee-hive shuts down:**
```
rbee-hive receives shutdown signal via SSH
rbee-hive sends POST http://mac.home.arpa:8001/shutdown to all workers
rbee-hive waits for workers to exit
rbee-hive exits
```

**4. Worker shuts down:**

**Worker shutdown narration (HTTP server closing - uses stdout):**
```
worker narrate("Shutting down gracefully")
  → stdout → rbee-hive captures
  → rbee-hive → SSE → queen-rbee (if still connected)
  → queen-rbee → stdout → rbee-keeper shell (already exited, not seen)
```

**VRAM freed:**
```
worker narrate("Freeing 669 MB VRAM")
  → stdout → rbee-hive captures
  → rbee-hive logs (queen-rbee already exited, not relayed)
```

**Worker exits:**
```
worker narrate("Worker exiting")
  → stdout → rbee-hive captures
  → rbee-hive logs (queen-rbee already exited, not relayed)
worker process exits
```

**Final state:**
- rbee-keeper: exited
- queen-rbee: exited
- rbee-hive: exited
- worker: exited
- VRAM: freed (available for games/other apps)

**Note:** Shutdown narration is typically NOT seen by user because rbee-keeper has already exited. This is by design - user got their result and moved on.

---

## Critical Corrections Applied

### ❌ WRONG (Original)
- "pool manager dies, worker lives"
- "ctl adds the worker details is last seen alive in the worker registry"
- "ctl runs a health check"
- "ctl runs execute"
- "ctl streams tokens to stdout"

### ✅ CORRECT (Updated)
- **rbee-hive is persistent daemon** (but dies when queen-rbee shuts down)
- **rbee-hive maintains worker registry** (in-memory, not ctl)
- **queen-rbee orchestrates** (not ctl)
- **rbee-keeper sends execute directly to worker** (bypasses rbee-hive)
- **rbee-keeper displays tokens to stdout, narration to stderr**
- **Cascading shutdown:** rbee-keeper → queen-rbee → rbee-hive → workers
- **Worker does NOT stay alive** after rbee-keeper exits

---

## Narration Flow Summary

**Before HTTP server ready (worker startup):**
```
worker narrate() → stdout → rbee-hive captures → SSE → queen-rbee → stdout → user shell
```

**During HTTP server active (inference):**
```
worker narrate() → SSE → queen-rbee → stdout → user shell
```

**After HTTP server closing (shutdown):**
```
worker narrate() → stdout → rbee-hive captures → SSE → queen-rbee → stdout → user shell
```

**All narration ends up in user's shell. The transport is just plumbing.**

---

**Updated by:** TEAM-038  
**Date:** 2025-10-10  
**Status:** ✅ CORRECTED