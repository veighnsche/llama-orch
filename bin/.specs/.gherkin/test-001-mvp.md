# Test-001: Cross-Node Inference Request Flow (MVP with Edge Cases)

## Topology

- **blep** (`blep.home.arpa`): Control node with `rbee-keeper`, `queen-rbee`, `rbee-hive`. Can run CPU workers.
- **workstation** (`workstation.home.arpa`): Compute node with `rbee-hive`, `llm-worker-rbee`. CUDA devices 0, 1, CPU.
- **mac** (`mac.home.arpa`): Compute node with `rbee-hive`, `llm-worker-rbee`. Metal backend only.

## Test Objective

From `blep`, run inference on `mac` using:
```bash
rbee-keeper infer --node mac --model hf:TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
  --prompt "write a short story" --max-tokens 20 --temperature 0.7
```

---

## Happy Path Flow

### **Phase 1: Worker Registry Check**

**rbee-keeper** queries local SQLite registry:
```sql
SELECT * FROM workers 
WHERE node = 'mac' 
  AND model_ref = 'hf:TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF'
  AND state IN ('idle', 'ready')
  AND last_health_check_unix > (NOW() - 60);
```

**IF** worker found and healthy:
- Skip to **Phase 6** (direct inference)

**ELSE**:
- Proceed to **Phase 2**

---

### **Phase 2: Pool Preflight**

**rbee-keeper** → **rbee-hive** on `mac`:

**2.1 Version Check**
```
GET http://mac.home.arpa:8080/v1/health
Authorization: Bearer <api_key>

Response:
{
  "status": "alive",
  "version": "0.1.0",
  "api_version": "v1"
}
```

**IF** version mismatch:
- **ABORT:** "Version mismatch: ctl=0.1.0, pool=0.2.0. Please upgrade."

**IF** connection fails (timeout 10s):
- **Retry** with exponential backoff (3 attempts)
- **ABORT:** "Cannot connect to mac.home.arpa:8080"

---

### **Phase 3: Model Provisioning**

**rbee-hive** checks local model catalog:
```sql
SELECT local_path FROM models 
WHERE reference = 'TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF'
  AND provider = 'hf';
```

**IF** model NOT found:

**3.1 Download Model**
```
rbee-hive → Hugging Face API
Stream progress via SSE to rbee-keeper
```

**SSE Stream:**
```
GET http://mac.home.arpa:8080/v1/models/download/progress?id=<download_id>

data: {"stage": "downloading", "bytes_downloaded": 1048576, "bytes_total": 5242880, "speed_mbps": 45.2}

data: {"stage": "downloading", "bytes_downloaded": 2097152, "bytes_total": 5242880, "speed_mbps": 48.1}

data: {"stage": "complete", "local_path": "/models/tinyllama-q4.gguf"}

data: [DONE]
```

**rbee-keeper** displays:
```
Downloading model... [████████████████████----] 80% (4.0 MB / 5.0 MB) @ 45.2 MB/s
```

**3.2 Register in Catalog**
```sql
INSERT INTO models (id, provider, reference, local_path, size_bytes, downloaded_at_unix)
VALUES ('tinyllama-q4', 'hf', 'TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF', 
        '/models/tinyllama-q4.gguf', 5242880, 1728508603);
```

---

### **Phase 4: Worker Preflight**

**rbee-hive** checks resources:

**4.1 RAM Check**
```rust
let model_size_mb = 5000;  // From catalog
let available_ram_mb = get_available_ram();

if available_ram_mb < model_size_mb * 1.2 {
    return Err("Insufficient RAM: need 6000 MB, have 4000 MB");
}
```

**4.2 Backend Check**
```rust
if !metal_available() {
    return Err("Metal backend not available");
}
```

---

### **Phase 5: Worker Startup**

**rbee-hive** spawns worker:
```bash
llm-worker-rbee \
  --model /models/tinyllama-q4.gguf \
  --backend metal \
  --device 0 \
  --port 8081 \
  --api-key <worker_api_key>
```

**Worker startup sequence:**
1. HTTP server binds to port 8081
2. Worker sends ready callback to pool:
   ```
   POST http://mac.home.arpa:8080/v1/workers/ready
   {
     "worker_id": "worker-abc123",
     "url": "http://mac.home.arpa:8081",
     "model_ref": "hf:TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
     "backend": "metal",
     "device": 0
   }
   ```
3. Model loading begins (async)

**rbee-hive** returns to **rbee-keeper**:
```json
{
  "worker_id": "worker-abc123",
  "url": "http://mac.home.arpa:8081",
  "state": "loading"
}
```

**Pool manager lifecycle:**
- **Remains running as persistent daemon**
- Monitors worker health every 30s
- Enforces idle timeout (5 minutes)

---

### **Phase 6: Worker Registration**

**rbee-keeper** updates local registry:
```sql
INSERT OR REPLACE INTO workers 
  (id, node, url, model_ref, backend, device, state, created_at_unix, last_health_check_unix)
VALUES 
  ('worker-abc123', 'mac', 'http://mac.home.arpa:8081', 
   'hf:TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF', 'metal', 0, 'loading', 
   1728508603, 1728508603);
```

---

### **Phase 7: Worker Health Check**

**rbee-keeper** polls worker:
```
GET http://mac.home.arpa:8081/v1/ready
Authorization: Bearer <worker_api_key>
```

**Response (loading):**
```json
{
  "ready": false,
  "state": "loading",
  "progress_url": "http://mac.home.arpa:8081/v1/loading/progress"
}
```

**rbee-keeper** streams loading progress:
```
GET http://mac.home.arpa:8081/v1/loading/progress

data: {"stage": "loading_to_vram", "layers_loaded": 12, "layers_total": 32, "vram_mb": 2048}

data: {"stage": "loading_to_vram", "layers_loaded": 24, "layers_total": 32, "vram_mb": 4096}

data: {"stage": "ready"}

data: [DONE]
```

**rbee-keeper** displays:
```
Loading model to VRAM... [████████████████████----] 75% (24/32 layers)
Model ready!
```

**Response (ready):**
```
GET /v1/ready
{
  "ready": true,
  "state": "idle",
  "model_loaded": true
}
```

---

### **Phase 8: Inference Execution**

**rbee-keeper** sends inference request:
```
POST http://mac.home.arpa:8081/v1/inference
Authorization: Bearer <worker_api_key>
Content-Type: application/json

{
  "prompt": "write a short story",
  "max_tokens": 20,
  "temperature": 0.7,
  "stream": true
}
```

**Worker response (SSE stream):**
```
HTTP/1.1 200 OK
Content-Type: text/event-stream

data: {"token": "Once", "index": 0}

data: {"token": " upon", "index": 1}

data: {"token": " a", "index": 2}

data: {"token": " time", "index": 3}

...

data: {"done": true, "total_tokens": 20, "duration_ms": 1234}

data: [DONE]
```

**rbee-keeper** streams to stdout:
```
Once upon a time, in a small village, there lived a curious cat named Whiskers who loved to explore.
```

**Worker state transition:**
- `idle` → `busy` (during inference)
- `busy` → `idle` (after completion)

**Pool manager actions:**
- Detects worker is idle
- Starts 5-minute keep-alive timer
- After 5 minutes: sends `POST /v1/admin/shutdown`
- Worker unloads model, releases VRAM, exits

---

## Edge Cases (MVP)

### **EC1: Connection Timeout**

**Scenario:** `mac` is unreachable

**rbee-keeper** behavior:
```
Attempt 1: Connecting to mac.home.arpa:8080... (timeout 10s)
Attempt 2: Connecting to mac.home.arpa:8080... (timeout 10s, delay 200ms)
Attempt 3: Connecting to mac.home.arpa:8080... (timeout 10s, delay 400ms)

Error: Cannot connect to mac.home.arpa:8080 after 3 attempts
Suggestion: Check if rbee-hive is running on mac
```

**Exit code:** 1

---

### **EC2: Model Download Failure**

**Scenario:** Network failure during download

**rbee-hive** behavior:
```
Downloading model... [████████------------] 40% (2.0 MB / 5.0 MB)
Error: Connection reset by peer

Retrying download (attempt 2/6)... delay 100ms
Downloading model... [████████████--------] 60% (3.0 MB / 5.0 MB)
```

**After 6 failed attempts:**
```
Error: Failed to download model after 6 attempts
Last error: Connection timeout
```

**rbee-keeper** receives error and displays to user.

---

### **EC3: Insufficient VRAM**

**Scenario:** Model requires 6 GB, only 4 GB available

**rbee-hive** preflight check:
```rust
let required_vram = 6000;
let available_vram = get_available_vram();  // 4000

if available_vram < required_vram {
    return Err(VramError {
        code: "VRAM_EXHAUSTED",
        message: "Insufficient VRAM: need 6000 MB, have 4000 MB",
        required: 6000,
        available: 4000,
    });
}
```

**rbee-keeper** displays:
```
Error: Insufficient VRAM on mac
  Required: 6000 MB
  Available: 4000 MB
  
Suggestion: Try a smaller quantized model (Q4 instead of Q8)
```

---

### **EC4: Worker Crash During Inference**

**Scenario:** Worker process dies mid-generation

**rbee-keeper** behavior:
```
Once upon a time, in a small village, there lived a curious cat
Error: SSE stream closed unexpectedly

Partial result saved to: /tmp/rbee-partial-abc123.txt
Tokens generated: 12 / 20
```

**Pool manager** detects worker exit:
- Removes worker from registry
- Logs crash event
- Does NOT restart automatically (user must retry)

---

### **EC5: Client Cancellation (Ctrl+C)**

**Scenario:** User presses Ctrl+C during inference

**rbee-keeper** behavior:
```rust
// Signal handler
signal::ctrl_c().await?;

// Send cancellation to worker
DELETE http://mac.home.arpa:8081/v1/inference/<request_id>

// Wait for acknowledgment (max 5s)
match timeout(5s, wait_for_ack()).await {
    Ok(_) => println!("\nCanceled."),
    Err(_) => println!("\nCanceled (worker may still be processing)."),
}

exit(130);  // 128 + SIGINT
```

**Worker** receives cancellation:
- Stops token generation immediately
- Releases slot
- Returns to `idle` state

---

### **EC6: Queue Full**

**Scenario:** Worker already processing a request

**rbee-keeper** sends inference request:
```
POST /v1/inference
```

**Worker** response:
```
HTTP/1.1 503 Service Unavailable
Content-Type: application/json

{
  "error": {
    "code": "ALL_SLOTS_BUSY",
    "message": "Worker is busy, try again later",
    "slots_total": 1,
    "slots_busy": 1
  }
}
```

**rbee-keeper** behavior:
```
Worker is busy, retrying in 1 second...
Worker is busy, retrying in 2 seconds...
Worker is busy, retrying in 4 seconds...

Error: Worker still busy after 3 retries
Suggestion: Wait for current request to complete or use a different node
```

---

### **EC7: Model Loading Timeout**

**Scenario:** Model takes too long to load (>5 minutes)

**rbee-keeper** polling loop:
```rust
let start = Instant::now();
let timeout = Duration::from_secs(300);  // 5 minutes

loop {
    let response = get("/v1/ready").await?;
    
    if response.ready {
        break;
    }
    
    if start.elapsed() > timeout {
        return Err("Model loading timeout after 5 minutes");
    }
    
    sleep(Duration::from_secs(2)).await;
}
```

**Error displayed:**
```
Error: Model loading timeout after 5 minutes
Worker state: loading (stuck at 28/32 layers)

Suggestion: Check worker logs on mac for errors
```

---

### **EC8: Version Mismatch**

**Scenario:** `rbee-keeper` v0.1.0, `rbee-hive` v0.2.0

**rbee-keeper** version check:
```rust
let ctl_version = "0.1.0";
let pool_version = response.version;  // "0.2.0"

if ctl_version != pool_version {
    return Err(VersionError {
        code: "VERSION_MISMATCH",
        message: format!(
            "Version mismatch: rbee-keeper={}, rbee-hive={}",
            ctl_version, pool_version
        ),
        ctl_version,
        pool_version,
    });
}
```

**Error displayed:**
```
Error: Version mismatch
  rbee-keeper: v0.1.0
  rbee-hive: v0.2.0
  
Please upgrade rbee-keeper to v0.2.0:
  cargo install rbee-keeper --version 0.2.0
```

---

### **EC9: Invalid API Key**

**Scenario:** Wrong API key configured

**rbee-keeper** request:
```
GET /v1/health
Authorization: Bearer wrong_key
```

**rbee-hive** response:
```
HTTP/1.1 401 Unauthorized
Content-Type: application/json

{
  "error": {
    "code": "INVALID_API_KEY",
    "message": "Invalid or missing API key"
  }
}
```

**rbee-keeper** displays:
```
Error: Authentication failed
  Invalid API key for mac.home.arpa
  
Check your configuration:
  ~/.rbee/config.yaml
```

---

### **EC10: Idle Timeout (Worker Auto-Shutdown)**

**Scenario:** Worker idle for 5 minutes, user launches game

**Timeline:**
```
T+0:00  - Inference completes, worker → idle
T+0:01  - Pool manager starts 5-minute timer
T+5:00  - Timer expires
T+5:00  - Pool manager: POST /v1/admin/shutdown
T+5:01  - Worker unloads model from VRAM
T+5:02  - Worker exits cleanly
T+5:02  - Pool manager removes from registry
T+5:03  - User launches game → VRAM available ✓
```

**Next inference request:**
```
rbee-keeper infer --node mac --model ...

No idle worker found, starting new worker...
Downloading model... (already cached, skip)
Loading model to VRAM... [████████████████████████] 100%
Model ready!
Once upon a time...
```

**Latency impact:**
- Cold start: ~15-30 seconds (model reload)
- Cached model: ~5-10 seconds (VRAM load only)

---

## Error Response Format

All errors follow this structure:
```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable message",
    "details": {
      "key": "value"
    }
  }
}
```

**Error codes:**
- `CONNECTION_TIMEOUT` - Cannot reach node
- `VERSION_MISMATCH` - Incompatible versions
- `MODEL_NOT_FOUND` - Model doesn't exist
- `DOWNLOAD_FAILED` - Model download error
- `VRAM_EXHAUSTED` - Insufficient VRAM
- `INVALID_API_KEY` - Authentication failure
- `ALL_SLOTS_BUSY` - Worker at capacity
- `WORKER_CRASHED` - Worker process died
- `LOADING_TIMEOUT` - Model load timeout
- `REQUEST_CANCELED` - User canceled request

---

## CLI Command Reference

```bash
# Basic inference
rbee-keeper infer \
  --node mac \
  --model hf:TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
  --prompt "write a short story" \
  --max-tokens 20 \
  --temperature 0.7

# With timeout
rbee-keeper infer ... --timeout 300

# With retry
rbee-keeper infer ... --retry 3

# List available workers
rbee-keeper workers list

# Check worker health
rbee-keeper workers health --node mac

# Manually shutdown worker
rbee-keeper workers shutdown --id worker-abc123

# View logs
rbee-keeper logs --node mac --follow
```

---

## Success Criteria

**Happy path:**
- ✓ Model downloads with progress bar
- ✓ Worker starts and loads model
- ✓ Inference streams tokens in real-time
- ✓ Worker auto-shuts down after 5 minutes idle
- ✓ Total latency < 30 seconds (cold start)

**Edge cases:**
- ✓ Connection failures retry with backoff
- ✓ Version mismatches detected and reported
- ✓ VRAM exhaustion prevents worker startup
- ✓ Worker crashes return partial results
- ✓ Ctrl+C cancels gracefully
- ✓ Busy workers return 503 with retry suggestion
- ✓ Invalid auth returns 401 with helpful message

---

## Next Steps

- [ ] Implement happy path flow
- [ ] Add error handling for all 10 edge cases
- [ ] Write BDD scenarios for each edge case
- [ ] Add integration tests
- [ ] Document API contracts in OpenAPI
