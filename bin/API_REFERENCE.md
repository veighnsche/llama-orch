# rbee API Reference

<!-- TEAM-185: Updated API reference with new inference parameters and worker spawn syntax -->

## rbee CLI

```bash
rbee queen start                              # Start queen-rbee daemon
rbee queen stop                               # Stop queen-rbee daemon
rbee hive start {id}                          # Start hive
rbee hive stop {id}                           # Stop hive
rbee hive list                                # List hives
rbee hive get {id}                            # Get hive details
rbee hive create --host H --port P            # Create hive
rbee hive update {id}                         # Update hive
rbee hive delete {id}                         # Delete hive
rbee worker --hive-id {id} spawn --model M --worker {cpu|cuda|metal} --device 0
rbee worker --hive-id {id} list               # List workers on hive
rbee worker --hive-id {id} get {id}           # Get worker details
rbee worker --hive-id {id} delete {id}        # Delete worker
rbee model --hive-id {id} download {model}    # Download model on hive
rbee model --hive-id {id} list                # List models on hive
rbee model --hive-id {id} get {id}            # Get model details
rbee model --hive-id {id} delete {id}         # Delete model
rbee infer --hive-id localhost --model M "prompt" \
  --max-tokens 20 --temperature 0.7 \
  --top-p 0.9 --top-k 50 \
  --device {cpu|cuda|metal} \
  --worker-id {id} \
  --stream true
```

**Config:** `~/.config/rbee/config.toml`
```toml
queen_port = 8500  # Persistent queen port configuration
```

---

## queen-rbee (Port 8500)

**Job-based architecture:** All operations go through POST /v1/jobs

```
GET  /health                     # Health check
POST /v1/heartbeat               # Receive hive heartbeat (callback)
POST /v1/jobs                    # Submit job (ALL operations)
GET  /v1/jobs/{id}/stream        # Stream job narration (SSE)
GET  /narration/stream           # Global narration stream (SSE)
```

**Job payloads:**
```json
// Hive operations
{"operation": "hive_start", "hive_id": "..."}
{"operation": "hive_stop", "hive_id": "..."}
{"operation": "hive_list"}
{"operation": "hive_get", "id": "..."}
{"operation": "hive_create", "host": "...", "port": 8600}
{"operation": "hive_update", "id": "..."}
{"operation": "hive_delete", "id": "..."}

// Worker operations (routed to hive)
{"operation": "worker_spawn", "hive_id": "localhost", "model": "...", "worker": "cpu|cuda|metal", "device": 0}
{"operation": "worker_list", "hive_id": "localhost"}
{"operation": "worker_get", "hive_id": "localhost", "id": "..."}
{"operation": "worker_delete", "hive_id": "localhost", "id": "..."}

// Model operations (routed to hive)
{"operation": "model_download", "hive_id": "localhost", "model": "..."}
{"operation": "model_list", "hive_id": "localhost"}
{"operation": "model_get", "hive_id": "localhost", "id": "..."}
{"operation": "model_delete", "hive_id": "localhost", "id": "..."}

// Inference (routed to hive)
{
  "operation": "infer",
  "hive_id": "localhost",           // Which hive to run on (default: localhost)
  "model": "...",                   // Model identifier
  "prompt": "...",                  // Input prompt
  "max_tokens": 20,                 // Max tokens to generate
  "temperature": 0.7,               // Sampling temperature
  "top_p": 0.9,                     // Nucleus sampling (optional)
  "top_k": 50,                      // Top-k sampling (optional)
  "device": "cuda",                 // Device type: cpu, cuda, or metal (filters compatible workers)
  "worker_id": "...",               // Specific worker ID (optional)
  "stream": true                    // Stream tokens as generated (default: true)
}
```

---

## rbee-hive (Port 8600)

**Job-based architecture:** All operations go through POST /v1/jobs

```
GET  /health                     # Health check
POST /v1/heartbeat               # Receive worker heartbeat (callback)
GET  /v1/devices                 # Device detection (direct query)
POST /v1/jobs                    # Submit job (ALL operations)
GET  /v1/jobs/{id}/stream        # Stream job narration (SSE)
GET  /metrics                    # Prometheus metrics
```

**Job payloads:**
```json
// Worker operations
{"operation": "worker_spawn", "model": "...", "worker": "cpu|cuda|metal", "device": 0}
{"operation": "worker_list"}
{"operation": "worker_get", "id": "..."}
{"operation": "worker_delete", "id": "..."}

// Model operations
{"operation": "model_download", "model": "..."}
{"operation": "model_list"}
{"operation": "model_get", "id": "..."}
{"operation": "model_delete", "id": "..."}

// Inference
{
  "operation": "infer",
  "model": "...",                   // Model identifier
  "prompt": "...",                  // Input prompt
  "max_tokens": 20,                 // Max tokens to generate
  "temperature": 0.7,               // Sampling temperature
  "top_p": 0.9,                     // Nucleus sampling (optional)
  "top_k": 50,                      // Top-k sampling (optional)
  "device": "cuda",                 // Device type: cpu, cuda, or metal (filters compatible workers)
  "worker_id": "...",               // Specific worker ID (optional)
  "stream": true                    // Stream tokens as generated (default: true)
}
```

---

## llm-worker-rbee (Dynamic Port)

```
GET  /health                        # Health check
POST /v1/inference                  # Create job, return job_id + sse_url
GET  /v1/inference/{job_id}/stream  # Stream results via SSE
```

---

## Flow

```
User → rbee CLI → queen-rbee → rbee-hive → llm-worker-rbee
```

## Heartbeat

```
worker → hive (30s) → queen (15s, aggregated)
```

## Callback

```
queen: POST /hive/start (spawn, return)
hive: starts up
hive: POST /heartbeat → queen (I'm ready!)
queen: GET /v1/devices → hive (detect)
queen: updates catalog (hive Online)
```
