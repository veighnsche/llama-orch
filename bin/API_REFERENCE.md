# rbee API Reference

**Date:** 2025-10-20  
**Version:** 0.1.0

---

## Table of Contents

1. [rbee CLI Commands](#rbee-cli-commands)
2. [queen-rbee HTTP Endpoints](#queen-rbee-http-endpoints)
3. [rbee-hive HTTP Endpoints](#rbee-hive-http-endpoints)
4. [llm-worker-rbee HTTP Endpoints](#llm-worker-rbee-http-endpoints)

---

## rbee CLI Commands

**Binary:** `rbee`  
**Purpose:** CLI tool for managing queen-rbee, hives, workers, and inference

### Queen Management

```bash
# Start queen-rbee daemon
rbee queen start

# Stop queen-rbee daemon
rbee queen stop
```

### Hive Management

```bash
# Start rbee-hive on localhost
rbee hive start

# Stop rbee-hive on localhost
rbee hive stop

# Start rbee-hive on remote hive
rbee hive start --host <HOST> <ACTION>

# Stop rbee-hive on remote hive
rbee hive stop --host <HOST> <ACTION>
```

---

## queen-rbee HTTP Endpoints

**Binary:** `queen-rbee`  
**Default Port:** 8500  
**Purpose:** Orchestrates hives, schedules jobs, receives heartbeats

### Health

```
GET /health
```

**Purpose:** Health check endpoint  
**Response:**
```json
{
  "status": "ok",
  "version": "0.1.0"
}
```

**Used by:** rbee-keeper to check if queen is running

---

### Shutdown

```
POST /shutdown
```

**Purpose:** Graceful shutdown of queen-rbee  
**Response:** `200 OK`

**Note:** Shuts down the HTTP server and exits

---

### Hive Start

```
POST /hive/start
```

**Purpose:** Start a hive (spawn hive process)  
**Request:** None (uses default localhost:8600)  
**Response:**
```json
{
  "hive_url": "http://localhost:8600",
  "hive_id": "localhost",
  "port": 8600
}
```

**Flow:**
1. Queen decides where to spawn (localhost for now)
2. Queen adds hive to catalog (status: Unknown)
3. Queen spawns hive process (fire and forget)
4. Hive sends heartbeat when ready (callback mechanism)

**Note:** Does NOT wait for hive to be ready!

---

### Heartbeat (Hive → Queen)

```
POST /heartbeat
```

**Purpose:** Receive heartbeat from hive  
**Request:**
```json
{
  "hive_id": "localhost",
  "timestamp": "2025-10-20T19:00:00Z",
  "workers": [
    {
      "worker_id": "worker-123",
      "state": "Idle",
      "model": null
    }
  ]
}
```

**Response:**
```json
{
  "status": "ok",
  "message": "Heartbeat received from localhost"
}
```

**Flow:**
1. Hive sends heartbeat (callback when ready)
2. Queen checks if first heartbeat
3. If first → trigger device detection
4. Update catalog with timestamp
5. Return acknowledgement

**Note:** This is the callback mechanism - hive signals it's ready!

---

### Job Create

```
POST /jobs
```

**Purpose:** Create a new inference job  
**Request:**
```json
{
  "model": "HF:author/model",
  "prompt": "Hello, world!",
  "max_tokens": 20,
  "temperature": 0.7
}
```

**Response:**
```json
{
  "job_id": "job-uuid",
  "sse_url": "/jobs/job-uuid/stream"
}
```

**Flow:**
1. Create job in registry
2. Check hive availability
3. Return job_id + SSE URL for streaming

---

### Job Stream

```
GET /jobs/{job_id}/stream
```

**Purpose:** Stream job results via Server-Sent Events (SSE)  
**Response:** SSE stream of tokens

**Example:**
```
data: Hello
data: ,
data:  world
data: !
data: [DONE]
```

**Note:** This is the second call in the dual-call pattern

---

## rbee-hive HTTP Endpoints

**Binary:** `rbee-hive`  
**Default Port:** 8600  
**Purpose:** Manages workers, receives worker heartbeats, sends heartbeats to queen

### Health

```
GET /health
```

**Purpose:** Health check endpoint  
**Response:**
```json
{
  "status": "ok",
  "version": "0.1.0"
}
```

---

### Shutdown

```
POST /v1/shutdown
```

**Purpose:** Graceful shutdown of rbee-hive  
**Response:** `200 OK`

**Note:** Protected endpoint (requires auth token in network mode)

---

### Heartbeat (Worker → Hive)

```
POST /v1/heartbeat
```

**Purpose:** Receive heartbeat from worker  
**Request:**
```json
{
  "worker_id": "worker-123",
  "timestamp_ms": 1729450800000,
  "health_status": "Healthy"
}
```

**Response:**
```json
{
  "status": "ok",
  "message": "Heartbeat received from worker-123"
}
```

**Flow:**
1. Worker sends heartbeat (I'm alive)
2. Hive updates worker registry
3. Return acknowledgement

---

### Device Detection

```
GET /v1/devices
```

**Purpose:** Get device capabilities (CPU, GPU, models, workers)  
**Response:**
```json
{
  "cpu": {
    "cores": 8,
    "ram_gb": 16
  },
  "gpus": [
    {
      "id": "gpu0",
      "name": "NVIDIA RTX 3090",
      "vram_gb": 24
    }
  ],
  "models": 5,
  "workers": 2
}
```

**Note:** Called by queen on first heartbeat

---

### Worker Spawn

```
POST /v1/workers/spawn
```

**Purpose:** Spawn a new worker process  
**Request:**
```json
{
  "model": "HF:author/model",
  "backend": "cuda",
  "device_id": 0
}
```

**Response:**
```json
{
  "worker_id": "worker-uuid",
  "status": "spawned"
}
```

---

### Worker Ready

```
POST /v1/workers/ready
```

**Purpose:** Worker signals it's ready for inference  
**Request:**
```json
{
  "worker_id": "worker-uuid"
}
```

**Response:** `200 OK`

---

### List Workers

```
GET /v1/workers/list
```

**Purpose:** List all workers in registry  
**Response:**
```json
{
  "workers": [
    {
      "worker_id": "worker-123",
      "state": "Idle",
      "model": "HF:author/model"
    }
  ]
}
```

---

### Model Download

```
POST /v1/models/download
```

**Purpose:** Download a model from Hugging Face  
**Request:**
```json
{
  "model": "HF:author/model"
}
```

**Response:**
```json
{
  "status": "downloading",
  "model": "HF:author/model"
}
```

---

### Download Progress

```
GET /v1/models/download/progress
```

**Purpose:** Get download progress for a model  
**Query:** `?model=HF:author/model`  
**Response:**
```json
{
  "model": "HF:author/model",
  "progress": 0.75,
  "status": "downloading"
}
```

---

### Capacity Check

```
GET /v1/capacity
```

**Purpose:** Check VRAM capacity for a model  
**Query:** `?model=HF:author/model&backend=cuda&device_id=0`  
**Response:**
```json
{
  "can_load": true,
  "vram_required_mb": 4096,
  "vram_available_mb": 8192
}
```

---

### Inference (Dual-Call Pattern)

```
POST /v1/inference
```

**Purpose:** Create inference job  
**Request:**
```json
{
  "model": "HF:author/model",
  "prompt": "Hello, world!",
  "max_tokens": 20,
  "temperature": 0.7
}
```

**Response:**
```json
{
  "job_id": "job-uuid",
  "sse_url": "/v1/inference/job-uuid/stream"
}
```

---

```
GET /v1/inference/{job_id}/stream
```

**Purpose:** Stream inference results via SSE  
**Response:** SSE stream of tokens

---

### Metrics

```
GET /metrics
```

**Purpose:** Prometheus metrics endpoint  
**Response:** Prometheus text format

---

## llm-worker-rbee HTTP Endpoints

**Binary:** `llm-worker-rbee`  
**Default Port:** Dynamic (assigned by hive)  
**Purpose:** Runs inference, sends heartbeats to hive

### Health

```
GET /health
```

**Purpose:** Health check endpoint  
**Response:**
```json
{
  "status": "ok",
  "version": "0.1.0"
}
```

---

### Inference Execute

```
POST /execute
```

**Purpose:** Execute inference request  
**Request:**
```json
{
  "prompt": "Hello, world!",
  "max_tokens": 20,
  "temperature": 0.7
}
```

**Response:** SSE stream of tokens

**Note:** Worker-specific endpoint, called by hive

---

## Architecture Summary

### Request Flow

```
User
  ↓
rbee CLI
  ↓
queen-rbee (8500)
  ↓
rbee-hive (8600)
  ↓
llm-worker-rbee (dynamic port)
```

### Heartbeat Flow

```
llm-worker-rbee
  ↓ (every 30s)
POST /v1/heartbeat
  ↓
rbee-hive
  ↓ (every 15s, aggregated)
POST /heartbeat
  ↓
queen-rbee
```

### Callback Mechanism

```
queen-rbee: POST /hive/start
  ↓ (spawn process, return immediately)
rbee-hive: starts up
  ↓ (when ready)
rbee-hive: POST /heartbeat → queen-rbee
  ↓
queen-rbee: triggers device detection
  ↓
queen-rbee: GET /v1/devices → rbee-hive
  ↓
queen-rbee: updates catalog (hive is Online)
```

**Key Point:** Queen doesn't wait - heartbeat is the callback!

---

## Port Summary

| Binary | Default Port | Configurable |
|--------|--------------|--------------|
| queen-rbee | 8500 | Yes (--port) |
| rbee-hive | 8600 | Yes (--port) |
| llm-worker-rbee | Dynamic | Assigned by hive |

---

## Authentication

- **queen-rbee:** No auth (local only)
- **rbee-hive:** Optional JWT auth (network mode)
- **llm-worker-rbee:** No auth (hive-only)

**Network Mode:** When hive is accessed over network, JWT token required for protected endpoints

---

**TEAM-164 OUT** 🎯
