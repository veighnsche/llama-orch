# Queen Job Operations Reference

**Date:** Oct 29, 2025  
**Status:** ✅ COMPLETE  
**Source:** `bin/97_contracts/operations-contract/src/lib.rs`

---

## Overview

**CRITICAL ARCHITECTURE:**

The Queen's job server (`POST /v1/jobs`) handles ONLY orchestration operations:
- **Status** - Live status from registries
- **Infer** - Scheduling and routing to workers

**Worker/Model lifecycle operations are NOT exposed through queen's job server.**

If you want to manage workers/models:
- **CLI:** rbee-keeper connects directly to hive's job server
- **GUI:** rbee-keeper opens hive's web UI in iframe

**NO PROXYING** - Queen doesn't forward operations to hive. Talk to hive directly.

---

## Queen's Job Operations (Public API)

### 1. **Status**
```json
{
  "operation": "status"
}
```
**Purpose:** Show live status of all hives and workers from registry  
**Handler:** Queen (queries registries)  
**Returns:** Current state of all hives and workers

---

### 2. **Infer**
```json
{
  "operation": "infer",
  "hive_id": "localhost",
  "model": "meta-llama/Llama-3.2-1B",
  "prompt": "Hello, how are you?",
  "max_tokens": 100,
  "temperature": 0.7,
  "top_p": 0.9,
  "stream": true
}
```
**Purpose:** Run inference  
**Handler:** Queen (scheduling + direct routing to worker)  
**Flow:**
1. Queen checks worker registry for available worker
2. If no worker: send `WorkerSpawn` job to hive, wait for heartbeat
3. Queen routes request DIRECTLY to worker (bypassing hive)
4. Queen relays SSE stream back to client

**CRITICAL:** Inference NEVER goes through hive. Queen routes directly to worker.

---

## Internal Queen Operations (NOT Exposed)

These operations exist in the codebase but are **internal to queen** for orchestration. They are NOT exposed through queen's job server.

### Internal Worker Spawn (Queen → Hive)
When queen needs a worker for inference, it internally sends a job to the hive:
```json
{
  "operation": "worker_spawn",
  "hive_id": "localhost",
  "model": "meta-llama/Llama-3.2-1B",
  "worker": "cpu",
  "device": 0
}
```
**This is sent directly to hive's job server, NOT through queen's job API.**

### Internal Model Download (Queen → Hive)
When queen needs a model, it internally sends a job to the hive:
```json
{
  "operation": "model_download",
  "hive_id": "localhost",
  "model": "meta-llama/Llama-3.2-1B"
}
```
**This is sent directly to hive's job server, NOT through queen's job API.**

---

## Hive Job Operations (Direct Access)

**To manage workers/models manually, connect directly to hive's job server.**

**Hive Job Server:** `http://localhost:7835/v1/jobs`

### Worker Operations (on Hive)

#### WorkerSpawn
```bash
curl -X POST http://localhost:7835/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "operation": "worker_spawn",
    "hive_id": "localhost",
    "model": "meta-llama/Llama-3.2-1B",
    "worker": "cpu",
    "device": 0
  }'
```

#### WorkerProcessList
```bash
curl -X POST http://localhost:7835/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "operation": "worker_process_list",
    "hive_id": "localhost"
  }'
```

#### WorkerProcessGet
```bash
curl -X POST http://localhost:7835/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "operation": "worker_process_get",
    "hive_id": "localhost",
    "worker_id": "worker-123"
  }'
```

#### WorkerProcessDelete
```bash
curl -X POST http://localhost:7835/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "operation": "worker_process_delete",
    "hive_id": "localhost",
    "worker_id": "worker-123"
  }'
```

### Model Operations (on Hive)

#### ModelDownload
```bash
curl -X POST http://localhost:7835/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "operation": "model_download",
    "hive_id": "localhost",
    "model": "meta-llama/Llama-3.2-1B"
  }'
```

#### ModelList
```bash
curl -X POST http://localhost:7835/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "operation": "model_list",
    "hive_id": "localhost"
  }'
```

#### ModelGet
```bash
curl -X POST http://localhost:7835/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "operation": "model_get",
    "hive_id": "localhost",
    "model": "meta-llama/Llama-3.2-1B"
  }'
```

#### ModelDelete
```bash
curl -X POST http://localhost:7835/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "operation": "model_delete",
    "hive_id": "localhost",
    "model": "meta-llama/Llama-3.2-1B"
  }'
```

---

## OpenAI-Compatible Endpoints

The Queen also provides OpenAI-compatible endpoints via `rbee-openai-adapter`.

### 1. **Chat Completions**

**Endpoint:** `POST /openai/v1/chat/completions`

**Request:**
```json
{
  "model": "llama-3-8b",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "temperature": 0.7,
  "max_tokens": 100,
  "stream": true
}
```

**Translation:**
- Converts to `Operation::Infer`
- Concatenates messages into single prompt
- Maps OpenAI parameters to rbee parameters
- Transforms response to OpenAI format

**Response (streaming):**
```
data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"llama-3-8b","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"llama-3-8b","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

data: [DONE]
```

---

### 2. **List Models**

**Endpoint:** `GET /openai/v1/models`

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "llama-3-8b",
      "object": "model",
      "created": 1234567890,
      "owned_by": "rbee"
    }
  ]
}
```

---

### 3. **Get Model**

**Endpoint:** `GET /openai/v1/models/{model}`

**Response:**
```json
{
  "id": "llama-3-8b",
  "object": "model",
  "created": 1234567890,
  "owned_by": "rbee"
}
```

---

### 4. **Completions (Legacy)**

**Endpoint:** `POST /openai/v1/completions`

**Purpose:** Legacy OpenAI completions endpoint (pre-chat)  
**Status:** Supported for backward compatibility

---

### 5. **Embeddings**

**Endpoint:** `POST /openai/v1/embeddings`

**Purpose:** Generate embeddings for text  
**Status:** Planned (not yet implemented)

---

## Architecture Summary

### rbee-keeper CLI

```
rbee-keeper CLI
  ├─→ Queen Job Server (http://localhost:7833/v1/jobs)
  │   ├─ Status
  │   └─ Infer
  │
  └─→ Hive Job Server (http://localhost:7835/v1/jobs)
      ├─ WorkerSpawn, WorkerProcessList, WorkerProcessGet, WorkerProcessDelete
      └─ ModelDownload, ModelList, ModelGet, ModelDelete
```

**NO PROXYING** - CLI talks directly to queen AND hive.

### rbee-keeper GUI

```
rbee-keeper GUI
  ├─→ Queen Web UI (iframe: http://localhost:7833/)
  ├─→ Hive Web UI (iframe: http://localhost:7835/)
  └─→ Worker Web UI (iframe: http://localhost:8080/)
```

**Direct SDK access** - GUI opens web UIs in iframes, uses SDK directly.

### Queen Internal Operations

When queen needs workers/models for inference:
```
Queen (internal) → Hive Job Server (http://localhost:7835/v1/jobs)
  ├─ WorkerSpawn (if no worker available)
  └─ ModelDownload (if model not available)
```

**These are internal queen operations, NOT exposed through queen's job API.**

### Inference Flow

```
Client → Queen (scheduling) → Worker (DIRECT)
      ↘ Hive (internal: spawn worker if needed)
```

**CRITICAL:** 
- Hive is NEVER in the inference path
- Queen routes directly to worker
- Hive only used for worker lifecycle (internal queen operation)

---

## Usage Examples

### Example 1: Manual Worker Management (via Hive)

```bash
# 1. Spawn worker (talk to HIVE directly)
curl -X POST http://localhost:7835/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "operation": "worker_spawn",
    "hive_id": "localhost",
    "model": "meta-llama/Llama-3.2-1B",
    "worker": "cpu",
    "device": 0
  }'

# 2. Wait for worker heartbeat (automatic)

# 3. Run inference (talk to QUEEN)
curl -X POST http://localhost:7833/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "operation": "infer",
    "hive_id": "localhost",
    "model": "meta-llama/Llama-3.2-1B",
    "prompt": "Hello!",
    "max_tokens": 50,
    "temperature": 0.7,
    "stream": true
  }'
```

### Example 1b: Automatic Worker Management (Queen Handles It)

```bash
# Just run inference - queen spawns worker if needed
curl -X POST http://localhost:7833/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "operation": "infer",
    "hive_id": "localhost",
    "model": "meta-llama/Llama-3.2-1B",
    "prompt": "Hello!",
    "max_tokens": 50,
    "temperature": 0.7,
    "stream": true
  }'

# Queen internally:
# 1. Checks worker registry
# 2. If no worker: sends WorkerSpawn to hive (internal)
# 3. Waits for worker heartbeat
# 4. Routes inference directly to worker
```

### Example 2: OpenAI-Compatible

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:7833/openai",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="llama-3-8b",
    messages=[
        {"role": "user", "content": "Hello!"}
    ],
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### Example 3: Check Status

```bash
# Get status from queen (queries registries)
curl -X POST http://localhost:7833/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "operation": "status"
  }'
```

### Example 4: List Models (via Hive)

```bash
# Talk to HIVE directly
curl -X POST http://localhost:7835/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "operation": "model_list",
    "hive_id": "localhost"
  }'
```

---

## References

- **Source:** `bin/97_contracts/operations-contract/src/lib.rs`
- **OpenAI Adapter:** `bin/15_queen_rbee_crates/rbee-openai-adapter/`
- **Job Router:** `bin/10_queen_rbee/src/job_router.rs`
- **Hive Forwarder:** `bin/10_queen_rbee/src/hive_forwarder.rs`

---

**Document Version:** 2.0  
**Last Updated:** Oct 29, 2025  
**Queen Public API:** 2 operations (Status, Infer) + 5 OpenAI-compatible  
**Hive Public API:** 8 operations (Worker/Model lifecycle)  
**NO PROXYING:** rbee-keeper talks directly to queen AND hive
