# worker-contract

**TEAM-270:** Worker contract types and API specification

## Overview

This crate defines the contract that **ALL worker implementations** must follow in the rbee system.

Workers can be:
- **Bespoke**: Custom implementations using Candle ML framework (e.g., `llm-worker-rbee`)
- **Adapters**: Wrappers around existing inference engines (e.g., `llama-cpp-adapter`, `vllm-adapter`)

All workers communicate with `queen-rbee` using this contract.

## Key Types

### WorkerInfo

Complete worker state, sent in heartbeats and returned by `/info` endpoint:

```rust
pub struct WorkerInfo {
    pub id: String,              // Unique worker ID
    pub model_id: String,        // Model being served
    pub device: String,          // Device (e.g., "GPU-0")
    pub port: u16,               // HTTP port
    pub status: WorkerStatus,    // Current status
    pub implementation: String,  // Worker type
    pub version: String,         // Worker version
}
```

### WorkerStatus

Current worker state:

```rust
pub enum WorkerStatus {
    Starting,  // Loading model
    Ready,     // Ready for inference
    Busy,      // Processing request
    Stopped,   // Gracefully stopped
}
```

### WorkerHeartbeat

Periodic status update sent to queen:

```rust
pub struct WorkerHeartbeat {
    pub worker: WorkerInfo,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}
```

## Heartbeat Protocol

- **Frequency:** Every 30 seconds
- **Endpoint:** `POST /v1/worker-heartbeat` on queen
- **Timeout:** 90 seconds (3 missed heartbeats)
- **Action:** Queen marks worker as unavailable

## Worker HTTP API

All workers must implement these endpoints:

### GET /health

Health check. Returns `200 OK` with body `"ok"`.

### GET /info

Returns worker information as JSON:

```json
{
  "id": "worker-abc123",
  "model_id": "meta-llama/Llama-2-7b",
  "device": "GPU-0",
  "port": 9301,
  "status": "ready",
  "implementation": "llm-worker-rbee",
  "version": "0.1.0"
}
```

### POST /v1/infer

Execute inference. Request:

```json
{
  "prompt": "Hello, world!",
  "max_tokens": 100,
  "temperature": 0.7,
  "stream": true
}
```

Response (non-streaming):

```json
{
  "text": "Hello! How can I help you today?",
  "tokens_generated": 8,
  "duration_ms": 250
}
```

Response (streaming via SSE):

```
data: {"token": "Hello"}
data: {"token": "!"}
data: [DONE]
```

## OpenAPI Specification

See `contracts/openapi/worker-api.yaml` for complete API specification.

## Usage Example

```rust
use worker_contract::{WorkerInfo, WorkerStatus, WorkerHeartbeat};
use chrono::Utc;

// Create worker info
let worker = WorkerInfo {
    id: "worker-abc123".to_string(),
    model_id: "meta-llama/Llama-2-7b".to_string(),
    device: "GPU-0".to_string(),
    port: 9301,
    status: WorkerStatus::Ready,
    implementation: "llm-worker-rbee".to_string(),
    version: "0.1.0".to_string(),
};

// Send heartbeat to queen
let heartbeat = WorkerHeartbeat::new(worker);
// POST to http://queen:8500/v1/worker-heartbeat
```

## Implementation Checklist

When implementing a new worker:

- [ ] Implement `GET /health` endpoint
- [ ] Implement `GET /info` endpoint
- [ ] Implement `POST /v1/infer` endpoint
- [ ] Send heartbeat to queen every 30 seconds
- [ ] Handle graceful shutdown (set status to `Stopped`)
- [ ] Report accurate status (`Starting` → `Ready` → `Busy` → `Ready`)

## Architecture

```
Hive spawns worker → Worker loads model → Worker reports ready → Worker accepts requests
                                   ↓
                           Heartbeat every 30s to queen
                                   ↓
                           Queen tracks worker in registry
                                   ↓
                           Queen routes inference to worker
```

## Extension Points

Future enhancements:

- **Multi-model workers**: Workers serving multiple models (vLLM, ComfyUI)
- **Dynamic VRAM**: Report changing VRAM usage in heartbeat
- **Workflow progress**: Report progress for long-running tasks (ComfyUI)
- **Batch inference**: Support for batched requests (vLLM)

## License

GPL-3.0-or-later
