# Multi-Modality Streaming Architecture Plan

**Author**: Specs Team  
**Date**: 2025-10-05  
**Status**: Planning Document  
**Purpose**: Address streaming architecture for multiple worker adapter types (text, image, audio, embeddings)

---

## 1. Problem Statement

### 1.1 Current Architecture Assumption

The current specs (`00_llama-orch.md`) assume **text-only streaming** via SSE (Server-Sent Events):

```
Client → Orchestrator → Worker (text-gen)
         ↓ SSE relay
         text tokens streamed back
```

**Key assumptions in current specs:**
- Workers stream text tokens via SSE
- Event types: `started`, `token`, `metrics`, `end`, `error`
- All workers use same streaming protocol
- Orchestrator acts as SSE relay (SYS-6.1.1: "MUST relay SSE streams from workers to clients")

### 1.2 The Gap

With **worker adapters** (see `bin/pool-managerd/.specs/10_worker_adapters.md`), we now support:

1. **Text generation workers** (LLMs) → Stream text tokens
2. **Image generation workers** (Stable Diffusion) → Return images (base64, binary, URL)
3. **Audio generation workers** → Return audio files
4. **Embedding workers** → Return vector embeddings (no streaming)
5. **Multimodal workers** → Mixed outputs (text + images)

**The problem**: Different modalities have different output formats and streaming requirements. The current "SSE text token relay" model doesn't fit all worker types.

---

## 2. Current Architecture Analysis

### 2.1 What the Specs Already Support

✅ **Pool manager bypassing for streaming** (SYS-4.3.2, SYS-5.4.x):
- Orchestrator receives worker URI from pool manager
- Orchestrator calls worker endpoints **directly** for inference
- Pool manager is **not** in the streaming path
- Quote from SYS-4.3.2: "The orchestrator directly calls worker endpoints to proxy/relay requests"

✅ **Worker adapter abstraction** (POOL-1xxx):
- Workers advertise **capabilities** (text-gen, image-gen, audio-gen, embedding)
- Orchestrator routes jobs based on **capability**, not worker type
- Adapters normalize worker state to common format

✅ **Capability-based routing** (POOL-1005):
- Jobs specify required capability
- Orchestrator selects worker with matching capability
- Worker type is abstracted away

### 2.2 What Needs to Change

❌ **Streaming protocol assumption**:
- Current: All workers use SSE with `token` events
- Reality: Image workers return binary data, embedding workers return JSON, etc.

❌ **Event type rigidity**:
- Current: Fixed event types (`started`, `token`, `metrics`, `end`, `error`)
- Reality: Image workers need `image_chunk` or `image_complete`, audio workers need `audio_chunk`

❌ **Content-Type handling**:
- Current: Assumes `text/event-stream` for all workers
- Reality: Image workers may use `application/octet-stream`, `multipart/form-data`, or `application/json`

---

## 3. Proposed Solution: Capability-Aware Streaming

### 3.1 Core Principle

**Workers expose capability-specific streaming protocols. Orchestrator acts as a protocol-aware relay based on job capability.**

**Orchestrator awareness**: The orchestrator MUST be aware of different worker types and their protocols to correctly relay responses. This awareness comes from:
1. **Worker metadata** (via pool manager heartbeat) — includes worker type and capabilities
2. **Job capability** (from client request) — determines expected protocol
3. **Protocol mapping** (in orchestrator config/code) — maps capability → streaming protocol

The orchestrator does NOT need to understand worker internals (CUDA vs Metal vs CPU), only the **protocol contract** each capability exposes.

### 3.2 Architecture

```
┌─────────────────────────────────────────────────────────┐
│ Orchestrator (Protocol-Aware Relay)                     │
│                                                          │
│  Worker metadata (from pool mgr) → Protocol mapping     │
│  Job capability → Select relay strategy                 │
│  • text-gen     → SSE relay (token events)              │
│  • image-gen    → Binary/JSON relay (image data)        │
│  • audio-gen    → Binary relay (audio chunks)           │
│  • embedding    → JSON response (no streaming)          │
│  • multimodal   → Mixed SSE (text + image events)       │
└────────────────────┬────────────────────────────────────┘
                     │ Direct HTTP call (bypass pool mgr)
                     ↓
┌────────────────────────────────────────────────────────┐
│ Worker (Capability-Specific Protocol)                  │
│                                                         │
│  • worker-orcd (text-gen)  → SSE: started, token*, end │
│  • sd-worker (image-gen)   → JSON: {image: "base64..."} │
│  • tts-worker (audio-gen)  → Binary stream: audio      │
│  • embed-worker (embedding)→ JSON: {embedding: [...]}  │
└─────────────────────────────────────────────────────────┘

Worker Types (array of implementations):
  - worker-orcd (bespoke CUDA, text-gen) ← M0, in 00_llama-orch.md
  - worker-aarmd (Apple Metal, text-gen) ← TODO, needs own spec
  - sd-worker (Stable Diffusion, image-gen) ← TODO, needs own spec
  - tts-worker (TTS, audio-gen) ← TODO, needs own spec
  - embed-worker (embeddings) ← TODO, needs own spec
```

### 3.3 Key Design Decisions

1. **Pool manager remains bypassed** ✅
   - Orchestrator still calls worker endpoints directly
   - Pool manager only handles lifecycle (spawn, stop, health)
   - No change to existing bypass architecture

2. **Orchestrator is protocol-aware, not worker-aware** ✅
   - Orchestrator knows: capability → protocol mapping
   - Orchestrator does NOT know: worker internals (CUDA kernels, Metal shaders, etc.)
   - Worker type is opaque; only capability and protocol matter

3. **Capability determines protocol** ✅
   - Job's `capability` field determines expected streaming protocol
   - Orchestrator selects relay strategy based on capability
   - Workers implement capability-specific endpoints

4. **Backward compatibility** ✅
   - Existing text-gen workers (worker-orcd) continue using SSE
   - New capabilities add new protocols, don't break existing ones

---

## 4. Detailed Design

### 4.1 Worker API Contract Extension

#### 4.1.1 Text Generation Workers (Existing)

**Capability**: `text-gen`  
**Protocol**: SSE (`text/event-stream`)  
**Endpoint**: `POST {worker_uri}/execute`

```
Response: SSE stream
event: started
data: {"job_id": "job-xyz"}

event: token
data: {"token": "Hello", "index": 0}

event: token
data: {"token": " world", "index": 1}

event: end
data: {"tokens_generated": 2}
```

**No changes needed** — existing protocol.

---

#### 4.1.2 Image Generation Workers (New)

**Capability**: `image-gen`  
**Protocol**: JSON response (non-streaming) OR SSE with image events  
**Endpoint**: `POST {worker_uri}/execute`

**Option A: JSON Response (Simple)**
```json
POST /execute
{
  "job_id": "job-xyz",
  "prompt": "a cat on a couch",
  "width": 1024,
  "height": 1024,
  "steps": 50
}

Response (200 OK):
{
  "job_id": "job-xyz",
  "images": [
    {
      "format": "png",
      "data": "base64_encoded_image_data...",
      "width": 1024,
      "height": 1024
    }
  ],
  "metadata": {
    "steps_taken": 50,
    "seed": 42
  }
}
```

**Option B: SSE with Progress (Advanced)**
```
Response: SSE stream
event: started
data: {"job_id": "job-xyz", "steps": 50}

event: progress
data: {"step": 10, "total": 50, "percent": 20}

event: progress
data: {"step": 25, "total": 50, "percent": 50}

event: image
data: {"format": "png", "data": "base64...", "width": 1024, "height": 1024}

event: end
data: {"steps_taken": 50}
```

**Recommendation**: Start with Option A (JSON response), add Option B (SSE progress) in future milestone.

---

#### 4.1.3 Embedding Workers (New)

**Capability**: `embedding`  
**Protocol**: JSON response (non-streaming)  
**Endpoint**: `POST {worker_uri}/execute`

```json
POST /execute
{
  "job_id": "job-xyz",
  "text": "Hello world",
  "model": "text-embedding-ada-002"
}

Response (200 OK):
{
  "job_id": "job-xyz",
  "embedding": [0.123, -0.456, 0.789, ...],  // 1536 dimensions
  "dimensions": 1536,
  "metadata": {
    "tokens_processed": 2
  }
}
```

**No streaming** — single JSON response.

---

#### 4.1.4 Audio Generation Workers (New)

**Capability**: `audio-gen`  
**Protocol**: Binary stream OR JSON with base64  
**Endpoint**: `POST {worker_uri}/execute`

**Option A: Binary Stream**
```
POST /execute
{
  "job_id": "job-xyz",
  "text": "Hello world",
  "voice": "en-US-male",
  "format": "mp3"
}

Response (200 OK):
Content-Type: audio/mpeg
Content-Length: 12345

<binary audio data>
```

**Option B: JSON with Base64**
```json
Response (200 OK):
{
  "job_id": "job-xyz",
  "audio": {
    "format": "mp3",
    "data": "base64_encoded_audio...",
    "duration_ms": 1500
  }
}
```

**Recommendation**: Option A (binary stream) for efficiency.

---

#### 4.1.5 Multimodal Workers (Future)

**Capability**: `multimodal`  
**Protocol**: SSE with mixed event types  
**Endpoint**: `POST {worker_uri}/execute`

```
Response: SSE stream
event: started
data: {"job_id": "job-xyz"}

event: token
data: {"token": "Here", "index": 0}

event: token
data: {"token": " is", "index": 1}

event: image
data: {"format": "png", "data": "base64...", "caption": "a cat"}

event: token
data: {"token": " a", "index": 2}

event: token
data: {"token": " cat", "index": 3}

event: end
data: {"tokens": 4, "images": 1}
```

---

### 4.2 Orchestrator Relay Strategy

#### 4.2.1 Capability-Based Relay Selection

```rust
// Orchestrator streaming module
async fn relay_worker_response(
    job: &Job,
    worker_uri: &str,
    client_stream: SseStream,
) -> Result<(), RelayError> {
    match job.capability {
        Capability::TextGen => {
            // SSE relay (existing)
            relay_sse_stream(worker_uri, client_stream).await
        }
        Capability::ImageGen => {
            // JSON relay
            relay_json_response(worker_uri, client_stream).await
        }
        Capability::Embedding => {
            // JSON relay (no streaming)
            relay_json_response(worker_uri, client_stream).await
        }
        Capability::AudioGen => {
            // Binary relay
            relay_binary_stream(worker_uri, client_stream).await
        }
        Capability::Multimodal => {
            // Mixed SSE relay
            relay_mixed_sse_stream(worker_uri, client_stream).await
        }
    }
}
```

#### 4.2.2 SSE Relay (Text Generation)

**Existing implementation** — no changes:

```rust
async fn relay_sse_stream(
    worker_uri: &str,
    client_stream: SseStream,
) -> Result<(), RelayError> {
    let worker_stream = reqwest::get(format!("{}/execute", worker_uri))
        .await?
        .bytes_stream();
    
    // Relay SSE events from worker to client
    // Add orchestrator metadata (correlation_id, queue_time, etc.)
    // Preserve event order
    
    for await event in worker_stream {
        let enriched_event = add_orchestrator_metadata(event);
        client_stream.send(enriched_event).await?;
    }
    
    Ok(())
}
```

#### 4.2.3 JSON Relay (Image/Embedding)

**New implementation**:

```rust
async fn relay_json_response(
    worker_uri: &str,
    client_stream: SseStream,
) -> Result<(), RelayError> {
    // Call worker, get JSON response
    let response = reqwest::post(format!("{}/execute", worker_uri))
        .json(&job_request)
        .send()
        .await?;
    
    let json_body = response.json::<serde_json::Value>().await?;
    
    // Wrap in SSE event for client
    let sse_event = SseEvent::new("result")
        .data(json_body.to_string());
    
    client_stream.send(sse_event).await?;
    
    Ok(())
}
```

**Client receives**:
```
event: result
data: {"job_id": "job-xyz", "images": [...]}
```

#### 4.2.4 Binary Relay (Audio)

**New implementation**:

```rust
async fn relay_binary_stream(
    worker_uri: &str,
    client_stream: SseStream,
) -> Result<(), RelayError> {
    let response = reqwest::post(format!("{}/execute", worker_uri))
        .json(&job_request)
        .send()
        .await?;
    
    // Option 1: Base64 encode and send as SSE
    let bytes = response.bytes().await?;
    let base64_data = base64::encode(&bytes);
    
    let sse_event = SseEvent::new("audio")
        .data(json!({
            "format": "mp3",
            "data": base64_data
        }).to_string());
    
    client_stream.send(sse_event).await?;
    
    // Option 2: Stream chunks as SSE events
    // (for large audio files)
    
    Ok(())
}
```

---

### 4.3 Client API Changes

#### 4.3.1 Job Submission (Extended)

**Add capability field**:

```json
POST /v2/tasks
{
  "session_id": "sess-abc",
  "capability": "image-gen",  // NEW: explicit capability
  "model": "stable-diffusion-xl",
  "prompt": "a cat on a couch",
  "width": 1024,
  "height": 1024,
  "steps": 50
}

Response (202 Accepted):
{
  "job_id": "job-xyz",
  "status": "queued",
  "capability": "image-gen",  // Echo capability
  "events_url": "/v2/tasks/job-xyz/events"
}
```

**Capability inference**:
- If `capability` is omitted, orchestrator infers from `model` (via catalog)
- If model is `stable-diffusion-*`, infer `image-gen`
- If model is `text-embedding-*`, infer `embedding`
- Default: `text-gen`

#### 4.3.2 SSE Event Types by Capability

**Text generation** (existing):
```
event: started
event: token
event: metrics
event: end
event: error
```

**Image generation** (new):
```
event: started
event: progress (optional)
event: result (JSON with image data)
event: end
event: error
```

**Embedding** (new):
```
event: started
event: result (JSON with embedding vector)
event: end
event: error
```

**Audio generation** (new):
```
event: started
event: audio (base64 or binary)
event: end
event: error
```

---

### 4.4 Worker Adapter Changes

#### 4.4.1 Adapter Metadata Extension

**Add protocol field**:

```rust
struct AdapterMetadata {
    name: String,
    version: String,
    capabilities: Vec<Capability>,
    protocol: StreamingProtocol,  // NEW
}

enum StreamingProtocol {
    Sse,           // text-gen
    Json,          // image-gen, embedding
    Binary,        // audio-gen
    MixedSse,      // multimodal
}
```

#### 4.4.2 Adapter Implementation Example

```rust
impl WorkerAdapter for ImageGenAdapter {
    fn metadata(&self) -> AdapterMetadata {
        AdapterMetadata {
            name: "stable-diffusion".to_string(),
            version: "1.0.0".to_string(),
            capabilities: vec![Capability::ImageGen],
            protocol: StreamingProtocol::Json,  // Image workers use JSON
        }
    }
    
    // ... spawn, health_check, etc.
}
```

---

## 5. Specification Organization

### 5.0 Spec File Structure (Reorganization)

**Current problem**: `00_llama-orch.md` contains worker-orcd specifics (VRAM-only, CUDA, etc.) but we're building an **array of worker implementations**.

**Proposed structure**:

```
bin/.specs/
├── 00_llama-orch.md          # System architecture (worker-agnostic)
│   ├── Worker contract (generic)
│   ├── Capability definitions
│   ├── Protocol requirements (high-level)
│   └── References to worker-specific specs
│
├── 00_streaming_protocols.md  # NEW: Protocol specifications
│   ├── text-gen protocol (SSE)
│   ├── image-gen protocol (JSON/Binary)
│   ├── audio-gen protocol (Binary)
│   ├── embedding protocol (JSON)
│   └── multimodal protocol (Mixed SSE)
│
├── 01_worker_orcd.md          # MOVE from 01_M0_worker_orcd.md
│   ├── Bespoke CUDA worker
│   ├── VRAM-only policy
│   ├── text-gen capability
│   └── SSE protocol implementation
│
├── 02_worker_aarmd.md         # TODO: Apple Metal worker
│   ├── Unified memory architecture
│   ├── text-gen capability
│   └── SSE protocol implementation
│
├── 03_worker_sd.md            # TODO: Stable Diffusion worker
│   ├── Image generation
│   ├── image-gen capability
│   └── JSON protocol implementation
│
├── 04_worker_tts.md           # TODO: TTS worker
│   ├── Audio generation
│   ├── audio-gen capability
│   └── Binary protocol implementation
│
└── 05_worker_embed.md         # TODO: Embedding worker
    ├── Vector embeddings
    ├── embedding capability
    └── JSON protocol implementation
```

**Rationale**:
- `00_llama-orch.md` becomes **worker-agnostic** system spec
- Each worker implementation gets its own spec (01, 02, 03, ...)
- `00_streaming_protocols.md` defines protocol contracts
- Orchestrator only needs to understand protocols, not worker internals

**Migration**:
1. Extract worker-orcd specifics from `00_llama-orch.md` → `01_worker_orcd.md`
2. Keep generic worker contract in `00_llama-orch.md`
3. Add protocol specs to `00_streaming_protocols.md`
4. Future workers (02, 03, ...) are TODO but have reserved spec slots

---

### 5.1 Changes to `00_llama-orch.md`

#### 5.1.1 Section 5.4 "Orchestrator → Worker (Direct)"

**Current** (SYS-5.4.1):
```
Response: SSE stream
- started
- token* (multiple)
- metrics* (periodic)
- end
- error (on failure or cancellation)
```

**Proposed**:
```
Response: Capability-dependent protocol

Text generation (capability: text-gen):
  SSE stream: started → token* → metrics* → end/error

Image generation (capability: image-gen):
  JSON response: {"images": [...], "metadata": {...}}
  OR SSE stream: started → progress* → result → end/error

Embedding (capability: embedding):
  JSON response: {"embedding": [...], "dimensions": N}

Audio generation (capability: audio-gen):
  Binary stream (audio/mpeg, audio/wav)
  OR JSON: {"audio": {"format": "mp3", "data": "base64..."}}

Multimodal (capability: multimodal):
  SSE stream: started → (token|image|audio)* → end/error
```

#### 5.1.2 Section 6.1.1 "Orchestrator Intelligence"

**Current**:
```
- MUST relay SSE streams from workers to clients
```

**Proposed**:
```
- MUST relay worker responses to clients using capability-appropriate protocol
- Orchestrator MUST be aware of worker capabilities and their protocol contracts
- Protocol selection based on job capability (from worker metadata via pool mgr):
  - text-gen: SSE relay (preserve token event order)
  - image-gen: JSON relay (wrap in SSE result event)
  - embedding: JSON relay (wrap in SSE result event)
  - audio-gen: Binary relay (base64 encode in SSE audio event)
  - multimodal: Mixed SSE relay (preserve event order)
- Orchestrator does NOT need to understand worker internals (CUDA, Metal, etc.)
- Only protocol contract matters for relay logic
```

#### 5.1.3 New Section: 5.4.3 "Capability-Specific Protocols"

**Add new section** after SYS-5.4.2:

```markdown
### 5.4.3 Capability-Specific Protocols [M3+] (SYS-5.4.3)

Workers MUST implement streaming protocols appropriate for their capability:

#### [SYS-5.4.3.1] Text Generation Protocol
- MUST use SSE (text/event-stream)
- MUST emit events: started, token, metrics, end/error
- Token events MUST include token text and index

#### [SYS-5.4.3.2] Image Generation Protocol
- MUST use JSON response OR SSE with result event
- JSON MUST include: images array, metadata
- Image data MUST be base64-encoded PNG/JPEG
- Optional: SSE progress events for long-running generation

#### [SYS-5.4.3.3] Embedding Protocol
- MUST use JSON response (no streaming)
- JSON MUST include: embedding array, dimensions
- Response MUST complete within timeout (default 30s)

#### [SYS-5.4.3.4] Audio Generation Protocol
- MUST use binary stream OR JSON with base64
- Binary stream MUST set Content-Type (audio/mpeg, audio/wav)
- JSON MUST include: format, data (base64), duration_ms

#### [SYS-5.4.3.5] Multimodal Protocol
- MUST use SSE with mixed event types
- MUST support events: token, image, audio, end/error
- Event order MUST reflect generation sequence
```

---

### 5.2 Changes to `10_worker_adapters.md`

#### 5.2.1 Section 3.4 "Image Generation Adapter"

**Enhance with protocol details**:

```markdown
### 3.4 Image Generation Adapter

**Worker spec**: `bin/.specs/03_worker_sd.md` (TODO)

**Requirements**:
- [POOL-1040] Adapter MUST spawn image generation worker (e.g., sd-worker)
- [POOL-1041] Adapter MUST translate text prompt → image request
- [POOL-1042] Adapter MUST handle image output (base64, URL, binary)
- [POOL-1043] Adapter MUST advertise capability: `image-gen`
- [POOL-1044] Adapter MUST advertise protocol: `Json` or `MixedSse`  // NEW
- [POOL-1045] Adapter MUST provide worker metadata to orchestrator (capability + protocol)

**Protocol**:
- Worker MUST respond with JSON containing base64-encoded images
- Orchestrator MUST wrap JSON response in SSE `result` event for client
- Worker MAY optionally support SSE with `progress` events
- Protocol contract defined in `00_streaming_protocols.md`
```

#### 5.2.2 Add New Adapters (TODO)

**Add sections** (reference future worker specs):
- 3.5 Embedding Adapter → `bin/.specs/05_worker_embed.md` (TODO)
- 3.6 Audio Generation Adapter → `bin/.specs/04_worker_tts.md` (TODO)
- 3.7 Multimodal Adapter → TBD (TODO)

**Note**: Each adapter references its worker spec for implementation details.

---

### 5.3 New Spec: `00_streaming_protocols.md`

**Create new spec** at `bin/.specs/00_streaming_protocols.md`:

```markdown
# Streaming Protocols Specification

**Purpose**: Define streaming protocols for different worker capabilities
**Audience**: Orchestrator developers, worker implementers, client SDK developers

## 1. Protocol Matrix

| Capability  | Protocol      | Content-Type           | Events                    | Workers              |
|-------------|---------------|------------------------|---------------------------|----------------------|
| text-gen    | SSE           | text/event-stream      | started, token, end       | worker-orcd, -aarmd  |
| image-gen   | JSON/SSE      | application/json       | started, result, end      | sd-worker (TODO)     |
| embedding   | JSON          | application/json       | started, result, end      | embed-worker (TODO)  |
| audio-gen   | Binary/JSON   | audio/*, application/json | started, audio, end    | tts-worker (TODO)    |
| multimodal  | Mixed SSE     | text/event-stream      | started, token/image, end | (TODO)               |

## 2. Protocol Definitions

[Detailed protocol specs for each capability]

## 3. Orchestrator Relay Behavior

**Orchestrator awareness**:
- Orchestrator MUST know capability → protocol mapping
- Orchestrator receives worker metadata (capability, worker_type) from pool manager
- Orchestrator selects relay strategy based on job capability
- Orchestrator does NOT need worker implementation details

[How orchestrator relays each protocol type]

## 4. Worker Implementation Requirements

**Each worker spec (01_worker_orcd.md, 02_worker_aarmd.md, etc.) MUST**:
- Declare supported capabilities
- Implement protocol contract for each capability
- Document any protocol extensions or deviations

## 5. Client SDK Guidance

[How clients should consume different protocols]
```

---

## 6. Implementation Roadmap

### 6.1 Phase 1: Foundation (M3.0)

**Goal**: Add protocol abstraction without breaking existing text-gen

1. ✅ Add `StreamingProtocol` enum to worker adapter metadata
2. ✅ Add `capability` field to job submission API (optional, inferred from model)
3. ✅ Refactor orchestrator relay to use capability-based dispatch
4. ✅ Keep text-gen using existing SSE relay (no changes)
5. ✅ Add protocol field to worker adapter trait

**Deliverables**:
- Updated `WorkerAdapter` trait with `protocol()` method
- Capability-based relay dispatcher in orchestrator
- Backward compatibility: text-gen workers unchanged

---

### 6.2 Phase 2: Image Generation (M3.5)

**Goal**: Add first non-text capability

1. ✅ Implement `ImageGenAdapter` in pool-managerd
2. ✅ Implement JSON relay in orchestrator
3. ✅ Add `image-gen` capability to job submission
4. ✅ Add SSE `result` event type for JSON responses
5. ✅ Test: Stable Diffusion worker integration

**Deliverables**:
- Working image generation pipeline
- JSON → SSE wrapping in orchestrator
- Client SDK support for image results

---

### 6.3 Phase 3: Embeddings (M4.0)

**Goal**: Add non-streaming capability

1. ✅ Implement `EmbeddingAdapter` in pool-managerd
2. ✅ Reuse JSON relay from Phase 2
3. ✅ Add `embedding` capability to job submission
4. ✅ Test: text-embedding-ada-002 worker

**Deliverables**:
- Working embedding pipeline
- Non-streaming JSON response handling

---

### 6.4 Phase 4: Audio & Multimodal (M4.5+)

**Goal**: Complete protocol matrix

1. ✅ Implement `AudioGenAdapter`
2. ✅ Implement binary relay in orchestrator
3. ✅ Implement `MultimodalAdapter`
4. ✅ Implement mixed SSE relay
5. ✅ Test: TTS worker, vision-language worker

**Deliverables**:
- Full protocol support matrix
- All capabilities operational

---

## 7. Testing Strategy

### 7.1 Protocol Contract Tests

**For each capability**:
- ✅ Worker emits correct protocol (SSE, JSON, binary)
- ✅ Orchestrator relays correctly
- ✅ Client receives expected format
- ✅ Error handling works across protocols

### 7.2 Integration Tests

**Cross-capability tests**:
- ✅ Submit text-gen job → receive SSE tokens
- ✅ Submit image-gen job → receive JSON with image
- ✅ Submit embedding job → receive JSON with vector
- ✅ Submit audio-gen job → receive binary audio
- ✅ Submit multimodal job → receive mixed SSE

### 7.3 Backward Compatibility Tests

**Ensure no regressions**:
- ✅ Existing text-gen workers work unchanged
- ✅ Existing clients receive SSE tokens as before
- ✅ Capability inference works (omit capability → infer from model)

---

## 8. Migration Guide

### 8.1 For Existing Deployments

**No breaking changes**:
1. Update orchestrator and pool-managerd binaries
2. Existing text-gen workers continue working (no code changes)
3. Existing clients continue working (SSE tokens unchanged)
4. New capabilities are opt-in (require new worker adapters)

### 8.2 For New Worker Implementations

**To add a new capability**:
1. Implement `WorkerAdapter` trait
2. Specify `capabilities` and `protocol` in metadata
3. Implement capability-specific endpoint (`/execute`)
4. Register adapter in pool-managerd config
5. Orchestrator automatically routes based on capability

---

## 9. Open Questions

### 9.1 Binary Streaming Efficiency

**Question**: Should we support chunked binary streaming for large audio/video files?

**Options**:
- A) Base64 encode in SSE (simple, less efficient)
- B) Direct binary stream with chunked transfer encoding (efficient, complex)
- C) Hybrid: SSE for small files, binary for large files

**Recommendation**: Start with A (base64 SSE), add B in future if needed.

---

### 9.2 Multimodal Event Ordering

**Question**: How to handle interleaved text + image generation?

**Example**: "Here is a cat [IMAGE] sitting on a couch"

**Options**:
- A) Sequential: emit all text, then all images
- B) Interleaved: emit text tokens, then image event, then more text tokens
- C) Parallel: emit text and image events concurrently

**Recommendation**: B (interleaved) — preserves generation order, best UX.

---

### 9.3 Protocol Negotiation

**Question**: Should workers advertise supported protocols in health endpoint?

**Proposal**:
```json
GET /health
{
  "status": "ready",
  "capabilities": ["text-gen"],
  "protocols": ["sse"],  // NEW
  "model_ref": "llama-7b",
  ...
}
```

**Benefit**: Orchestrator can validate protocol compatibility before routing.

**Recommendation**: Add in Phase 2 (image-gen).

---

## 10. Summary

### 10.1 Key Takeaways

✅ **Pool manager bypass is preserved** — orchestrator still calls workers directly

✅ **Orchestrator is protocol-aware, not worker-aware**:
   - Knows: capability → protocol mapping
   - Does NOT know: worker internals (CUDA, Metal, etc.)
   - Receives worker metadata (capability, protocol) from pool manager

✅ **Capability determines protocol** — text-gen uses SSE, image-gen uses JSON, etc.

✅ **Backward compatible** — existing text-gen workers (worker-orcd) unchanged

✅ **Extensible** — new capabilities add new protocols without breaking existing ones

✅ **Spec organization supports worker array**:
   - `00_llama-orch.md` = worker-agnostic system spec
   - `01_worker_orcd.md` = bespoke CUDA worker (M0)
   - `02-05_worker_*.md` = future workers (TODO)
   - `00_streaming_protocols.md` = protocol contracts

### 10.2 Specification Changes Needed

1. **00_llama-orch.md** (make worker-agnostic):
   - Extract worker-orcd specifics → move to `01_worker_orcd.md`
   - Update SYS-5.4.1 (add capability-specific protocols)
   - Update SYS-6.1.1 (orchestrator protocol awareness)
   - Add SYS-5.4.3 (protocol specifications)
   - Keep generic worker contract only

2. **01_worker_orcd.md** (rename from 01_M0_worker_orcd.md):
   - Move from milestone-scoped to worker-scoped spec
   - Keep all worker-orcd specifics (VRAM-only, CUDA, etc.)
   - Reference `00_streaming_protocols.md` for SSE protocol

3. **10_worker_adapters.md**:
   - Add protocol field to adapter metadata
   - Add worker spec references (01, 02, 03, ...)
   - Enhance image-gen adapter spec
   - Add embedding, audio, multimodal adapter specs (TODO)

4. **New spec: 00_streaming_protocols.md**:
   - Protocol matrix (capability → protocol → workers)
   - Detailed protocol definitions
   - Orchestrator relay behavior (protocol-aware, not worker-aware)
   - Worker implementation requirements
   - Client guidance

5. **Future worker specs** (TODO):
   - `02_worker_aarmd.md` (Apple Metal, text-gen)
   - `03_worker_sd.md` (Stable Diffusion, image-gen)
   - `04_worker_tts.md` (TTS, audio-gen)
   - `05_worker_embed.md` (Embeddings)

### 10.3 Next Steps

1. ✅ Review this plan with team
2. ✅ Reorganize spec files:
   - Make `00_llama-orch.md` worker-agnostic
   - Rename `01_M0_worker_orcd.md` → `01_worker_orcd.md` (worker-scoped)
   - Create `00_streaming_protocols.md`
   - Reserve slots for future workers (02-05)
3. ✅ Update specifications:
   - `00_llama-orch.md` (orchestrator protocol awareness)
   - `10_worker_adapters.md` (protocol metadata)
   - `00_streaming_protocols.md` (protocol contracts)
4. ✅ Implement Phase 1 (protocol abstraction)
5. ✅ Implement Phase 2 (image-gen, first non-text worker)
6. ✅ Iterate on remaining phases (audio, embeddings, multimodal)

---

**End of Plan**
