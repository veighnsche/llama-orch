# 🎯 Narration Architecture: Stdout vs SSE

**Status**: ✅ **CLARIFIED**  
**Date**: 2025-10-09  
**Author**: Narration Core Team 🎀

---

## 📋 Executive Summary

Narration events serve **two different purposes** at **two different times** in the worker's lifecycle:

1. **Stdout Narration** - Worker lifecycle events (startup, model loading, shutdown)
   - Captured by pool-manager
   - Used for operational monitoring
   - ~8 events per worker lifetime

2. **SSE Narration** - Per-request events (inference pipeline, progress, completion)
   - Streamed to user via orchestrator
   - Used for real-time user feedback
   - ~7 events per inference request

**Both are needed!** They are not redundant - they serve different audiences at different times.

---

## 🔍 Worker Lifecycle Timeline

```
┌─────────────────────────────────────────────────────────────────┐
│ Phase 1: Worker Startup (NO HTTP connection)                    │
│ Audience: Pool-Manager (via stdout)                             │
├─────────────────────────────────────────────────────────────────┤
│ 1. Pool-manager spawns worker binary                            │
│ 2. Worker process starts                                        │
│ 3. Device initialization                                        │
│ 4. Model loading                                                │
│ 5. HTTP server starts                                           │
│ 6. Worker calls back to pool-manager "I'm ready!"               │
│                                                                  │
│ Narration Output: stdout → Pool-manager captures                │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ Phase 2: Worker Ready (HTTP server running)                     │
│ Audience: None (waiting for requests)                           │
├─────────────────────────────────────────────────────────────────┤
│ Worker is idle, waiting for orchestrator to send /execute       │
│                                                                  │
│ Narration Output: None (no events during idle)                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ Phase 3: Inference Request (HTTP connection active)             │
│ Audience: User (via orchestrator, through SSE stream)           │
├─────────────────────────────────────────────────────────────────┤
│ 1. Orchestrator sends POST /execute                             │
│ 2. Worker validates request                                     │
│ 3. Worker starts inference                                      │
│ 4. Worker tokenizes prompt                                      │
│ 5. Worker resets cache                                          │
│ 6. Worker generates tokens (streaming)                          │
│ 7. Worker completes inference                                   │
│ 8. Worker sends final SSE event                                 │
│                                                                  │
│ Narration Output: SSE stream → Orchestrator → User's screen     │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ Phase 4: Worker Shutdown (HTTP connection may be closed)        │
│ Audience: Pool-Manager (via stdout)                             │
├─────────────────────────────────────────────────────────────────┤
│ 1. Pool-manager sends SIGTERM or POST /shutdown                 │
│ 2. Worker gracefully shuts down HTTP server                     │
│ 3. Worker frees VRAM                                            │
│ 4. Worker exits                                                 │
│                                                                  │
│ Narration Output: stdout → Pool-manager captures                │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 Narration Event Classification

### Category 1: Stdout-Only Events (Worker Lifecycle)

**These happen when there is NO active HTTP request:**

| File | Line | Actor | Action | Event | Why Stdout? |
|------|------|-------|--------|-------|-------------|
| `main.rs` | 76-84 | `llorch-candled` | `startup` | "Starting Candle worker on port 8080" | Binary just started, no HTTP yet |
| `device.rs` | 18-25 | `device-manager` | `device_init` | "Initialized CPU device" | During startup, before HTTP |
| `device.rs` | 37-45 | `device-manager` | `device_init` | "Initialized CUDA device 0" | During startup, before HTTP |
| `device.rs` | 58-66 | `device-manager` | `device_init` | "Initialized Apple Metal device 0" | During startup, before HTTP |
| `main.rs` | 95-103 | `model-loader` | `model_load` | "Loading Llama model from /models/..." | During startup, before HTTP |
| `inference.rs` | 58-66 | `model-loader` | `model_load` | "Loaded Llama model (7000 MB, vocab: 32000)" | During startup, before HTTP |
| `main.rs` | 119-128 | `llorch-candled` | `callback_ready` | "Reporting ready to pool-managerd" | During startup, callback to pool-manager |
| `startup.rs` | 33-42 | `llorch-candled` | `callback_ready` | "Calling pool-managerd at http://..." | During startup, callback to pool-manager |
| `startup.rs` | 48-57 | `llorch-candled` | `error` | "Pool manager callback failed: 500" | During startup, callback error |
| `server.rs` | 83-90 | `http-server` | `server_start` | "HTTP server initialized on 0.0.0.0:8080" | Server lifecycle, not request-specific |
| `server.rs` | 126-133 | `http-server` | `server_bind` | "HTTP server listening on 0.0.0.0:8080" | Server lifecycle, not request-specific |
| `server.rs` | 108-116 | `http-server` | `error` | "Failed to bind to 0.0.0.0:8080" | Server lifecycle error |
| `server.rs` | 160-167 | `http-server` | `server_shutdown` | "HTTP server shutting down gracefully" | Server lifecycle, shutdown |

**Total: 13 stdout-only events**

**Audience**: Pool-manager (operational monitoring)  
**Output**: stdout → captured by pool-manager → pool-manager logs  
**Purpose**: Track worker lifecycle, diagnose startup/shutdown issues

---

### Category 2: SSE Events (Per-Request)

**These happen DURING an active `/execute` HTTP request:**

| File | Line | Actor | Action | Event | Why SSE? |
|------|------|-------|--------|-------|----------|
| `execute.rs` | 36-45 | `http-server` | `error` | "Validation failed for job job-123" | During request, user needs to see |
| `execute.rs` | 52-60 | `http-server` | `execute_request` | "Inference request validated for job job-123" | During request, user wants feedback |
| `execute.rs` | 81-90 | `candle-backend` | `error` | "Inference failed for job job-123: ..." | During request, user needs error |
| `inference.rs` | 158-165 | `candle-backend` | `inference_start` | "Starting inference (prompt: 15 chars, max_tokens: 50)" | During inference, user wants progress |
| `inference.rs` | 176-184 | `tokenizer` | `tokenize` | "Tokenized prompt (15 tokens)" | During inference, user wants details |
| `inference.rs` | 192-199 | `candle-backend` | `cache_reset` | "Reset KV cache before inference" | During inference, technical detail |
| `inference.rs` | 295-303 | `candle-backend` | `token_generate` | "Generated 10 tokens" | During inference, progress update |
| `inference.rs` | 325-334 | `candle-backend` | `inference_complete` | "Inference completed (50 tokens in 250 ms, 200 tok/s)" | During inference, completion status |

**Total: 8 SSE events per request**

**Audience**: End user (via orchestrator)  
**Output**: SSE stream → orchestrator → user's screen  
**Purpose**: Real-time feedback on inference progress, show what's happening

---

### Category 3: Hybrid Events (Could Be Either)

**These could go to stdout OR SSE depending on context:**

| File | Line | Actor | Action | Event | Context |
|------|------|-------|--------|-------|---------|
| `health.rs` | 43-50 | `http-server` | `health_check` | "Health check: healthy (VRAM: 8000 MB)" | During `/health` request |
| `inference.rs` | 87-94 | `candle-backend` | `warmup` | "Starting GPU warmup" | During startup (stdout) OR on-demand (SSE) |
| `inference.rs` | 124-132 | `candle-backend` | `warmup` | "GPU warmup complete (50 ms)" | During startup (stdout) OR on-demand (SSE) |

**Decision**: Currently stdout-only, but could be SSE if warmup is triggered during a request.

---

## 🎯 Implementation Strategy

### Current State (Partially Correct)

✅ **Stdout narration is implemented correctly:**
- All lifecycle events go to stdout
- Pool-manager can capture them
- Works for operational monitoring

❌ **SSE narration is NOT implemented:**
- Per-request events only go to stdout
- User cannot see them in real-time
- Missing `narration` event type in SSE

---

### Required Changes

#### 1. Add Narration Event Type to SSE

**File**: `src/http/sse.rs`

```rust
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InferenceEvent {
    Started { ... },
    Token { ... },
    Metrics { ... },
    
    /// NEW: Narration event for user-facing progress updates
    Narration {
        actor: String,
        action: String,
        target: String,
        human: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        cute: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        story: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        correlation_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        job_id: Option<String>,
    },
    
    End { ... },
    Error { ... },
}
```

#### 2. Create SSE Channel for Narration

**File**: `src/http/execute.rs`

```rust
pub async fn handle_execute<B: InferenceBackend>(
    State(backend): State<Arc<Mutex<B>>>,
    Json(req): Json<ExecuteRequest>,
) -> Result<Sse<EventStream>, ValidationErrorResponse> {
    // Create channel for narration events
    let (narration_tx, mut narration_rx) = tokio::sync::mpsc::unbounded_channel();
    
    // Store in request-local context (thread-local or async-local)
    NARRATION_SENDER.with(|sender| {
        *sender.borrow_mut() = Some(narration_tx.clone());
    });
    
    // ... rest of handler
}
```

#### 3. Modify Narration Function

**File**: `narration-core/src/lib.rs` (or create worker-specific wrapper)

```rust
pub fn narrate(fields: NarrationFields) {
    // 1. Always log to stdout (for pool-manager)
    tracing::event!(Level::INFO, ...);
    
    // 2. If we're in an HTTP request context, also emit SSE
    if let Some(tx) = get_current_narration_sender() {
        let _ = tx.send(InferenceEvent::Narration {
            actor: fields.actor.to_string(),
            action: fields.action.to_string(),
            target: fields.target,
            human: fields.human,
            cute: fields.cute,
            story: fields.story,
            correlation_id: fields.correlation_id,
            job_id: fields.job_id,
        });
    }
}
```

#### 4. Merge Narration Events into SSE Stream

**File**: `src/http/execute.rs`

```rust
// Merge narration events with token events
let narration_stream = tokio_stream::wrappers::UnboundedReceiverStream::new(narration_rx);
let token_stream = /* ... existing token stream ... */;

// Interleave them
let merged_stream = stream::select(narration_stream, token_stream);
```

---

## 📊 Event Flow Diagrams

### Stdout Flow (Worker Lifecycle)

```
┌─────────────────┐
│ Pool-Manager    │
│                 │
│ 1. Spawns       │
│    worker       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Worker Binary   │
│                 │
│ narrate()       │
│   ↓             │
│ tracing::event()│
│   ↓             │
│ stdout          │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Pool-Manager    │
│ Captures stdout │
│                 │
│ Logs:           │
│ "Worker started"│
│ "Model loaded"  │
│ "Server ready"  │
└─────────────────┘
```

### SSE Flow (Per-Request)

```
┌─────────────────┐
│ User's Screen   │
│                 │
│ "Starting..."   │
│ "Tokenizing..." │
│ "Generated 10"  │
└────────▲────────┘
         │
         │ SSE stream
         │
┌────────┴────────┐
│ Orchestrator    │
│ Relays events   │
└────────▲────────┘
         │
         │ SSE stream
         │
┌────────┴────────┐
│ Worker          │
│ /execute        │
│                 │
│ narrate()       │
│   ↓             │
│ SSE channel     │
│   ↓             │
│ HTTP response   │
└─────────────────┘
```

---

## ✅ Benefits of Dual Output

### Stdout (Pool-Manager)
- ✅ Operational monitoring
- ✅ Worker lifecycle tracking
- ✅ Startup/shutdown diagnostics
- ✅ Model loading verification
- ✅ Server health monitoring

### SSE (User)
- ✅ Real-time inference progress
- ✅ Transparency (user sees what's happening)
- ✅ Better UX (not just waiting for tokens)
- ✅ Debugging (user can see where it's slow)
- ✅ Trust (user sees the system working)

---

## 🚨 Critical Distinction

**NOT redundant!** They serve different purposes:

| Aspect | Stdout | SSE |
|--------|--------|-----|
| **When** | Worker lifecycle (startup/shutdown) | During inference request |
| **Who** | Pool-manager (operator) | End user (via orchestrator) |
| **Why** | Operational monitoring | User experience |
| **What** | "Worker started", "Model loaded" | "Tokenizing...", "Generated 10 tokens" |
| **Frequency** | ~13 events per worker lifetime | ~8 events per request |

---

## 📝 Implementation Checklist

### Phase 1: SSE Narration Events
- [ ] Add `Narration` variant to `InferenceEvent` enum
- [ ] Create narration channel in execute handler
- [ ] Store channel in request-local context
- [ ] Modify `narrate()` to check for SSE channel
- [ ] Merge narration events into SSE stream
- [ ] Test: User sees narration events in real-time

### Phase 2: Stdout Narration (Already Done)
- [x] Worker startup events
- [x] Device initialization events
- [x] Model loading events
- [x] Pool-manager callback events
- [x] Server lifecycle events

### Phase 3: Documentation
- [ ] Update OpenAPI spec with narration events
- [ ] Document event ordering
- [ ] Add examples to API docs
- [ ] Update orchestrator relay logic

---

## 🎯 Expected User Experience

### User's Screen (Orchestrator PC)

```
┌─────────────────────────────────────────────────────────┐
│ Inference Request: "Write a haiku about GPUs"           │
├─────────────────────────────────────────────────────────┤
│ [Narration Panel]                                       │
│ ✅ Inference request validated for job job-123          │
│ 🚀 Starting inference (prompt: 28 chars, max_tokens: 50)│
│ 🍰 Tokenized prompt (7 tokens)                          │
│ 🧹 Reset KV cache before inference                      │
│ 🎯 Generated 10 tokens                                  │
│ 🎯 Generated 20 tokens                                  │
│ 🎉 Inference completed (42 tokens in 250 ms, 168 tok/s) │
│                                                          │
│ [Token Stream] (goes to AI agent)                       │
│ Silicon dreams ignite                                   │
│ Parallel cores dance as one                             │
│ CUDA's warm embrace                                     │
└─────────────────────────────────────────────────────────┘
```

### Pool-Manager Logs

```
[2025-10-09T13:27:00Z] INFO worker-gpu0-r1: Starting Candle worker on port 8080
[2025-10-09T13:27:00Z] INFO worker-gpu0-r1: Initialized CUDA device 0
[2025-10-09T13:27:01Z] INFO worker-gpu0-r1: Loaded Llama model (7000 MB, vocab: 32000)
[2025-10-09T13:27:01Z] INFO worker-gpu0-r1: HTTP server listening on 0.0.0.0:8080
[2025-10-09T13:27:01Z] INFO worker-gpu0-r1: Calling pool-managerd at http://localhost:9000/ready
```

**Both are valuable! Different audiences, different purposes.**

---

*Documented by the Narration Core Team 🎀*  
*May your stdout flow to pool-manager and your SSE flow to users! 💝*
