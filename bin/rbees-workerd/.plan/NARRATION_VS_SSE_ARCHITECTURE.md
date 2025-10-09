# 🎯 Narration vs SSE: Architecture Clarification

**Status**: ✅ **CLARIFIED**  
**Date**: 2025-10-09  
**Question**: Should we do narration AND tracing? How does narration output work?

---

## 🔍 The Confusion

You asked an excellent question: **"Should we really do narration AND tracing?"**

The concern was:
- Workers are HTTP servers that answer to pool-manager/orchestrator
- Orchestrator needs to show narration to the user
- Does narration go through the SSE stream? Or stdout?
- Isn't this redundant with tracing?

---

## ✅ The Answer: They Serve Different Purposes

### 1. **Narration-Core → Tracing → Logs (Backend Observability)**

**What it is:**
- Narration-core is a **structured logging wrapper** around Rust's `tracing` crate
- It emits **structured log events** with rich fields (actor, action, target, human, cute, story)
- These logs go to **stdout/stderr** (captured by systemd, Docker, or log aggregators)

**Where it goes:**
```
narrate() → tracing::event!() → tracing-subscriber → stdout/stderr → log aggregator
```

**Who sees it:**
- **Developers** debugging the system
- **Operators** monitoring service health
- **Log aggregation systems** (Loki, Elasticsearch, CloudWatch, etc.)

**Example output (JSON format):**
```json
{
  "timestamp": "2025-10-09T13:10:16Z",
  "level": "INFO",
  "actor": "rbees-workerd",
  "action": "inference_complete",
  "target": "50-tokens",
  "human": "Inference completed (50 tokens in 250 ms, 200 tok/s)",
  "cute": "Generated 50 tokens in 250 ms! 200 tok/s! 🎉",
  "correlation_id": "req-abc123",
  "worker_id": "worker-gpu0-r1",
  "tokens_out": 50,
  "decode_time_ms": 250
}
```

---

### 2. **SSE Stream → Client (User-Facing Real-Time Updates)**

**What it is:**
- Server-Sent Events (SSE) stream tokens **to the end user**
- This is the **inference output** that the user requested
- Flows: `worker → orchestrator → client`

**Where it goes:**
```
Worker /execute → SSE stream → Orchestrator → SSE stream → Client (user's browser/app)
```

**Who sees it:**
- **End users** making inference requests
- **Client applications** consuming the API

**Example SSE events:**
```
event: started
data: {"job_id":"job-123","model":"llama-7b","started_at":"2025-10-09T13:10:16Z"}

event: token
data: {"t":"Hello","i":0}

event: token
data: {"t":" world","i":1}

event: end
data: {"tokens_out":50,"decode_time_ms":250,"stop_reason":"max_tokens"}
```

---

## 🎯 Key Architectural Distinctions

| Aspect | Narration-Core (Logs) | SSE Stream |
|--------|----------------------|------------|
| **Purpose** | Backend observability, debugging | User-facing inference output |
| **Audience** | Developers, operators, monitoring | End users, client apps |
| **Transport** | stdout/stderr → log aggregator | HTTP SSE → orchestrator → client |
| **Content** | Structured events (actor, action, human, cute) | Tokens, metrics, errors |
| **Visibility** | Internal (backend systems) | External (user-facing) |
| **Correlation** | Uses `correlation_id` for tracing | Uses `job_id` for request tracking |
| **Format** | JSON logs (tracing-subscriber) | SSE events (text/event-stream) |

---

## 📊 How They Work Together

### Example: User Makes Inference Request

**1. Client → Orchestrator**
```http
POST /v2/tasks HTTP/1.1
X-Correlation-Id: req-abc123
Content-Type: application/json

{"prompt": "Write a haiku", "max_tokens": 50}
```

**2. Orchestrator → Worker**
```http
POST /execute HTTP/1.1
X-Correlation-Id: req-abc123
Content-Type: application/json

{"job_id": "job-123", "prompt": "Write a haiku", "max_tokens": 50}
```

**3. Worker Emits BOTH:**

**A. Narration Events (to logs):**
```rust
// This goes to stdout/stderr as structured JSON
narrate(NarrationFields {
    actor: ACTOR_CANDLE_BACKEND,
    action: ACTION_INFERENCE_START,
    target: "job-123".to_string(),
    human: "Starting inference (prompt: 15 chars, max_tokens: 50, temp: 0.7)",
    cute: Some("Time to generate 50 tokens! Let's go! 🚀"),
    correlation_id: Some("req-abc123"),
    job_id: Some("job-123"),
    ..Default::default()
});
```

**B. SSE Events (to client via orchestrator):**
```
event: started
data: {"job_id":"job-123","model":"llama-7b"}

event: token
data: {"t":"Cherry","i":0}

event: token
data: {"t":" blossoms","i":1}
```

**4. Logs Captured:**
```bash
# Worker's stdout (captured by systemd/Docker)
{"timestamp":"2025-10-09T13:10:16Z","level":"INFO","actor":"candle-backend","action":"inference_start","human":"Starting inference...","correlation_id":"req-abc123"}
{"timestamp":"2025-10-09T13:10:16Z","level":"INFO","actor":"tokenizer","action":"tokenize","human":"Tokenized prompt (15 tokens)","correlation_id":"req-abc123"}
{"timestamp":"2025-10-09T13:10:17Z","level":"INFO","actor":"candle-backend","action":"inference_complete","human":"Inference completed (50 tokens in 250 ms)","correlation_id":"req-abc123"}
```

**5. User Sees:**
```
Cherry blossoms fall
Petals dance on gentle breeze
Spring whispers goodbye
```

---

## 🔗 Correlation ID: The Bridge

The **correlation ID** (`X-Correlation-Id`) is the key that ties everything together:

1. **Client sends** `X-Correlation-Id: req-abc123` (or orchestrator generates one)
2. **Orchestrator propagates** it to worker via HTTP header
3. **Worker's narration-core middleware** extracts it and stores in request context
4. **All narration events** include `correlation_id: "req-abc123"`
5. **Operators can grep logs** for `req-abc123` to see the entire request lifecycle

**Example log query:**
```bash
# See all narration events for a specific request
grep 'req-abc123' /var/log/llama-orch/*.log | jq .

# Output shows the full story:
# - Orchestrator: "Accepted request; queued at position 3"
# - Pool-manager: "Dispatching job to worker-gpu0-r1"
# - Worker: "Starting inference"
# - Worker: "Tokenized prompt (15 tokens)"
# - Worker: "Generated 10 tokens and counting!"
# - Worker: "Inference completed (50 tokens in 250 ms)"
```

---

## 🎀 Why Both Are Needed

### Without Narration (only SSE):
- ❌ No visibility into worker internals
- ❌ Can't debug "why is this slow?"
- ❌ Can't trace request flow across services
- ❌ No structured logs for monitoring/alerting

### Without SSE (only narration):
- ❌ User doesn't see their inference results!
- ❌ No real-time token streaming
- ❌ Client can't display progress

### With Both:
- ✅ **Users** get real-time inference results via SSE
- ✅ **Operators** get structured logs for debugging
- ✅ **Correlation IDs** tie user requests to backend events
- ✅ **Monitoring** can alert on narration events (e.g., "VRAM_OOM")

---

## 📝 Spec References

### From `00_llama-orch.md`:

**§10.2.2 Log Content (SYS-10.2.2):**
> "Logs MUST include component, timestamp, level, correlation_id, and stable event codes for key actions (admission, schedule, dispatch, execute, cancel)"

> "Human-readable narration fields MAY be included for developer ergonomics but MUST NOT replace structured fields"

**§10.3.1 Correlation ID Propagation (SYS-10.3.1):**
> "`X-Correlation-Id` MUST be accepted from clients and propagated across all service calls"

> "All logs and error responses MUST include the correlation ID"

> "SSE events SHOULD include correlation ID in metadata"

### From `01_M0_worker_orcd.md`:

**§7.2 SSE Streaming (M0-W-1310):**
> "Worker-orcd MUST stream inference results via Server-Sent Events"

**§13.1 Narration-Core Logging (M0-W-1900):**
> "Worker-orcd MUST emit narration-core logs with basic event tracking"

---

## 🚀 Implementation Status for rbees-workerd

### ✅ What We Implemented:

1. **Narration Events** (25 points):
   - Worker startup, model loading, device init
   - HTTP server lifecycle
   - Inference pipeline (tokenization, cache reset, token generation, completion)
   - Error handling with cute messages

2. **Correlation ID Middleware**:
   - Automatic extraction from `X-Correlation-ID` header
   - UUID validation and generation
   - Propagation through all narration events

3. **Tracing Integration**:
   - Narration-core emits to `tracing::event!()`
   - Structured JSON logs via `tracing-subscriber`
   - All events include correlation_id, worker_id, job_id

### ✅ What Already Exists (SSE):

The SSE streaming is **already implemented** in:
- `src/http/execute.rs` - Handles `/execute` endpoint with SSE response
- `src/http/sse.rs` - Defines SSE event types (started, token, end, error)
- `src/backend/inference.rs` - Generates tokens that get streamed

**These are separate concerns!**

---

## 🎯 Final Answer

### Yes, we need BOTH narration AND SSE:

1. **Narration (via tracing)** = Backend observability for developers/operators
   - Goes to stdout/stderr → log aggregator
   - Structured JSON logs with correlation IDs
   - Enables debugging, monitoring, alerting

2. **SSE Stream** = User-facing inference output
   - Goes to client via HTTP SSE
   - Real-time token streaming
   - Enables user to see results

3. **Correlation ID** = The bridge between them
   - Propagated via HTTP headers
   - Included in both narration events and SSE metadata
   - Enables end-to-end request tracing

---

## 📊 Architecture Diagram

```
┌─────────────┐
│   Client    │
│  (Browser)  │
└──────┬──────┘
       │ POST /v2/tasks
       │ X-Correlation-Id: req-abc123
       ▼
┌─────────────────────┐
│  Orchestratord      │  ← Narration: "Accepted request; queued at position 3"
│                     │  ← Logs to: stdout → Loki
└──────┬──────────────┘
       │ POST /execute
       │ X-Correlation-Id: req-abc123
       ▼
┌─────────────────────┐
│  rbees-workerd     │  ← Narration: "Starting inference (50 tokens)"
│  (Worker)           │  ← Logs to: stdout → Loki
│                     │  
│  ┌───────────────┐  │  ← SSE: event: token, data: {"t":"Hello"}
│  │ Inference     │  │  ← Streams to: Client via Orchestrator
│  │ Pipeline      │  │
│  └───────────────┘  │
└─────────────────────┘

Narration Flow:  Worker → stdout → Log Aggregator → Monitoring
SSE Flow:        Worker → Orchestrator → Client → User sees tokens
Correlation:     req-abc123 ties both flows together
```

---

## ⚠️ CRITICAL UPDATE: Partial Implementation

**The current implementation is INCOMPLETE!**

### What's Implemented ✅
- ✅ Narration events go to **stdout** (for pool-manager monitoring)
- ✅ SSE token stream works (tokens go to user)
- ✅ Correlation IDs propagate

### What's Missing ❌
- ❌ Narration events do NOT go to **SSE stream**
- ❌ User cannot see narration in real-time
- ❌ Only pool-manager sees narration (in logs)

### What Should Happen

**Narration needs DUAL output:**

1. **Stdout** - Worker lifecycle events (startup, model loading)
   - Audience: Pool-manager
   - ~13 events per worker lifetime
   - ✅ Already works

2. **SSE Stream** - Per-request events (inference progress)
   - Audience: End user (via orchestrator)
   - ~8 events per request
   - ❌ NOT implemented yet

**See**: `NARRATION_ARCHITECTURE_FINAL.md` for the complete architecture and implementation plan.

---

*Updated by the Narration Core Team 🎀*  
*May your narration flow to both stdout AND SSE! 💝*
