# 🔌 Narration Wiring & Stream Separation Explained

**Date**: 2025-10-09  
**Questions**:
1. How is narration-core wired up to HTTP?
2. How does the worker distinguish between narration logs and SSE token streams?

---

## ✅ Answer 1: Narration is NOT Wired to HTTP

**Narration-core is completely separate from the HTTP response!**

### The Wiring (or Lack Thereof)

**In `main.rs` (line 64):**
```rust
#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing (JSON format for structured logging)
    tracing_subscriber::fmt().with_target(false).json().init();
    
    // ... rest of the code
}
```

**That's it!** This is the ONLY "wiring" needed. Here's what happens:

1. **`tracing_subscriber::fmt().json().init()`** sets up the global tracing subscriber
2. When you call `narrate()`, it calls `tracing::event!()`
3. The tracing subscriber captures the event and formats it as JSON
4. The JSON is written to **stdout** (standard output)
5. The process manager (systemd, Docker, etc.) captures stdout and sends it to logs

### Narration Flow Diagram

```
narrate(NarrationFields {...})
    ↓
tracing::event!(Level::INFO, actor = ..., human = ...)
    ↓
tracing_subscriber (global singleton)
    ↓
JSON formatter
    ↓
stdout (file descriptor 1)
    ↓
Process manager (systemd/Docker)
    ↓
Log file or log aggregator (Loki, CloudWatch, etc.)
```

**Key Point**: Narration **never touches** the HTTP response. It goes straight to stdout.

---

## ✅ Answer 2: Complete Separation of Concerns

The worker distinguishes between narration and SSE streams because **they use completely different code paths**:

### Path 1: Narration Events (to stdout)

**Location**: Anywhere in the code  
**Mechanism**: `narrate()` function  
**Output**: stdout as JSON logs

**Example from `src/http/execute.rs` (lines 52-60):**
```rust
// This goes to STDOUT as a JSON log
narrate(NarrationFields {
    actor: ACTOR_HTTP_SERVER,
    action: ACTION_EXECUTE_REQUEST,
    target: req.job_id.clone(),
    human: format!("Inference request validated for job {}", req.job_id),
    cute: Some(format!("Job {} looks good, let's go! ✅", req.job_id)),
    job_id: Some(req.job_id.clone()),
    ..Default::default()
});
```

**Output to stdout:**
```json
{"timestamp":"2025-10-09T13:13:27Z","level":"INFO","actor":"http-server","action":"execute_request","target":"job-123","human":"Inference request validated for job job-123","cute":"Job job-123 looks good, let's go! ✅","job_id":"job-123"}
```

---

### Path 2: SSE Token Stream (to HTTP response)

**Location**: `src/http/execute.rs`  
**Mechanism**: Axum SSE response  
**Output**: HTTP response body (text/event-stream)

**Example from `src/http/execute.rs` (lines 105-126):**
```rust
// Convert result to SSE events
let mut events = vec![InferenceEvent::Started {
    job_id: req.job_id.clone(),
    model: "model".to_string(),
    started_at: "0".to_string(),
}];

for (i, token) in result.tokens.iter().enumerate() {
    events.push(InferenceEvent::Token { 
        t: token.clone(), 
        i: i as u32 
    });
}

events.push(InferenceEvent::End {
    tokens_out: result.tokens.len() as u32,
    decode_time_ms: result.decode_time_ms,
    stop_reason: result.stop_reason,
    stop_sequence_matched: result.stop_sequence_matched,
});

// This goes to HTTP RESPONSE BODY
let stream: EventStream = Box::new(
    stream::iter(events).map(|event| 
        Ok(Event::default().json_data(&event).unwrap())
    )
);

Ok(Sse::new(stream))  // Returns HTTP response
```

**Output to HTTP response:**
```
HTTP/1.1 200 OK
Content-Type: text/event-stream

event: started
data: {"type":"started","job_id":"job-123","model":"model","started_at":"0"}

event: token
data: {"type":"token","t":"Hello","i":0}

event: token
data: {"type":"token","t":" world","i":1}

event: end
data: {"type":"end","tokens_out":50,"decode_time_ms":250,"stop_reason":"MAX_TOKENS"}
```

---

## 🎯 Side-by-Side Comparison

### During One Inference Request

**What the code does:**

```rust
// src/http/execute.rs - handle_execute()

// 1. NARRATION: Log that we're starting (goes to stdout)
narrate(NarrationFields {
    actor: ACTOR_HTTP_SERVER,
    action: ACTION_EXECUTE_REQUEST,
    human: "Inference request validated for job job-123",
    // ...
});

// 2. Execute inference
let result = backend.lock().await.execute(&req.prompt, &config).await?;

// 3. NARRATION: Log completion (goes to stdout)
// (This happens inside backend.execute() - see inference.rs)
narrate(NarrationFields {
    actor: ACTOR_CANDLE_BACKEND,
    action: ACTION_INFERENCE_COMPLETE,
    human: "Inference completed (50 tokens in 250 ms)",
    // ...
});

// 4. SSE: Build response events (goes to HTTP response)
let events = vec![
    InferenceEvent::Started { ... },
    InferenceEvent::Token { t: "Hello", i: 0 },
    InferenceEvent::Token { t: " world", i: 1 },
    InferenceEvent::End { tokens_out: 50, ... },
];

// 5. SSE: Return HTTP response
Ok(Sse::new(stream))
```

**What gets output:**

**To stdout (captured by logs):**
```json
{"timestamp":"2025-10-09T13:13:27.001Z","level":"INFO","actor":"http-server","action":"execute_request","human":"Inference request validated for job job-123"}
{"timestamp":"2025-10-09T13:13:27.010Z","level":"INFO","actor":"candle-backend","action":"inference_start","human":"Starting inference (prompt: 15 chars, max_tokens: 50)"}
{"timestamp":"2025-10-09T13:13:27.020Z","level":"INFO","actor":"tokenizer","action":"tokenize","human":"Tokenized prompt (15 tokens)"}
{"timestamp":"2025-10-09T13:13:27.270Z","level":"INFO","actor":"candle-backend","action":"inference_complete","human":"Inference completed (50 tokens in 250 ms, 200 tok/s)"}
```

**To HTTP response (client sees this):**
```
event: started
data: {"type":"started","job_id":"job-123","model":"llama-7b","started_at":"2025-10-09T13:13:27Z"}

event: token
data: {"type":"token","t":"Hello","i":0}

event: token
data: {"type":"token","t":" world","i":1}

event: end
data: {"type":"end","tokens_out":50,"decode_time_ms":250,"stop_reason":"MAX_TOKENS"}
```

---

## 🔍 How They're Distinguished

### 1. Different Data Structures

**Narration:**
```rust
// Defined in narration-core/src/lib.rs
pub struct NarrationFields {
    pub actor: &'static str,
    pub action: &'static str,
    pub target: String,
    pub human: String,
    pub cute: Option<String>,
    pub story: Option<String>,
    pub correlation_id: Option<String>,
    // ... 20+ more fields
}
```

**SSE Events:**
```rust
// Defined in src/http/sse.rs
pub enum InferenceEvent {
    Started { job_id: String, model: String, started_at: String },
    Token { t: String, i: u32 },
    Metrics { tokens_per_sec: f32, vram_bytes: u64 },
    End { tokens_out: u32, decode_time_ms: u64, stop_reason: StopReason },
    Error { code: String, message: String },
}
```

**Completely different types!**

---

### 2. Different Output Mechanisms

**Narration:**
- Calls `tracing::event!()` macro
- Goes through `tracing_subscriber`
- Writes to **file descriptor 1** (stdout)
- Never touches HTTP response

**SSE:**
- Returns `Sse<EventStream>` from handler
- Axum serializes it to HTTP response
- Writes to **HTTP socket** (TCP connection)
- Never touches stdout

---

### 3. Different Consumers

**Narration logs are consumed by:**
- Log aggregators (Loki, Elasticsearch, CloudWatch)
- Monitoring systems (Prometheus + Loki)
- Developers debugging issues
- Operators investigating incidents

**SSE events are consumed by:**
- End users (browser, mobile app)
- Client applications
- Orchestrator (relays to client)

---

## 📊 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    llorch-candled Worker                     │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  HTTP Handler: handle_execute()                        │ │
│  │                                                         │ │
│  │  1. narrate("Inference request validated")             │ │
│  │     ↓                                                   │ │
│  │     tracing::event!() ──→ stdout ──→ Logs              │ │
│  │                                                         │ │
│  │  2. backend.execute(&prompt, &config)                  │ │
│  │     ↓                                                   │ │
│  │     narrate("Tokenized prompt") ──→ stdout ──→ Logs    │ │
│  │     narrate("Inference complete") ──→ stdout ──→ Logs  │ │
│  │                                                         │ │
│  │  3. Build SSE events                                   │ │
│  │     events = [Started, Token, Token, ..., End]         │ │
│  │                                                         │ │
│  │  4. Return Sse::new(stream)                            │ │
│  │     ↓                                                   │ │
│  │     HTTP Response ──→ Client                           │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  Two completely separate output paths:                      │
│  • Narration → stdout → Logs (backend observability)        │
│  • SSE → HTTP response → Client (user-facing results)       │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎯 Key Insights

### 1. **No Wiring Needed**
Narration-core doesn't need to be "wired up" to HTTP. It's wired to `tracing`, which is wired to stdout. That's it.

### 2. **Fire-and-Forget**
When you call `narrate()`, it immediately writes to stdout and returns. It doesn't wait for anything, doesn't block, doesn't affect the HTTP response.

### 3. **Completely Independent**
You could:
- Remove all `narrate()` calls → SSE still works
- Remove SSE streaming → narration still works
- They don't know about each other

### 4. **Same Request, Different Outputs**
One HTTP request produces:
- **Multiple narration events** → stdout → logs
- **One SSE stream** → HTTP response → client

### 5. **Correlation ID is the Bridge**
The only connection between them is the `correlation_id`:
- Extracted from `X-Correlation-Id` HTTP header
- Included in narration events
- Can be included in SSE metadata (optional)
- Allows operators to correlate logs with user requests

---

## 🔧 Practical Example

### User makes request:
```bash
curl -N -H "X-Correlation-Id: req-abc123" \
  http://localhost:8080/execute \
  -d '{"job_id":"job-123","prompt":"Hello","max_tokens":5}'
```

### Worker stdout (logs):
```json
{"timestamp":"2025-10-09T13:13:27.001Z","level":"INFO","actor":"http-server","action":"execute_request","correlation_id":"req-abc123","job_id":"job-123","human":"Inference request validated for job job-123"}
{"timestamp":"2025-10-09T13:13:27.010Z","level":"INFO","actor":"candle-backend","action":"inference_start","correlation_id":"req-abc123","human":"Starting inference (prompt: 5 chars, max_tokens: 5)"}
{"timestamp":"2025-10-09T13:13:27.050Z","level":"INFO","actor":"candle-backend","action":"inference_complete","correlation_id":"req-abc123","human":"Inference completed (5 tokens in 40 ms, 125 tok/s)"}
```

### HTTP response (user sees):
```
HTTP/1.1 200 OK
Content-Type: text/event-stream
X-Correlation-Id: req-abc123

event: started
data: {"type":"started","job_id":"job-123","model":"llama-7b","started_at":"2025-10-09T13:13:27Z"}

event: token
data: {"type":"token","t":"Hello","i":0}

event: token
data: {"type":"token","t":" world","i":1}

event: end
data: {"type":"end","tokens_out":5,"decode_time_ms":40,"stop_reason":"MAX_TOKENS"}
```

### Operator can correlate:
```bash
# Find all events for this request
grep "req-abc123" /var/log/llama-orch/worker.log

# See the full story:
# - Request validated
# - Inference started
# - Inference completed in 40ms
```

---

## ⚠️ CRITICAL UPDATE: This Explanation Was INCOMPLETE!

### What This Document Got Wrong

**Original claim**: "Narration goes to stdout, SSE goes to HTTP, they're completely separate."

**Reality**: Narration needs to go to BOTH stdout AND SSE!

### Corrected Architecture

**Narration has TWO outputs:**

1. **Stdout (Current Implementation ✅)**
   - Worker lifecycle events (startup, shutdown, model loading)
   - Captured by pool-manager
   - ~13 events per worker lifetime
   - **This part works correctly**

2. **SSE Stream (NOT YET IMPLEMENTED ❌)**
   - Per-request events (inference progress, token generation)
   - Streamed to user via orchestrator
   - ~8 events per inference request
   - **This part is MISSING**

### What Needs to Change

The `narrate()` function should:
```rust
pub fn narrate(fields: NarrationFields) {
    // 1. ALWAYS log to stdout (for pool-manager)
    tracing::event!(Level::INFO, ...);
    
    // 2. IF in HTTP request context, ALSO emit SSE event
    if let Some(sse_tx) = get_current_sse_sender() {
        sse_tx.send(InferenceEvent::Narration { ... });
    }
}
```

### Complete Architecture

**See**: `NARRATION_ARCHITECTURE_FINAL.md` for:
- Which events go to stdout only (13 events)
- Which events go to SSE (8 events)
- How to implement dual output
- Complete implementation plan

---

*Corrected by the Narration Core Team 🎀*  
*We apologize for the initial confusion! 💝*
