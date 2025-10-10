# TEAM-038 Narration Architecture Decision

**Team:** TEAM-038 (Implementation Team)  
**Date:** 2025-10-10T14:30  
**Status:** ⚠️ PARTIALLY CORRECTED - See TEAM_038_NARRATION_FLOW_CORRECTED.md  
**Priority:** CRITICAL - Affects User Experience

---

## ⚠️ IMPORTANT: This Document Has Been Corrected

**This document contains the original analysis with a critical misunderstanding:**
- ❌ WRONG: "Stdout narration is for pool-manager (operators)"
- ✅ CORRECT: "All narration is for users. Transport varies by HTTP server state."

**For the CORRECTED architecture, see:**
- `TEAM_038_NARRATION_FLOW_CORRECTED.md` - Complete corrected flow
- `TEAM_039_HANDOFF_NARRATION.md` - Implementation handoff with corrections

**Key corrections made:**
1. Audience is ALWAYS the user (not operators)
2. pool-manager → rbee-hive (terminology update)
3. Transport mechanism explained correctly (stdout → rbee-hive → SSE → queen-rbee → user)

---

## 🎯 Executive Summary

After studying all narration plans from the Narration Core Team, I recommend:

1. ✅ **Keep narration** - It serves a critical user-facing purpose
2. ✅ **Narration can PARTIALLY replace tracing** - For user-visible events
3. ✅ **Dual output is essential** - stdout (operators) + SSE (users)
4. ❌ **Tracing is NOT overkill** - Still needed for internal debugging

**Key Insight:** Narration is for USERS FIRST, developers second. It must flow to rbee-keeper's shell via SSE.

---

## 📊 Current State Analysis

### What Narration Core Team Built

**✅ Implemented:**
- Narration events with cute messages
- Correlation ID propagation
- Structured logging via tracing
- 25 narration points across worker lifecycle

**❌ Missing (CRITICAL):**
- Narration events do NOT go to SSE stream
- User cannot see narration in real-time
- Only operators see narration (in logs)

**⚠️ CORRECTION:** The audience is ALWAYS the user. What's missing is the transport mechanism (SSE) during inference. See `TEAM_038_NARRATION_FLOW_CORRECTED.md` for the correct architecture.

### The Architecture Gap

**Current Flow:**
```
narrate() → tracing::event!() → stdout → log files
```

**Required Flow:**
```
narrate() → BOTH:
  1. tracing::event!() → stdout → log files (operators)
  2. SSE event → orchestrator → rbee-keeper shell (users)
```

---

## 🎯 The User's Vision (100% Correct)

### What User Wants to See in rbee-keeper Shell

```bash
$ rbee-keeper infer --node mac --model tinyllama --prompt "hello"

[Narration] 🌅 Worker waking up on mac...
[Narration] 📦 Loading model tinyllama-q4.gguf...
[Narration] 🛏️ Model loaded! 669 MB cozy in VRAM!
[Narration] 🚀 Starting inference (prompt: 5 chars, max_tokens: 20)
[Narration] 🍰 Tokenized prompt (1 token)
[Narration] 🧹 Reset KV cache for fresh start
[Tokens]    Hello world, this is a test...
[Narration] 🎯 Generated 10 tokens
[Narration] 🎉 Inference complete! 20 tokens in 150ms (133 tok/s)

✅ Done!
```

**Key Points:**
1. Narration goes to **stderr** (so user sees it)
2. Tokens go to **stdout** (so AI agent can pipe it)
3. Narration is **optional** (can be disabled with `--quiet`)
4. Narration shows **what's happening behind the scenes**

---

## 🔍 Detailed Analysis

### 1. Narration vs Tracing: Are They Redundant?

**NO! They serve different purposes:**

| Aspect | Narration | Tracing |
|--------|-----------|---------|
| **Audience** | Users (via rbee-keeper) | Developers (via logs) |
| **Purpose** | Show what's happening | Debug internal state |
| **Format** | Human-friendly cute messages | Structured technical logs |
| **Output** | SSE → rbee-keeper shell | stdout → log aggregator |
| **Content** | "Loading model... 📦" | `model_path=/models/..., size_mb=669` |
| **When** | User-visible events | All events (including internal) |

**Example:**

**Narration (User sees):**
```
🚀 Starting inference (prompt: 15 chars, max_tokens: 50)
```

**Tracing (Developer sees in logs):**
```json
{
  "timestamp": "2025-10-10T14:30:00Z",
  "level": "INFO",
  "actor": "candle-backend",
  "action": "inference_start",
  "correlation_id": "req-abc123",
  "job_id": "job-123",
  "prompt_length": 15,
  "max_tokens": 50,
  "temperature": 0.7,
  "top_p": 0.9,
  "model_path": "/models/tinyllama-q4.gguf",
  "device": "cuda:0",
  "vram_allocated_mb": 669
}
```

**Verdict:** Both are needed. Narration is user-facing, tracing is developer-facing.

---

### 2. Can Narration Replace Tracing?

**PARTIALLY - Only for user-visible events:**

#### Events That Should Be NARRATION ONLY (SSE + stdout)
These are user-facing and should flow to rbee-keeper:

1. ✅ Model loading progress
2. ✅ Inference start/complete
3. ✅ Token generation progress
4. ✅ Errors (user-friendly messages)
5. ✅ Worker startup (if user spawned it)

#### Events That Should Be TRACING ONLY (stdout only)
These are internal and users don't need to see:

1. ✅ HTTP request parsing details
2. ✅ Memory allocation details
3. ✅ Cache management internals
4. ✅ Device initialization details
5. ✅ Performance profiling data
6. ✅ Debug-level state dumps

#### Events That Need BOTH
These should emit narration (for users) AND tracing (for developers):

1. ✅ Inference errors (narration: "VRAM exhausted", tracing: full stack trace)
2. ✅ Model loading (narration: "Loading...", tracing: file paths, sizes, timings)
3. ✅ Worker crashes (narration: "Worker crashed", tracing: panic details)

**Verdict:** Narration can replace tracing for ~30% of events (user-visible ones). The other 70% still need tracing for debugging.

---

### 3. Stdout vs SSE: The Critical Distinction

**The Narration Core Team identified TWO types of narration:**

#### Type 1: Stdout Narration (Worker Lifecycle)
**Audience:** USER (via rbee-keeper shell)  
**When:** Worker startup, shutdown, health checks  
**Count:** ~13 events per worker lifetime
**Transport:** stdout → rbee-hive captures → SSE → queen-rbee → stdout → user shell

**Examples:**
- "Worker starting on port 8001"
- "Initialized CUDA device 0"
- "Loaded model (669 MB)"
- "HTTP server listening on 0.0.0.0:8001"
- "Calling rbee-hive ready callback"

**Why stdout:** These happen BEFORE HTTP connection exists, or AFTER it's closed. rbee-hive captures stdout and converts to SSE for queen-rbee.

#### Type 2: SSE Narration (Per-Request)
**Audience:** USER (via rbee-keeper shell)  
**When:** During inference request  
**Count:** ~8 events per request
**Transport:** SSE → queen-rbee → stdout → user shell

**Examples:**
- "Starting inference (prompt: 15 chars)"
- "Tokenized prompt (15 tokens)"
- "Generated 10 tokens"
- "Inference complete (50 tokens in 250ms)"

**Why SSE:** These happen DURING an active HTTP request, worker HTTP server is ready, so narration flows directly through SSE to queen-rbee.

---

## 🎯 Recommended Architecture

### Dual Output System

```
┌─────────────────────────────────────────────────────────────┐
│ narrate(fields)                                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│ 1. ALWAYS emit to tracing (for developers/operators)        │
│    tracing::event!(Level::INFO, ...)                        │
│    → stdout → log aggregator                                │
│                                                              │
│ 2. IF in HTTP request context, ALSO emit to SSE (for users) │
│    if let Some(tx) = get_sse_sender() {                     │
│        tx.send(InferenceEvent::Narration { ... })           │
│    }                                                         │
│    → SSE stream → orchestrator → rbee-keeper → user's shell │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Flow Diagram

```
User runs: rbee-keeper infer --node mac --model tinyllama --prompt "hello"
    ↓
rbee-keeper connects to queen-rbee
    ↓
queen-rbee connects to rbee-hive (mac)
    ↓
rbee-hive spawns llm-worker-rbee
    ↓
Worker emits narration:
    ├─→ stdout: {"actor":"llm-worker-rbee","action":"startup",...}
    │   → rbee-hive logs (operator sees)
    │
    └─→ (no SSE yet, HTTP server not started)

Worker HTTP server starts
    ↓
rbee-hive sends POST /execute to worker
    ↓
Worker starts inference, emits narration:
    ├─→ stdout: {"actor":"candle-backend","action":"inference_start",...}
    │   → rbee-hive logs (operator sees)
    │
    └─→ SSE: event: narration, data: {"human":"Starting inference...","cute":"🚀"}
        → queen-rbee → rbee-keeper → user's shell (user sees)

Worker generates tokens:
    ├─→ SSE: event: token, data: {"t":"Hello","i":0}
    │   → rbee-keeper stdout (AI agent pipes this)
    │
    └─→ SSE: event: narration, data: {"human":"Generated 10 tokens","cute":"🎯"}
        → rbee-keeper stderr (user sees progress)

Worker completes:
    ├─→ stdout: {"actor":"candle-backend","action":"inference_complete",...}
    │   → rbee-hive logs (operator sees)
    │
    └─→ SSE: event: narration, data: {"human":"Complete! 20 tokens in 150ms","cute":"🎉"}
        → rbee-keeper stderr (user sees)
```

---

## 🔧 Implementation Requirements

### 1. Add Narration Event Type to SSE

**File:** `bin/llm-worker-rbee/src/http/sse.rs`

```rust
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InferenceEvent {
    Started { ... },
    Token { ... },
    Metrics { ... },
    
    /// NEW: Narration event for user-facing progress updates
    /// TEAM-038: Added for user visibility in rbee-keeper shell
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

### 2. Create SSE Channel in Execute Handler

**File:** `bin/llm-worker-rbee/src/http/execute.rs`

```rust
pub async fn handle_execute<B: InferenceBackend>(
    State(backend): State<Arc<Mutex<B>>>,
    Json(req): Json<ExecuteRequest>,
) -> Result<Sse<EventStream>, ValidationErrorResponse> {
    // TEAM-038: Create channel for narration events
    let (narration_tx, narration_rx) = tokio::sync::mpsc::unbounded_channel();
    
    // Store in request-local context
    REQUEST_NARRATION_SENDER.with(|sender| {
        *sender.borrow_mut() = Some(narration_tx.clone());
    });
    
    // ... rest of handler
    
    // Merge narration events with token events
    let narration_stream = tokio_stream::wrappers::UnboundedReceiverStream::new(narration_rx);
    let token_stream = /* ... existing token stream ... */;
    let merged_stream = stream::select(narration_stream, token_stream);
    
    Ok(Sse::new(merged_stream))
}
```

### 3. Modify Narration Function for Dual Output

**File:** `libs/narration-core/src/lib.rs` (or worker-specific wrapper)

```rust
pub fn narrate(fields: NarrationFields) {
    // TEAM-038: Dual output for narration
    
    // 1. ALWAYS emit to tracing (for operators/developers)
    tracing::event!(
        Level::INFO,
        actor = fields.actor,
        action = fields.action,
        target = fields.target,
        human = fields.human,
        cute = fields.cute.as_deref(),
        story = fields.story.as_deref(),
        correlation_id = fields.correlation_id.as_deref(),
        job_id = fields.job_id.as_deref(),
    );
    
    // 2. IF in HTTP request context, ALSO emit to SSE (for users)
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

// Thread-local storage for SSE sender
thread_local! {
    static REQUEST_NARRATION_SENDER: RefCell<Option<UnboundedSender<InferenceEvent>>> 
        = RefCell::new(None);
}

fn get_current_narration_sender() -> Option<UnboundedSender<InferenceEvent>> {
    REQUEST_NARRATION_SENDER.with(|sender| sender.borrow().clone())
}
```

### 4. Update rbee-keeper to Display Narration

**File:** `bin/rbee-keeper/src/commands/infer.rs`

```rust
// TEAM-038: Handle narration events separately from tokens
match event.type {
    "narration" => {
        // Display to stderr (user sees, doesn't interfere with stdout)
        if !args.quiet {
            eprintln!("[{}] {}", event.actor, event.cute.unwrap_or(event.human));
        }
    }
    "token" => {
        // Display to stdout (AI agent can pipe this)
        print!("{}", event.t);
        io::stdout().flush()?;
    }
    "end" => {
        println!(); // Newline after tokens
        if !args.quiet {
            eprintln!("✅ Complete! {} tokens in {}ms", 
                event.tokens_out, event.decode_time_ms);
        }
    }
    _ => {}
}
```

---

## 🎯 Answers to Your Questions

### Q1: Should we do narration AND tracing?

**YES, but with clarification:**

- **Narration** = User-facing events (cute messages, progress updates)
  - Goes to: SSE stream → rbee-keeper shell (users see)
  - Also goes to: stdout → logs (operators see)
  
- **Tracing** = Developer-facing events (technical details, debug info)
  - Goes to: stdout → logs (developers see)
  - Never goes to SSE (users don't need this)

**Overlap:** ~30% of events emit both narration AND tracing. The other 70% are tracing-only (internal details).

### Q2: Is tracing overkill?

**NO - Tracing is essential for:**

1. **Debugging production issues** - Full technical details
2. **Performance profiling** - Timing, memory, resource usage
3. **Correlation across services** - Trace requests end-to-end
4. **Alerting/monitoring** - Structured logs for automation
5. **Internal state dumps** - Cache state, model state, etc.

**Narration cannot replace tracing because:**
- Narration is simplified for users (hides complexity)
- Narration doesn't include all technical details
- Narration is optional (can be disabled with `--quiet`)
- Tracing is always on (needed for debugging)

### Q3: Narration is for users first, developers second?

**100% CORRECT!**

**Primary audience:** Users running rbee-keeper  
**Secondary audience:** Operators monitoring logs

**This means:**
1. ✅ Narration MUST go to SSE stream (users see it in real-time)
2. ✅ Narration MUST have cute/friendly messages (not just technical)
3. ✅ Narration MUST be optional (`--quiet` flag)
4. ✅ Narration MUST be separate from token stream (tokens to stdout, narration to stderr)

### Q4: Combination of stdout and SSE?

**YES - Dual output is essential:**

**Stdout (always):**
- Worker lifecycle events (startup, shutdown)
- All narration events (for debugging/logs)
- Captured by rbee-hive → converted to SSE → queen-rbee → user shell

**SSE (during requests):**
- Per-request narration events (inference progress)
- Token stream (inference output)
- Flows to user via rbee-keeper

**Both are needed because:**
- Stdout: Used when HTTP server not ready (startup/shutdown) - rbee-hive captures and converts to SSE
- SSE: Used when HTTP server active (inference) - direct to queen-rbee
- Both end up in user's shell via queen-rbee

### Q5: Distinction between inference tokens (product) and narration (byproduct)?

**CRITICAL DISTINCTION:**

**Inference Tokens (THE PRODUCT):**
- What the user asked for
- Goes to stdout (AI agent can pipe it)
- Example: `"Hello world, this is a test..."`
- SSE event type: `token`

**Narration (THE BYPRODUCT):**
- Progress updates, what's happening
- Goes to stderr (user sees, doesn't interfere with stdout)
- Example: `"🚀 Starting inference..."`, `"🎯 Generated 10 tokens"`
- SSE event type: `narration`

**In rbee-keeper:**
```bash
# Tokens go to stdout (can be piped)
rbee-keeper infer ... > output.txt

# Narration goes to stderr (user sees)
[Narration] 🚀 Starting inference...
[Narration] 🎯 Generated 10 tokens
[Narration] 🎉 Complete!

# output.txt contains only tokens:
Hello world, this is a test...
```

---

## 📊 Event Classification

### Category 1: Stdout-Only Events (13 events)
**Audience:** Pool-manager (operators)  
**When:** Worker lifecycle (startup, shutdown)  
**Output:** stdout → logs

1. Worker startup
2. Device initialization (CPU/CUDA/Metal)
3. Model loading
4. HTTP server start/bind
5. Pool-manager callback
6. Server shutdown

### Category 2: SSE + Stdout Events (8 events)
**Audience:** Users (via rbee-keeper) + Operators (via logs)  
**When:** During inference request  
**Output:** SSE → rbee-keeper + stdout → logs

1. Inference start
2. Tokenization
3. Cache reset
4. Token generation progress
5. Inference complete
6. Validation errors
7. Inference errors

### Category 3: Tracing-Only Events (internal)
**Audience:** Developers (debugging)  
**When:** Always  
**Output:** stdout → logs (never SSE)

1. HTTP request parsing details
2. Memory allocation details
3. Performance profiling
4. Debug-level state dumps
5. Internal cache management

---

## ✅ Final Recommendations

### 1. Keep Narration ✅
**Reason:** Essential for user experience. Users need to see what's happening.

### 2. Keep Tracing ✅
**Reason:** Essential for debugging. Developers need technical details.

### 3. Implement Dual Output ✅
**Reason:** Narration serves two audiences (users + operators) at different times.

**Implementation:**
- Stdout: Always emit (for operators)
- SSE: Emit during requests (for users)

### 4. Narration Can Partially Replace Tracing ✅
**Reason:** For user-visible events, narration is sufficient. But internal events still need tracing.

**Split:**
- 30% of events: Narration only (user-visible)
- 40% of events: Narration + Tracing (user-visible + technical details)
- 30% of events: Tracing only (internal debugging)

### 5. Narration is Optional ✅
**Reason:** Users should be able to disable narration with `--quiet` flag.

**Implementation:**
```bash
# With narration (default)
rbee-keeper infer ...
[Narration] 🚀 Starting inference...
Hello world...

# Without narration (quiet mode)
rbee-keeper infer --quiet ...
Hello world...
```

---

## 🚀 Next Steps for TEAM-039

### Priority 1: Implement SSE Narration
1. Add `Narration` event type to `InferenceEvent` enum
2. Create SSE channel in execute handler
3. Modify `narrate()` to emit to both stdout and SSE
4. Test that narration appears in SSE stream

### Priority 2: Update rbee-keeper
1. Handle `narration` SSE events
2. Display narration to stderr
3. Display tokens to stdout
4. Add `--quiet` flag to disable narration

### Priority 3: Update Orchestrator
1. Relay narration events from worker to client
2. Preserve event ordering (narration + tokens)
3. Handle narration in SSE stream

### Priority 4: Documentation
1. Update OpenAPI spec with narration events
2. Document event ordering
3. Add examples to API docs
4. Update user guide with narration examples

---

## 📚 References

- **Narration Plans:** `bin/llm-worker-rbee/.plan/`
- **Critical Gap:** `CRITICAL_NARRATION_MISSING.md`
- **Architecture:** `NARRATION_ARCHITECTURE_FINAL.md`
- **Stdout vs SSE:** `NARRATION_VS_SSE_ARCHITECTURE.md`
- **Wiring:** `NARRATION_WIRING_EXPLAINED.md`

---

**TEAM-038 Recommendation Complete ✅**

**Summary:**
- ✅ Keep narration (essential for UX)
- ✅ Keep tracing (essential for debugging)
- ✅ Implement dual output (stdout + SSE)
- ✅ Narration is for users first, developers second
- ✅ Tokens (product) and narration (byproduct) are separate streams

**The Narration Core Team was RIGHT - narration MUST flow to users via SSE!** 🎀
