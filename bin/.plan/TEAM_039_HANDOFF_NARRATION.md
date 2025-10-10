# TEAM-039 HANDOFF: Narration Implementation

**From:** TEAM-038 (Implementation Team)  
**To:** TEAM-039 (Next Implementation Team)  
**Date:** 2025-10-10T14:32  
**Status:** 🔴 CRITICAL - User Experience Blocker  
**Priority:** P0 - Must implement before any inference testing

---

## 🎯 Mission

Implement dual-output narration system so users can see what's happening in real-time when running `rbee-keeper infer`.

**Current State:** Narration only goes to logs (operators see it)  
**Required State:** Narration goes to logs AND SSE stream (users see it in rbee-keeper shell)

---

## 📋 Background

### What TEAM-038 Discovered

After analyzing all narration plans from the Narration Core Team, we confirmed:

1. ✅ **Narration is essential** - Users need to see what's happening
2. ✅ **Tracing is NOT overkill** - Still needed for debugging (70% of events)
3. ✅ **Dual output required** - stdout (operators) + SSE (users)
4. ✅ **Narration is for users FIRST** - Must flow to rbee-keeper shell

### The Architecture Gap

**Current (WRONG):**
```
narrate() → tracing::event!() → stdout → logs
```
❌ Users can't see narration in real-time

**Required (CORRECT):**
```
narrate() → BOTH:
  1. tracing::event!() → stdout → logs (operators)
  2. SSE event → orchestrator → rbee-keeper shell (users)
```
✅ Users see narration in real-time

---

## 🎯 Your Tasks (Priority Order)

### Priority 1: Add Narration Event Type to SSE ⚡ CRITICAL

**File:** `bin/llm-worker-rbee/src/http/sse.rs`

**What to do:**
Add `Narration` variant to the `InferenceEvent` enum.

**Code:**
```rust
// TEAM-039: Added narration event type for user-facing progress updates
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InferenceEvent {
    Started {
        job_id: String,
        model: String,
        started_at: String,
    },
    
    Token {
        t: String,
        i: u32,
    },
    
    Metrics {
        tokens_out: u32,
        decode_time_ms: u64,
        tokens_per_sec: f64,
    },
    
    /// NEW: Narration event for user-facing progress updates
    /// Shows what's happening behind the scenes (model loading, tokenization, etc.)
    /// Goes to rbee-keeper shell so users can see progress
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
    
    End {
        tokens_out: u32,
        decode_time_ms: u64,
        stop_reason: String,
        stop_sequence_matched: Option<String>,
    },
    
    Error {
        code: String,
        message: String,
    },
}
```

**Why:** SSE stream needs to carry narration events, not just tokens.

**Test:**
```bash
# After implementation, SSE stream should include:
event: narration
data: {"type":"narration","actor":"candle-backend","action":"inference_start","human":"Starting inference...","cute":"🚀"}

event: token
data: {"type":"token","t":"Hello","i":0}

event: narration
data: {"type":"narration","actor":"candle-backend","action":"token_generate","human":"Generated 10 tokens","cute":"🎯"}
```

---

### Priority 2: Create SSE Channel in Execute Handler ⚡ CRITICAL

**File:** `bin/llm-worker-rbee/src/http/execute.rs`

**What to do:**
Create a channel that narration events can flow through during inference requests.

**Code:**
```rust
use tokio::sync::mpsc::{unbounded_channel, UnboundedSender};
use tokio_stream::wrappers::UnboundedReceiverStream;

// TEAM-039: Thread-local storage for narration SSE sender
thread_local! {
    static NARRATION_SENDER: RefCell<Option<UnboundedSender<InferenceEvent>>> 
        = RefCell::new(None);
}

pub async fn handle_execute<B: InferenceBackend>(
    State(backend): State<Arc<Mutex<B>>>,
    Json(req): Json<ExecuteRequest>,
) -> Result<Sse<EventStream>, ValidationErrorResponse> {
    // TEAM-039: Create channel for narration events
    let (narration_tx, narration_rx) = unbounded_channel();
    
    // Store in thread-local context so narrate() can access it
    NARRATION_SENDER.with(|sender| {
        *sender.borrow_mut() = Some(narration_tx.clone());
    });
    
    // ... existing validation and inference code ...
    
    // TEAM-039: Merge narration events with token events
    let narration_stream = UnboundedReceiverStream::new(narration_rx);
    let token_stream = /* ... existing token stream ... */;
    
    // Interleave narration and token events
    let merged_stream = stream::select(narration_stream, token_stream);
    
    // Clean up thread-local after request completes
    let cleanup_stream = merged_stream.chain(stream::once(async {
        NARRATION_SENDER.with(|sender| {
            *sender.borrow_mut() = None;
        });
        // Return a dummy event that gets filtered out
        InferenceEvent::End { /* ... */ }
    }));
    
    Ok(Sse::new(cleanup_stream))
}

// TEAM-039: Helper function to get current narration sender
pub fn get_narration_sender() -> Option<UnboundedSender<InferenceEvent>> {
    NARRATION_SENDER.with(|sender| sender.borrow().clone())
}
```

**Why:** We need a way to send narration events into the SSE stream during inference.

**Test:**
```rust
#[tokio::test]
async fn test_narration_channel() {
    let (tx, mut rx) = unbounded_channel();
    
    // Simulate narration event
    tx.send(InferenceEvent::Narration {
        actor: "test".to_string(),
        action: "test".to_string(),
        target: "test".to_string(),
        human: "Test narration".to_string(),
        cute: Some("🎯".to_string()),
        story: None,
        correlation_id: None,
        job_id: None,
    }).unwrap();
    
    // Verify event received
    let event = rx.recv().await.unwrap();
    assert!(matches!(event, InferenceEvent::Narration { .. }));
}
```

---

### Priority 3: Modify Narration Function for Dual Output ⚡ CRITICAL

**File:** `libs/narration-core/src/lib.rs` OR create `bin/llm-worker-rbee/src/narration.rs`

**What to do:**
Make `narrate()` emit to BOTH stdout (tracing) AND SSE (if in request context).

**Code:**
```rust
// TEAM-039: Dual-output narration function
use crate::http::execute::get_narration_sender;
use crate::http::sse::InferenceEvent;

pub fn narrate(fields: NarrationFields) {
    // 1. ALWAYS emit to tracing (for operators/developers)
    //    This goes to stdout → logs
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
        worker_id = fields.worker_id.as_deref(),
    );
    
    // 2. IF in HTTP request context, ALSO emit to SSE (for users)
    //    This goes to SSE stream → orchestrator → rbee-keeper shell
    if let Some(tx) = get_narration_sender() {
        let event = InferenceEvent::Narration {
            actor: fields.actor.to_string(),
            action: fields.action.to_string(),
            target: fields.target.clone(),
            human: fields.human.clone(),
            cute: fields.cute.clone(),
            story: fields.story.clone(),
            correlation_id: fields.correlation_id.clone(),
            job_id: fields.job_id.clone(),
        };
        
        // Send to SSE stream (ignore errors if channel closed)
        let _ = tx.send(event);
    }
}
```

**Why:** Narration needs to go to BOTH places - logs (for operators) and SSE (for users).

**Test:**
```rust
#[test]
fn test_narration_dual_output() {
    // Set up SSE channel
    let (tx, mut rx) = unbounded_channel();
    NARRATION_SENDER.with(|sender| {
        *sender.borrow_mut() = Some(tx);
    });
    
    // Emit narration
    narrate(NarrationFields {
        actor: ACTOR_CANDLE_BACKEND,
        action: ACTION_INFERENCE_START,
        target: "job-123".to_string(),
        human: "Starting inference".to_string(),
        cute: Some("🚀".to_string()),
        ..Default::default()
    });
    
    // Verify SSE event received
    let event = rx.try_recv().unwrap();
    assert!(matches!(event, InferenceEvent::Narration { .. }));
    
    // Verify tracing event emitted (check logs)
    // Note: This requires tracing-subscriber test setup
}
```

---

### Priority 4: Update rbee-keeper to Display Narration 🔥 HIGH

**File:** `bin/rbee-keeper/src/commands/infer.rs`

**What to do:**
Handle narration SSE events and display them to stderr (so they don't interfere with stdout tokens).

**Code:**
```rust
// TEAM-039: Handle SSE events from worker
pub async fn handle_infer(args: InferArgs) -> Result<()> {
    // ... connect to queen-rbee, get worker URL ...
    
    // Stream SSE events from worker
    let mut event_stream = connect_sse(&worker_url).await?;
    
    while let Some(event) = event_stream.next().await {
        match event.type {
            // TEAM-039: Display narration to stderr (user sees progress)
            "narration" => {
                if !args.quiet {
                    let emoji = event.cute.as_deref().unwrap_or("");
                    let message = event.human.as_deref().unwrap_or("Processing...");
                    eprintln!("[{}] {} {}", event.actor, emoji, message);
                }
            }
            
            // Tokens go to stdout (AI agent can pipe this)
            "token" => {
                print!("{}", event.t);
                io::stdout().flush()?;
            }
            
            // Metrics (optional display)
            "metrics" => {
                if !args.quiet {
                    eprintln!("[Metrics] {} tokens in {}ms ({:.1} tok/s)", 
                        event.tokens_out, 
                        event.decode_time_ms,
                        event.tokens_per_sec);
                }
            }
            
            // End of inference
            "end" => {
                println!(); // Newline after tokens
                if !args.quiet {
                    eprintln!("✅ Complete! {} tokens in {}ms", 
                        event.tokens_out, event.decode_time_ms);
                }
                break;
            }
            
            // Error
            "error" => {
                eprintln!("❌ Error: {}", event.message);
                return Err(anyhow!("Inference failed: {}", event.message));
            }
            
            _ => {
                // Unknown event type, ignore
            }
        }
    }
    
    Ok(())
}
```

**Add `--quiet` flag:**
```rust
// In cli.rs
#[derive(Parser)]
pub struct InferArgs {
    // ... existing args ...
    
    /// Disable narration output (only show tokens)
    #[arg(long)]
    pub quiet: bool,
}
```

**Why:** Users need to see narration in their shell, separate from token output.

**Test:**
```bash
# With narration (default)
$ rbee-keeper infer --node mac --model tinyllama --prompt "hello"
[candle-backend] 🚀 Starting inference (prompt: 5 chars, max_tokens: 20)
[tokenizer] 🍰 Tokenized prompt (1 token)
[candle-backend] 🧹 Reset KV cache for fresh start
Hello world, this is a test...
[candle-backend] 🎯 Generated 10 tokens
[candle-backend] 🎉 Inference complete! 20 tokens in 150ms (133 tok/s)
✅ Complete! 20 tokens in 150ms

# Without narration (quiet mode)
$ rbee-keeper infer --node mac --model tinyllama --prompt "hello" --quiet
Hello world, this is a test...

# Piping tokens to file (narration goes to stderr, doesn't interfere)
$ rbee-keeper infer ... > output.txt
[candle-backend] 🚀 Starting inference...
# output.txt contains only: Hello world, this is a test...
```

---

### Priority 5: Update Orchestrator to Relay Narration 🔥 HIGH

**File:** `bin/queen-rbee/src/routes/tasks.rs` (or wherever SSE relay happens)

**What to do:**
Relay narration events from worker to client (rbee-keeper).

**Code:**
```rust
// TEAM-039: Relay all SSE events from worker to client
pub async fn stream_task_events(
    task_id: String,
) -> Result<Sse<EventStream>, StatusCode> {
    // Get worker URL for this task
    let worker_url = get_worker_url_for_task(&task_id)?;
    
    // Connect to worker's SSE stream
    let worker_stream = connect_to_worker_sse(&worker_url).await?;
    
    // Relay all events to client (including narration)
    let relay_stream = worker_stream.map(|event| {
        // Pass through all event types: started, token, narration, metrics, end, error
        Ok(Event::default().json_data(&event).unwrap())
    });
    
    Ok(Sse::new(relay_stream))
}
```

**Why:** Orchestrator is the middleman - it needs to pass narration events through to rbee-keeper.

**Test:**
```bash
# Verify orchestrator relays narration events
curl -N http://localhost:8080/v2/tasks/job-123/events

event: narration
data: {"type":"narration","actor":"candle-backend","action":"inference_start",...}

event: token
data: {"type":"token","t":"Hello","i":0}

event: narration
data: {"type":"narration","actor":"candle-backend","action":"token_generate",...}
```

---

### Priority 6: Update OpenAPI Spec 📝 MEDIUM

**File:** `contracts/openapi/worker.yaml`

**What to do:**
Document the new `narration` event type in the SSE stream.

**Code:**
```yaml
# TEAM-039: Added narration event type
/execute:
  post:
    summary: Execute inference request
    responses:
      '200':
        description: SSE stream of inference events
        content:
          text/event-stream:
            schema:
              oneOf:
                - $ref: '#/components/schemas/StartedEvent'
                - $ref: '#/components/schemas/TokenEvent'
                - $ref: '#/components/schemas/MetricsEvent'
                - $ref: '#/components/schemas/NarrationEvent'  # NEW
                - $ref: '#/components/schemas/EndEvent'
                - $ref: '#/components/schemas/ErrorEvent'

components:
  schemas:
    NarrationEvent:
      type: object
      required:
        - type
        - actor
        - action
        - target
        - human
      properties:
        type:
          type: string
          enum: [narration]
        actor:
          type: string
          description: Component emitting the narration
          example: "candle-backend"
        action:
          type: string
          description: Action being performed
          example: "inference_start"
        target:
          type: string
          description: Target of the action
          example: "job-123"
        human:
          type: string
          description: Human-readable message
          example: "Starting inference (prompt: 15 chars, max_tokens: 50)"
        cute:
          type: string
          nullable: true
          description: Cute/friendly version of the message
          example: "Time to generate 50 tokens! Let's go! 🚀"
        story:
          type: string
          nullable: true
          description: Story-style narration
        correlation_id:
          type: string
          nullable: true
        job_id:
          type: string
          nullable: true
```

**Why:** API consumers need to know about narration events.

---

## 🎯 Event Classification Reference

### CRITICAL CORRECTION: All Narration is for the USER

**WRONG:** "Stdout narration is for operators"  
**CORRECT:** "All narration is for users. Transport varies by HTTP server state."

### Stdout → rbee-hive → SSE Events (13 events)
**Audience:** USER (via rbee-keeper shell)  
**When:** Worker lifecycle (startup, shutdown) - HTTP server not ready  
**Output:** stdout → rbee-hive captures → SSE → queen-rbee → stdout → user shell

1. Worker startup (`llm-worker-rbee`, `startup`)
2. Device init CPU (`device-manager`, `device_init`)
3. Device init CUDA (`device-manager`, `device_init`)
4. Device init Metal (`device-manager`, `device_init`)
5. Model load start (`model-loader`, `model_load`)
6. Model load complete (`model-loader`, `model_load`)
7. Pool callback start (`llm-worker-rbee`, `callback_ready`)
8. Pool callback complete (`llm-worker-rbee`, `callback_ready`)
9. Pool callback error (`llm-worker-rbee`, `error`)
10. HTTP server init (`http-server`, `server_start`)
11. HTTP server bind (`http-server`, `server_bind`)
12. HTTP server bind error (`http-server`, `error`)
13. HTTP server shutdown (`http-server`, `server_shutdown`)

**Why stdout:** HTTP server not ready yet, so narration goes through stdout. rbee-hive captures and converts to SSE for queen-rbee.

### SSE → queen-rbee → stdout Events (8 events per request)
**Audience:** USER (via rbee-keeper shell)  
**When:** During inference request - HTTP server active  
**Output:** SSE → queen-rbee → stdout → user shell

1. Validation error (`http-server`, `error`)
2. Request validated (`http-server`, `execute_request`)
3. Inference error (`candle-backend`, `error`)
4. Inference start (`candle-backend`, `inference_start`)
5. Tokenize (`tokenizer`, `tokenize`)
6. Cache reset (`candle-backend`, `cache_reset`)
7. Token progress (`candle-backend`, `token_generate`)
8. Inference complete (`candle-backend`, `inference_complete`)

**Why SSE:** HTTP server is active, so narration goes directly through SSE to queen-rbee.

---

## 🔄 The Complete Transport Flow

### Understanding the Plumbing

**All narration is for the USER. The transport is just plumbing.**

```
Phase 1: rbee-hive Startup
  rbee-hive narrate() → stdout → SSH → queen-rbee → stdout → user shell

Phase 2: Worker Startup (HTTP not ready)
  worker narrate() → stdout → rbee-hive captures → SSE → queen-rbee → stdout → user shell

Phase 3: Inference (HTTP active)
  worker narrate() → SSE → queen-rbee → stdout → user shell

Phase 4: Worker Shutdown (HTTP closing)
  worker narrate() → stdout → rbee-hive captures → SSE → queen-rbee → stdout → user shell
```

### Key Implementation: rbee-hive Must Capture Worker Stdout

**CRITICAL:** rbee-hive must capture worker stdout during startup/shutdown and convert to SSE.

```rust
// In rbee-hive worker spawning code
let mut child = Command::new("llm-worker-rbee")
    .stdout(Stdio::piped())  // Capture stdout
    .spawn()?;

let stdout = child.stdout.take().unwrap();

// Read stdout and convert to SSE events
tokio::spawn(async move {
    let reader = BufReader::new(stdout);
    let mut lines = reader.lines();
    
    while let Some(line) = lines.next_line().await.unwrap() {
        // Parse JSON narration from stdout
        if let Ok(narration) = parse_narration_json(&line) {
            // Convert to SSE and send to queen-rbee
            send_sse_to_queen_rbee(narration).await;
        }
    }
});
```

---

## ✅ Acceptance Criteria

### Must Have (P0)
- [ ] `Narration` event type added to `InferenceEvent` enum
- [ ] SSE channel created in execute handler (worker)
- [ ] `narrate()` emits to both stdout and SSE (based on HTTP state)
- [ ] **rbee-hive captures worker stdout** and converts to SSE
- [ ] **queen-rbee merges narration from SSH and SSE sources**
- [ ] rbee-keeper displays narration to stderr
- [ ] rbee-keeper displays tokens to stdout
- [ ] `--quiet` flag works (disables narration)

### Should Have (P1)
- [ ] OpenAPI spec updated with narration events
- [ ] Unit tests for narration channel
- [ ] Integration test: narration appears in SSE stream
- [ ] Integration test: rbee-keeper displays narration correctly

### Nice to Have (P2)
- [ ] Narration events include emoji by default
- [ ] Narration can be filtered by actor (e.g., only show inference events)
- [ ] Narration timestamps displayed in rbee-keeper

---

## 🧪 Testing Plan

### Unit Tests

**Test 1: Narration event serialization**
```rust
#[test]
fn test_narration_event_serialization() {
    let event = InferenceEvent::Narration {
        actor: "test".to_string(),
        action: "test_action".to_string(),
        target: "test-target".to_string(),
        human: "Test message".to_string(),
        cute: Some("🎯".to_string()),
        story: None,
        correlation_id: Some("req-123".to_string()),
        job_id: Some("job-123".to_string()),
    };
    
    let json = serde_json::to_string(&event).unwrap();
    assert!(json.contains("\"type\":\"narration\""));
    assert!(json.contains("\"actor\":\"test\""));
}
```

**Test 2: Narration channel**
```rust
#[tokio::test]
async fn test_narration_channel_flow() {
    let (tx, mut rx) = unbounded_channel();
    
    // Send narration event
    tx.send(InferenceEvent::Narration {
        actor: "test".to_string(),
        action: "test".to_string(),
        target: "test".to_string(),
        human: "Test".to_string(),
        cute: None,
        story: None,
        correlation_id: None,
        job_id: None,
    }).unwrap();
    
    // Receive and verify
    let event = rx.recv().await.unwrap();
    assert!(matches!(event, InferenceEvent::Narration { .. }));
}
```

### Integration Tests

**Test 3: End-to-end narration flow**
```bash
# Start worker
cargo run --bin llm-worker-rbee -- --port 8001 --model /models/tinyllama-q4.gguf

# Send inference request, capture SSE stream
curl -N http://localhost:8001/execute \
  -H "Content-Type: application/json" \
  -d '{"job_id":"test-123","prompt":"hello","max_tokens":10}' \
  | grep "event: narration"

# Should see narration events in output
```

**Test 4: rbee-keeper displays narration**
```bash
# Run inference with narration
rbee-keeper infer --node mac --model tinyllama --prompt "hello" 2>&1 | tee output.log

# Verify narration in stderr
grep "🚀" output.log
grep "🎯" output.log
grep "🎉" output.log

# Verify tokens in stdout
rbee-keeper infer ... 2>/dev/null | grep "Hello"
```

---

## 🚨 Common Pitfalls

### Pitfall 1: Thread-Local Context
**Problem:** `thread_local!` doesn't work across async boundaries  
**Solution:** Use `tokio::task_local!` or pass channel through request extensions

### Pitfall 2: Channel Cleanup
**Problem:** Narration sender not cleaned up after request  
**Solution:** Use RAII pattern or explicit cleanup in finally block

### Pitfall 3: Event Ordering
**Problem:** Narration events arrive out of order with tokens  
**Solution:** Use `stream::select` to interleave events in order

### Pitfall 4: Stdout vs Stderr
**Problem:** Narration goes to stdout, interferes with token piping  
**Solution:** Always use stderr for narration in rbee-keeper

### Pitfall 5: Quiet Mode
**Problem:** `--quiet` flag doesn't suppress all narration  
**Solution:** Check `args.quiet` before every `eprintln!` call

---

## 📚 References

### Documentation
- **Decision Doc:** `bin/.specs/TEAM_038_NARRATION_DECISION.md`
- **Narration Plans:** `bin/llm-worker-rbee/.plan/`
- **Critical Gap:** `bin/llm-worker-rbee/.plan/CRITICAL_NARRATION_MISSING.md`
- **Architecture:** `bin/llm-worker-rbee/.plan/NARRATION_ARCHITECTURE_FINAL.md`

### Code Locations
- **SSE Types:** `bin/llm-worker-rbee/src/http/sse.rs`
- **Execute Handler:** `bin/llm-worker-rbee/src/http/execute.rs`
- **Narration Core:** `libs/narration-core/src/lib.rs`
- **rbee-keeper Infer:** `bin/rbee-keeper/src/commands/infer.rs`

### Specs
- **Worker Spec:** `requirements/01_M0_worker_orcd.md` (§13.1 Narration)
- **Orchestrator Spec:** `requirements/00_llama-orch.md` (§10.2.2 Logs)
- **OpenAPI:** `contracts/openapi/worker.yaml`

---

## 🎯 Expected User Experience (After Implementation)

```bash
$ rbee-keeper infer --node mac --model tinyllama --prompt "write a haiku"

[llm-worker-rbee] 🌅 Worker waking up to help with inference!
[device-manager] 🖥️ Initialized Metal device 0
[model-loader] 📦 Loading model from /models/tinyllama-q4.gguf
[model-loader] 🛏️ Model loaded! 669 MB cozy in VRAM!
[http-server] 🚀 HTTP server ready on port 8001
[candle-backend] 🚀 Starting inference (prompt: 14 chars, max_tokens: 50)
[tokenizer] 🍰 Tokenized prompt (4 tokens)
[candle-backend] 🧹 Reset KV cache for fresh start
Cherry blossoms fall
Petals dance on gentle breeze
Spring whispers goodbye
[candle-backend] 🎯 Generated 10 tokens
[candle-backend] 🎯 Generated 20 tokens
[candle-backend] 🎉 Inference complete! 42 tokens in 250ms (168 tok/s)

✅ Done!
```

**With `--quiet`:**
```bash
$ rbee-keeper infer --node mac --model tinyllama --prompt "write a haiku" --quiet

Cherry blossoms fall
Petals dance on gentle breeze
Spring whispers goodbye
```

**Piping tokens:**
```bash
$ rbee-keeper infer ... > haiku.txt
[candle-backend] 🚀 Starting inference...
[candle-backend] 🎉 Complete!

$ cat haiku.txt
Cherry blossoms fall
Petals dance on gentle breeze
Spring whispers goodbye
```

---

## ✅ Definition of Done

**This task is complete when:**

1. ✅ User runs `rbee-keeper infer` and sees narration events in real-time
2. ✅ Narration goes to stderr (doesn't interfere with stdout tokens)
3. ✅ `--quiet` flag suppresses narration
4. ✅ Tokens can be piped to file without narration
5. ✅ All 8 per-request narration events appear in SSE stream
6. ✅ Orchestrator relays narration events to client
7. ✅ OpenAPI spec documents narration events
8. ✅ Unit tests pass for narration channel
9. ✅ Integration test shows narration in rbee-keeper shell

---

---

## 🎯 CRITICAL UNDERSTANDING

### The Audience Never Changes

**ALL narration is for the USER in rbee-keeper shell.**

The transport mechanism changes based on HTTP server state:
- **Before HTTP ready:** stdout → (captured by parent) → SSE → queen-rbee → stdout → user
- **During HTTP active:** SSE → queen-rbee → stdout → user
- **After HTTP closed:** stdout → (captured by parent) → SSE → queen-rbee → stdout → user

### The Three-Tier Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ TIER 1: rbee-keeper (User's Shell)                          │
│ - Displays ALL narration to stderr                          │
│ - Displays tokens to stdout                                 │
│ - User sees everything in real-time                         │
└─────────────────────────────────────────────────────────────┘
                              ↑
                              │ stdout (all narration)
                              │
┌─────────────────────────────┴───────────────────────────────┐
│ TIER 2: queen-rbee (Orchestrator)                           │
│ - Merges narration from SSH (rbee-hive startup)             │
│ - Merges narration from SSE (worker startup via rbee-hive)  │
│ - Merges narration from SSE (inference via worker)          │
│ - Outputs ALL to stdout for rbee-keeper                     │
└──────────────────────────────────────────────────────────────┘
                    ↑                           ↑
                    │ SSH stdout                │ SSE (HTTP)
                    │                           │
┌───────────────────┴──────┐    ┌──────────────┴──────────────┐
│ TIER 3a: rbee-hive       │    │ TIER 3b: llm-worker-rbee    │
│ - Captures worker stdout │    │ - Emits narration to stdout │
│ - Converts to SSE        │    │   (when HTTP not ready)     │
│ - Sends to queen-rbee    │    │ - Emits narration to SSE    │
│                          │    │   (when HTTP active)        │
└──────────────────────────┘    └─────────────────────────────┘
```

### Key Implementation Points

1. **Worker:** Emit to stdout (always) + SSE (when HTTP active)
2. **rbee-hive:** Capture worker stdout, convert to SSE, send to queen-rbee
3. **queen-rbee:** Merge all narration sources, output to stdout
4. **rbee-keeper:** Display narration to stderr, tokens to stdout

---

**TEAM-038 Handoff Complete ✅**

**Priority:** P0 - This is a critical user experience feature. Users need to see what's happening!

**Estimated Effort:** 3-4 days for full implementation + testing (increased due to stdout capture complexity)

**Critical Files:**
- `bin/.specs/TEAM_038_NARRATION_FLOW_CORRECTED.md` - Corrected architecture
- `bin/.specs/TEAM_038_NARRATION_DECISION.md` - Original analysis
- `bin/llm-worker-rbee/.plan/` - Narration Core Team plans

**Questions?** See the corrected architecture document for the complete flow.

Good luck, TEAM-039! Remember: **All narration is for the user. The transport is just plumbing.** 🎀💝
