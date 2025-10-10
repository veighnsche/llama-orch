# TEAM-038 Narration Flow - CORRECTED Architecture

**Team:** TEAM-038 (Implementation Team)  
**Date:** 2025-10-10T14:41  
**Status:** ✅ CORRECTED  
**Priority:** CRITICAL - Fixes misunderstanding about narration flow

---

## 🚨 Critical Correction

**WRONG ASSUMPTION:**
> "Stdout narration is for pool-manager (operators)"

**CORRECT UNDERSTANDING:**
> "ALL narration is for the USER. The transport mechanism changes based on whether HTTP server is ready."

---

## 🎯 The Real Architecture

### Key Insight: Narration is ALWAYS for the User

**The audience never changes - it's always the USER in rbee-keeper shell.**

**What changes is the TRANSPORT:**
- **Before HTTP server ready:** stdout → SSH → queen-rbee → stdout → user shell
- **During HTTP server active:** SSE → queen-rbee → stdout → user shell
- **After HTTP server closed:** stdout → SSH → queen-rbee → stdout → user shell

---

## 📊 Complete Narration Flow

### Phase 1: rbee-hive Startup (SSH Transport)

```
User runs: rbee-keeper infer --node mac --model tinyllama --prompt "hello"
    ↓
rbee-keeper → queen-rbee (HTTP)
    ↓
queen-rbee starts rbee-hive via SSH on "mac"
    ↓
rbee-hive starts up, emits narration:
    narrate("rbee-hive starting on port 9200")
    ↓
    stdout → SSH tunnel → queen-rbee
    ↓
    queen-rbee → stdout → rbee-keeper shell
    ↓
    USER SEES: [rbee-hive] 🌅 Starting pool manager on port 9200
```

**Transport:** stdout → SSH → queen-rbee → stdout → user shell

**Why:** rbee-hive HTTP server not ready yet, so narration goes through SSH stdout.

---

### Phase 2: Worker Startup (SSH Transport)

```
rbee-hive HTTP server ready
    ↓
queen-rbee sends task to rbee-hive (HTTP POST /v1/workers/spawn)
    ↓
rbee-hive spawns llm-worker-rbee process
    ↓
Worker starts up, emits narration:
    narrate("Worker starting on port 8001")
    narrate("Initialized Metal device 0")
    narrate("Loading model from /models/tinyllama-q4.gguf")
    narrate("Model loaded! 669 MB in VRAM")
    narrate("HTTP server starting on port 8001")
    ↓
    stdout → rbee-hive captures (parent process)
    ↓
    rbee-hive → SSE → queen-rbee (HTTP)
    ↓
    queen-rbee → stdout → rbee-keeper shell
    ↓
    USER SEES:
    [llm-worker-rbee] 🌅 Worker starting on port 8001
    [device-manager] 🖥️ Initialized Metal device 0
    [model-loader] 📦 Loading model...
    [model-loader] 🛏️ Model loaded! 669 MB in VRAM
    [http-server] 🚀 HTTP server starting on port 8001
```

**Transport:** stdout → rbee-hive → SSE → queen-rbee → stdout → user shell

**Why:** Worker HTTP server not ready yet, so narration goes through stdout. rbee-hive captures it and converts to SSE for queen-rbee.

---

### Phase 3: Inference Request (SSE Transport)

```
Worker HTTP server ready
    ↓
queen-rbee sends inference request to worker (HTTP POST /execute)
    ↓
Worker processes request, emits narration:
    narrate("Starting inference (prompt: 5 chars)")
    narrate("Tokenized prompt (1 token)")
    narrate("Reset KV cache")
    narrate("Generated 10 tokens")
    narrate("Inference complete! 20 tokens in 150ms")
    ↓
    SSE → queen-rbee (HTTP)
    ↓
    queen-rbee → stdout → rbee-keeper shell
    ↓
    USER SEES:
    [candle-backend] 🚀 Starting inference...
    [tokenizer] 🍰 Tokenized prompt (1 token)
    [candle-backend] 🧹 Reset KV cache
    Hello world, this is a test...
    [candle-backend] 🎯 Generated 10 tokens
    [candle-backend] 🎉 Complete! 20 tokens in 150ms
```

**Transport:** SSE → queen-rbee → stdout → user shell

**Why:** Worker HTTP server is active, so narration goes through SSE directly to queen-rbee.

---

### Phase 4: Worker Shutdown (SSH Transport)

```
Inference complete, worker idle timeout reached
    ↓
rbee-hive sends shutdown to worker (HTTP POST /shutdown OR SIGTERM)
    ↓
Worker shuts down, emits narration:
    narrate("Shutting down gracefully")
    narrate("Freeing VRAM")
    narrate("Worker exiting")
    ↓
    stdout → rbee-hive captures (parent process)
    ↓
    rbee-hive → SSE → queen-rbee (HTTP)
    ↓
    queen-rbee → stdout → rbee-keeper shell
    ↓
    USER SEES:
    [http-server] 👋 Shutting down gracefully
    [device-manager] 🧹 Freeing 669 MB VRAM
    [llm-worker-rbee] 💤 Worker exiting
```

**Transport:** stdout → rbee-hive → SSE → queen-rbee → stdout → user shell

**Why:** Worker HTTP server may be closing, so narration goes through stdout. rbee-hive captures and converts to SSE.

---

## 🔄 Transport Decision Logic

### Rule: HTTP Server State Determines Transport

```rust
fn narrate(fields: NarrationFields) {
    // ALWAYS emit to tracing (for debugging/logs)
    tracing::event!(Level::INFO, ...);
    
    // Determine transport based on HTTP server state
    if http_server_is_active() {
        // HTTP server ready → use SSE
        if let Some(sse_tx) = get_sse_sender() {
            sse_tx.send(InferenceEvent::Narration { ... });
        }
    } else {
        // HTTP server not ready → use stdout
        // Parent process (rbee-hive) will capture and convert to SSE
        // (stdout already emitted via tracing above)
    }
}
```

### Timeline

```
Worker Process Lifecycle:
├─ Phase 1: Startup (HTTP server NOT ready)
│  └─ Transport: stdout → rbee-hive → SSE → queen-rbee → stdout → user
│
├─ Phase 2: HTTP Server Active
│  └─ Transport: SSE → queen-rbee → stdout → user
│
└─ Phase 3: Shutdown (HTTP server closing/closed)
   └─ Transport: stdout → rbee-hive → SSE → queen-rbee → stdout → user
```

---

## 📊 Complete Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│ USER'S SHELL (rbee-keeper)                                      │
│ [rbee-hive] 🌅 Starting pool manager...                         │
│ [llm-worker-rbee] 🌅 Worker starting...                         │
│ [model-loader] 📦 Loading model...                              │
│ [candle-backend] 🚀 Starting inference...                       │
│ Hello world, this is a test...                                  │
│ [candle-backend] 🎉 Complete!                                   │
└─────────────────────────────────────────────────────────────────┘
                              ↑
                              │ stdout (all narration ends here)
                              │
┌─────────────────────────────┴───────────────────────────────────┐
│ QUEEN-RBEE (Orchestrator on "blep")                             │
│                                                                  │
│ Receives narration from multiple sources:                       │
│ 1. SSH stdout from rbee-hive startup                            │
│ 2. SSE from rbee-hive (worker startup narration)                │
│ 3. SSE from worker (inference narration)                        │
│                                                                  │
│ Converts ALL to stdout for rbee-keeper shell                    │
└──────────────────────────────────────────────────────────────────┘
                    ↑                           ↑
                    │ SSH stdout                │ SSE (HTTP)
                    │                           │
┌───────────────────┴──────┐    ┌──────────────┴──────────────────┐
│ RBEE-HIVE (on "mac")     │    │ LLM-WORKER-RBEE (on "mac")      │
│                          │    │                                  │
│ Phase 1: Startup         │    │ Phase 1: Startup                 │
│   narrate() → stdout ────┼────┤   narrate() → stdout             │
│   (via SSH to queen-rbee)│    │   (captured by rbee-hive)        │
│                          │    │                                  │
│ Phase 2: Running         │    │ Phase 2: HTTP Server Active      │
│   Captures worker stdout │    │   narrate() → SSE ───────────────┤
│   Converts to SSE ───────┼────┤   (direct to queen-rbee)         │
│   Sends to queen-rbee    │    │                                  │
│                          │    │ Phase 3: Shutdown                │
│ Phase 3: Shutdown        │    │   narrate() → stdout             │
│   narrate() → stdout ────┼────┤   (captured by rbee-hive)        │
│   (via SSH to queen-rbee)│    │                                  │
└──────────────────────────┘    └──────────────────────────────────┘
```

---

## 🎯 Corrected Event Classification

### Type 1: Stdout → SSH → SSE → stdout (rbee-hive Startup)
**Audience:** USER (via rbee-keeper shell)  
**When:** rbee-hive starting up  
**Transport:** stdout → SSH → queen-rbee → stdout → user shell

**Events:**
1. rbee-hive startup
2. rbee-hive HTTP server bind
3. rbee-hive ready

### Type 2: Stdout → SSE → stdout (Worker Startup)
**Audience:** USER (via rbee-keeper shell)  
**When:** Worker starting up (HTTP server not ready)  
**Transport:** stdout → rbee-hive → SSE → queen-rbee → stdout → user shell

**Events:**
1. Worker startup
2. Device initialization (CPU/CUDA/Metal)
3. Model loading
4. HTTP server starting
5. Pool callback to rbee-hive

### Type 3: SSE → stdout (Inference)
**Audience:** USER (via rbee-keeper shell)  
**When:** Worker HTTP server active, processing request  
**Transport:** SSE → queen-rbee → stdout → user shell

**Events:**
1. Inference start
2. Tokenization
3. Cache reset
4. Token generation progress
5. Inference complete

### Type 4: Stdout → SSE → stdout (Worker Shutdown)
**Audience:** USER (via rbee-keeper shell)  
**When:** Worker shutting down (HTTP server closing)  
**Transport:** stdout → rbee-hive → SSE → queen-rbee → stdout → user shell

**Events:**
1. HTTP server shutdown
2. VRAM freed
3. Worker exit

---

## 🔧 Implementation Requirements

### 1. rbee-hive Must Capture Worker Stdout

**File:** `bin/rbee-hive/src/worker_manager.rs`

```rust
// TEAM-039: Capture worker stdout and convert to SSE
pub async fn spawn_worker(args: WorkerArgs) -> Result<WorkerHandle> {
    let mut child = Command::new("llm-worker-rbee")
        .args(&args.to_args())
        .stdout(Stdio::piped())  // Capture stdout
        .stderr(Stdio::piped())  // Capture stderr
        .spawn()?;
    
    let stdout = child.stdout.take().unwrap();
    let (narration_tx, narration_rx) = unbounded_channel();
    
    // Spawn task to read worker stdout and convert to SSE
    tokio::spawn(async move {
        let reader = BufReader::new(stdout);
        let mut lines = reader.lines();
        
        while let Some(line) = lines.next_line().await.unwrap() {
            // Parse JSON narration event from stdout
            if let Ok(event) = parse_narration_from_stdout(&line) {
                // Convert to SSE event
                let sse_event = InferenceEvent::Narration {
                    actor: event.actor,
                    action: event.action,
                    target: event.target,
                    human: event.human,
                    cute: event.cute,
                    story: event.story,
                    correlation_id: event.correlation_id,
                    job_id: event.job_id,
                };
                
                // Send to queen-rbee via SSE
                let _ = narration_tx.send(sse_event);
            }
        }
    });
    
    Ok(WorkerHandle {
        child,
        narration_rx,
    })
}

fn parse_narration_from_stdout(line: &str) -> Result<NarrationEvent> {
    // Parse JSON log line from tracing-subscriber
    let log: serde_json::Value = serde_json::from_str(line)?;
    
    Ok(NarrationEvent {
        actor: log["actor"].as_str().unwrap_or("unknown").to_string(),
        action: log["action"].as_str().unwrap_or("unknown").to_string(),
        target: log["target"].as_str().unwrap_or("").to_string(),
        human: log["human"].as_str().unwrap_or("").to_string(),
        cute: log["cute"].as_str().map(String::from),
        story: log["story"].as_str().map(String::from),
        correlation_id: log["correlation_id"].as_str().map(String::from),
        job_id: log["job_id"].as_str().map(String::from),
    })
}
```

### 2. rbee-hive Must Stream Narration to queen-rbee

**File:** `bin/rbee-hive/src/http/workers.rs`

```rust
// TEAM-039: Stream worker narration events to queen-rbee
#[axum::debug_handler]
pub async fn stream_worker_narration(
    Path(worker_id): Path<String>,
) -> Result<Sse<EventStream>, StatusCode> {
    let worker = get_worker(&worker_id).ok_or(StatusCode::NOT_FOUND)?;
    
    // Stream narration events from worker
    let stream = worker.narration_rx.map(|event| {
        Ok(Event::default().json_data(&event).unwrap())
    });
    
    Ok(Sse::new(stream))
}
```

### 3. queen-rbee Must Merge All Narration Sources

**File:** `bin/queen-rbee/src/routes/tasks.rs`

```rust
// TEAM-039: Merge narration from multiple sources
pub async fn stream_task_events(
    task_id: String,
) -> Result<Sse<EventStream>, StatusCode> {
    let task = get_task(&task_id)?;
    
    // Source 1: rbee-hive startup narration (via SSH stdout)
    let hive_startup_stream = capture_ssh_stdout(&task.node);
    
    // Source 2: Worker startup narration (via rbee-hive SSE)
    let worker_startup_stream = connect_to_hive_sse(&task.node, &task.worker_id);
    
    // Source 3: Inference narration (via worker SSE)
    let inference_stream = connect_to_worker_sse(&task.worker_url);
    
    // Merge all streams in order
    let merged_stream = stream::select_all(vec![
        hive_startup_stream,
        worker_startup_stream,
        inference_stream,
    ]);
    
    Ok(Sse::new(merged_stream))
}

fn capture_ssh_stdout(node: &str) -> impl Stream<Item = InferenceEvent> {
    // Capture stdout from SSH session
    // Parse JSON logs and convert to narration events
    // ...
}
```

### 4. rbee-keeper Displays All Narration to stderr

**File:** `bin/rbee-keeper/src/commands/infer.rs`

```rust
// TEAM-039: Display all narration to stderr (user sees it)
pub async fn handle_infer(args: InferArgs) -> Result<()> {
    let mut event_stream = connect_to_queen_rbee(&args).await?;
    
    while let Some(event) = event_stream.next().await {
        match event.type {
            "narration" => {
                // Display to stderr (doesn't interfere with stdout tokens)
                if !args.quiet {
                    let emoji = event.cute.as_deref().unwrap_or("");
                    let message = event.human.as_deref().unwrap_or("");
                    eprintln!("[{}] {} {}", event.actor, emoji, message);
                }
            }
            "token" => {
                // Display to stdout (AI agent can pipe this)
                print!("{}", event.t);
                io::stdout().flush()?;
            }
            "end" => {
                println!();
                if !args.quiet {
                    eprintln!("✅ Complete! {} tokens in {}ms", 
                        event.tokens_out, event.decode_time_ms);
                }
                break;
            }
            _ => {}
        }
    }
    
    Ok(())
}
```

---

## 🎯 Key Takeaways

### 1. Audience is ALWAYS the User
- ❌ NOT: "Stdout narration is for operators"
- ✅ CORRECT: "All narration is for users, transport varies"

### 2. Transport Changes Based on HTTP Server State
- **Before HTTP ready:** stdout → (SSH or parent capture) → SSE → stdout → user
- **During HTTP active:** SSE → stdout → user
- **After HTTP closed:** stdout → (SSH or parent capture) → SSE → stdout → user

### 3. rbee-hive is a Bridge
- Captures worker stdout during startup/shutdown
- Converts stdout to SSE for queen-rbee
- Relays worker SSE during inference

### 4. queen-rbee is the Aggregator
- Receives narration from multiple sources (SSH, SSE)
- Merges all narration streams
- Outputs to stdout for rbee-keeper shell

### 5. rbee-keeper is the Display
- Receives all narration via stdout from queen-rbee
- Displays narration to stderr (user sees)
- Displays tokens to stdout (AI agent pipes)

---

## 📊 Corrected Terminology

**OLD (WRONG):**
- "pool-manager" → ❌ This is outdated
- "Stdout narration is for operators" → ❌ This is wrong

**NEW (CORRECT):**
- "rbee-hive" → ✅ Pool manager renamed
- "All narration is for users" → ✅ Audience never changes
- "Transport varies by HTTP state" → ✅ stdout vs SSE

---

## ✅ Summary

**The Complex But Logical Flow:**

1. **rbee-hive startup** → stdout → SSH → queen-rbee → stdout → user
2. **Worker startup** → stdout → rbee-hive → SSE → queen-rbee → stdout → user
3. **Inference** → SSE → queen-rbee → stdout → user
4. **Worker shutdown** → stdout → rbee-hive → SSE → queen-rbee → stdout → user

**All narration ends up in the user's shell, but the path varies based on HTTP server state.**

---

**TEAM-038 Correction Complete ✅**

**This corrects the misunderstanding about narration audience and transport mechanisms.**

**All narration is for the USER. The transport is just plumbing.** 🎀
