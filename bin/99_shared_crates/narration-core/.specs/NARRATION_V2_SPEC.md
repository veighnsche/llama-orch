# Narration Core V2 Specification

**Version:** 2.0.0-draft  
**Date:** 2025-10-26  
**Status:** DESIGN PHASE  
**Author:** TEAM-297

---

## Executive Summary

Narration V2 is a **complete redesign** of the observability/narration system based on the current rbee architecture where:

1. **rbee-keeper** does user orchestration + hive lifecycle (via SSH)
2. **queen-rbee** does API orchestration + scheduling (job-server)
3. **rbee-hive** does worker lifecycle (starts/stops workers)
4. **llm-worker-rbee** does inference

**Key Problem:** The current narration system was designed for a different architecture and now has:
- ❌ Mandatory `job_id` on every narration (was a bandaid fix)
- ❌ Fragile SSE sink (requires channels to exist before narration)
- ❌ No support for pre-job-server narration (startup phase)
- ❌ No support for capturing child process stdout (worker startup)
- ❌ Complex routing that doesn't match the actual architecture

---

## Design Principles

### 1. **Context-Aware Routing**
Narration automatically routes based on execution context, NOT manual configuration:
- **Standalone process** (no job context) → stdout
- **Job server context** (job_id present) → SSE + stdout
- **Child process** (spawned by parent) → stdout (parent captures)

### 2. **Zero-Configuration**
No explicit mode selection. The system detects:
- Am I in a job?
- Am I a daemon with an HTTP server?
- Am I a child process?

### 3. **Process Boundary Transparency**
When a parent spawns a child:
- Child emits narration to stdout (always)
- Parent captures child's stdout
- Parent converts to narration events (if in job context)

### 4. **Dual-Output is Optional**
Not everything needs both stdout AND SSE:
- CLI operations before job submission: stdout only
- Daemon startup: stdout only (captured by supervisor)
- Job operations: SSE + stdout

---

## Narration Flows

### Flow 1: Keeper Starts Queen (Local Only)

```
┌─────────────┐
│ rbee-keeper │
└──────┬──────┘
       │
       │ 1. Spawn queen process
       ▼
┌─────────────┐
│ queen-rbee  │
│             │
│ Narration:  │
│   stdout ───┼──→ keeper captures → displays to user
│             │
└─────────────┘

Context: NO job server, NO SSE, just process startup
```

**Narration emitter:**
```rust
NARRATE.action("start")
    .human("🐝 Queen starting on port 7833")
    .emit();  // Goes to stdout (no job_id = no SSE)
```

**Keeper side:**
```rust
// Spawn queen with stdout captured
let mut child = Command::new("queen-rbee")
    .stdout(Stdio::piped())
    .spawn()?;

// Stream queen's stdout to terminal
let stdout = child.stdout.take().unwrap();
tokio::spawn(async move {
    let reader = BufReader::new(stdout);
    let mut lines = reader.lines();
    while let Some(line) = lines.next_line().await? {
        println!("{}", line);  // Show to user
    }
});
```

---

### Flow 2: Keeper Starts Hive via SSH

#### Scenario A: Localhost Hive

```
┌─────────────┐
│ rbee-keeper │
└──────┬──────┘
       │
       │ 1. Spawn rbee-hive (local)
       ▼
┌─────────────┐
│ rbee-hive   │
│             │
│ Narration:  │
│   stdout ───┼──→ keeper captures → displays to user
│             │
└─────────────┘

Context: NO job server, just daemon startup
```

#### Scenario B: Remote Hive

```
┌─────────────┐
│ rbee-keeper │
└──────┬──────┘
       │
       │ 1. SSH to remote
       │ 2. Start rbee-hive
       ▼
┌─────────────────────────┐
│ SSH Session (remote)    │
│                         │
│  ┌─────────────┐        │
│  │ rbee-hive   │        │
│  │             │        │
│  │ Narration:  │        │
│  │   stdout ───┼────────┼──→ SSH session → keeper → user
│  │             │        │
│  └─────────────┘        │
└─────────────────────────┘

Context: NO job server, stdout over SSH
```

---

### Flow 3: CLI Job Through Job-Server (Download Model Example)

```
┌─────────────┐
│ rbee-keeper │──────────────────────────────────────────┐
└──────┬──────┘                                           │
       │                                                  │
       │ PHASE 1: Before job submission                  │
       │ ─────────────────────────────                   │
       │                                                  │
       │ NARRATE.action("job_submit")                    │
       │   .emit();  ──→ stdout only                     │
       │                                                  │
       │ PHASE 2: Job submission                         │
       │ ───────────────────                             │
       │                                                  │
       │ POST /v1/jobs                                    │
       ▼                                                  │
┌─────────────┐                                           │
│ queen-rbee  │                                           │
│             │                                           │
│ 1. Create job channel (job_id = "abc123")             │
│ 2. Spawn task to handle operation                      │
│ 3. Forward to hive                                     │
│             │                                           │
│             │ POST /v1/jobs (to hive)                  │
│             ▼                                           │
│      ┌─────────────┐                                   │
│      │ rbee-hive   │                                   │
│      │             │                                    │
│      │ 1. Create hive job channel (hive_job_id)       │
│      │ 2. Execute operation (model download)           │
│      │                                                  │
│      │ NARRATE.action("model_download")                │
│      │   .job_id(hive_job_id)  ──→ SSE to queen       │
│      │                                                  │
│      └─────┬───────┘                                   │
│            │                                            │
│            │ SSE events                                 │
│            ▼                                            │
│ Queen receives SSE ──→ forwards to keeper's job channel│
│                                                         │
└─────────────┬───────────────────────────────────────────┘
              │
              │ GET /v1/jobs/abc123/stream
              ▼
       Keeper receives SSE → prints to stdout
```

**Key insight:** Narration switches from stdout-only to SSE-enabled **when job_id is present**.

---

### Flow 4: Hive Starts Worker (Most Complex!)

```
┌─────────────┐
│ rbee-hive   │
└──────┬──────┘
       │
       │ 1. User submits "spawn worker" job
       │ 2. Hive creates job channel (job_id = "hive-job-xyz")
       │
       │ 3. Spawn worker process
       ▼
┌──────────────────┐
│ llm-worker-rbee  │
│                  │
│ STARTUP PHASE:   │
│ ─────────────    │
│                  │
│ NARRATE          │
│   .action("startup")
│   .emit();  ──→ stdout │
│                  │
│ (NO HTTP yet!)   │
│ (NO SSE!)        │
│                  │
│ Loading model... │
│ NARRATE          │
│   .action("model_load")
│   .emit();  ──→ stdout │
│                  │
└─────────┬────────┘
          │
          │ stdout
          ▼
    ┌─────────────┐
    │ rbee-hive   │
    │ (parent)    │
    │             │
    │ Captures    │
    │ worker      │
    │ stdout!     │
    │             │
    │ Parses      │
    │ narration   │
    │ events      │
    │             │
    │ Converts    │
    │ to SSE ─────┼──→ Emits with job_id("hive-job-xyz")
    │             │
    └─────────────┘
```

**Critical requirement:** Hive must:
1. Spawn worker with `Stdio::piped()`
2. Capture worker's stdout line-by-line
3. Parse narration events from stdout
4. Re-emit as narration with the job_id

**Worker side:**
```rust
// Worker has NO job_id during startup
NARRATE.action("startup")
    .human("🐝 Worker starting...")
    .emit();  // Goes to stdout (no job_id)

NARRATE.action("model_load")
    .human("📦 Loading model...")
    .emit();  // Goes to stdout (no job_id)
```

**Hive side:**
```rust
// Spawn worker with captured stdout
let mut child = Command::new("llm-worker-rbee")
    .stdout(Stdio::piped())
    .spawn()?;

let stdout = child.stdout.take().unwrap();

// Stream worker's stdout and convert to narration
tokio::spawn(async move {
    let reader = BufReader::new(stdout);
    let mut lines = reader.lines();
    
    while let Some(line) = lines.next_line().await? {
        // Parse narration event from stdout
        if let Some(narration_event) = parse_narration_line(&line) {
            // Re-emit with job_id for SSE routing
            NARRATE.action(narration_event.action)
                .job_id(&job_id)  // ← Hive's job_id
                .human(&narration_event.message)
                .emit();  // Now goes to SSE!
        } else {
            // Non-narration output, just log it
            tracing::debug!("Worker output: {}", line);
        }
    }
});
```

---

### Flow 5: Tauri GUI Usage

```
┌──────────────┐
│ Tauri GUI    │
│ (rbee-keeper)│
└──────┬───────┘
       │
       │ 1. User clicks "Start Hive"
       │ 2. Tauri command invoked
       │
       ▼
┌──────────────────────┐
│ tauri_commands.rs    │
│                      │
│ Uses JobClient to    │
│ submit to queen      │
│                      │
│ SSE events received  │
│ via callback         │
│                      │
│ Emits to frontend ───┼──→ GUI displays progress
│                      │
└──────────────────────┘
```

**Key difference:** Instead of `println!()`, the line handler emits events to Tauri:

```rust
#[tauri::command]
async fn hive_start(hive_id: String) -> Result<String, String> {
    let client = JobClient::new("http://localhost:7833");
    
    client.submit_and_stream(
        Operation::HiveStart { alias: hive_id },
        |line| {
            // Emit to Tauri frontend instead of stdout
            emit_narration_event(&line)?;
            Ok(())
        }
    ).await
    .map_err(|e| e.to_string())
}
```

---

## Technical Design

### 1. Narration Context Detection

```rust
pub enum NarrationContext {
    /// Standalone process, no job, no HTTP
    Standalone,
    
    /// Daemon with HTTP server + job channels
    JobServer { job_id: String },
    
    /// Child process (worker spawned by hive)
    ChildProcess,
}

impl NarrationContext {
    /// Detect current execution context
    pub fn detect() -> Self {
        // Check if job_id is in thread-local storage
        if let Some(job_id) = JOB_CONTEXT.with(|ctx| ctx.borrow().clone()) {
            return Self::JobServer { job_id };
        }
        
        // Check if we're a child process (PPID != init)
        if is_child_process() {
            return Self::ChildProcess;
        }
        
        Self::Standalone
    }
}
```

### 2. Narration Emitter (Simplified API)

```rust
/// Emit narration based on context
pub fn emit(fields: NarrationFields) {
    match NarrationContext::detect() {
        NarrationContext::Standalone => {
            // Just output to stdout
            emit_to_stdout(&fields);
        }
        
        NarrationContext::JobServer { job_id } => {
            // Dual output: stdout + SSE
            emit_to_stdout(&fields);
            sse_sink::send(&job_id, &fields);
        }
        
        NarrationContext::ChildProcess => {
            // Output to stdout (parent will capture)
            emit_to_stdout(&fields);
        }
    }
}

/// Format narration for stdout
fn emit_to_stdout(fields: &NarrationFields) {
    eprintln!("[{:<10}] {:<15}: {}", 
        fields.actor, 
        fields.action, 
        fields.human
    );
}
```

### 3. Builder API (No Mandatory job_id!)

```rust
pub struct Narration {
    fields: NarrationFields,
}

impl Narration {
    pub fn new(actor: &'static str, action: &'static str, target: impl Into<String>) -> Self {
        Self {
            fields: NarrationFields {
                actor,
                action,
                target: target.into(),
                human: String::new(),
                job_id: None,  // ← Optional!
                ..Default::default()
            },
        }
    }
    
    /// Optional job_id (only needed for SSE routing in job-server context)
    pub fn job_id(mut self, id: impl Into<String>) -> Self {
        self.fields.job_id = Some(id.into());
        self
    }
    
    /// Emit narration (context-aware routing)
    pub fn emit(self) {
        emit(self.fields);
    }
}
```

### 4. Thread-Local Job Context

```rust
use std::cell::RefCell;

thread_local! {
    static JOB_CONTEXT: RefCell<Option<String>> = RefCell::new(None);
}

/// Set job context for current thread
pub fn set_job_context(job_id: String) {
    JOB_CONTEXT.with(|ctx| {
        *ctx.borrow_mut() = Some(job_id);
    });
}

/// Clear job context
pub fn clear_job_context() {
    JOB_CONTEXT.with(|ctx| {
        *ctx.borrow_mut() = None;
    });
}
```

### 5. Child Process Output Capture

```rust
/// Helper for spawning processes with narration capture
pub struct NarrationCapture {
    job_id: Option<String>,
}

impl NarrationCapture {
    pub fn new(job_id: Option<String>) -> Self {
        Self { job_id }
    }
    
    /// Spawn command with stdout capture
    pub async fn spawn(&self, mut command: Command) -> Result<Child> {
        command.stdout(Stdio::piped());
        command.stderr(Stdio::piped());
        
        let mut child = command.spawn()?;
        
        // Capture stdout
        if let Some(stdout) = child.stdout.take() {
            let job_id = self.job_id.clone();
            tokio::spawn(async move {
                Self::stream_output(stdout, job_id).await;
            });
        }
        
        Ok(child)
    }
    
    /// Stream and parse child's output
    async fn stream_output(stdout: ChildStdout, job_id: Option<String>) {
        let reader = BufReader::new(stdout);
        let mut lines = reader.lines();
        
        while let Some(line) = lines.next_line().await.ok().flatten() {
            // Try to parse as narration event
            if let Some(event) = Self::parse_narration(&line) {
                // Re-emit with job_id if we have one
                if let Some(ref jid) = job_id {
                    NARRATE.action(event.action)
                        .job_id(jid)
                        .human(&event.message)
                        .emit();
                } else {
                    // No job context, just print
                    println!("{}", line);
                }
            } else {
                // Not narration, just print
                println!("{}", line);
            }
        }
    }
    
    /// Parse narration event from stdout line
    fn parse_narration(line: &str) -> Option<NarrationEvent> {
        // Match format: "[actor     ] action         : message"
        let re = Regex::new(r"\[(.{10})\] (.{15}): (.+)").ok()?;
        let caps = re.captures(line)?;
        
        Some(NarrationEvent {
            actor: caps[1].trim().to_string(),
            action: caps[2].trim().to_string(),
            message: caps[3].to_string(),
        })
    }
}
```

---

## Migration Strategy

### Phase 1: Add Context Detection (Non-Breaking)
- Add `NarrationContext::detect()`
- Add thread-local job context
- Keep existing APIs working

### Phase 2: Make job_id Optional (Breaking)
- Update `Narration` builder to make `job_id` optional
- Update all call sites to remove mandatory `.job_id()`
- Context-aware routing automatically detects job context

### Phase 3: Add Process Capture Utilities
- Add `NarrationCapture` helper
- Update hive to use it for worker spawning
- Update keeper to use it for queen/hive spawning

### Phase 4: Remove Old SSE Sink
- Remove global channel code
- Remove `create_job_channel()` complexity
- Simplify to just thread-local context

---

## Benefits

### ✅ Simpler API
```rust
// Before (V1):
NARRATE.action("start")
    .job_id(&job_id)  // ← Always required, even when not needed!
    .human("Starting")
    .emit();

// After (V2):
NARRATE.action("start")
    .human("Starting")
    .emit();  // ← Automatically routes based on context!
```

### ✅ Works for Startup Narration
```rust
// Worker startup (no job context yet)
NARRATE.action("startup")
    .human("Worker starting")
    .emit();  // → stdout (no SSE, no job_id needed)

// Later, after HTTP server starts and job context is set:
set_job_context("job-abc");
NARRATE.action("model_load")
    .human("Loading model")
    .emit();  // → stdout + SSE (context detected!)
```

### ✅ Process Boundary Transparency
```rust
// Hive spawns worker
let capture = NarrationCapture::new(Some(job_id.clone()));
let child = capture.spawn(Command::new("llm-worker-rbee")).await?;

// Worker's stdout is automatically captured and converted to SSE!
```

### ✅ No More Fragile Channel Creation
```rust
// V1: Must create channel BEFORE any narration
create_job_channel(job_id.clone(), 1000);
NARRATE.action("start").job_id(&job_id).emit();  // Would drop if channel not created!

// V2: Just set context, narration works
set_job_context(job_id.clone());
NARRATE.action("start").emit();  // Works regardless of channel state
```

---

## Open Questions

1. **How to handle stderr vs stdout?**
   - Proposal: stderr for logs, stdout for narration
   - Child processes emit narration to stdout, logs to stderr

2. **Should we support correlation_id in V2?**
   - Proposal: Yes, but also auto-detected from thread-local context

3. **How to handle multi-threaded job handlers?**
   - Proposal: Thread-local context is inherited by spawned tasks

4. **Backward compatibility?**
   - Proposal: V2 API is parallel, can be adopted incrementally

---

## Next Steps

1. Prototype `NarrationContext::detect()`
2. Implement thread-local job context
3. Create `NarrationCapture` utility
4. Migrate one flow (e.g., hive → worker) to validate design
5. Full migration across all components

---

## Conclusion

Narration V2 redesigns the observability system to match the **actual architecture** of rbee:
- Context-aware routing (no manual mode selection)
- Optional job_id (only needed for SSE)
- Process boundary transparency (child stdout capture)
- Simpler API (less boilerplate)

This solves the fundamental mismatch between the narration system and the job-client/server architecture.
