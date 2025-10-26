# Narration V2: Detailed Implementation Plan

**Version:** 2.0.0-implementation  
**Date:** 2025-10-26  
**Status:** IMPLEMENTATION READY  
**Author:** TEAM-297

---

## Current Implementation Analysis

### What We Have Today

#### 1. Job Client/Server Split

**job-client** (`bin/99_shared_crates/job-client/src/lib.rs`):
```rust
pub struct JobClient {
    base_url: String,
    client: reqwest::Client,
}

impl JobClient {
    pub async fn submit_and_stream<F>(
        &self,
        operation: Operation,
        mut line_handler: F,  // ← Receives raw string lines
    ) -> Result<String>
}
```

**job-server** (`bin/99_shared_crates/job-server/src/lib.rs`):
```rust
pub struct JobRegistry<T> {
    jobs: Arc<Mutex<HashMap<String, Job<T>>>>,
}

pub struct Job<T> {
    pub job_id: String,
    pub state: JobState,
    pub token_receiver: Option<TokenReceiver<T>>,  // ← For inference tokens
    pub payload: Option<serde_json::Value>,
}
```

#### 2. Narration SSE Sink (SEPARATE from JobRegistry!)

**sse_sink** (`bin/99_shared_crates/narration-core/src/sse_sink.rs`):
```rust
static SSE_CHANNEL_REGISTRY: Lazy<SseChannelRegistry> = ...;

pub struct SseChannelRegistry {
    senders: Arc<Mutex<HashMap<String, mpsc::Sender<NarrationEvent>>>>,
    receivers: Arc<Mutex<HashMap<String, mpsc::Receiver<NarrationEvent>>>>,
}

// CRITICAL FUNCTIONS:
pub fn create_job_channel(job_id: String, capacity: usize)
pub fn take_job_receiver(job_id: &str) -> Option<mpsc::Receiver<NarrationEvent>>
pub fn send(fields: &NarrationFields)  // ← Requires fields.job_id!
```

#### 3. Thread-Local Context (EXISTS but underused!)

**context.rs** (`bin/99_shared_crates/narration-core/src/context.rs`):
```rust
tokio::task_local! {
    static NARRATION_CONTEXT: RefCell<NarrationContext>;
}

pub struct NarrationContext {
    pub job_id: Option<String>,
    pub correlation_id: Option<String>,
}

pub async fn with_narration_context<F>(ctx: NarrationContext, f: F) -> F::Output
```

#### 4. Job Router Pattern

**queen/job_router.rs** and **hive/job_router.rs**:
```rust
pub async fn create_job(state: JobState, payload: serde_json::Value) -> Result<JobResponse> {
    let job_id = state.registry.create_job();
    state.registry.set_payload(&job_id, payload);
    
    // ← CRITICAL: Must create channel BEFORE any narration!
    sse_sink::create_job_channel(job_id.clone(), 1000);
    
    NARRATE.action("job_create")
        .job_id(&job_id)  // ← REQUIRED or narration is dropped!
        .emit();
    
    Ok(JobResponse { job_id, sse_url })
}

pub async fn execute_job(job_id: String, state: JobState) -> impl Stream<Item = String> {
    job_server::execute_and_stream(job_id, registry, |job_id, payload| {
        route_operation(job_id, payload, ...)
    }).await
}
```

#### 5. SSE Endpoint

**http/jobs.rs**:
```rust
pub async fn handle_stream_job(
    Path(job_id): Path<String>,
    State(state): State<SchedulerState>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    // Take narration receiver (MPSC, can only be done once)
    let sse_rx_opt = sse_sink::take_job_receiver(&job_id);
    
    // Trigger execution (spawns in background)
    let _token_stream = job_router::execute_job(job_id.clone(), state).await;
    
    // Stream events from receiver
    let stream = async_stream::stream! {
        let Some(mut sse_rx) = sse_rx_opt else {
            yield Ok(Event::default().data("ERROR: Job channel not found"));
            return;
        };
        
        loop {
            match sse_rx.recv().await {
                Some(event) => yield Ok(Event::default().data(&event.formatted)),
                None => {
                    yield Ok(Event::default().data("[DONE]"));
                    break;
                }
            }
        }
        
        sse_sink::remove_job_channel(&job_id);
    };
    
    Sse::new(stream)
}
```

---

## The Core Problem

### Two Separate Channel Systems

**System 1: Narration Channels** (sse_sink)
- Purpose: Stream human-readable narration events
- Storage: `SSE_CHANNEL_REGISTRY` (global static)
- Creation: `create_job_channel(job_id, capacity)`
- Usage: `NARRATE.action().job_id().emit()` → `sse_sink::send()`

**System 2: Token Channels** (JobRegistry)
- Purpose: Stream inference tokens/results
- Storage: `JobRegistry<T>.token_receiver`
- Creation: `registry.set_token_receiver(job_id, rx)`
- Usage: Worker emits tokens → flows through receiver

**Both delivered through the SAME HTTP SSE endpoint!**

### The Fragility

```rust
// CURRENT: create_job_channel() MUST be called first
create_job_channel(job_id.clone(), 1000);  // ← Step 1: Create channel
NARRATE.action("start").job_id(&job_id).emit();  // ← Step 2: Narrate

// If you reverse the order:
NARRATE.action("start").job_id(&job_id).emit();  // ← DROPPED! (no channel)
create_job_channel(job_id.clone(), 1000);  // ← Too late!
```

### What About Worker Startup?

```rust
// Worker starts (NO job context yet!)
fn main() {
    NARRATE.action("startup").emit();  // ← NO job_id, goes to stdout
    
    load_model();  // During this, narration goes to stdout
    
    // HTTP server starts
    start_http_server();  // Now what? Still no job_id!
}
```

**The hive needs to capture worker's stdout and convert it to narration!**

---

## Solution: Dual-Mode Narration

### Core Concept

Narration operates in **TWO MODES** based on execution context:

1. **Standalone Mode** - No job, no SSE, just stdout/stderr
2. **Job Mode** - In job context, emit to both stdout AND SSE

**Key insight:** Don't try to force everything through job channels!

### Mode Detection

```rust
pub enum NarrationMode {
    /// Standalone: stdout only (no job context)
    Standalone,
    
    /// Job context: stdout + SSE (job_id known)
    Job { job_id: String },
}

impl NarrationMode {
    /// Auto-detect from thread-local context
    pub fn detect() -> Self {
        // Check thread-local first
        if let Some(ctx) = context::get_context() {
            if let Some(job_id) = ctx.job_id {
                return Self::Job { job_id };
            }
        }
        
        Self::Standalone
    }
}
```

### Narration Emitter (Refactored)

```rust
/// Emit narration (mode-aware)
pub fn narrate(fields: NarrationFields) {
    // ALWAYS emit to stdout (regardless of mode)
    emit_to_stdout(&fields);
    
    // Detect mode
    let mode = NarrationMode::detect();
    
    match mode {
        NarrationMode::Standalone => {
            // Just stdout (already done above)
        }
        
        NarrationMode::Job { job_id } => {
            // ALSO emit to SSE (if channel exists)
            // If channel doesn't exist, that's OK! We already got stdout.
            let _ = sse_sink::try_send(&job_id, &fields);
        }
    }
}

fn emit_to_stdout(fields: &NarrationFields) {
    eprintln!("[{:<10}] {:<15}: {}", 
        fields.actor, 
        fields.action, 
        fields.human
    );
}
```

### SSE Sink (Refactored - No Mandatory Channel Creation!)

```rust
/// Try to send to SSE channel (non-blocking, failure OK)
pub fn try_send(job_id: &str, fields: &NarrationFields) -> bool {
    let senders = SSE_CHANNEL_REGISTRY.senders.lock().unwrap();
    
    if let Some(tx) = senders.get(job_id) {
        let event = NarrationEvent::from(fields.clone());
        let _ = tx.try_send(event);
        return true;
    }
    
    false  // No channel, that's OK!
}

/// Create channel (optional now!)
pub fn create_job_channel(job_id: String, capacity: usize) {
    // Same implementation as before
    let (tx, rx) = mpsc::channel(capacity);
    SSE_CHANNEL_REGISTRY.senders.lock().unwrap().insert(job_id.clone(), tx);
    SSE_CHANNEL_REGISTRY.receivers.lock().unwrap().insert(job_id, rx);
}
```

**Key change:** `try_send()` instead of `send()` - failure is OK!

---

## Implementation Plan

### Phase 1: Make SSE Optional (Non-Breaking)

**Goal:** Narration works even if SSE channel doesn't exist

#### Step 1.1: Refactor `sse_sink::send()`

```rust
// OLD (current):
pub fn send(fields: &NarrationFields) {
    let Some(job_id) = &fields.job_id else {
        return;  // DROP if no job_id (fail-fast)
    };
    
    let event = NarrationEvent::from(fields.clone());
    SSE_CHANNEL_REGISTRY.send_to_job(job_id, event);
    // ↑ Drops if channel doesn't exist
}

// NEW:
pub fn send(fields: &NarrationFields) {
    // Try thread-local context first
    let job_id = fields.job_id.as_ref()
        .or_else(|| context::get_context().and_then(|ctx| ctx.job_id.as_ref()));
    
    if let Some(job_id) = job_id {
        let event = NarrationEvent::from(fields.clone());
        let _ = SSE_CHANNEL_REGISTRY.try_send_to_job(job_id, event);
        // ↑ Failure OK! Stdout already has it.
    }
}
```

#### Step 1.2: Update `SseChannelRegistry`

```rust
impl SseChannelRegistry {
    // NEW: Non-blocking send (failure OK)
    pub fn try_send_to_job(&self, job_id: &str, event: NarrationEvent) -> bool {
        let senders = self.senders.lock().unwrap();
        if let Some(tx) = senders.get(job_id) {
            let _ = tx.try_send(event);  // Drop if full/closed
            return true;
        }
        false  // No channel, that's OK!
    }
}
```

**Impact:** Narration no longer requires `create_job_channel()` to be called first!

#### Step 1.3: Add Mode Detection

```rust
// In lib.rs
pub enum NarrationMode {
    Standalone,
    Job { job_id: String },
}

impl NarrationMode {
    pub fn detect() -> Self {
        if let Some(ctx) = context::get_context() {
            if let Some(job_id) = ctx.job_id {
                return Self::Job { job_id };
            }
        }
        Self::Standalone
    }
}
```

**Testing:**
```rust
#[test]
fn test_narration_without_channel() {
    // NO create_job_channel() call!
    
    // This should NOT panic or fail
    NARRATE.action("test")
        .job_id("nonexistent-job")
        .human("This goes to stdout only")
        .emit();
    
    // Success! (before, this would fail-fast)
}
```

---

### Phase 2: Use Thread-Local Context Consistently

**Goal:** Eliminate manual `.job_id()` calls everywhere

#### Step 2.1: Update Job Router to Set Context

```rust
// In job_router.rs
async fn route_operation(
    job_id: String,
    payload: serde_json::Value,
    state: JobState,
) -> Result<()> {
    // Set thread-local context
    let ctx = NarrationContext::new().with_job_id(&job_id);
    
    with_narration_context(ctx, async move {
        // Parse operation
        let operation: Operation = serde_json::from_value(payload)?;
        
        // NOW narration automatically includes job_id!
        NARRATE.action("route_job")
            .context(operation.name())
            .human("Executing operation: {}")
            .emit();  // ← NO .job_id() needed!
        
        // Dispatch to handler
        match operation {
            Operation::HiveStart { alias } => {
                // ALL narration in this handler automatically has job_id!
                execute_hive_start(alias).await
            }
            // ...
        }
    }).await
}
```

#### Step 2.2: Remove Manual `.job_id()` Calls

```rust
// BEFORE (current):
NARRATE.action("hive_start")
    .job_id(&job_id)  // ← Manual, error-prone
    .human("Starting hive")
    .emit();

// AFTER:
NARRATE.action("hive_start")
    .human("Starting hive")
    .emit();  // ← job_id auto-injected from context!
```

**Impact:** Less boilerplate, no forgotten `job_id` calls!

---

### Phase 3: Add Process Stdout Capture

**Goal:** Capture child process stdout and convert to narration

#### Step 3.1: Create `ProcessNarrationCapture`

```rust
// NEW FILE: bin/99_shared_crates/narration-core/src/process_capture.rs

use tokio::process::{Command, ChildStdout};
use tokio::io::{AsyncBufReadExt, BufReader};
use regex::Regex;

pub struct ProcessNarrationCapture {
    job_id: Option<String>,
}

impl ProcessNarrationCapture {
    pub fn new(job_id: Option<String>) -> Self {
        Self { job_id }
    }
    
    /// Spawn command with stdout capture
    pub async fn spawn(&self, mut command: Command) -> Result<tokio::process::Child> {
        command.stdout(std::process::Stdio::piped());
        command.stderr(std::process::Stdio::piped());
        
        let mut child = command.spawn()?;
        
        // Capture stdout
        if let Some(stdout) = child.stdout.take() {
            let job_id = self.job_id.clone();
            tokio::spawn(async move {
                Self::stream_and_parse(stdout, job_id).await;
            });
        }
        
        // Capture stderr
        if let Some(stderr) = child.stderr.take() {
            let job_id = self.job_id.clone();
            tokio::spawn(async move {
                Self::stream_and_parse(stderr, job_id).await;
            });
        }
        
        Ok(child)
    }
    
    /// Stream and parse child's output
    async fn stream_and_parse(output: ChildStdout, job_id: Option<String>) {
        let reader = BufReader::new(output);
        let mut lines = reader.lines();
        
        while let Ok(Some(line)) = lines.next_line().await {
            // Try to parse as narration event
            if let Some(event) = Self::parse_narration(&line) {
                if let Some(ref jid) = job_id {
                    // Re-emit with job_id for SSE routing
                    NARRATE.action(event.action.as_str())
                        .job_id(jid)
                        .human(&event.message)
                        .emit();
                } else {
                    // No job context, just print
                    eprintln!("{}", line);
                }
            } else {
                // Not narration, just print
                eprintln!("{}", line);
            }
        }
    }
    
    /// Parse narration event from stdout line
    fn parse_narration(line: &str) -> Option<ParsedNarrationEvent> {
        // Match format: "[actor     ] action         : message"
        let re = Regex::new(r"\[(.{10})\] (.{15}): (.+)").ok()?;
        let caps = re.captures(line)?;
        
        Some(ParsedNarrationEvent {
            actor: caps[1].trim().to_string(),
            action: caps[2].trim().to_string(),
            message: caps[3].to_string(),
        })
    }
}

struct ParsedNarrationEvent {
    actor: String,
    action: String,
    message: String,
}
```

#### Step 3.2: Use in Hive for Worker Spawning

```rust
// In hive/worker_lifecycle.rs
pub async fn spawn_worker(
    job_id: &str,
    worker_binary: &str,
    model_path: &str,
    device: &str,
) -> Result<()> {
    NARRATE.action("worker_spawn")
        .job_id(job_id)
        .human("🚀 Spawning worker...")
        .emit();
    
    // Create capture helper
    let capture = ProcessNarrationCapture::new(Some(job_id.to_string()));
    
    // Spawn with stdout capture
    let mut command = Command::new(worker_binary);
    command.arg("--model").arg(model_path);
    command.arg("--device").arg(device);
    // ... more args
    
    let child = capture.spawn(command).await?;
    
    // Worker's stdout is automatically:
    // 1. Captured by parent
    // 2. Parsed for narration events
    // 3. Re-emitted with job_id for SSE
    
    // Wait for health check
    tokio::time::sleep(Duration::from_secs(2)).await;
    
    NARRATE.action("worker_ready")
        .job_id(job_id)
        .human("✅ Worker ready")
        .emit();
    
    Ok(())
}
```

**Impact:** Worker startup narration now flows through SSE!

---

### Phase 4: Keeper Lifecycle Management

**Goal:** Keeper can spawn queen/hive and display their stdout

#### Step 4.1: Keeper Spawns Queen (Localhost)

```rust
// In rbee-keeper/src/handlers/queen.rs
pub async fn start_queen() -> Result<()> {
    NARRATE.action("queen_start")
        .human("Starting queen-rbee...")
        .emit();  // → stdout only (no job)
    
    // Spawn with stdout capture
    let mut command = Command::new("queen-rbee");
    command.arg("--port").arg("7833");
    command.stdout(Stdio::piped());
    
    let mut child = command.spawn()?;
    
    // Stream queen's stdout to terminal
    if let Some(stdout) = child.stdout.take() {
        tokio::spawn(async move {
            let reader = BufReader::new(stdout);
            let mut lines = reader.lines();
            
            while let Ok(Some(line)) = lines.next_line().await {
                println!("{}", line);  // Show to user
            }
        });
    }
    
    // Wait for health check
    wait_for_health("http://localhost:7833").await?;
    
    NARRATE.action("queen_ready")
        .human("✅ Queen ready")
        .emit();
    
    Ok(())
}
```

#### Step 4.2: Keeper Spawns Hive (SSH)

```rust
// In rbee-keeper/src/handlers/hive.rs
pub async fn start_hive(hive_id: &str) -> Result<()> {
    let hive_config = get_hive_config(hive_id)?;
    
    if hive_config.is_localhost() {
        // Spawn locally (same as queen)
        start_hive_local(&hive_config).await
    } else {
        // Spawn via SSH
        start_hive_ssh(&hive_config).await
    }
}

async fn start_hive_ssh(config: &HiveConfig) -> Result<()> {
    NARRATE.action("hive_start_ssh")
        .context(&config.host)
        .human("Starting hive on {}...")
        .emit();
    
    // SSH command
    let mut command = Command::new("ssh");
    command.arg(format!("{}@{}", config.user, config.host));
    command.arg("rbee-hive");
    command.arg("--port").arg(config.port.to_string());
    command.stdout(Stdio::piped());
    
    let mut child = command.spawn()?;
    
    // Stream SSH output to terminal
    if let Some(stdout) = child.stdout.take() {
        tokio::spawn(async move {
            let reader = BufReader::new(stdout);
            let mut lines = reader.lines();
            
            while let Ok(Some(line)) = lines.next_line().await {
                println!("{}", line);  // Show to user
            }
        });
    }
    
    Ok(())
}
```

---

## Migration Strategy

### Week 1: Phase 1 (SSE Optional)
- [ ] Refactor `sse_sink::send()` to use `try_send()`
- [ ] Add `NarrationMode::detect()`
- [ ] Update tests to work without `create_job_channel()`
- [ ] Verify no regressions

### Week 2: Phase 2 (Thread-Local Context)
- [ ] Update job routers to use `with_narration_context()`
- [ ] Remove manual `.job_id()` calls (100+ locations)
- [ ] Add context-aware tests
- [ ] Verify SSE routing still works

### Week 3: Phase 3 (Process Capture)
- [ ] Implement `ProcessNarrationCapture`
- [ ] Update hive worker spawning
- [ ] Test worker startup narration through SSE
- [ ] Document pattern

### Week 4: Phase 4 (Keeper Lifecycle)
- [ ] Update keeper queen lifecycle
- [ ] Update keeper hive lifecycle
- [ ] Test SSH stdout capture
- [ ] End-to-end testing

---

## Benefits Summary

### ✅ No More Fragile Channel Creation
```rust
// BEFORE: Must create channel first
create_job_channel(job_id.clone(), 1000);  // ← Forget this = broken!
NARRATE.action("start").job_id(&job_id).emit();

// AFTER: Just narrate
NARRATE.action("start").emit();  // ← Works regardless!
```

### ✅ Less Boilerplate
```rust
// BEFORE: Manual job_id everywhere
NARRATE.action("hive_start").job_id(&job_id).human("Starting").emit();
NARRATE.action("hive_check").job_id(&job_id).human("Checking").emit();
NARRATE.action("hive_ready").job_id(&job_id).human("Ready").emit();

// AFTER: Context auto-injects
with_narration_context(ctx.with_job_id(job_id), async {
    NARRATE.action("hive_start").human("Starting").emit();
    NARRATE.action("hive_check").human("Checking").emit();
    NARRATE.action("hive_ready").human("Ready").emit();
}).await
```

### ✅ Worker Startup Works
```rust
// BEFORE: Worker startup narration lost (no job context)
fn main() {
    NARRATE.action("startup").emit();  // → stdout only, hive can't see it
}

// AFTER: Hive captures and converts
let capture = ProcessNarrationCapture::new(Some(job_id));
let child = capture.spawn(command).await?;
// → Worker's stdout parsed and emitted with job_id for SSE!
```

### ✅ Keeper Lifecycle Works
```rust
// BEFORE: Queen/hive startup narration invisible
// AFTER: Keeper captures stdout and displays to user
let child = spawn_with_stdout_capture(command).await?;
// → User sees queen/hive startup in real-time
```

---

## Backward Compatibility

### Phase 1-2: Fully Compatible
- `create_job_channel()` still works (just not required)
- Manual `.job_id()` still works (just redundant with context)
- Existing code continues working

### Phase 3-4: Additive Only
- New `ProcessNarrationCapture` is optional
- Existing spawn code unchanged
- New code can use capture helper

### Migration Path
1. Deploy Phase 1-2 (no code changes required!)
2. Gradually adopt thread-local context
3. Add process capture where needed
4. No breaking changes at any step

---

## Testing Strategy

### Unit Tests
- `test_narration_without_channel()` - narration works without SSE
- `test_thread_local_context()` - context auto-injection
- `test_process_capture_parsing()` - parse narration from stdout

### Integration Tests
- `test_job_flow_with_context()` - full job with context
- `test_worker_startup_capture()` - hive captures worker narration
- `test_keeper_queen_stdout()` - keeper displays queen startup

### End-to-End Tests
- CLI job through job-server
- Worker spawn with stdout capture
- Keeper lifecycle operations

---

## Open Questions

1. **Should we deprecate `create_job_channel()`?**
   - Proposal: Keep it but make it optional (best of both worlds)

2. **How to handle stderr vs stdout?**
   - Proposal: Both captured, narration on stdout, logs on stderr

3. **Should we add structured logging in addition to narration?**
   - Proposal: Keep separate - narration for users, logs for operators

4. **Migration timeline?**
   - Proposal: 4 weeks, non-breaking, gradual adoption

---

## Next Steps

1. **Review this plan** - Confirm architecture aligns with vision
2. **Prototype Phase 1** - Prove SSE optional approach works
3. **Update spec** - Incorporate findings into V2 spec
4. **Begin implementation** - Start with Phase 1 (lowest risk)

---

## Conclusion

The solution is **NOT** a separate side-channel for narration. The current architecture is sound:
- Narration and tokens use separate MPSC channels
- Both delivered through the same SSE endpoint
- The problem is timing and context, not the channel architecture

**The fix:**
1. Make SSE delivery optional (stdout is primary)
2. Use thread-local context consistently
3. Add process stdout capture for child processes
4. Separate job narration from process narration

This maintains the current architecture while fixing the fragility and making it work across all scenarios (job context, process startup, CLI operations, etc.).
