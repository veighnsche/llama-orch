# Component Responsibilities - FINAL CLARIFICATION

**Date:** 2025-10-09T17:34:00+02:00  
**Updated:** 2025-10-10T14:02 (TEAM-037 - queen-rbee orchestration)  
**By:** User (Vince) + TEAM-024 + TEAM-037  
**Status:** NORMATIVE  
**Priority:** CRITICAL - Read before building anything!

---

## 🚨 CRITICAL: rbee-keeper is a TESTING TOOL 🚨

**rbee-keeper is NOT for production!**
- Testing: rbee-keeper spawns queen-rbee, runs test, kills everything
- Production: llama-orch SDK → queen-rbee directly

**WHENEVER queen-rbee DIES, ALL hives and workers DIE gracefully!**

---

## TL;DR - The 4 Binaries

| Binary | Crate | Type | Purpose | Production? | Status |
|--------|-------|------|---------|-------------|--------|
| **queen-rbee** | bin/queen-rbee | Daemon (HTTP) | THE BRAIN - orchestrates, controls hives via SSH | ✅ YES | M1 ❌ |
| **llm-worker-rbee** | bin/llm-worker-rbee | Daemon (HTTP) | WORKER - loads model, generates tokens | ✅ YES | M0 ✅ |
| **rbee-keeper** | bin/rbee-keeper | CLI | **TESTING TOOL** - integration tester | ❌ NO | M0 ✅ |
| **rbee-hive** | bin/rbee-hive | Daemon (HTTP) | POOL MANAGER - spawns workers, health monitoring | ✅ YES | M0 ✅ |

## Component Responsibilities

| Component | Type | Has Registry? | Has Catalog? | Stateful? |
|-----------|------|---------------|--------------|-----------|
| **queen-rbee** | Daemon | ✅ Worker Registry (global) | ❌ | ✅ YES (SQLite) |
| **llm-worker-rbee** | Daemon | ❌ | ❌ | ❌ NO (stateless) |
| **llorch** | CLI | ❌ | ❌ | ❌ NO |
| **rbee-hive** | CLI | ✅ Worker metadata (local) | ✅ Model + Backend Catalog | ❌ NO (filesystem) |

---

## rbee-hive (rbee-hive) - Local Pool CLI

### What It Needs (M0):

#### 1. Model Catalog ✅ DONE
**Location:** `.test-models/catalog.json`

**Schema:**
```json
{
  "models": [
    {
      "id": "qwen-0.5b",
      "name": "Qwen2.5 0.5B Instruct",
      "repo": "Qwen/Qwen2.5-0.5B-Instruct",
      "architecture": "qwen",
      "downloaded": true,
      "size_gb": 0.9,
      "path": ".test-models/qwen-0.5b"
    }
  ]
}
```

**Purpose:**
- Track which models are available
- Track download status
- Track model metadata

**Status:** ✅ IMPLEMENTED

---

#### 2. Backend Catalog (NEW - M0)
**Location:** `.runtime/backends.json` or detect dynamically

**Schema:**
```json
{
  "backends": [
    {
      "name": "cpu",
      "available": true,
      "priority": 3
    },
    {
      "name": "metal",
      "available": true,
      "priority": 1,
      "note": "Mac has Metal only (no CUDA)"
    },
    {
      "name": "cuda",
      "available": false,
      "reason": "No NVIDIA GPU detected"
    }
  ]
}
```

**Purpose:**
- Know which backends are available on this pool
- Prevent spawning workers on unavailable backends
- **IMPORTANT:** Mac has Metal only (accelerate framework disabled)
- All machines have CPU

**Detection Logic:**
```rust
// Detect available backends
fn detect_backends() -> Vec<Backend> {
    let mut backends = vec![Backend::Cpu]; // Always available
    
    #[cfg(target_os = "macos")]
    if has_metal() {
        backends.push(Backend::Metal);
    }
    
    #[cfg(not(target_os = "macos"))]
    if has_cuda() {
        backends.push(Backend::Cuda);
    }
    
    backends
}
```

**Status:** ❌ NOT IMPLEMENTED (M0 task)

---

#### 3. Worker Registry (LOCAL ONLY - M0)
**Location:** `.runtime/workers/` (PID files + JSON metadata)

**Purpose:**
- Track workers spawned by THIS rbee-hive
- Kill workers when requested
- **Kill orphaned workers** (M0 requirement!)

**Schema (per worker):**
```json
{
  "worker_id": "worker-metal-abc123",
  "pid": 12345,
  "model": "qwen-0.5b",
  "backend": "metal",
  "port": 8001,
  "spawned_at": "2025-10-09T17:00:00Z",
  "status": "running"
}
```

**Orphan Detection:**
```rust
// Find orphaned workers (process dead but file exists)
fn find_orphaned_workers() -> Vec<WorkerInfo> {
    let mut orphans = vec![];
    
    for worker_file in read_dir(".runtime/workers/")? {
        let worker = read_worker_info(&worker_file)?;
        
        // Check if process is still alive
        if !is_process_alive(worker.pid) {
            orphans.push(worker);
        }
    }
    
    orphans
}

// Kill orphaned workers
fn kill_orphaned_workers() -> Result<()> {
    let orphans = find_orphaned_workers()?;
    
    for orphan in orphans {
        println!("Found orphaned worker: {}", orphan.worker_id);
        
        // Try to kill process (may already be dead)
        let _ = kill_process(orphan.pid);
        
        // Remove metadata file
        remove_file(format!(".runtime/workers/{}.json", orphan.worker_id))?;
        
        println!("Cleaned up orphaned worker: {}", orphan.worker_id);
    }
    
    Ok(())
}
```

**Commands:**
```bash
# List workers (including orphaned)
rbee-hive worker list

# Kill orphaned workers
rbee-hive worker cleanup

# Kill all workers
rbee-hive worker stop-all
```

**Status:** ⚠️ PARTIAL (has spawn/stop, needs orphan cleanup)

---

#### 4. Robust Cancellation (M0)
**Problem:** Workers don't support cancellation!

**Solution:** Add cancellation to worker

**Worker API:**
```rust
// POST /cancel
{
  "job_id": "job-123"
}
```

**Worker Implementation:**
```rust
// In worker inference loop
async fn execute_inference(job_id: String, prompt: String) -> Result<()> {
    let cancel_flag = Arc::new(AtomicBool::new(false));
    
    // Register cancellation handler
    register_cancel_handler(job_id.clone(), cancel_flag.clone());
    
    // Generate tokens
    for token in generate_tokens(prompt) {
        // Check cancellation
        if cancel_flag.load(Ordering::Relaxed) {
            return Ok(()); // Stop generation
        }
        
        // Stream token
        stream_token(token).await?;
    }
    
    Ok(())
}

// Cancellation endpoint
async fn handle_cancel(job_id: String) -> Result<()> {
    if let Some(cancel_flag) = get_cancel_handler(&job_id) {
        cancel_flag.store(true, Ordering::Relaxed);
        Ok(())
    } else {
        Err(anyhow!("Job not found"))
    }
}
```

**Status:** ❌ NOT IMPLEMENTED (M0 task)

---

### rbee-hive Summary

**What rbee-hive HAS:**
- ✅ Model catalog (tracks models)
- ⏳ Backend catalog (needs implementation)
- ⏳ Worker registry (local, needs orphan cleanup)

**What rbee-hive DOES NOT HAVE:**
- ❌ Global worker registry (that's orchestrator's job)
- ❌ Stateful database (uses filesystem)
- ❌ HTTP server (it's a CLI)

**Why rbee-hive is stateless:**
- CLI runs on-demand, exits after command
- All state in filesystem (catalog.json, worker files)
- No long-running process

---

## queen-rbee - The Brain (HTTP Daemon)

### What It Needs (M1):

#### 1. Worker Registry (GLOBAL - M1)
**Location:** SQLite database (persistent)

**Schema:**
```sql
CREATE TABLE workers (
    worker_id TEXT PRIMARY KEY,
    pool_host TEXT NOT NULL,
    port INTEGER NOT NULL,
    model TEXT NOT NULL,
    backend TEXT NOT NULL,
    status TEXT NOT NULL, -- 'healthy', 'unhealthy', 'dead'
    last_health_check TIMESTAMP,
    registered_at TIMESTAMP NOT NULL,
    metadata JSON
);
```

**Purpose:**
- Track ALL workers across ALL pools
- Route inference requests to workers
- Health check workers periodically
- Remove dead workers

**Why SQLite:**
- Orchestrator is stateful (long-running daemon)
- Workers can crash
- Pool-ctl is short-lived (can't maintain registry)
- Need persistence across orchestrator restarts

**Health Checking:**
```rust
// Periodic health check (every 30s)
async fn health_check_workers() {
    let workers = db.get_all_workers().await?;
    
    for worker in workers {
        let health = check_worker_health(&worker).await;
        
        match health {
            Ok(_) => {
                db.update_worker_status(&worker.id, "healthy").await?;
            }
            Err(_) => {
                db.update_worker_status(&worker.id, "unhealthy").await?;
                
                // If unhealthy for >5 minutes, mark as dead
                if worker.unhealthy_duration() > Duration::from_secs(300) {
                    db.update_worker_status(&worker.id, "dead").await?;
                    
                    // Optionally: Call rbee-hive to cleanup orphan
                    cleanup_orphaned_worker(&worker).await?;
                }
            }
        }
    }
}

// Cleanup orphaned worker via rbee-hive
async fn cleanup_orphaned_worker(worker: &Worker) -> Result<()> {
    // SSH to pool and run cleanup
    ssh_exec(
        &worker.pool_host,
        "rbee-hive worker cleanup"
    ).await
}
```

**Status:** ❌ NOT IMPLEMENTED (M1 task)

---

#### 2. Rhai Scripting Engine (M1)
**Purpose:** User can script orchestration logic

**Why Rhai:**
- Embedded scripting language for Rust
- Safe (no unsafe operations)
- Fast
- Easy to integrate

**Use Cases:**
```rust
// User-defined scheduling script
fn schedule_job(job, workers) {
    // Custom logic
    if job.priority == "high" {
        return workers.filter(|w| w.backend == "cuda")[0];
    } else {
        return workers.least_loaded();
    }
}

// User-defined admission control
fn should_admit(job, queue_depth) {
    if queue_depth > 100 {
        return false; // Reject
    }
    if job.max_tokens > 2000 {
        return false; // Too expensive
    }
    return true;
}

// User-defined retry policy
fn should_retry(job, attempt, error) {
    if attempt > 3 {
        return false;
    }
    if error.kind == "timeout" {
        return true;
    }
    return false;
}
```

**Integration:**
```rust
use rhai::Engine;

struct Orchestrator {
    engine: Engine,
    // ... other fields
}

impl Orchestrator {
    fn new() -> Self {
        let mut engine = Engine::new();
        
        // Register custom types
        engine.register_type::<Job>();
        engine.register_type::<Worker>();
        
        // Register functions
        engine.register_fn("schedule_job", schedule_job);
        engine.register_fn("should_admit", should_admit);
        
        // Load user script
        engine.eval_file("orchestration.rhai")?;
        
        Self { engine }
    }
    
    async fn schedule(&self, job: Job) -> Result<Worker> {
        let workers = self.get_healthy_workers().await?;
        
        // Call user script
        let worker = self.engine.call_fn::<Worker>(
            "schedule_job",
            (job, workers)
        )?;
        
        Ok(worker)
    }
}
```

**Status:** ❌ NOT IMPLEMENTED (M1 task)

---

#### 3. Prompt Constructor (SHARED - M1)
**Purpose:** Format prompts for different chat templates

**Why Needed:**
- Different models use different chat formats
- Qwen uses `<|im_start|>user\n{prompt}<|im_end|>`
- Llama uses `[INST]{prompt}[/INST]`
- Mistral uses different format
- Need consistent interface

**Location:** `bin/shared-crates/prompt-constructor/`

**API:**
```rust
pub enum ChatTemplate {
    Qwen,
    Llama,
    Mistral,
    Phi,
    Raw, // No template
}

pub struct PromptConstructor {
    template: ChatTemplate,
}

impl PromptConstructor {
    pub fn new(template: ChatTemplate) -> Self {
        Self { template }
    }
    
    pub fn format_chat(&self, messages: &[ChatMessage]) -> String {
        match self.template {
            ChatTemplate::Qwen => self.format_qwen(messages),
            ChatTemplate::Llama => self.format_llama(messages),
            ChatTemplate::Mistral => self.format_mistral(messages),
            ChatTemplate::Phi => self.format_phi(messages),
            ChatTemplate::Raw => messages.last().unwrap().content.clone(),
        }
    }
    
    fn format_qwen(&self, messages: &[ChatMessage]) -> String {
        let mut result = String::new();
        
        for msg in messages {
            match msg.role {
                Role::System => {
                    result.push_str(&format!(
                        "<|im_start|>system\n{}<|im_end|>\n",
                        msg.content
                    ));
                }
                Role::User => {
                    result.push_str(&format!(
                        "<|im_start|>user\n{}<|im_end|>\n",
                        msg.content
                    ));
                }
                Role::Assistant => {
                    result.push_str(&format!(
                        "<|im_start|>assistant\n{}<|im_end|>\n",
                        msg.content
                    ));
                }
            }
        }
        
        // Add assistant start token
        result.push_str("<|im_start|>assistant\n");
        
        result
    }
    
    fn format_llama(&self, messages: &[ChatMessage]) -> String {
        // Llama format: [INST] {prompt} [/INST]
        let user_msg = messages.iter()
            .filter(|m| m.role == Role::User)
            .last()
            .unwrap();
        
        format!("[INST] {} [/INST]", user_msg.content)
    }
    
    // ... other formats
}

pub struct ChatMessage {
    pub role: Role,
    pub content: String,
}

pub enum Role {
    System,
    User,
    Assistant,
}
```

**Usage in queen-rbee:**
```rust
// When routing to worker
let constructor = PromptConstructor::new(ChatTemplate::Qwen);
let formatted_prompt = constructor.format_chat(&request.messages);

// Send to worker
worker.execute(formatted_prompt).await?;
```

**Usage in rbee-hive:**
```rust
// When testing locally
let constructor = PromptConstructor::new(ChatTemplate::Qwen);
let formatted_prompt = constructor.format_chat(&[
    ChatMessage {
        role: Role::User,
        content: "Hello".to_string(),
    }
]);

// Test worker
llorch_infer(formatted_prompt).await?;
```

**Status:** ❌ NOT IMPLEMENTED (M1 task)

**Note:** Both queen-rbee AND rbee-hive need this!

---

### queen-rbee Summary

**What queen-rbee HAS:**
- ✅ Worker Registry (global, SQLite)
- ✅ Rhai scripting (user-defined orchestration)
- ✅ Prompt constructor (shared with rbee-hive)
- ✅ HTTP server (client API)
- ✅ Queue management
- ✅ Scheduling
- ✅ SSE relay

**What queen-rbee DOES NOT HAVE:**
- ❌ Model catalog (that's rbee-hive's job)
- ❌ Worker spawning (that's rbee-hive's job)
- ❌ Model downloads (that's rbee-hive's job)

**Why queen-rbee is stateful:**
- Long-running daemon
- Maintains worker registry
- Maintains queue state
- Needs persistence (SQLite)

---

## llm-worker-rbee - Worker (HTTP Daemon)

### What It Needs (M0):

#### 1. Cancellation Support (M0 - CRITICAL!)
**Problem:** Workers don't support cancellation!

**API:**
```rust
// POST /cancel
{
  "job_id": "job-123"
}

// Response
{
  "status": "cancelled",
  "job_id": "job-123"
}
```

**Implementation:**
```rust
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::sync::RwLock;

struct Worker {
    // Map of job_id -> cancel flag
    cancel_flags: Arc<RwLock<HashMap<String, Arc<AtomicBool>>>>,
}

impl Worker {
    async fn execute(&self, job_id: String, prompt: String) -> Result<()> {
        // Create cancel flag
        let cancel_flag = Arc::new(AtomicBool::new(false));
        
        // Register it
        self.cancel_flags.write().await.insert(
            job_id.clone(),
            cancel_flag.clone()
        );
        
        // Generate tokens
        for token in self.generate_tokens(&prompt) {
            // Check cancellation EVERY token
            if cancel_flag.load(Ordering::Relaxed) {
                // Send cancellation event
                self.send_event(InferenceEvent::End {
                    stop_reason: "CANCELLED".to_string(),
                    tokens_out: current_token_count,
                }).await?;
                
                // Clean up
                self.cancel_flags.write().await.remove(&job_id);
                
                return Ok(());
            }
            
            // Stream token
            self.send_event(InferenceEvent::Token {
                t: token,
                i: current_token_count,
            }).await?;
        }
        
        // Clean up
        self.cancel_flags.write().await.remove(&job_id);
        
        Ok(())
    }
    
    async fn cancel(&self, job_id: String) -> Result<()> {
        let flags = self.cancel_flags.read().await;
        
        if let Some(flag) = flags.get(&job_id) {
            flag.store(true, Ordering::Relaxed);
            Ok(())
        } else {
            Err(anyhow!("Job not found or already completed"))
        }
    }
}

// HTTP endpoint
async fn handle_cancel(
    State(worker): State<Arc<Worker>>,
    Json(req): Json<CancelRequest>,
) -> Result<Json<CancelResponse>, StatusCode> {
    worker.cancel(req.job_id.clone()).await
        .map(|_| Json(CancelResponse {
            status: "cancelled".to_string(),
            job_id: req.job_id,
        }))
        .map_err(|_| StatusCode::NOT_FOUND)
}
```

**Status:** ❌ NOT IMPLEMENTED (M0 - HIGH PRIORITY!)

---

#### 2. What Worker DOES NOT HAVE
- ❌ Model catalog (stateless, loads ONE model)
- ❌ Worker registry (doesn't track other workers)
- ❌ Scheduling logic (dumb executor)
- ❌ Queue (accepts one request at a time)

**Why worker is stateless:**
- Loads ONE model
- Executes ONE request at a time
- No persistent state
- Can be killed and restarted anytime

---

## Critical Clarifications

### 1. Worker Registry Ownership

**WRONG:**
```
rbee-hive has worker registry → tracks all workers
```

**CORRECT:**
```
queen-rbee has worker registry → tracks all workers (global)
rbee-hive has worker metadata → tracks local workers only (for cleanup)
```

**Why:**
- queen-rbee is stateful (SQLite)
- queen-rbee needs to route requests
- rbee-hive is stateless (filesystem)
- rbee-hive only needs to kill local workers

---

### 2. Orchestrator vs CLI

**queen-rbee (THE BRAIN):**
- ✅ Makes intelligent decisions
- ✅ Schedules jobs
- ✅ Routes to workers
- ✅ User-scriptable (Rhai)
- ✅ Stateful (SQLite)

**rbee-keeper (PRECISE COMMANDS):**
- ✅ Executes specific commands
- ✅ No intelligence
- ✅ No scheduling
- ✅ No scripting
- ✅ Stateless (CLI)

**Example:**
```bash
# rbee-keeper: Precise command
llorch pool worker spawn metal --host mac --model qwen --gpu 0

# queen-rbee: Smart decision (via Rhai script)
# User writes script:
fn schedule_job(job, workers) {
    // Orchestrator decides which worker based on:
    // - Load
    // - Model availability
    // - Backend preference
    // - Custom logic
    return best_worker;
}
```

---

### 3. Orphaned Worker Cleanup Flow

**Problem:** Workers crash during development

**Solution:**
```
1. Worker crashes (process dies)
2. Metadata file remains in .runtime/workers/
3. queen-rbee health check fails
4. queen-rbee marks worker as "dead" in registry
5. queen-rbee calls rbee-hive to cleanup:
   ssh mac.home.arpa "rbee-hive worker cleanup"
6. rbee-hive finds orphaned workers (PID dead, file exists)
7. rbee-hive removes metadata files
8. queen-rbee removes worker from registry
```

**Commands:**
```bash
# Manual cleanup on pool
rbee-hive worker cleanup

# Orchestrator triggers cleanup
# (happens automatically via health checks)
```

---

## Implementation Priority (M0 Tasks)

### HIGH PRIORITY (M0 - Do First):
1. ✅ Model catalog (DONE)
2. ⏳ Backend catalog (detect available backends)
3. ⏳ Worker cancellation (POST /cancel endpoint)
4. ⏳ Orphaned worker cleanup (rbee-hive worker cleanup)

### MEDIUM PRIORITY (M1 - After CP4):
1. ⏳ queen-rbee HTTP server
2. ⏳ Worker registry (SQLite)
3. ⏳ Rhai scripting engine
4. ⏳ Prompt constructor (shared crate)

### LOW PRIORITY (M2 - Polish):
1. ⏳ Advanced scheduling
2. ⏳ Multi-tenant support
3. ⏳ Metrics & observability

---

## Summary Table

| Feature | rbee-hive | queen-rbee | worker |
|---------|----------|---------------|--------|
| Model Catalog | ✅ YES | ❌ NO | ❌ NO |
| Backend Catalog | ✅ YES | ❌ NO | ❌ NO |
| Worker Registry (local) | ✅ YES | ❌ NO | ❌ NO |
| Worker Registry (global) | ❌ NO | ✅ YES | ❌ NO |
| Orphan Cleanup | ✅ YES | ✅ YES (triggers) | ❌ NO |
| Cancellation | ❌ NO | ✅ YES (routes) | ✅ YES (executes) |
| Rhai Scripting | ❌ NO | ✅ YES | ❌ NO |
| Prompt Constructor | ✅ YES | ✅ YES | ❌ NO |
| Stateful | ❌ NO | ✅ YES | ❌ NO |
| Database | ❌ NO | ✅ SQLite | ❌ NO |
| HTTP Server | ❌ NO | ✅ YES | ✅ YES |

---

## Key Insights (User's Clarifications)

1. ✅ **Mac has Metal only** (accelerate disabled)
2. ✅ **All machines have CPU**
3. ✅ **Orphaned workers will happen a lot** (need cleanup)
4. ✅ **Workers don't have cancellation** (need to add)
5. ✅ **queen-rbee needs Rhai** (user-scriptable)
6. ✅ **queen-rbee is THE BRAIN** (makes decisions)
7. ✅ **rbee-keeper has precise commands** (no intelligence)
8. ✅ **Prompt constructor needed** (chat templates)
9. ✅ **Worker registry in queen-rbee ONLY** (not rbee-hive)
10. ✅ **queen-rbee is stateful** (SQLite for persistence)

---

**Signed:** User (Vince) + TEAM-024  
**Date:** 2025-10-09T17:34:00+02:00  
**Status:** NORMATIVE - Follow this for all implementation

