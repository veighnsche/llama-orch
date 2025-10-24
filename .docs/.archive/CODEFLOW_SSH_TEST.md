# Complete Codeflow Analysis: SSH Test Operation

**Command:** `./rbee hive ssh-test --ssh-host workstation.arpa.hoe --ssh-user vince`

**Date:** 2025-01-21  
**Status:** ✅ WORKING (SSH test executed successfully)

---

## Architecture Overview

```
User → ./rbee (xtask wrapper)
         ↓
    rbee-keeper (CLI client)
         ↓ HTTP POST /v1/jobs
    queen-rbee (orchestrator daemon)
         ↓
    job-router → hive-lifecycle
         ↓
    ssh-client (test connection)
         ↓ SSE stream
    rbee-keeper (display results)
```

---

## Phase 1: Command Entry (`./rbee`)

**File:** `xtask/src/main.rs:73`
- Delegates to `tasks::rbee::run_rbee_keeper(args)`

**File:** `xtask/src/tasks/rbee.rs`
1. Check if `target/debug/rbee-keeper` needs rebuild
2. Compare mtime vs source files in `bin/00_rbee_keeper/`
3. Build if stale: `cargo build --bin rbee-keeper`
4. Forward command: `target/debug/rbee-keeper hive ssh-test ...`

---

## Phase 2: rbee-keeper CLI Parsing

**File:** `bin/00_rbee_keeper/src/main.rs`

### Entry Point (lines 253-257)
```rust
#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    handle_command(cli).await
}
```

### Command Parsing (lines 360-362)
```rust
HiveAction::SshTest { ssh_host, ssh_port, ssh_user } => {
    Operation::SshTest { ssh_host, ssh_port, ssh_user }
}
```

**Parsed:**
- `ssh_host`: `"workstation.arpa.hoe"`
- `ssh_port`: `22` (default)
- `ssh_user`: `"vince"`

---

## Phase 3: Queen Lifecycle Management

**File:** `bin/00_rbee_keeper/src/queen_lifecycle.rs`

### 3.1 Ensure Queen Running (lines 100-106)
- 30-second timeout with visual countdown
- Checks health: `GET http://localhost:8500/health`

### 3.2 If Queen Not Running:
1. **Find binary** (lines 124-130)
   - Uses `daemon-lifecycle::DaemonManager::find_in_target("queen-rbee")`
   - Output: `Found binary at: target/debug/queen-rbee`

2. **Spawn process** (lines 132-141)
   ```rust
   let args = vec!["--port".to_string(), "8500".to_string()];
   let manager = DaemonManager::new(queen_binary, args);
   let mut _child = manager.spawn().await?;
   ```
   - Output: `Daemon spawned with PID: 215718`

3. **Poll health** (lines 143-154)
   - Exponential backoff: 100ms, 200ms, 400ms, 800ms, 1600ms, 3200ms
   - Output: `Queen health check succeeded after 360.527071ms`

---

## Phase 4: Job Submission

**File:** `bin/00_rbee_keeper/src/job_client.rs:34-46`

```rust
pub async fn submit_and_stream_job(
    client: &reqwest::Client,
    queen_url: &str,
    operation: Operation,
) -> Result<()> {
    let job_payload = serde_json::to_value(&operation)?;
    let queen_handle = ensure_queen_running(queen_url).await?;
    
    let res = client.post(format!("{}/v1/jobs", queen_url))
        .json(&job_payload)
        .send()
        .await?;
```

**HTTP Request:**
```http
POST http://localhost:8500/v1/jobs
Content-Type: application/json

{
  "operation": "ssh_test",
  "ssh_host": "workstation.arpa.hoe",
  "ssh_port": 22,
  "ssh_user": "vince"
}
```

**Response:**
```json
{
  "job_id": "job-7ba316ca-83f3-4748-befa-229e75661b92",
  "sse_url": "/v1/jobs/job-7ba316ca-83f3-4748-befa-229e75661b92/stream"
}
```

---

## Phase 5: Queen-rbee Job Processing

**File:** `bin/10_queen_rbee/src/http/jobs.rs:42-51`

```rust
pub async fn handle_create_job(
    State(state): State<SchedulerState>,
    Json(payload): Json<serde_json::Value>,
) -> Result<Json<JobResponse>, (StatusCode, String)> {
    crate::job_router::create_job(state.into(), payload)
        .await
        .map(Json)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
}
```

**File:** `bin/10_queen_rbee/src/job_router.rs:56-67`

```rust
pub async fn create_job(state: JobState, payload: serde_json::Value) -> Result<JobResponse> {
    let job_id = state.registry.create_job();
    let sse_url = format!("/v1/jobs/{}/stream", job_id);
    state.registry.set_payload(&job_id, payload);
    
    Ok(JobResponse { job_id, sse_url })
}
```

**Uses:** `job-registry` crate
- Generate UUID: `job-7ba316ca-...`
- Create job record with `JobState::Queued`
- Store payload for deferred execution

---

## Phase 6: SSE Stream Connection

**File:** `bin/00_rbee_keeper/src/job_client.rs:67-84`

```rust
let sse_full_url = format!("{}{}", queen_url, sse_url);
let response = client.get(&sse_full_url).send().await?;

Narration::new(ACTOR_RBEE_KEEPER, ACTION_JOB_STREAM, job_id)
    .operation(operation_name)
    .human("📡 Streaming results...")
    .emit();

let mut stream = response.bytes_stream();
```

**HTTP Request:**
```http
GET http://localhost:8500/v1/jobs/job-7ba316ca-83f3-4748-befa-229e75661b92/stream
Accept: text/event-stream
```

---

## Phase 7: Job Execution (Deferred)

**File:** `bin/10_queen_rbee/src/http/jobs.rs:58-69`

```rust
pub async fn handle_stream_job(
    Path(job_id): Path<String>,
    State(state): State<SchedulerState>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let token_stream = crate::job_router::execute_job(job_id, state.into()).await;
    let event_stream = token_stream.map(|data| Ok(Event::default().data(data)));
    Sse::new(event_stream)
}
```

**File:** `bin/99_shared_crates/job-registry/src/lib.rs:274-326`

Key pattern: `execute_and_stream`
1. Retrieve payload from registry
2. Spawn execution in background (`tokio::spawn`)
3. Return stream immediately (non-blocking)

---

## Phase 8: Operation Routing

**File:** `bin/10_queen_rbee/src/job_router.rs:89-129`

```rust
async fn route_operation(
    payload: serde_json::Value,
    registry: Arc<JobRegistry<String>>,
    hive_catalog: Arc<HiveCatalog>,
) -> Result<()> {
    let operation: Operation = serde_json::from_value(payload)?;
    
    match operation {
        Operation::SshTest { ssh_host, ssh_port, ssh_user } => {
            let request = SshTestRequest { ssh_host, ssh_port, ssh_user };
            let response = execute_ssh_test(request).await?;
            
            if !response.success {
                return Err(anyhow::anyhow!("SSH connection failed: {}", 
                    response.error.unwrap_or_else(|| "Unknown error".to_string())));
            }
            
            Narration::new(ACTOR_QUEEN_ROUTER, "ssh_test_complete", "success")
                .human(format!("✅ SSH test successful: {}", 
                    response.test_output.unwrap_or_default()))
                .emit();
        }
        // ... other operations ...
    }
    
    Ok(())
}
```

**Uses:** `rbee-operations` crate for type-safe contract

---

## Phase 9: SSH Test Execution

**File:** `bin/15_queen_rbee_crates/hive-lifecycle/src/lib.rs:237-276`

```rust
pub async fn execute_ssh_test(request: SshTestRequest) -> Result<SshTestResponse> {
    let target = format!("{}@{}:{}", request.ssh_user, request.ssh_host, request.ssh_port);
    
    let config = SshConfig {
        host: request.ssh_host,
        port: request.ssh_port,
        user: request.ssh_user,
        timeout_secs: 5,
    };
    
    let result = test_ssh_connection(config).await?;
    
    let response = SshTestResponse {
        success: result.success,
        error: result.error,
        test_output: result.test_output,
    };
    
    Ok(response)
}
```

**Uses:** `queen-rbee-ssh-client` crate
- Connect to `vince@workstation.arpa.hoe:22`
- Run test command: `echo test`
- Timeout: 5 seconds

---

## Phase 10: Stream Results Back

**File:** `bin/00_rbee_keeper/src/job_client.rs:86-115`

```rust
while let Some(chunk) = stream.next().await {
    let text = String::from_utf8_lossy(&chunk);
    
    for line in text.lines() {
        if let Some(data) = line.strip_prefix("data: ") {
            println!("{}", data);
            
            if data.contains("[DONE]") {
                Narration::new(ACTOR_RBEE_KEEPER, ACTION_JOB_COMPLETE, job_id)
                    .human("✅ Complete")
                    .emit();
                return Ok(());
            }
        }
    }
}
```

**SSE Output:**
```
data: [🐝 hive-lifecycle] 🔐 Testing SSH connection to vince@workstation.arpa.hoe:22
data: [🐝 hive-lifecycle] ✅ SSH test successful
data: [👑 queen-router] ✅ SSH test successful: test
data: [DONE]
```

---

## Key Architectural Patterns

### 1. Dual-Call Pattern (Job Registry)
- **POST** `/v1/jobs` → Create job, return `job_id` + `sse_url`
- **GET** `/v1/jobs/{job_id}/stream` → Execute job, stream results

### 2. Command Pattern (Operations)
- All operations in `rbee-operations` crate
- Type-safe serialization/deserialization
- Single source of truth for client ↔ server contract

### 3. Lifecycle Management (Daemon Lifecycle)
- Shared `daemon-lifecycle` crate for process spawning
- Used by: `rbee-keeper` → `queen-rbee` → `rbee-hive` → workers

### 4. Observability (Narration)
- Structured logging with Actor/Action/Target pattern
- SSE streaming for real-time feedback

### 5. Fire-and-Forget (Async Spawning)
- Spawn processes asynchronously
- Return immediately, don't wait for readiness
- Use heartbeats for health monitoring

---

## Shared Crates

### Core Infrastructure
1. **`daemon-lifecycle`** - Process spawning
2. **`job-registry`** - Job state and streaming
3. **`rbee-operations`** - Operation types
4. **`observability-narration-core`** - Structured logging

### Queen-rbee Specific
5. **`queen-rbee-hive-catalog`** - SQLite catalog
6. **`queen-rbee-hive-registry`** - RAM registry
7. **`queen-rbee-hive-lifecycle`** - Hive operations
8. **`queen-rbee-ssh-client`** - SSH operations

---

## Data Flow Summary

```
User Command
    ↓
xtask wrapper (build check)
    ↓
rbee-keeper (CLI client)
    ↓ POST /v1/jobs
queen-rbee (orchestrator)
    ↓ job-registry (store payload)
    ↓ GET /v1/jobs/{id}/stream
job-router (execute)
    ↓
hive-lifecycle (business logic)
    ↓
ssh-client (SSH test)
    ↓ SSE stream
rbee-keeper (display)
```

---

## Performance Characteristics

### Latency (from logs)
- Queen startup: ~360ms (cold start)
- Job creation: <10ms
- SSE connection: <50ms
- SSH test: ~100-500ms
- **Total (cold):** ~500-900ms
- **Total (warm):** ~150-550ms

### Resource Usage
- `rbee-keeper`: Ephemeral (exits after command)
- `queen-rbee`: Long-lived daemon (~10MB RAM)
- `job-registry`: In-memory HashMap

---

## Future Operations

Same codeflow applies to:

### Hive Operations
- `hive_install` - Install hive binary
- `hive_uninstall` - Stop and cleanup
- `hive_update` - Update config/capabilities
- `hive_start` - Spawn hive process
- `hive_stop` - Graceful shutdown
- `hive_list` - Query catalog
- `hive_get` - Get details

### Worker/Model/Inference (forwarded to hive)
- `worker_spawn/list/get/delete`
- `model_download/list/get/delete`
- `infer` - Run inference

**Pattern:** Queen orchestrates, hives execute.

---

## Team History

- **TEAM-151:** Migrated to numbered architecture
- **TEAM-154:** Created job-registry
- **TEAM-185:** Consolidated queen-lifecycle
- **TEAM-186:** Created rbee-operations
- **TEAM-187:** Added SshTest operation
- **TEAM-188:** Implemented SshTest

---

## Conclusion

Clean separation of concerns:
1. **CLI layer** - User interface, HTTP client
2. **Orchestration layer** - Job routing, hive management
3. **Business logic layer** - Hive operations, SSH
4. **Infrastructure layer** - Reusable components

Architecture supports type-safe operations, real-time streaming, and graceful error handling.
