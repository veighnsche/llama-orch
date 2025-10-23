# rbee Architecture Overview - Part 4: Data Flow & Protocols

**Version:** 1.0.0  
**Date:** October 23, 2025  
**Status:** Living Document

---

## Request Flow Patterns

### Pattern 1: Hive Lifecycle (queen-handled)

**Operations:** HiveInstall, HiveStart, HiveStop, HiveStatus, HiveUninstall

**Flow:**
```
rbee-keeper → queen-rbee → hive-lifecycle crate
                         (no HTTP forwarding)
```

**Detailed Steps:**

1. **User Command:**
   ```bash
   $ rbee-keeper hive start localhost
   ```

2. **rbee-keeper submits operation:**
   ```
   POST http://localhost:8500/v1/jobs
   Body: {
     "operation": {
       "HiveStart": {
         "alias": "localhost"
       }
     }
   }
   Response: { "job_id": "uuid-123" }
   ```

3. **rbee-keeper connects to SSE:**
   ```
   GET http://localhost:8500/v1/jobs/uuid-123/stream
   ```

4. **queen-rbee receives job:**
   ```rust
   // POST /v1/jobs handler
   let job_id = Uuid::new_v4().to_string();
   let receiver = registry.create_job(job_id.clone());
   
   // Spawn operation handler
   tokio::spawn(async move {
       route_operation(operation, job_id, state).await
   });
   
   Ok(Json(CreateJobResponse { job_id }))
   ```

5. **queen-rbee routes operation:**
   ```rust
   // job_router.rs
   match operation {
       Operation::HiveStart { alias } => {
           execute_hive_start(
               HiveStartRequest { alias, job_id },
               config
           ).await
       }
   }
   ```

6. **hive-lifecycle executes:**
   ```rust
   // hive-lifecycle/src/start.rs
   pub async fn execute_hive_start(
       request: HiveStartRequest,
       config: Arc<RbeeConfig>,
   ) -> Result<()> {
       // 1. Get hive config
       let hive = config.get_hive(&request.alias)?;
       
       // 2. Check if already running
       NARRATE.action("hive_health_check")
           .job_id(&request.job_id)
           .emit();
       
       // 3. Start hive process
       NARRATE.action("hive_start")
           .job_id(&request.job_id)
           .human("🚀 Starting hive...")
           .emit();
       
       // 4. Wait for health check
       // ... SSH or local process management
       
       // 5. Success
       NARRATE.action("hive_start")
           .job_id(&request.job_id)
           .human("✅ Hive started")
           .emit();
       
       Ok(())
   }
   ```

7. **SSE events flow to client:**
   ```
   data: {"event": "hive_health_check", "message": "Checking hive health..."}
   data: {"event": "hive_start", "message": "🚀 Starting hive..."}
   data: {"event": "hive_start", "message": "✅ Hive started"}
   data: [DONE]
   ```

8. **rbee-keeper displays output:**
   ```
   Checking hive health...
   🚀 Starting hive...
   ✅ Hive started
   [DONE]
   ```

**Key Insight:** Queen executes directly, no HTTP forwarding.

---

### Pattern 2: Worker/Model Operations (hive-forwarded)

**Operations:** WorkerSpawn, WorkerList, ModelDownload, ModelList

**Flow:**
```
rbee-keeper → queen-rbee → rbee-hive (HTTP)
              (routes)      (executes)
```

**Detailed Steps:**

1. **User Command:**
   ```bash
   $ rbee-keeper worker spawn --model llama-3-8b --device GPU-0
   ```

2. **rbee-keeper submits:**
   ```
   POST http://localhost:8500/v1/jobs
   Body: {
     "operation": {
       "WorkerSpawn": {
         "hive_id": "localhost",
         "model": "llama-3-8b",
         "device": "GPU-0"
       }
     }
   }
   ```

3. **queen-rbee routes:**
   ```rust
   // job_router.rs
   match operation {
       op if op.should_forward_to_hive() => {
           hive_forwarder::forward_to_hive(&job_id, op, config).await
       }
   }
   ```

4. **hive_forwarder submits to hive:**
   ```rust
   // hive_forwarder.rs
   pub async fn forward_to_hive(
       job_id: &str,
       operation: Operation,
       config: Arc<RbeeConfig>,
   ) -> Result<()> {
       let hive_url = "http://localhost:9000";
       let client = JobClient::new(hive_url);
       
       client.submit_and_stream(operation, |line| {
           // Forward SSE events to queen's registry
           NARRATE.action("hive_forward")
               .job_id(job_id)
               .context(&line)
               .emit();
           Ok(())
       }).await
   }
   ```

5. **rbee-hive receives:**
   ```
   POST http://localhost:9000/v1/jobs
   Body: { "operation": { "WorkerSpawn": { ... } } }
   ```

6. **rbee-hive routes:**
   ```rust
   // hive/job_router.rs
   match operation {
       Operation::WorkerSpawn { .. } => {
           // TODO: execute_worker_spawn().await
           Err(anyhow!("Not yet implemented"))
       }
   }
   ```

7. **SSE chain:**
   ```
   rbee-hive → queen-rbee → rbee-keeper
   (emits)     (forwards)   (displays)
   ```

**Key Insight:** Two HTTP hops, but SSE events flow through.

#### SSE Routing in Integrated Mode (local-hive)

**Question:** When queen calls hive crates directly (no HTTP), will narration work?

**Answer:** ✅ **YES!** Here's why:

```rust
// Integrated mode flow
pub async fn forward_to_hive(...) -> Result<()> {
    #[cfg(feature = "local-hive")]
    {
        if is_localhost_hive(hive_id, &config) {
            // Direct Rust call to hive crate
            return forward_via_local_hive(job_id, operation).await;
        }
    }
    // ... HTTP path
}

#[cfg(feature = "local-hive")]
async fn forward_via_local_hive(
    job_id: &str,  // ← Queen's job_id passed to hive crate
    operation: Operation,
) -> Result<()> {
    match operation {
        Operation::WorkerSpawn { model, device, .. } => {
            // Call hive crate with queen's job_id
            rbee_hive_worker_lifecycle::spawn_worker(
                job_id,  // ← CRITICAL: Queen's job_id
                &model,
                &device,
            ).await?;
        }
    }
    Ok(())
}

// In hive crate:
pub async fn spawn_worker(
    job_id: &str,  // ← Receives queen's job_id
    model: &str,
    device: &str,
) -> Result<()> {
    // Emit narration with queen's job_id
    NARRATE
        .action("worker_spawn")
        .job_id(job_id)  // ← Uses queen's job_id!
        .emit();
    
    // Narration goes to SSE_BROADCASTER (global static)
    // In integrated mode: same process = same SSE_BROADCASTER
    // Channel exists! Events flow to queen's SSE stream ✅
}
```

**Why It Works:**

1. **SSE_BROADCASTER is per-process:** Each process has one global instance
2. **Integrated mode = one process:** Queen and hive crates share SSE_BROADCASTER
3. **Queen creates job channel:** Channel exists in (single) SSE_BROADCASTER
4. **Hive crates use queen's job_id:** Passed as parameter, not generated
5. **Narration flows correctly:** Same process = same SSE_BROADCASTER = works!

**See:** `.arch/SSE_ROUTING_ANALYSIS.md` for detailed analysis.

---

### Pattern 3: Inference (direct to worker)

**Operations:** Infer

**Flow:**
```
rbee-keeper → queen-rbee → llm-worker-rbee (direct HTTP)
              (schedules)   (executes)
```

**Detailed Steps:**

1. **User Command:**
   ```bash
   $ rbee-keeper infer --prompt "Hello" --model llama-3-8b
   ```

2. **rbee-keeper submits:**
   ```
   POST http://localhost:8500/v1/jobs
   Body: {
     "operation": {
       "Infer": {
         "prompt": "Hello",
         "model": "llama-3-8b",
         "max_tokens": 100
       }
     }
   }
   ```

3. **queen-rbee schedules:**
   ```rust
   // job_router.rs
   Operation::Infer { prompt, model, .. } => {
       // TODO: IMPLEMENT INFERENCE SCHEDULING
       // 1. Query hive for available workers
       // 2. Select worker (load balancing)
       // 3. Direct HTTP POST to worker
       // 4. Stream tokens back to client
       Err(anyhow!("Not yet implemented"))
   }
   ```

4. **queen-rbee → worker (direct HTTP):**
   ```rust
   // Future implementation
   let worker_url = select_worker(model).await?;
   
   let response = reqwest::Client::new()
       .post(format!("{}/v1/inference", worker_url))
       .json(&InferenceRequest {
           prompt,
           max_tokens,
           temperature,
       })
       .send()
       .await?;
   
   // Stream tokens via SSE
   let stream = response.bytes_stream();
   // ... relay to client
   ```

5. **llm-worker-rbee executes:**
   ```rust
   // POST /v1/inference
   pub async fn handle_inference(
       State(state): State<WorkerState>,
       Json(request): Json<InferenceRequest>,
   ) -> Result<Sse<impl Stream>> {
       let tokens = state.model.generate(&request.prompt)?;
       
       let stream = tokens.map(|token| {
           Event::default().data(token)
       });
       
       Ok(Sse::new(stream))
   }
   ```

**Key Insight:** Queen circumvents hive for performance!

---

## SSE Streaming Protocol

### Dual-Call Pattern

**All operations use the same pattern:**

1. **POST /v1/jobs** - Submit job, get job_id
2. **GET /v1/jobs/{job_id}/stream** - Connect to SSE stream

### SSE Event Format

```
data: {"event": "action_name", "message": "Human-readable message"}
data: {"event": "action_name", "message": "Another event"}
data: [DONE]
```

### SSE Lifecycle

```
Client connects → GET /v1/jobs/{job_id}/stream
                  ↓
Server creates receiver from registry
                  ↓
Operation handler sends events via registry
                  ↓
Events flow to client via SSE
                  ↓
Operation completes → [DONE] sent
                  ↓
Client disconnects
```

### Code Implementation

**Server-Side (job-server):**
```rust
pub struct JobRegistry<T> {
    jobs: Arc<RwLock<HashMap<String, Job<T>>>>,
}

pub struct Job<T> {
    pub id: String,
    pub channel: mpsc::Sender<T>,
}

impl<T> JobRegistry<T> {
    pub fn send_event(&self, job_id: &str, event: T) -> Result<()> {
        let jobs = self.jobs.read().unwrap();
        let job = jobs.get(job_id).ok_or(anyhow!("Job not found"))?;
        job.channel.send(event)?;
        Ok(())
    }
}
```

**Narration Integration:**
```rust
// SSE sink (inside narration-core)
pub struct SseSink {
    registry: Arc<JobRegistry<String>>,
}

impl SseSink {
    pub fn send(&self, job_id: &str, message: &str) {
        self.registry.send_event(job_id, message.to_string()).ok();
    }
}

// Narration emit
impl Narration {
    pub fn emit(self) {
        if let Some(job_id) = &self.job_id {
            SSE_SINK.send(job_id, &self.message);
        } else {
            eprintln!("{}", self.message);
        }
    }
}
```

**Client-Side (job-client):**
```rust
pub async fn submit_and_stream<F>(
    &self,
    operation: Operation,
    line_handler: F,
) -> Result<String>
where
    F: Fn(String) -> Result<()>
{
    // 1. Submit job
    let response: CreateJobResponse = self.client
        .post(format!("{}/v1/jobs", self.base_url))
        .json(&CreateJobRequest { operation })
        .send()
        .await?
        .json()
        .await?;
    
    // 2. Connect to SSE
    let mut stream = self.client
        .get(format!("{}/v1/jobs/{}/stream", self.base_url, response.job_id))
        .send()
        .await?
        .bytes_stream();
    
    // 3. Process events
    while let Some(chunk) = stream.next().await {
        let data = chunk?;
        let line = String::from_utf8_lossy(&data);
        
        // Strip SSE prefix
        let line = line.strip_prefix("data: ").unwrap_or(&line);
        
        // Check for done
        if line == "[DONE]" {
            break;
        }
        
        // Handle event
        line_handler(line.to_string())?;
    }
    
    Ok(response.job_id)
}
```

### SSE Error Handling

**Connection Lost:**
```rust
// Client should retry with exponential backoff
let mut retries = 0;
loop {
    match connect_sse().await {
        Ok(stream) => break stream,
        Err(e) if retries < 3 => {
            retries += 1;
            tokio::time::sleep(Duration::from_secs(2_u64.pow(retries))).await;
        }
        Err(e) => return Err(e),
    }
}
```

**Job Not Found:**
```rust
// Server returns 404
if !registry.has_job(&job_id) {
    return Err((StatusCode::NOT_FOUND, "Job not found"));
}
```

---

## Heartbeat Architecture (TEAM-261)

### Old Architecture (REMOVED)

```
Worker → Hive (aggregation) → Queen
         POST /v1/heartbeat    POST /v1/heartbeat
```

**Problems:**
- Complex (aggregation logic)
- Distributed state
- Extra hop
- No direct worker visibility

### New Architecture (TEAM-261)

```
Worker → Queen (direct)
         POST /v1/worker-heartbeat
```

**Benefits:**
- Simple (no aggregation)
- Single source of truth (queen)
- Direct communication
- Queen knows all workers

### Worker Heartbeat Implementation

**Worker Side:**
```rust
// bin/30_llm_worker_rbee/src/heartbeat.rs
pub async fn send_heartbeat_to_queen(
    worker_id: &str,
    queen_url: &str,
    health_status: HealthStatus,
) -> Result<()> {
    let payload = WorkerHeartbeatPayload {
        worker_id: worker_id.to_string(),
        timestamp: chrono::Utc::now().to_rfc3339(),
        health_status,
    };
    
    let client = reqwest::Client::new();
    client
        .post(format!("{}/v1/worker-heartbeat", queen_url))
        .json(&payload)
        .send()
        .await?;
    
    Ok(())
}

pub fn start_heartbeat_task(
    worker_id: String,
    queen_url: String,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(30));
        
        loop {
            interval.tick().await;
            
            if let Err(e) = send_heartbeat_to_queen(
                &worker_id,
                &queen_url,
                HealthStatus::Healthy,
            ).await {
                eprintln!("Failed to send heartbeat: {}", e);
            }
        }
    })
}
```

**Queen Side:**
```rust
// bin/10_queen_rbee/src/http/heartbeat.rs
pub async fn handle_worker_heartbeat(
    State(state): State<HeartbeatState>,
    Json(payload): Json<WorkerHeartbeatPayload>,
) -> Result<Json<HttpHeartbeatAcknowledgement>> {
    eprintln!(
        "💓 Worker heartbeat: worker_id={}, timestamp={}, health_status={:?}",
        payload.worker_id, payload.timestamp, payload.health_status
    );
    
    // TODO: Update worker registry
    // state.worker_registry.update_worker_state(&payload.worker_id, payload);
    
    Ok(Json(HttpHeartbeatAcknowledgement {
        status: "ok".to_string(),
        message: format!("Heartbeat received from worker {}", payload.worker_id),
    }))
}
```

### Heartbeat Payload

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerHeartbeatPayload {
    pub worker_id: String,
    pub timestamp: String,  // ISO 8601
    pub health_status: HealthStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HealthStatus {
    Healthy,
    Degraded,
}
```

### Heartbeat Interval

- **Frequency:** 30 seconds
- **Timeout:** 90 seconds (3 missed heartbeats)
- **Action:** Mark worker as unavailable

### Worker Registry (TODO)

```rust
pub struct WorkerRegistry {
    workers: Arc<RwLock<HashMap<String, WorkerInfo>>>,
}

pub struct WorkerInfo {
    pub worker_id: String,
    pub last_heartbeat: Instant,
    pub health_status: HealthStatus,
    pub model: String,
    pub device: String,
    pub url: String,
}

impl WorkerRegistry {
    pub fn update_worker(&self, payload: WorkerHeartbeatPayload) {
        let mut workers = self.workers.write().unwrap();
        workers.insert(
            payload.worker_id.clone(),
            WorkerInfo {
                worker_id: payload.worker_id,
                last_heartbeat: Instant::now(),
                health_status: payload.health_status,
                // ... other fields from registry or config
            },
        );
    }
    
    pub fn get_available_workers(&self, model: &str) -> Vec<WorkerInfo> {
        let workers = self.workers.read().unwrap();
        workers
            .values()
            .filter(|w| {
                w.model == model
                    && w.health_status == HealthStatus::Healthy
                    && w.last_heartbeat.elapsed() < Duration::from_secs(90)
            })
            .cloned()
            .collect()
    }
}
```

---

## Operation Routing Logic

### Decision Tree

```
Operation received → What type?
    ↓
    ├─ Hive lifecycle? (Install, Start, Stop, Status, Uninstall)
    │  └→ Execute in queen (hive-lifecycle crate)
    │
    ├─ Worker/Model ops? (WorkerSpawn, ModelDownload, etc.)
    │  └→ Forward to hive via HTTP (hive_forwarder)
    │
    ├─ Inference? (Infer)
    │  └→ Schedule to worker, direct HTTP (TODO)
    │
    └─ Unknown?
       └→ Return error
```

### Code Implementation

```rust
pub async fn route_operation(
    operation: Operation,
    job_id: String,
    state: SchedulerState,
) -> Result<()> {
    match operation {
        // Hive lifecycle (queen-handled)
        Operation::HiveInstall { alias, binary_path } => {
            execute_hive_install(
                HiveInstallRequest { alias, binary_path, job_id },
                state.config,
            ).await
        }
        
        // ... more hive operations
        
        // Inference (queen schedules)
        Operation::Infer { prompt, model, .. } => {
            // TODO: IMPLEMENT INFERENCE SCHEDULING
            Err(anyhow!("Not yet implemented"))
        }
        
        // Worker/Model operations (forward to hive)
        op if op.should_forward_to_hive() => {
            hive_forwarder::forward_to_hive(&job_id, op, state.config).await
        }
        
        // Unknown operations
        _ => Err(anyhow!("Operation '{}' not supported", operation_name)),
    }
}
```

### Routing Rules

| Operation | Handler | HTTP Hops | Why |
|-----------|---------|-----------|-----|
| HiveInstall | Queen | 0 | Queen manages hives |
| HiveStart | Queen | 0 | Queen knows hive locations |
| WorkerSpawn | Hive | 1 | Hive manages local workers |
| ModelDownload | Hive | 1 | Hive manages local models |
| Infer | Worker | 1 | Direct for performance |

---

## Next: Part 5 - Development Patterns

The next document covers development patterns and practices:
- Crate Structure
- BDD Testing
- Character-Driven Development
- Code Organization

**See:** `.arch/04_DEVELOPMENT_PART_5.md`
