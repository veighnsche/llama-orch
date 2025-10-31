# Hive Job Server - Three Client Architecture

## Overview

The **rbee-hive job server** (`POST /v1/jobs` + SSE streaming) is designed to be consumed by **three distinct clients**, each with different use cases and interaction patterns.

---

## 🎯 The Three Clients

### 1. 📦 The SDK (Hive UI)
**Package:** `@rbee/rbee-hive-sdk` (WASM)  
**Consumer:** Hive UI (`bin/20_rbee_hive/ui/app`)  
**Use Case:** Direct user interaction with local hive

### 2. 👑 The Queen (Orchestration)
**Binary:** `queen-rbee`  
**Module:** `bin/10_queen_rbee/src/hive_forwarder.rs`  
**Use Case:** Automated scheduling and multi-hive orchestration

### 3. 🐝 The Keeper CLI
**Binary:** `rbee-keeper`  
**Module:** `bin/00_rbee_keeper/src/job_client.rs`  
**Use Case:** Manual operations and system management

---

## 📦 Client 1: The SDK (Hive UI)

### Purpose
Direct browser-based interaction with the local hive. Users can spawn workers, download models, and monitor operations through a web UI.

### Architecture
```
Browser (Hive UI)
    ↓ (WASM)
@rbee/rbee-hive-sdk
    ↓ (HTTP)
POST /v1/jobs → Hive Job Server
    ↓ (SSE)
GET /v1/jobs/{job_id}/stream
```

### Implementation

**SDK Client:**
```rust
// bin/20_rbee_hive/ui/packages/rbee-hive-sdk/src/client.rs

pub struct HiveClient {
    base_url: String,
    hive_id: String,
}

impl HiveClient {
    pub async fn submit_and_stream<F>(
        &self,
        operation: JsValue,
        line_handler: F,
    ) -> Result<(), JsValue>
    where
        F: Fn(String),
    {
        // 1. POST /v1/jobs with operation
        // 2. Extract job_id from response
        // 3. Connect to SSE stream
        // 4. Call line_handler for each event
    }
}
```

**React Hook Usage:**
```typescript
// bin/20_rbee_hive/ui/packages/rbee-hive-react/src/index.ts

export function useWorkers() {
  const { data, isLoading } = useQuery({
    queryKey: ['hive-workers'],
    queryFn: async () => {
      await ensureWasmInit()
      const op = OperationBuilder.workerList(hiveId)
      const lines: string[] = []
      
      await client.submitAndStream(op, (line: string) => {
        if (line !== '[DONE]') {
          lines.push(line)
        }
      })
      
      return JSON.parse(lines[lines.length - 1])
    },
  })
}
```

**UI Component:**
```tsx
// bin/20_rbee_hive/ui/app/src/App.tsx

function App() {
  const { workers, loading } = useWorkers()
  const { spawnWorker, isPending } = useHiveOperations()
  
  return (
    <button onClick={() => spawnWorker({ 
      modelId: 'llama-3.2-1b',
      workerType: 'cuda',
      deviceId: 0
    })}>
      Spawn Worker
    </button>
  )
}
```

### Characteristics
- **Direct connection:** Browser → Hive (no intermediary)
- **Single hive:** Always operates on the local hive
- **User-initiated:** All operations triggered by user clicks
- **Real-time feedback:** SSE streams show progress in UI
- **WASM-based:** Compiled Rust code running in browser

---

## 👑 Client 2: The Queen (Orchestration)

### Purpose
Automated scheduling and orchestration across multiple hives. Queen forwards operations to the appropriate hive based on scheduling decisions.

### Architecture
```
Queen Scheduler
    ↓
Hive Forwarder (hive_forwarder.rs)
    ↓
rbee-job-client (shared crate)
    ↓ (HTTP)
POST /v1/jobs → Hive Job Server
    ↓ (SSE)
GET /v1/jobs/{job_id}/stream
```

### Implementation

**Hive Forwarder:**
```rust
// bin/10_queen_rbee/src/hive_forwarder.rs

pub async fn forward_to_hive(
    job_id: &str,
    operation: Operation,
    hive_registry: Arc<TelemetryRegistry>,
) -> Result<()> {
    // 1. Determine target hive from operation
    let hive_id = extract_hive_id(&operation)?;
    
    // 2. Get hive URL from registry
    let hive_info = hive_registry.get_hive(&hive_id)
        .ok_or_else(|| anyhow!("Hive not found: {}", hive_id))?;
    
    let hive_url = format!("http://{}:{}", hive_info.hostname, hive_info.port);
    
    // 3. Forward operation to hive
    let client = JobClient::new(hive_url);
    client.submit_and_stream(operation, |line| {
        // Forward output to Queen's job stream
        println!("{}", line);
        Ok(())
    }).await?;
    
    Ok(())
}
```

**Job Router:**
```rust
// bin/10_queen_rbee/src/job_router.rs

async fn route_operation(
    job_id: String,
    payload: serde_json::Value,
    registry: Arc<JobRegistry<String>>,
    hive_registry: Arc<TelemetryRegistry>,
) -> Result<()> {
    let operation: Operation = serde_json::from_value(payload)?;
    
    match operation {
        // Infer stays in Queen for scheduling
        Operation::Infer { .. } => {
            let scheduler = SimpleScheduler::new(hive_registry.clone());
            let schedule_result = scheduler.schedule(job_request).await?;
            // ... scheduling logic
        }
        
        // Worker/Model operations forwarded to hive
        op if op.should_forward_to_hive() => {
            forward_to_hive(&job_id, op, hive_registry).await?
        }
        
        _ => anyhow::bail!("Unknown operation")
    }
    
    Ok(())
}
```

**Shared Job Client:**
```rust
// bin/99_shared_crates/rbee-job-client/src/lib.rs

pub struct JobClient {
    base_url: String,
    client: reqwest::Client,
}

impl JobClient {
    pub async fn submit_and_stream<F>(
        &self,
        operation: Operation,
        line_handler: F,
    ) -> Result<String>
    where
        F: Fn(&str) -> Result<()>,
    {
        // 1. POST /v1/jobs
        let response = self.client
            .post(format!("{}/v1/jobs", self.base_url))
            .json(&operation)
            .send()
            .await?;
        
        let job_id = response.json::<JobResponse>().await?.job_id;
        
        // 2. Connect to SSE stream
        let stream_url = format!("{}/v1/jobs/{}/stream", self.base_url, job_id);
        let mut event_source = EventSource::get(&stream_url);
        
        // 3. Process events
        while let Some(event) = event_source.next().await {
            match event {
                Ok(Event::Message(msg)) => {
                    line_handler(&msg.data)?;
                    if msg.data == "[DONE]" {
                        break;
                    }
                }
                Err(e) => return Err(e.into()),
            }
        }
        
        Ok(job_id)
    }
}
```

### Characteristics
- **Multi-hive:** Can forward to any registered hive
- **Automated:** Operations triggered by scheduling logic
- **Transparent forwarding:** Client sees Queen, but work happens on hive
- **Scheduling intelligence:** Chooses best hive for inference
- **Centralized control:** Single point for orchestration

---

## 🐝 Client 3: The Keeper CLI

### Purpose
Manual operations and system management. Administrators use the CLI to perform operations on specific hives.

### Architecture
```
Keeper CLI (Tauri app)
    ↓
Job Client (job_client.rs)
    ↓
rbee-job-client (shared crate)
    ↓ (HTTP)
POST /v1/jobs → Hive Job Server
    ↓ (SSE)
GET /v1/jobs/{job_id}/stream
```

### Implementation

**Keeper Job Client:**
```rust
// bin/00_rbee_keeper/src/job_client.rs

pub async fn submit_hive_operation(
    hive_url: &str,
    operation: Operation,
) -> Result<String> {
    let client = JobClient::new(hive_url);
    
    client.submit_and_stream(operation, |line| {
        // Emit narration events for UI
        NARRATE
            .action("hive_operation_progress")
            .context(line)
            .human("{}")
            .emit();
        Ok(())
    }).await
}
```

**CLI Command:**
```rust
// bin/00_rbee_keeper/src/main.rs

#[derive(Subcommand)]
enum HiveAction {
    WorkerSpawn {
        #[arg(long)]
        hive: String,
        #[arg(long)]
        model: String,
        #[arg(long)]
        worker_type: String,
        #[arg(long)]
        device: u32,
    },
}

async fn handle_hive_action(action: HiveAction) -> Result<()> {
    match action {
        HiveAction::WorkerSpawn { hive, model, worker_type, device } => {
            let hive_url = format!("http://{}:7835", hive);
            let operation = Operation::WorkerSpawn {
                hive_id: hive.clone(),
                model,
                worker: worker_type,
                device,
            };
            
            let job_id = submit_hive_operation(&hive_url, operation).await?;
            println!("Worker spawn job submitted: {}", job_id);
            Ok(())
        }
    }
}
```

**Tauri Command:**
```rust
// bin/00_rbee_keeper/src-tauri/src/main.rs

#[tauri::command]
async fn spawn_worker(
    hive_id: String,
    model_id: String,
    worker_type: String,
    device_id: u32,
) -> Result<String, String> {
    let hive_url = format!("http://{}:7835", hive_id);
    let operation = Operation::WorkerSpawn {
        hive_id,
        model: model_id,
        worker: worker_type,
        device: device_id,
    };
    
    submit_hive_operation(&hive_url, operation)
        .await
        .map_err(|e| e.to_string())
}
```

### Characteristics
- **Manual control:** User explicitly chooses target hive
- **System management:** Administrative operations
- **Direct targeting:** Specifies exact hive for operation
- **Narration events:** Progress shown in Keeper UI
- **Tauri-based:** Desktop app with Rust backend

---

## 🔄 Comparison Matrix

| Aspect | SDK (Hive UI) | Queen (Orchestration) | Keeper CLI |
|--------|---------------|----------------------|------------|
| **Language** | TypeScript + WASM | Rust | Rust |
| **Environment** | Browser | Server | Desktop (Tauri) |
| **Hive Selection** | Always local | Automatic (scheduling) | Manual (user choice) |
| **Use Case** | User operations | Automated orchestration | Admin management |
| **Connection** | Direct | Forwarded | Direct |
| **Multi-hive** | No (single hive) | Yes (all hives) | Yes (user selects) |
| **Shared Code** | Custom WASM client | `rbee-job-client` | `rbee-job-client` |

---

## 📊 Data Flow Examples

### Example 1: User Spawns Worker via Hive UI

```
1. User clicks "Spawn Worker" in Hive UI
   ↓
2. React hook calls useHiveOperations.spawnWorker()
   ↓
3. useMutation calls HiveClient.submitAndStream()
   ↓
4. WASM SDK sends POST /v1/jobs to localhost:7835
   ↓
5. Hive job server creates job, returns job_id
   ↓
6. WASM SDK connects to GET /v1/jobs/{job_id}/stream
   ↓
7. SSE events streamed back to browser
   ↓
8. React hook updates UI with progress
   ↓
9. Worker spawned, UI shows success
```

### Example 2: Queen Schedules Inference

```
1. Client sends inference request to Queen
   ↓
2. Queen's job router receives Operation::Infer
   ↓
3. SimpleScheduler finds best hive for model
   ↓
4. Scheduler determines: hive "gpu-0" has model loaded
   ↓
5. Queen forwards Operation::Infer to gpu-0 via hive_forwarder
   ↓
6. hive_forwarder uses JobClient to POST /v1/jobs to gpu-0:7835
   ↓
7. Hive job server on gpu-0 processes inference
   ↓
8. SSE events streamed back to Queen
   ↓
9. Queen forwards events to original client
   ↓
10. Client receives inference result
```

### Example 3: Admin Downloads Model via Keeper

```
1. Admin opens Keeper, selects "Download Model"
   ↓
2. Admin chooses target hive: "gpu-1"
   ↓
3. Keeper creates Operation::ModelDownload
   ↓
4. Keeper's job_client sends POST /v1/jobs to gpu-1:7835
   ↓
5. Hive job server starts download
   ↓
6. SSE events streamed back to Keeper
   ↓
7. Keeper UI shows download progress
   ↓
8. Model downloaded, Keeper shows success
```

---

## 🔧 Shared Infrastructure

### rbee-job-client Crate

**Location:** `bin/99_shared_crates/rbee-job-client`

**Purpose:** Shared job submission and SSE streaming logic used by both Queen and Keeper.

**Why Shared:**
- ✅ Single source of truth for job submission pattern
- ✅ Consistent error handling
- ✅ Automatic [DONE] detection
- ✅ SSE prefix stripping
- ✅ Bugs fixed in one place

**Not Used By SDK:**
- SDK is WASM (JavaScript environment)
- SDK has its own implementation in TypeScript/Rust WASM
- Cannot share Rust crates between WASM and native

---

## 🎯 Design Principles

### 1. Uniform API
All three clients use the same hive job server API:
- `POST /v1/jobs` - Submit operation
- `GET /v1/jobs/{job_id}/stream` - Stream results

### 2. Client-Specific Concerns
Each client handles its own concerns:
- **SDK:** Browser environment, WASM compilation, React state
- **Queen:** Multi-hive routing, scheduling, forwarding
- **Keeper:** CLI parsing, Tauri commands, narration events

### 3. Separation of Concerns
- **Hive:** Executes operations (workers, models, inference)
- **Queen:** Orchestrates across hives (scheduling, routing)
- **Clients:** Initiate operations (user/admin/automated)

### 4. Extensibility
Adding a new operation:
1. Add to `Operation` enum in `rbee-operations`
2. Implement handler in hive's `job_router.rs`
3. All three clients automatically support it

---

## 📝 Adding a New Client

If you need to add a fourth client (e.g., Python SDK, REST API gateway), follow this pattern:

### Step 1: Job Submission
```
POST http://{hive}:7835/v1/jobs
Content-Type: application/json

{
  "operation": "worker_spawn",
  "hive_id": "gpu-0",
  "model": "llama-3.2-1b",
  "worker": "cuda",
  "device": 0
}
```

### Step 2: Get Job ID
```json
{
  "job_id": "job_abc123"
}
```

### Step 3: Stream Results
```
GET http://{hive}:7835/v1/jobs/job_abc123/stream
Accept: text/event-stream

event: message
data: [INFO] Starting worker spawn...

event: message
data: [INFO] Loading model...

event: message
data: [DONE]
```

### Step 4: Parse Output
- Lines starting with `[INFO]`, `[ERROR]`, etc. are progress
- Last line before `[DONE]` often contains JSON result
- `[DONE]` signals completion

---

## 🔗 Related Documentation

- **Job Server Implementation:** `bin/20_rbee_hive/src/http/jobs.rs`
- **Operation Types:** `bin/99_shared_crates/rbee-operations/src/lib.rs`
- **Queen Forwarding:** `bin/10_queen_rbee/src/hive_forwarder.rs`
- **Shared Job Client:** `bin/99_shared_crates/rbee-job-client/src/lib.rs`
- **SDK Client:** `bin/20_rbee_hive/ui/packages/rbee-hive-sdk/src/client.rs`
- **Keeper Client:** `bin/00_rbee_keeper/src/job_client.rs`

---

**Last Updated:** TEAM-377 | 2025-10-31  
**Maintainer:** rbee Team
