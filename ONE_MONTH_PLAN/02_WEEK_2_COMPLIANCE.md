# Week 2: EU Compliance — Days 8-14

**Goal**: EU audit toggle + basic web UI  
**Deliverable**: GDPR-compliant inference with audit logs

---

## 🎉 EVEN BETTER NEWS: Audit System Already Built!

**What's Already Done:**
- ✅ **audit-logging crate** (895 lines of comprehensive docs!)
- ✅ **3 modes**: Disabled (homelab), Local (single-node), Platform (marketplace)
- ✅ **32 pre-defined event types** (including GDPR events!)
- ✅ **Query API** for data export (no need to parse logs manually)
- ✅ **Hash chain integrity** (tamper-evident)
- ✅ **Flush modes** (Immediate, Batched, Hybrid)
- ✅ **Already integrated** in queen-rbee (Week 1)

**Time Saved:** 3 days (audit system + query API ready)

**What This Means:**
- Day 8: Just test what you already have (2 hours instead of full day)
- Day 9: Use existing query API for GDPR endpoints (2 hours instead of full day)
- Days 10-14: Focus on web UI and polish

**You're WAY ahead of schedule!**

---

## Day 8 (Monday): Audit Toggle

**GOOD NEWS:** The audit-logging crate already exists with 895 lines of documentation!

### Morning Session (09:00-13:00)

**Task 1: Verify existing audit-logging crate (30 min)**
```bash
# Check what you already have
cat bin/shared-crates/audit-logging/README.md

# 895 lines of comprehensive docs!
# - 3 modes: Disabled, Local, Platform
# - 32 pre-defined event types
# - Query API for GDPR exports
# - Hash chain integrity
# - Flush modes (Immediate, Batched, Hybrid)
```

**Task 2: Test audit toggle (1 hour)**
```bash
# Already integrated in Week 1!
# Just verify it works

# Test disabled mode (homelab)
LLORCH_EU_AUDIT=false cargo run
curl -X POST http://localhost:8080/v2/tasks \
  -H "Content-Type: application/json" \
  -d '{"model":"tinyllama","prompt":"test"}'

# Verify: No audit log created (zero overhead)

# Test enabled mode (EU compliance)
LLORCH_EU_AUDIT=true cargo run
curl -X POST http://localhost:8080/v2/tasks \
  -H "Content-Type: application/json" \
  -d '{"model":"tinyllama","prompt":"test"}'

# ✅ Verify: Audit log created with events
cat /var/log/llorch/audit/queen-rbee/*.audit
```

**Task 3: Document what you already have (30 min)**
```markdown
# EU Audit Mode

## What's Already Built

✅ audit-logging crate (895 lines of docs)
✅ 3 modes: Disabled, Local, Platform
✅ 32 pre-defined event types
✅ Query API for GDPR exports
✅ Hash chain integrity
✅ Flush modes (Immediate, Batched, Hybrid)
✅ Integrated in queen-rbee (Week 1)

## Usage

# Enable EU audit
export LLORCH_EU_AUDIT=true
cargo run

# Disable (homelab mode)
export LLORCH_EU_AUDIT=false
cargo run
```

**Task 4: Add audit middleware (OPTIONAL - 1 hour)**
```rust
// In queen-rbee/src/middleware.rs
use axum::{
    extract::Request,
    middleware::Next,
    response::Response,
};
use std::sync::Arc;

pub async fn audit_middleware(
    req: Request,
    next: Next,
    audit_logger: Arc<AuditLogger>,
) -> Response {
    let method = req.method().clone();
    let uri = req.uri().clone();
    let ip = req.headers()
        .get("x-forwarded-for")
        .and_then(|h| h.to_str().ok())
        .map(|s| s.to_string());
    
    let response = next.run(req).await;
    
    // Log the request
    let event = AuditEvent {
        timestamp: Utc::now().to_rfc3339(),
        event_type: "http_request".to_string(),
        user_id: None,  // TODO: Extract from auth header
        resource_id: uri.path().to_string(),
        action: method.to_string(),
        metadata: serde_json::json!({
            "status": response.status().as_u16(),
        }),
        ip_address: ip,
    };
    
    let _ = audit_logger.log(event);
    
    response
}
```

### Afternoon Session (14:00-18:00)

**Task 4: Wire up middleware (1 hour)**
```rust
// In queen-rbee/src/main.rs
use axum::middleware;
use std::sync::Arc;

#[tokio::main]
async fn main() {
    let eu_audit_enabled = env::var("LLORCH_EU_AUDIT")
        .unwrap_or_else(|_| "false".to_string()) == "true";
    
    let audit_logger = Arc::new(AuditLogger::new(eu_audit_enabled)?);
    
    let mut app = Router::new()
        .route("/health", get(health))
        .route("/v2/tasks", post(submit_task))
        .route("/workers/register", post(register_worker))
        .with_state(state);
    
    if eu_audit_enabled {
        let audit_logger_clone = audit_logger.clone();
        app = app.layer(middleware::from_fn(move |req, next| {
            audit_middleware(req, next, audit_logger_clone.clone())
        }));
        
        tracing::info!("Audit log: {}", 
            env::var("LLORCH_AUDIT_LOG_PATH")
                .unwrap_or_else(|_| "/var/log/llorch/audit.log".to_string())
        );
    }
    
    // ... start server
}
```

**Task 5: Test audit logging (2 hours)**
```bash
# Test with audit enabled
LLORCH_EU_AUDIT=true \
LLORCH_AUDIT_LOG_PATH=/tmp/llorch-audit.log \
cargo run

# Submit a job
curl -X POST http://localhost:8080/v2/tasks \
  -H "Content-Type: application/json" \
  -d '{"model":"tinyllama","prompt":"test"}'

# Check audit log
cat /tmp/llorch-audit.log
# Should see JSON entries

# Test with audit disabled
LLORCH_EU_AUDIT=false cargo run

# Submit a job
curl -X POST http://localhost:8080/v2/tasks \
  -H "Content-Type: application/json" \
  -d '{"model":"tinyllama","prompt":"test"}'

# Verify no audit log created
```

**Task 6: Document toggle (1 hour)**
```markdown
# EU Audit Mode

## Enable

export LLORCH_EU_AUDIT=true
export LLORCH_AUDIT_LOG_PATH=/var/log/llorch/audit.log
cargo run

## Disable (default)

export LLORCH_EU_AUDIT=false
cargo run

## Audit Log Format

Each line is a JSON object:
{
  "timestamp": "2025-10-09T12:00:00Z",
  "event_type": "http_request",
  "user_id": null,
  "resource_id": "/v2/tasks",
  "action": "POST",
  "metadata": {"status": 200},
  "ip_address": "192.168.1.100"
}
```

**Day 8 Deliverable**: EU audit toggle working

---

## Day 9 (Tuesday): GDPR Endpoints

**GOOD NEWS:** The audit-logging crate already has a query API!

### Morning Session (09:00-13:00)

**Task 1: Use existing query API (1 hour)**
```rust
// In queen-rbee/src/gdpr.rs
use axum::{Json, extract::Query};
use serde::{Deserialize, Serialize};
use audit_logging::{AuditLogger, AuditQuery};  // ✅ Existing crate!

#[derive(Deserialize)]
pub struct ExportQuery {
    user_id: String,
}

#[derive(Serialize)]
pub struct ExportResponse {
    user_id: String,
    jobs: Vec<Job>,
    audit_events: Vec<AuditEvent>,
    created_at: String,
}

pub async fn gdpr_export(
    State(state): State<AppState>,
    Query(query): Query<ExportQuery>,
) -> Result<Json<ExportResponse>, AppError> {
    // Get all jobs for user
    let jobs = state.jobs.lock().unwrap()
        .iter()
        .filter(|j| j.user_id.as_ref() == Some(&query.user_id))
        .cloned()
        .collect();
    
    // ✅ Use existing query API!
    let audit_events = state.audit_logger.query(AuditQuery {
        actor: Some(query.user_id.clone()),
        start_time: None,  // All time
        end_time: None,
        event_types: vec![],  // All types
        limit: 10000,
    }).await?;
    
    Ok(Json(ExportResponse {
        user_id: query.user_id,
        jobs,
        audit_events,
        created_at: Utc::now().to_rfc3339(),
    }))
}
```

**What you get for FREE:**
- ✅ Query API (already implemented!)
- ✅ Filter by actor, time, event type
- ✅ Pagination support
- ✅ No need to parse log files manually

**Task 2: Data deletion endpoint (1 hour)**
```rust
// In queen-rbee/src/gdpr.rs
#[derive(Deserialize)]
pub struct DeleteRequest {
    user_id: String,
    reason: String,
}

#[derive(Serialize)]
pub struct DeleteResponse {
    user_id: String,
    jobs_deleted: usize,
    status: String,
}

pub async fn gdpr_delete(
    State(state): State<AppState>,
    Json(req): Json<DeleteRequest>,
) -> Result<Json<DeleteResponse>, AppError> {
    // Soft delete jobs (mark as deleted)
    let mut jobs = state.jobs.lock().unwrap();
    let mut deleted_count = 0;
    
    for job in jobs.iter_mut() {
        if job.user_id.as_ref() == Some(&req.user_id) {
            job.status = "deleted".to_string();
            deleted_count += 1;
        }
    }
    
    // Log deletion in audit log
    let audit_logger = state.audit_logger.clone();
    audit_logger.log(AuditEvent {
        timestamp: Utc::now().to_rfc3339(),
        event_type: "gdpr_deletion".to_string(),
        user_id: Some(req.user_id.clone()),
        resource_id: "user_data".to_string(),
        action: "DELETE".to_string(),
        metadata: serde_json::json!({
            "reason": req.reason,
            "jobs_deleted": deleted_count,
        }),
        ip_address: None,
    })?;
    
    Ok(Json(DeleteResponse {
        user_id: req.user_id,
        jobs_deleted: deleted_count,
        status: "deleted".to_string(),
    }))
}
```

### Afternoon Session (14:00-18:00)

**Task 3: Consent endpoint (1 hour)**
```rust
// In queen-rbee/src/gdpr.rs
#[derive(Deserialize)]
pub struct ConsentRequest {
    user_id: String,
    consent_type: String,
    granted: bool,
}

pub async fn gdpr_consent(
    State(state): State<AppState>,
    Json(req): Json<ConsentRequest>,
) -> Result<&'static str, AppError> {
    // Log consent in audit log
    state.audit_logger.log(AuditEvent {
        timestamp: Utc::now().to_rfc3339(),
        event_type: "gdpr_consent".to_string(),
        user_id: Some(req.user_id),
        resource_id: "consent".to_string(),
        action: if req.granted { "GRANT" } else { "REVOKE" }.to_string(),
        metadata: serde_json::json!({
            "consent_type": req.consent_type,
            "granted": req.granted,
        }),
        ip_address: None,
    })?;
    
    Ok("OK")
}
```

**Task 4: Wire up GDPR endpoints (1 hour)**
```rust
// In queen-rbee/src/main.rs
if eu_audit_enabled {
    app = app
        .route("/gdpr/export", get(gdpr_export))
        .route("/gdpr/delete", post(gdpr_delete))
        .route("/gdpr/consent", post(gdpr_consent));
    
    tracing::info!("GDPR endpoints enabled");
}
```

**Task 5: Test GDPR endpoints (2 hours)**
```bash
# Export data
curl "http://localhost:8080/gdpr/export?user_id=test-user"

# Delete data
curl -X POST http://localhost:8080/gdpr/delete \
  -H "Content-Type: application/json" \
  -d '{"user_id":"test-user","reason":"user_request"}'

# Record consent
curl -X POST http://localhost:8080/gdpr/consent \
  -H "Content-Type: application/json" \
  -d '{"user_id":"test-user","consent_type":"data_processing","granted":true}'
```

**Day 9 Deliverable**: GDPR endpoints functional

---

## Day 10 (Wednesday): Data Residency

### Morning Session (09:00-13:00)

**Task 1: Add region to worker registration (1 hour)**
```rust
// Update Worker struct
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Worker {
    id: String,
    host: String,
    port: u16,
    model: String,
    region: String,  // NEW: "EU", "US", "ASIA", etc.
}

// Update registration endpoint
async fn register_worker(
    State(state): State<AppState>,
    Json(mut worker): Json<Worker>,
) -> &'static str {
    // Auto-detect region from IP if not provided
    if worker.region.is_empty() {
        worker.region = detect_region(&worker.host).unwrap_or("UNKNOWN".to_string());
    }
    
    tracing::info!("Worker registered: {} in region {}", worker.id, worker.region);
    state.workers.lock().unwrap().push(worker);
    "OK"
}

fn detect_region(host: &str) -> Option<String> {
    // Simple implementation - can be improved with GeoIP
    if host.ends_with(".eu") || host.contains("europe") {
        Some("EU".to_string())
    } else if host.ends_with(".us") || host.contains("america") {
        Some("US".to_string())
    } else {
        None
    }
}
```

**Task 2: Add EU-only filtering (2 hours)**
```rust
// In queen-rbee/src/main.rs
async fn submit_task(
    State(state): State<AppState>,
    Json(req): Json<TaskRequest>,
) -> Result<Json<TaskResponse>, AppError> {
    let job_id = Uuid::new_v4().to_string();
    
    let eu_audit_enabled = env::var("LLORCH_EU_AUDIT")
        .unwrap_or_else(|_| "false".to_string()) == "true";
    
    // Find worker with matching model
    let workers = state.workers.lock().unwrap();
    let mut candidates: Vec<_> = workers.iter()
        .filter(|w| w.model == req.model)
        .collect();
    
    // If EU audit enabled, filter to EU-only workers
    if eu_audit_enabled {
        candidates.retain(|w| w.region == "EU");
        
        if candidates.is_empty() {
            return Err(AppError::NoEUWorkerAvailable(req.model));
        }
    }
    
    let worker = candidates.first()
        .ok_or_else(|| AppError::NoWorkerAvailable(req.model))?
        .clone();
    
    drop(workers);
    
    tracing::info!("Dispatching job {} to worker {} in region {}", 
        job_id, worker.id, worker.region);
    
    // ... rest of dispatch logic
}
```

### Afternoon Session (14:00-18:00)

**Task 3: Update llm-worker-rbee registration (1 hour)**
```rust
// In llm-worker-rbee/src/main.rs
async fn register_with_orchestrator(config: &Config) -> Result<()> {
    let client = reqwest::Client::new();
    
    let region = env::var("LLORCH_REGION")
        .unwrap_or_else(|_| "EU".to_string());  // Default to EU
    
    let registration = serde_json::json!({
        "id": config.worker_id,
        "host": get_local_ip()?,
        "port": config.port,
        "model": extract_model_name(&config.model_path),
        "region": region,
    });
    
    println!("📡 Registering with orchestrator (region: {})", region);
    
    // ... rest of registration
}
```

**Task 4: Test data residency (2 hours)**
```bash
# Start orchestrator with EU audit
LLORCH_EU_AUDIT=true cargo run

# Spawn EU worker
LLORCH_REGION=EU rbee pool worker spawn metal --model tinyllama --host mac.home.arpa

# Spawn US worker (for testing)
LLORCH_REGION=US rbee pool worker spawn metal --model tinyllama --host workstation.home.arpa

# Submit job - should only go to EU worker
llorch jobs submit --model tinyllama --prompt "test"

# Verify in logs that EU worker was selected
```

**Task 5: Document data residency (1 hour)**
```markdown
# Data Residency

## EU-Only Mode

When EU audit is enabled, only workers in the EU region are used.

### Enable EU-Only

export LLORCH_EU_AUDIT=true
cargo run

### Worker Registration

Workers must specify their region:

export LLORCH_REGION=EU
llm-worker-rbee --worker-id worker-1 --model model.gguf

### Supported Regions

- EU: European Union
- US: United States
- ASIA: Asia Pacific
- UNKNOWN: Region not detected
```

**Day 10 Deliverable**: EU-only worker filtering working

---

## Day 11 (Thursday): Web UI Start

### Morning Session (09:00-13:00)

**Task 1: Create Vue 3 project (1 hour)**
```bash
cd /home/vince/Projects/llama-orch
mkdir -p frontend/llorch-ui
cd frontend/llorch-ui

# Create Vite + Vue 3 project
npm create vite@latest . -- --template vue-ts
npm install
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p
```

**Task 2: Setup TailwindCSS (30 min)**
```js
// tailwind.config.js
export default {
  content: [
    "./index.html",
    "./src/**/*.{vue,js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}
```

```css
/* src/assets/main.css */
@tailwind base;
@tailwind components;
@tailwind utilities;
```

**Task 3: Create job submission form (2 hours)**
```vue
<!-- src/components/JobSubmit.vue -->
<template>
  <div class="max-w-2xl mx-auto p-6">
    <h1 class="text-3xl font-bold mb-6">Submit Inference Job</h1>
    
    <form @submit.prevent="submitJob" class="space-y-4">
      <div>
        <label class="block text-sm font-medium mb-2">Model</label>
        <select v-model="form.model" class="w-full px-4 py-2 border rounded">
          <option value="tinyllama">TinyLlama 1.1B</option>
          <option value="qwen">Qwen 0.5B</option>
          <option value="phi3">Phi-3 Mini</option>
        </select>
      </div>
      
      <div>
        <label class="block text-sm font-medium mb-2">Prompt</label>
        <textarea 
          v-model="form.prompt" 
          class="w-full px-4 py-2 border rounded h-32"
          placeholder="Enter your prompt..."
        ></textarea>
      </div>
      
      <button 
        type="submit" 
        class="w-full bg-blue-600 text-white px-6 py-3 rounded hover:bg-blue-700"
        :disabled="loading"
      >
        {{ loading ? 'Submitting...' : 'Submit Job' }}
      </button>
    </form>
    
    <div v-if="result" class="mt-6 p-4 bg-green-50 border border-green-200 rounded">
      <p class="font-medium">Job Submitted!</p>
      <p class="text-sm text-gray-600">Job ID: {{ result.job_id }}</p>
      <p class="text-sm text-gray-600">Status: {{ result.status }}</p>
    </div>
    
    <div v-if="error" class="mt-6 p-4 bg-red-50 border border-red-200 rounded">
      <p class="font-medium text-red-800">Error</p>
      <p class="text-sm text-red-600">{{ error }}</p>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'

const form = ref({
  model: 'tinyllama',
  prompt: ''
})

const loading = ref(false)
const result = ref<any>(null)
const error = ref<string | null>(null)

async function submitJob() {
  loading.value = true
  error.value = null
  result.value = null
  
  try {
    const response = await fetch('http://localhost:8080/v2/tasks', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(form.value),
    })
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`)
    }
    
    result.value = await response.json()
  } catch (e: any) {
    error.value = e.message
  } finally {
    loading.value = false
  }
}
</script>
```

### Afternoon Session (14:00-18:00)

**Task 4: Create token streaming display (2 hours)**
```vue
<!-- src/components/TokenStream.vue -->
<template>
  <div class="max-w-2xl mx-auto p-6">
    <h2 class="text-2xl font-bold mb-4">Live Inference</h2>
    
    <div class="bg-gray-50 border rounded p-4 min-h-[200px] font-mono text-sm">
      <div v-if="streaming" class="text-blue-600 mb-2">
        ⚡ Streaming tokens...
      </div>
      <div v-if="tokens.length === 0 && !streaming" class="text-gray-400">
        No output yet. Submit a job to see tokens stream here.
      </div>
      <div>{{ tokens.join('') }}</div>
    </div>
    
    <div v-if="complete" class="mt-4 text-green-600">
      ✅ Generation complete ({{ tokens.length }} tokens)
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'

const tokens = ref<string[]>([])
const streaming = ref(false)
const complete = ref(false)

// TODO: Wire up SSE streaming from queen-rbee
// For now, mock data
function startStreaming() {
  streaming.value = true
  complete.value = false
  tokens.value = []
  
  const mockTokens = "Hello world! This is a test of token streaming.".split(' ')
  let i = 0
  
  const interval = setInterval(() => {
    if (i < mockTokens.length) {
      tokens.value.push(mockTokens[i] + ' ')
      i++
    } else {
      clearInterval(interval)
      streaming.value = false
      complete.value = true
    }
  }, 200)
}

// Expose for testing
defineExpose({ startStreaming })
</script>
```

**Task 5: Create basic layout (1 hour)**
```vue
<!-- src/App.vue -->
<template>
  <div class="min-h-screen bg-gray-100">
    <nav class="bg-white shadow">
      <div class="max-w-7xl mx-auto px-4 py-4">
        <h1 class="text-2xl font-bold">llama-orch</h1>
        <p class="text-sm text-gray-600">EU-Native LLM Inference</p>
      </div>
    </nav>
    
    <main class="py-8">
      <JobSubmit />
      <TokenStream ref="streamRef" />
    </main>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import JobSubmit from './components/JobSubmit.vue'
import TokenStream from './components/TokenStream.vue'

const streamRef = ref()
</script>
```

**Task 6: Test UI (1 hour)**
```bash
npm run dev

# Open http://localhost:5173
# Test job submission
# Verify form works
```

**Day 11 Deliverable**: Basic web UI with job submission

---

## Day 12 (Friday): Web UI Finish

### Morning Session (09:00-13:00)

**Task 1: Add audit log viewer (2 hours)**
```vue
<!-- src/components/AuditLog.vue -->
<template>
  <div class="max-w-4xl mx-auto p-6">
    <h2 class="text-2xl font-bold mb-4">Audit Log</h2>
    
    <div v-if="!euAuditEnabled" class="p-4 bg-yellow-50 border border-yellow-200 rounded">
      <p class="text-yellow-800">
        EU audit mode is disabled. Enable with LLORCH_EU_AUDIT=true
      </p>
    </div>
    
    <div v-else class="space-y-2">
      <div 
        v-for="event in events" 
        :key="event.timestamp"
        class="p-4 bg-white border rounded"
      >
        <div class="flex justify-between items-start">
          <div>
            <span class="font-mono text-sm text-gray-500">
              {{ new Date(event.timestamp).toLocaleString() }}
            </span>
            <p class="font-medium">{{ event.event_type }}</p>
            <p class="text-sm text-gray-600">
              {{ event.action }} {{ event.resource_id }}
            </p>
          </div>
          <span 
            class="px-2 py-1 text-xs rounded"
            :class="getStatusClass(event.metadata.status)"
          >
            {{ event.metadata.status }}
          </span>
        </div>
      </div>
      
      <div v-if="events.length === 0" class="text-center text-gray-400 py-8">
        No audit events yet
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'

const euAuditEnabled = ref(false)
const events = ref<any[]>([])

onMounted(async () => {
  // Check if EU audit is enabled
  try {
    const response = await fetch('http://localhost:8080/health')
    euAuditEnabled.value = response.headers.get('x-eu-audit') === 'true'
  } catch (e) {
    console.error('Failed to check EU audit status', e)
  }
  
  // Load audit events (mock for now)
  events.value = [
    {
      timestamp: new Date().toISOString(),
      event_type: 'http_request',
      action: 'POST',
      resource_id: '/v2/tasks',
      metadata: { status: 200 }
    }
  ]
})

function getStatusClass(status: number) {
  if (status >= 200 && status < 300) {
    return 'bg-green-100 text-green-800'
  } else if (status >= 400) {
    return 'bg-red-100 text-red-800'
  }
  return 'bg-gray-100 text-gray-800'
}
</script>
```

**Task 2: Add job history (1 hour)**
```vue
<!-- src/components/JobHistory.vue -->
<template>
  <div class="max-w-4xl mx-auto p-6">
    <h2 class="text-2xl font-bold mb-4">Job History</h2>
    
    <div class="space-y-2">
      <div 
        v-for="job in jobs" 
        :key="job.id"
        class="p-4 bg-white border rounded"
      >
        <div class="flex justify-between items-start">
          <div>
            <p class="font-mono text-sm text-gray-500">{{ job.id }}</p>
            <p class="font-medium">{{ job.model }}</p>
            <p class="text-sm text-gray-600 truncate max-w-md">
              {{ job.prompt }}
            </p>
          </div>
          <span 
            class="px-2 py-1 text-xs rounded"
            :class="getStatusClass(job.status)"
          >
            {{ job.status }}
          </span>
        </div>
      </div>
      
      <div v-if="jobs.length === 0" class="text-center text-gray-400 py-8">
        No jobs yet. Submit your first job above!
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'

const jobs = ref<any[]>([])

onMounted(() => {
  // TODO: Fetch jobs from API
  // For now, mock data
  jobs.value = []
})

function getStatusClass(status: string) {
  switch (status) {
    case 'completed': return 'bg-green-100 text-green-800'
    case 'running': return 'bg-blue-100 text-blue-800'
    case 'queued': return 'bg-yellow-100 text-yellow-800'
    case 'failed': return 'bg-red-100 text-red-800'
    default: return 'bg-gray-100 text-gray-800'
  }
}
</script>
```

### Afternoon Session (14:00-18:00)

**Task 3: Add CORS to queen-rbee (30 min)**
```rust
// In queen-rbee/src/main.rs
use tower_http::cors::{CorsLayer, Any};

let app = Router::new()
    .route("/health", get(health))
    .route("/v2/tasks", post(submit_task))
    .route("/workers/register", post(register_worker))
    .layer(
        CorsLayer::new()
            .allow_origin(Any)
            .allow_methods(Any)
            .allow_headers(Any)
    )
    .with_state(state);
```

**Task 4: Wire up components (1 hour)**
```vue
<!-- src/App.vue -->
<template>
  <div class="min-h-screen bg-gray-100">
    <nav class="bg-white shadow">
      <div class="max-w-7xl mx-auto px-4 py-4">
        <h1 class="text-2xl font-bold">llama-orch</h1>
        <p class="text-sm text-gray-600">EU-Native LLM Inference</p>
      </div>
    </nav>
    
    <main class="py-8 space-y-8">
      <JobSubmit @job-submitted="onJobSubmitted" />
      <TokenStream ref="streamRef" />
      <JobHistory :jobs="jobs" />
      <AuditLog v-if="euAuditEnabled" />
    </main>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import JobSubmit from './components/JobSubmit.vue'
import TokenStream from './components/TokenStream.vue'
import JobHistory from './components/JobHistory.vue'
import AuditLog from './components/AuditLog.vue'

const streamRef = ref()
const jobs = ref<any[]>([])
const euAuditEnabled = ref(false)

onMounted(async () => {
  // Check EU audit status
  try {
    const response = await fetch('http://localhost:8080/health')
    euAuditEnabled.value = response.headers.get('x-eu-audit') === 'true'
  } catch (e) {
    console.error('Failed to check EU audit status', e)
  }
})

function onJobSubmitted(job: any) {
  jobs.value.unshift(job)
  streamRef.value?.startStreaming()
}
</script>
```

**Task 5: Build and deploy (2 hours)**
```bash
# Build for production
npm run build

# Serve static files from queen-rbee
# Add static file serving to queen-rbee
```

```rust
// In queen-rbee/src/main.rs
use tower_http::services::ServeDir;

let app = Router::new()
    .route("/health", get(health))
    .route("/v2/tasks", post(submit_task))
    .route("/workers/register", post(register_worker))
    .nest_service("/", ServeDir::new("../frontend/llorch-ui/dist"))
    .with_state(state);
```

**Day 12 Deliverable**: Complete web UI deployed

---

## Days 13-14 (Weekend): Polish & Test

### Saturday

**Task 1: Fix UI bugs (3 hours)**
- Form validation
- Error handling
- Loading states
- Responsive design

**Task 2: Improve styling (2 hours)**
- Better colors
- Consistent spacing
- Professional look

**Task 3: Add help text (1 hour)**
- Tooltips
- Placeholder text
- Instructions

### Sunday

**Task 1: End-to-end testing (3 hours)**
```bash
# Full flow test
LLORCH_EU_AUDIT=true cargo run

# Open UI
open http://localhost:8080

# Submit job
# Verify audit log shows entry
# Verify GDPR export works
```

**Task 2: Documentation (2 hours)**
```markdown
# Web UI

## Access

http://localhost:8080

## Features

- Job submission
- Token streaming
- Job history
- Audit log viewer (EU mode only)

## EU Mode

Enable EU audit mode:

export LLORCH_EU_AUDIT=true
cargo run

The UI will show:
- Audit log viewer
- GDPR export button
- EU-only badge
```

**Task 3: Prepare for Week 3 (1 hour)**
- Review marketing plan
- Sketch landing page
- Prepare outreach list

---

## Week 2 Success Criteria

- [ ] LLORCH_EU_AUDIT toggle works
- [ ] Audit logging to file
- [ ] GDPR endpoints (export, delete, consent)
- [ ] EU-only worker filtering
- [ ] Web UI deployed
- [ ] Job submission via UI
- [ ] Audit log viewer
- [ ] Job history
- [ ] Clean, professional design
- [ ] Full end-to-end flow tested

---

**Version**: 1.0  
**Status**: EXECUTE  
**Last Updated**: 2025-10-09
