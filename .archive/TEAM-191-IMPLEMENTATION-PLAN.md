# TEAM-191: Implementation Plan - Operations Split & Job Server

**Date:** 2025-10-21  
**Status:** 📋 Planning

## Overview

Split operations into two enums and create shared job-server infrastructure.

## Phase 1: Split Operations ⚡ (2-3 hours)

### Step 1.1: Create hive-operations crate
```bash
mkdir -p bin/99_shared_crates/hive-operations/src
```

**File:** `bin/99_shared_crates/hive-operations/Cargo.toml`
```toml
[package]
name = "hive-operations"
version = "0.1.0"
edition = "2021"

[dependencies]
serde = { version = "1.0", features = ["derive"] }
```

**File:** `bin/99_shared_crates/hive-operations/src/lib.rs`
```rust
//! Hive operations (queen-rbee → rbee-hive contract)
//!
//! TEAM-191: Split from rbee-operations
//! These operations are forwarded from queen to hive

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "operation", rename_all = "snake_case")]
pub enum HiveOperation {
    // Worker operations
    WorkerSpawn {
        model: String,
        worker: String,
        device: u32,
    },
    WorkerList,
    WorkerGet {
        id: String,
    },
    WorkerDelete {
        id: String,
    },

    // Model operations
    ModelDownload {
        model: String,
    },
    ModelList,
    ModelGet {
        id: String,
    },
    ModelDelete {
        id: String,
    },

    // Inference operation
    Infer {
        model: String,
        prompt: String,
        max_tokens: u32,
        temperature: f32,
        #[serde(skip_serializing_if = "Option::is_none")]
        top_p: Option<f32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        top_k: Option<u32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        device: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        worker_id: Option<String>,
        #[serde(default = "default_stream")]
        stream: bool,
    },
}

fn default_stream() -> bool {
    true
}

impl HiveOperation {
    pub fn name(&self) -> &'static str {
        match self {
            HiveOperation::WorkerSpawn { .. } => "worker_spawn",
            HiveOperation::WorkerList => "worker_list",
            HiveOperation::WorkerGet { .. } => "worker_get",
            HiveOperation::WorkerDelete { .. } => "worker_delete",
            HiveOperation::ModelDownload { .. } => "model_download",
            HiveOperation::ModelList => "model_list",
            HiveOperation::ModelGet { .. } => "model_get",
            HiveOperation::ModelDelete { .. } => "model_delete",
            HiveOperation::Infer { .. } => "infer",
        }
    }
}
```

### Step 1.2: Update rbee-operations to QueenOperation

**File:** `bin/99_shared_crates/rbee-operations/src/lib.rs`

Remove hive operations, rename to `QueenOperation`:
```rust
//! Queen operations (rbee-keeper → queen-rbee contract)
//!
//! TEAM-191: Split from unified Operation enum
//! These operations are handled by queen-rbee

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "operation", rename_all = "snake_case")]
pub enum QueenOperation {
    // System-wide operations
    Status,

    // Hive lifecycle management
    SshTest {
        ssh_host: String,
        #[serde(default = "default_ssh_port")]
        ssh_port: u16,
        ssh_user: String,
    },
    HiveInstall {
        hive_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        ssh_host: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        ssh_port: Option<u16>,
        #[serde(skip_serializing_if = "Option::is_none")]
        ssh_user: Option<String>,
        #[serde(default = "default_port")]
        port: u16,
        #[serde(skip_serializing_if = "Option::is_none")]
        binary_path: Option<String>,
    },
    HiveUninstall {
        hive_id: String,
        #[serde(default)]
        catalog_only: bool,
    },
    HiveUpdate {
        hive_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        ssh_host: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        ssh_port: Option<u16>,
        #[serde(skip_serializing_if = "Option::is_none")]
        ssh_user: Option<String>,
        #[serde(default)]
        refresh_capabilities: bool,
    },
    HiveStart {
        #[serde(default = "default_hive_id")]
        hive_id: String,
    },
    HiveStop {
        #[serde(default = "default_hive_id")]
        hive_id: String,
    },
    HiveList,
    HiveGet {
        #[serde(default = "default_hive_id")]
        hive_id: String,
    },
    HiveStatus {
        #[serde(default = "default_hive_id")]
        hive_id: String,
    },
    
    // TEAM-191: Hive operations (forwarded to rbee-hive)
    // These contain hive_id for routing
    ForwardToHive {
        hive_id: String,
        #[serde(flatten)]
        operation: hive_operations::HiveOperation,
    },
}
```

### Step 1.3: Update queen-rbee imports

**Files to update:**
- `bin/10_queen_rbee/src/job_router.rs`
- `bin/00_rbee_keeper/src/main.rs`

Change:
```rust
use rbee_operations::Operation;
```

To:
```rust
use rbee_operations::QueenOperation;
use hive_operations::HiveOperation;
```

## Phase 2: Create job-server Crate 🏗️ (4-6 hours)

### Step 2.1: Create crate structure

```bash
mkdir -p bin/99_shared_crates/job-server/src
```

**File:** `bin/99_shared_crates/job-server/Cargo.toml`
```toml
[package]
name = "job-server"
version = "0.1.0"
edition = "2021"

[dependencies]
# HTTP framework
axum = { workspace = true }
tokio = { workspace = true }

# Streaming
futures = { workspace = true }
async-stream = "0.3"

# Serialization
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

# Error handling
anyhow = "1.0"

# Shared crates
job-registry = { path = "../job-registry" }
observability-narration-core = { path = "../narration-core" }
```

### Step 2.2: Define core traits

**File:** `bin/99_shared_crates/job-server/src/traits.rs`
```rust
//! Core traits for job-based server architecture

use anyhow::Result;
use job_registry::JobRegistry;
use serde::de::DeserializeOwned;
use std::sync::Arc;

/// Trait for routing operations to handlers
pub trait OperationRouter: Send + Sync + 'static {
    /// The operation type this router handles
    type Operation: DeserializeOwned + Send + Sync;
    
    /// Route an operation to its handler
    async fn route_operation(&self, operation: Self::Operation) -> Result<()>;
}

/// Trait for job state (required for HTTP handlers)
pub trait JobState: Clone + Send + Sync + 'static {
    /// Get the job registry
    fn registry(&self) -> Arc<JobRegistry<String>>;
    
    /// Get the router for this state
    fn router(&self) -> Arc<dyn OperationRouter<Operation = Self::Operation>>;
    
    /// The operation type
    type Operation: DeserializeOwned + Send + Sync;
}
```

### Step 2.3: Implement HTTP handlers

**File:** `bin/99_shared_crates/job-server/src/http.rs`
```rust
//! Generic HTTP handlers for job-based architecture

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::sse::{Event, Sse},
    Json,
};
use futures::stream::Stream;
use observability_narration_core::sse_sink;
use serde::{Deserialize, Serialize};
use std::convert::Infallible;

use crate::traits::JobState;

/// Response from job creation
#[derive(Debug, Serialize, Deserialize)]
pub struct JobResponse {
    pub job_id: String,
    pub sse_url: String,
}

/// POST /v1/jobs - Create a new job
pub async fn handle_create_job<S>(
    State(state): State<S>,
    Json(payload): Json<serde_json::Value>,
) -> Result<Json<JobResponse>, (StatusCode, String)>
where
    S: JobState,
{
    // Create job in registry
    let job_id = state.registry().create_job(payload);
    
    // Return job ID and SSE URL
    let response = JobResponse {
        job_id: job_id.clone(),
        sse_url: format!("/v1/jobs/{}/stream", job_id),
    };
    
    Ok(Json(response))
}

/// GET /v1/jobs/{job_id}/stream - Stream job results via SSE
pub async fn handle_stream_job<S>(
    Path(job_id): Path<String>,
    State(state): State<S>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>>
where
    S: JobState,
{
    // Subscribe to SSE broadcaster
    let mut sse_rx = sse_sink::subscribe().expect("SSE sink not initialized");
    
    // Trigger job execution
    let registry = state.registry();
    let router = state.router();
    
    tokio::spawn(async move {
        if let Some(payload) = registry.get_job(&job_id) {
            // Parse operation
            if let Ok(operation) = serde_json::from_value::<S::Operation>(payload) {
                // Route to handler
                let _ = router.route_operation(operation).await;
            }
        }
    });
    
    // Give background task time to start
    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
    
    // Stream SSE events
    let stream = async_stream::stream! {
        let mut last_event_time = std::time::Instant::now();
        let completion_timeout = std::time::Duration::from_millis(2000);
        let mut received_first_event = false;
        
        loop {
            let timeout_fut = tokio::time::sleep(completion_timeout);
            tokio::pin!(timeout_fut);
            
            tokio::select! {
                result = sse_rx.recv() => {
                    match result {
                        Ok(event) => {
                            received_first_event = true;
                            last_event_time = std::time::Instant::now();
                            let formatted = format!("[{}] {}", event.actor, event.human);
                            yield Ok(Event::default().data(formatted));
                        }
                        Err(_) => {
                            if received_first_event {
                                yield Ok(Event::default().data("[DONE]"));
                            }
                            break;
                        }
                    }
                }
                _ = &mut timeout_fut, if received_first_event => {
                    if last_event_time.elapsed() >= completion_timeout {
                        yield Ok(Event::default().data("[DONE]"));
                        break;
                    }
                }
            }
        }
    };
    
    Sse::new(stream)
}
```

### Step 2.4: Create lib.rs

**File:** `bin/99_shared_crates/job-server/src/lib.rs`
```rust
//! Generic job-based server infrastructure
//!
//! TEAM-191: Shared between queen-rbee and rbee-hive

pub mod http;
pub mod traits;

pub use http::{handle_create_job, handle_stream_job, JobResponse};
pub use traits::{JobState, OperationRouter};
```

## Phase 3: Refactor queen-rbee 🔄 (3-4 hours)

### Step 3.1: Implement traits

**File:** `bin/10_queen_rbee/src/job_router.rs`

Add trait implementations:
```rust
use job_server::{OperationRouter, JobState as JobStateTrait};

impl OperationRouter for JobState {
    type Operation = QueenOperation;
    
    async fn route_operation(&self, operation: QueenOperation) -> Result<()> {
        route_operation(
            serde_json::to_value(&operation)?,
            self.registry.clone(),
            self.hive_catalog.clone(),
            self.hive_registry.clone(),
        ).await
    }
}

impl JobStateTrait for JobState {
    type Operation = QueenOperation;
    
    fn registry(&self) -> Arc<JobRegistry<String>> {
        self.registry.clone()
    }
    
    fn router(&self) -> Arc<dyn OperationRouter<Operation = QueenOperation>> {
        Arc::new(self.clone())
    }
}
```

### Step 3.2: Update HTTP handlers

**File:** `bin/10_queen_rbee/src/http/jobs.rs`

Replace custom handlers with job-server:
```rust
use job_server::{handle_create_job, handle_stream_job};

// Remove custom implementations
// Use job-server handlers directly in router
```

### Step 3.3: Update router

**File:** `bin/10_queen_rbee/src/main.rs`

```rust
use job_server::{handle_create_job, handle_stream_job};

Router::new()
    .route("/v1/jobs", post(handle_create_job::<JobState>))
    .route("/v1/jobs/{job_id}/stream", get(handle_stream_job::<JobState>))
```

## Phase 4: Implement rbee-hive Job Architecture 🐝 (6-8 hours)

### Step 4.1: Add dependencies

**File:** `bin/20_rbee_hive/Cargo.toml`
```toml
[dependencies]
# Job infrastructure
job-registry = { path = "../99_shared_crates/job-registry" }
job-server = { path = "../99_shared_crates/job-server" }
hive-operations = { path = "../99_shared_crates/hive-operations" }
```

### Step 4.2: Create job state

**File:** `bin/20_rbee_hive/src/job_router.rs`
```rust
use hive_operations::HiveOperation;
use job_server::{OperationRouter, JobState as JobStateTrait};
use job_registry::JobRegistry;
use std::sync::Arc;

#[derive(Clone)]
pub struct JobState {
    pub registry: Arc<JobRegistry<String>>,
    // TODO: Add worker_registry, model_catalog when implemented
}

impl OperationRouter for JobState {
    type Operation = HiveOperation;
    
    async fn route_operation(&self, operation: HiveOperation) -> Result<()> {
        match operation {
            HiveOperation::WorkerSpawn { .. } => {
                // TODO: Implement
            }
            HiveOperation::WorkerList => {
                // TODO: Implement
            }
            // ... other operations
        }
        Ok(())
    }
}

impl JobStateTrait for JobState {
    type Operation = HiveOperation;
    
    fn registry(&self) -> Arc<JobRegistry<String>> {
        self.registry.clone()
    }
    
    fn router(&self) -> Arc<dyn OperationRouter<Operation = HiveOperation>> {
        Arc::new(self.clone())
    }
}
```

### Step 4.3: Add HTTP endpoints

**File:** `bin/20_rbee_hive/src/main.rs`
```rust
use job_server::{handle_create_job, handle_stream_job};

// Initialize job registry
let job_registry = Arc::new(JobRegistry::new());

let job_state = JobState {
    registry: job_registry,
};

// Create router
let app = Router::new()
    .route("/health", get(health_check))
    .route("/v1/jobs", post(handle_create_job::<JobState>))
    .route("/v1/jobs/{job_id}/stream", get(handle_stream_job::<JobState>))
    .with_state(job_state);
```

## Phase 5: Implement Operation Forwarding 🔀 (4-6 hours)

### Step 5.1: Create forwarding client

**File:** `bin/10_queen_rbee/src/hive_client.rs`
```rust
//! Client for forwarding operations to hive

use anyhow::Result;
use hive_operations::HiveOperation;
use job_server::JobResponse;
use queen_rbee_hive_catalog::HiveRecord;

pub struct HiveClient {
    client: reqwest::Client,
}

impl HiveClient {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
        }
    }
    
    pub async fn forward_operation(
        &self,
        hive: &HiveRecord,
        operation: HiveOperation,
    ) -> Result<()> {
        // 1. POST to hive's /v1/jobs
        let response = self.client
            .post(format!("http://{}:{}/v1/jobs", hive.host, hive.port))
            .json(&operation)
            .send()
            .await?;
        
        let job_response: JobResponse = response.json().await?;
        
        // 2. Stream from hive's SSE endpoint
        let mut sse_stream = self.client
            .get(format!("http://{}:{}{}", hive.host, hive.port, job_response.sse_url))
            .send()
            .await?
            .bytes_stream();
        
        // 3. Forward events (they'll go through sse_sink automatically)
        // Events from hive are already formatted, just pass through
        
        Ok(())
    }
}
```

### Step 5.2: Update queen router

**File:** `bin/10_queen_rbee/src/job_router.rs`
```rust
match operation {
    QueenOperation::Status => { /* ... */ }
    QueenOperation::HiveList => { /* ... */ }
    
    QueenOperation::ForwardToHive { hive_id, operation } => {
        // Get hive from catalog
        let hive = state.hive_catalog.get(&hive_id)
            .ok_or_else(|| anyhow::anyhow!("Hive {} not found", hive_id))?;
        
        // Forward to hive
        let client = HiveClient::new();
        client.forward_operation(&hive, operation).await?;
    }
}
```

## Testing Strategy

### Unit Tests
- ✅ Operation serialization/deserialization
- ✅ Trait implementations
- ✅ HTTP handler responses

### Integration Tests
- ✅ Queen job creation and streaming
- ✅ Hive job creation and streaming
- ✅ Queen → Hive forwarding
- ✅ End-to-end: rbee-keeper → queen → hive

### Manual Tests
```bash
# Test queen operations
./rbee status
./rbee hive list

# Test hive operations (once forwarding is implemented)
./rbee worker list
./rbee model list
./rbee infer --model llama-3-8b --prompt "hello"
```

## Rollout Plan

1. **Phase 1**: Split operations (no breaking changes to functionality)
2. **Phase 2**: Create job-server (isolated, no integration yet)
3. **Phase 3**: Refactor queen to use job-server (test thoroughly)
4. **Phase 4**: Add job architecture to hive (test in isolation)
5. **Phase 5**: Enable forwarding (test end-to-end)

## Success Criteria

- ✅ All existing operations still work
- ✅ Queen and hive use identical job architecture
- ✅ ~350 lines of code eliminated
- ✅ Type-safe operation separation
- ✅ Forwarding works end-to-end
- ✅ All tests passing

## Estimated Timeline

- **Phase 1**: 2-3 hours
- **Phase 2**: 4-6 hours
- **Phase 3**: 3-4 hours
- **Phase 4**: 6-8 hours
- **Phase 5**: 4-6 hours

**Total: 19-27 hours** (~3-4 days)

---

**Ready to start with Phase 1?**
