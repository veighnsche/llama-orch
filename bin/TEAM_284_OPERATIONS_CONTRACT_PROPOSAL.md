# TEAM-284: Operations Contract Proposal

**Date:** Oct 24, 2025  
**Status:** 💡 **PROPOSAL**

## Problem: Duplication Between Queen and Hive

You're absolutely right - there's significant duplication:

### Current Architecture (DUPLICATED!)

```text
rbee-operations (shared)
    ↓
    ├─→ queen-rbee/hive_forwarder.rs (forwards operations to hive)
    │   - Knows which operations go to hive
    │   - HTTP client code
    │   - Job submission logic
    │
    └─→ rbee-hive/job_router.rs (receives operations from queen)
        - Knows which operations it handles
        - HTTP server code
        - Same job routing logic
```

**The duplication:**
1. Both know about the same operations
2. Both have job routing logic
3. Both handle Operation enum
4. Both use job-server pattern
5. No shared contract for the HTTP API

## What Should Be in a Contract?

### Operations Contract Should Include:

```text
operations-contract/
├── types.rs
│   └── Operation enum (ALREADY EXISTS in rbee-operations!)
│
├── requests.rs (NEW!)
│   ├── WorkerSpawnRequest
│   ├── ModelDownloadRequest
│   └── etc.
│
├── responses.rs (NEW!)
│   ├── WorkerSpawnResponse
│   ├── ModelListResponse
│   └── etc.
│
├── api.rs (NEW!)
│   ├── POST /v1/jobs endpoint spec
│   ├── GET /v1/jobs/{id}/stream endpoint spec
│   └── Error response format
│
└── routing.rs (NEW!)
    └── Trait for operation handlers
```

## Proposed Refactor

### 1. Rename rbee-operations → operations-contract

```toml
# bin/99_shared_crates/operations-contract/
[package]
name = "operations-contract"
```

**Why:** It's not just operations, it's the contract between queen and hive!

### 2. Add Request/Response Types

```rust
// operations-contract/src/requests.rs

/// Request to spawn a worker
pub struct WorkerSpawnRequest {
    pub hive_id: String,
    pub model: String,
    pub worker: String,
    pub device: u32,
}

/// Request to download a model
pub struct ModelDownloadRequest {
    pub hive_id: String,
    pub model: String,
}

// etc.
```

```rust
// operations-contract/src/responses.rs

/// Response from worker spawn
pub struct WorkerSpawnResponse {
    pub worker_id: String,
    pub port: u16,
    pub status: String,
}

/// Response from model list
pub struct ModelListResponse {
    pub models: Vec<ModelInfo>,
}

// etc.
```

### 3. Add API Specification

```rust
// operations-contract/src/api.rs

/// Hive API specification
///
/// All hives must implement these endpoints
pub struct HiveApiSpec;

impl HiveApiSpec {
    /// Create job endpoint
    pub const CREATE_JOB: &'static str = "/v1/jobs";
    
    /// Stream job results endpoint
    pub const STREAM_JOB: &'static str = "/v1/jobs/{id}/stream";
    
    /// Job response format
    pub fn job_response(job_id: String) -> JobResponse {
        JobResponse {
            job_id: job_id.clone(),
            sse_url: format!("/v1/jobs/{}/stream", job_id),
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct JobResponse {
    pub job_id: String,
    pub sse_url: String,
}
```

### 4. Add Operation Handler Trait

```rust
// operations-contract/src/routing.rs

/// Trait for handling operations
///
/// Both queen and hive implement this trait
#[async_trait]
pub trait OperationHandler {
    /// Handle an operation
    async fn handle(&self, operation: Operation) -> Result<OperationResult>;
}

/// Result of an operation
pub enum OperationResult {
    Success(serde_json::Value),
    Error(String),
    Stream(Box<dyn Stream<Item = String>>),
}
```

## Benefits of This Refactor

### 1. Single Source of Truth

```text
BEFORE:
- queen knows: "WorkerSpawn should go to hive"
- hive knows: "I handle WorkerSpawn"
- NO shared contract

AFTER:
- operations-contract knows: "WorkerSpawn is a hive operation"
- Both queen and hive use the contract
- Shared request/response types
```

### 2. Type Safety

```rust
// BEFORE (queen):
let payload = json!({
    "operation": "worker_spawn",
    "hive_id": hive_id,
    "model": model,
    // ... hope we got the fields right!
});

// AFTER (queen):
let request = WorkerSpawnRequest {
    hive_id,
    model,
    worker,
    device,
};
let operation = Operation::WorkerSpawn(request);
// Compiler ensures we have all fields!
```

### 3. Easier to Add Operations

```text
BEFORE (3 places to update):
1. rbee-operations/src/lib.rs - Add Operation variant
2. queen-rbee/hive_forwarder.rs - Add to should_forward_to_hive()
3. rbee-hive/job_router.rs - Add match arm

AFTER (1 place to update):
1. operations-contract/src/lib.rs - Add Operation variant with request/response
   - should_forward_to_hive() is automatic (trait method)
   - Both queen and hive use the same types
```

### 4. Testability

```rust
// operations-contract/tests/roundtrip.rs

#[test]
fn test_worker_spawn_roundtrip() {
    let request = WorkerSpawnRequest { /* ... */ };
    let operation = Operation::WorkerSpawn(request);
    
    // Serialize (queen sends)
    let json = serde_json::to_string(&operation).unwrap();
    
    // Deserialize (hive receives)
    let received: Operation = serde_json::from_str(&json).unwrap();
    
    // Verify
    assert_eq!(operation, received);
}
```

## Comparison to Other Contracts

### We Already Have This Pattern!

```text
worker-contract/
├── types.rs → WorkerInfo
├── heartbeat.rs → WorkerHeartbeat
└── api.rs → Worker API endpoints

hive-contract/
├── types.rs → HiveInfo
├── heartbeat.rs → HiveHeartbeat
└── api.rs → Hive API endpoints

operations-contract/ (PROPOSED)
├── types.rs → Operation enum
├── requests.rs → Request types
├── responses.rs → Response types
├── api.rs → Job API endpoints
└── routing.rs → Handler trait
```

**Same pattern, different purpose!**

## Current State Analysis

### What's Already Shared

✅ `rbee-operations` - Operation enum  
✅ `job-server` - Job registry pattern  
✅ `job-client` - HTTP client for jobs  

### What's Duplicated

❌ Job routing logic (queen forwards, hive routes)  
❌ Request/response types (implicit in Operation variants)  
❌ API endpoint paths (hardcoded in multiple places)  
❌ Error handling (different in queen vs hive)  

### What's Missing

❌ Shared request/response types  
❌ API specification  
❌ Handler trait  
❌ Validation logic  

## Implementation Plan

### Phase 1: Add Request/Response Types

1. Create `operations-contract/src/requests.rs`
2. Create `operations-contract/src/responses.rs`
3. Update Operation enum to use these types
4. Update queen and hive to use typed requests

### Phase 2: Add API Specification

1. Create `operations-contract/src/api.rs`
2. Define endpoint constants
3. Define response formats
4. Update queen and hive to use constants

### Phase 3: Add Handler Trait

1. Create `operations-contract/src/routing.rs`
2. Define OperationHandler trait
3. Implement trait in queen (forwarding)
4. Implement trait in hive (execution)

### Phase 4: Consolidate Logic

1. Move `should_forward_to_hive()` to trait method
2. Remove duplication between queen and hive
3. Add shared validation logic
4. Add shared error handling

## Example: Worker Spawn (Before vs After)

### BEFORE

```rust
// rbee-operations/src/lib.rs
pub enum Operation {
    WorkerSpawn {
        hive_id: String,
        model: String,
        worker: String,
        device: u32,
    },
}

// queen-rbee/hive_forwarder.rs
pub async fn forward_to_hive(operation: Operation) -> Result<()> {
    let json = serde_json::to_value(&operation)?;
    // ... HTTP POST to hive
}

// rbee-hive/job_router.rs
match operation {
    Operation::WorkerSpawn { hive_id, model, worker, device } => {
        // ... spawn worker
    }
}
```

### AFTER

```rust
// operations-contract/src/requests.rs
#[derive(Serialize, Deserialize)]
pub struct WorkerSpawnRequest {
    pub hive_id: String,
    pub model: String,
    pub worker: String,
    pub device: u32,
}

// operations-contract/src/responses.rs
#[derive(Serialize, Deserialize)]
pub struct WorkerSpawnResponse {
    pub worker_id: String,
    pub port: u16,
}

// operations-contract/src/lib.rs
pub enum Operation {
    WorkerSpawn(WorkerSpawnRequest),
}

impl Operation {
    pub fn is_hive_operation(&self) -> bool {
        matches!(self, Operation::WorkerSpawn(_) | /* ... */)
    }
}

// queen-rbee/hive_forwarder.rs
pub async fn forward_to_hive(operation: Operation) -> Result<WorkerSpawnResponse> {
    let json = serde_json::to_value(&operation)?;
    let response: WorkerSpawnResponse = client.post(url).json(&json).send().await?.json().await?;
    Ok(response)
}

// rbee-hive/job_router.rs
match operation {
    Operation::WorkerSpawn(request) => {
        let response = spawn_worker(request).await?;
        Ok(OperationResult::Success(serde_json::to_value(response)?))
    }
}
```

## Recommendation

**YES, you should absolutely refactor this!**

1. ✅ Rename `rbee-operations` → `operations-contract`
2. ✅ Add request/response types
3. ✅ Add API specification
4. ✅ Add handler trait
5. ✅ Remove duplication

This will make the codebase:
- **More maintainable** - Single source of truth
- **More type-safe** - Compiler-checked contracts
- **Easier to extend** - Add operations in one place
- **Better tested** - Shared test suite

## Next Steps

1. Create proposal document (this file)
2. Get feedback on approach
3. Implement Phase 1 (request/response types)
4. Migrate queen and hive to use new types
5. Implement remaining phases

**This is exactly the kind of refactoring that makes sense!**
