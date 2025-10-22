# Crate Interface Standard - Robust Proposal

**Version:** 1.0  
**Date:** 2025-10-20  
**Status:** PROPOSAL

---

## Executive Summary

**Problem:** Inconsistent crate interfaces make HTTP wrappers unpredictable and hard to maintain.

**Solution:** Establish a **standard interface pattern** that ALL crates must follow.

**Benefits:**
- Predictable HTTP wrapper shape
- Consistent error handling
- Easy to understand and maintain
- Self-documenting code

---

## Table of Contents

1. [Current State Analysis](#current-state-analysis)
2. [The Standard Pattern](#the-standard-pattern)
3. [Crate Categories](#crate-categories)
4. [Implementation Guide](#implementation-guide)
5. [Migration Plan](#migration-plan)
6. [Examples](#examples)

---

## Current State Analysis

### Existing Crates Inventory

#### Queen-Specific Crates (`15_queen_rbee_crates/`)
1. **hive-lifecycle** - Hive orchestration
2. **scheduler** - Job orchestration
3. **hive-catalog** - Hive persistence (CRUD)
4. **hive-registry** - Hive state management
5. **worker-registry** - Worker state management
6. **health** - Health check
7. **preflight** - Pre-flight checks
8. **ssh-client** - SSH operations

#### Shared Crates (`99_shared_crates/`)
1. **heartbeat** - Heartbeat protocol
2. **job-registry** - Job state management
3. **daemon-lifecycle** - Process spawning
4. **narration-core** - Observability
5. **timeout-enforcer** - Timeout management
6. **rbee-http-client** - HTTP client (stub)
7. **auth-min** - Authentication
8. **jwt-guardian** - JWT handling
9. **audit-logging** - Audit trails
10. **input-validation** - Input validation
11. **secrets-management** - Secret handling
12. **model-catalog** - Model metadata
13. **sse-relay** - SSE streaming
14. **deadline-propagation** - Deadline tracking
15. **rbee-types** - Shared types
16. **hive-core** - Core hive logic

### Current Interface Patterns

```rust
// Pattern 1: Returns primitive (INCONSISTENT)
ensure_hive_running(...) -> Result<String>

// Pattern 2: Returns structured response (GOOD)
orchestrate_job(...) -> Result<JobResponse>

// Pattern 3: Returns structured response with generics (GOOD)
handle_hive_heartbeat<C, D>(...) -> Result<HeartbeatAcknowledgement>

// Pattern 4: CRUD operations (GOOD)
add_hive(...) -> Result<()>
get_hive(...) -> Result<Option<HiveRecord>>
list_hives(...) -> Result<Vec<HiveRecord>>

// Pattern 5: Registry operations (GOOD)
create_job() -> String
set_token_receiver(...) -> ()
take_token_receiver(...) -> Option<TokenReceiver<T>>
```

---

## The Standard Pattern

### Core Principle

**Every crate entrypoint must follow ONE of these patterns:**

1. **Command Pattern** - Do something, return structured result
2. **Query Pattern** - Get something, return data
3. **CRUD Pattern** - Standard database operations
4. **Registry Pattern** - State management operations

---

### Pattern 1: Command Pattern

**Use when:** Performing an action that returns a result

```rust
// In crate (pure business logic):
#[derive(Debug, Clone)]
pub struct XxxResponse {
    // All relevant output data
    pub field1: Type1,
    pub field2: Type2,
}

#[derive(Debug, Clone)]
pub struct XxxRequest {
    // All input parameters
    pub param1: Type1,
    pub param2: Type2,
}

pub async fn execute_xxx(
    dependencies: Dependencies,  // Arc<T> for shared state
    request: XxxRequest,
) -> Result<XxxResponse> {
    // Business logic
    Ok(XxxResponse { ... })
}
```

**HTTP wrapper:**
```rust
// In http.rs:
#[derive(Deserialize)]
pub struct HttpXxxRequest {
    pub param1: Type1,
    pub param2: Type2,
}

#[derive(Serialize)]
pub struct HttpXxxResponse {
    pub field1: Type1,
    pub field2: Type2,
}

pub async fn handle_xxx(
    State(deps): State<Dependencies>,
    Json(req): Json<HttpXxxRequest>,
) -> Result<Json<HttpXxxResponse>, (StatusCode, String)> {
    let request = crate::XxxRequest {
        param1: req.param1,
        param2: req.param2,
    };
    
    let response = crate::execute_xxx(deps, request).await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    
    Ok(Json(HttpXxxResponse {
        field1: response.field1,
        field2: response.field2,
    }))
}
```

---

### Pattern 2: Query Pattern

**Use when:** Retrieving data without side effects

```rust
// In crate:
pub async fn get_xxx(
    store: Arc<XxxStore>,
    id: &str,
) -> Result<Option<Xxx>> {
    // Query logic
}

pub async fn list_xxx(
    store: Arc<XxxStore>,
    filter: Option<XxxFilter>,
) -> Result<Vec<Xxx>> {
    // List logic
}
```

**HTTP wrapper:**
```rust
// In http.rs:
pub async fn handle_get_xxx(
    State(store): State<Arc<XxxStore>>,
    Path(id): Path<String>,
) -> Result<Json<Option<Xxx>>, (StatusCode, String)> {
    crate::get_xxx(store, &id).await
        .map(Json)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
}
```

---

### Pattern 3: CRUD Pattern

**Use when:** Managing persistent entities

```rust
// In crate:
pub struct XxxCatalog {
    pool: SqlitePool,
}

impl XxxCatalog {
    // CREATE
    pub async fn add_xxx(&self, xxx: XxxRecord) -> Result<()> { ... }
    
    // READ
    pub async fn get_xxx(&self, id: &str) -> Result<Option<XxxRecord>> { ... }
    pub async fn list_xxx(&self) -> Result<Vec<XxxRecord>> { ... }
    
    // UPDATE
    pub async fn update_xxx(&self, xxx: XxxRecord) -> Result<()> { ... }
    pub async fn update_xxx_status(&self, id: &str, status: XxxStatus) -> Result<()> { ... }
    
    // DELETE
    pub async fn remove_xxx(&self, id: &str) -> Result<()> { ... }
}
```

**HTTP wrappers:**
```rust
// POST /xxx
pub async fn handle_create_xxx(...) -> Result<(StatusCode, Json<XxxRecord>), ...>

// GET /xxx/:id
pub async fn handle_get_xxx(...) -> Result<Json<Option<XxxRecord>>, ...>

// GET /xxx
pub async fn handle_list_xxx(...) -> Result<Json<Vec<XxxRecord>>, ...>

// PUT /xxx/:id
pub async fn handle_update_xxx(...) -> Result<StatusCode, ...>

// DELETE /xxx/:id
pub async fn handle_delete_xxx(...) -> Result<StatusCode, ...>
```

---

### Pattern 4: Registry Pattern

**Use when:** Managing in-memory state

```rust
// In crate:
pub struct XxxRegistry<T> {
    items: Arc<Mutex<HashMap<String, T>>>,
}

impl<T> XxxRegistry<T> {
    pub fn new() -> Self { ... }
    
    // Create
    pub fn create_xxx(&self) -> String { ... }  // Returns ID
    
    // Read
    pub fn has_xxx(&self, id: &str) -> bool { ... }
    pub fn get_xxx(&self, id: &str) -> Option<T> { ... }
    
    // Update
    pub fn update_xxx(&self, id: &str, value: T) { ... }
    
    // Delete
    pub fn remove_xxx(&self, id: &str) -> Option<T> { ... }
    
    // Utility
    pub fn count(&self) -> usize { ... }
    pub fn ids(&self) -> Vec<String> { ... }
}
```

**Note:** Registries are usually NOT exposed via HTTP directly. They're used by Command Pattern functions.

---

## Crate Categories

### Category A: Orchestration Crates

**Purpose:** Execute complex business logic workflows

**Pattern:** Command Pattern

**Examples:**
- `hive-lifecycle` - Orchestrate hive spawning
- `scheduler` - Orchestrate job distribution
- `preflight` - Execute pre-flight checks

**Standard Interface:**
```rust
pub struct XxxRequest { ... }
pub struct XxxResponse { ... }

pub async fn execute_xxx(
    deps: Dependencies,
    request: XxxRequest,
) -> Result<XxxResponse>
```

---

### Category B: Data Management Crates

**Purpose:** Persist and retrieve data

**Pattern:** CRUD Pattern

**Examples:**
- `hive-catalog` - Hive persistence
- `model-catalog` - Model metadata
- `audit-logging` - Audit trails

**Standard Interface:**
```rust
pub struct XxxCatalog { ... }

impl XxxCatalog {
    pub async fn add_xxx(&self, xxx: Xxx) -> Result<()>
    pub async fn get_xxx(&self, id: &str) -> Result<Option<Xxx>>
    pub async fn list_xxx(&self) -> Result<Vec<Xxx>>
    pub async fn update_xxx(&self, xxx: Xxx) -> Result<()>
    pub async fn remove_xxx(&self, id: &str) -> Result<()>
}
```

---

### Category C: State Management Crates

**Purpose:** Manage in-memory state

**Pattern:** Registry Pattern

**Examples:**
- `job-registry` - Job state
- `hive-registry` - Hive state
- `worker-registry` - Worker state

**Standard Interface:**
```rust
pub struct XxxRegistry<T> { ... }

impl<T> XxxRegistry<T> {
    pub fn create_xxx(&self) -> String
    pub fn get_xxx(&self, id: &str) -> Option<T>
    pub fn update_xxx(&self, id: &str, value: T)
    pub fn remove_xxx(&self, id: &str) -> Option<T>
}
```

---

### Category D: Protocol Crates

**Purpose:** Implement communication protocols

**Pattern:** Command Pattern with traits

**Examples:**
- `heartbeat` - Heartbeat protocol
- `sse-relay` - SSE streaming
- `deadline-propagation` - Deadline tracking

**Standard Interface:**
```rust
pub trait XxxHandler {
    async fn handle_xxx(&self, payload: XxxPayload) -> Result<XxxResponse>;
}

pub async fn handle_xxx<H: XxxHandler>(
    handler: Arc<H>,
    payload: XxxPayload,
) -> Result<XxxResponse>
```

---

### Category E: Utility Crates

**Purpose:** Provide helper functionality

**Pattern:** Function-based (no HTTP)

**Examples:**
- `daemon-lifecycle` - Process spawning
- `timeout-enforcer` - Timeout management
- `narration-core` - Observability
- `auth-min` - Authentication
- `jwt-guardian` - JWT handling
- `input-validation` - Validation
- `secrets-management` - Secret handling

**Standard Interface:**
```rust
// Pure functions, no HTTP wrappers needed
pub fn do_xxx(params: Params) -> Result<Output>
pub async fn do_xxx_async(params: Params) -> Result<Output>
```

---

## Implementation Guide

### Step 1: Classify Your Crate

Determine which category your crate belongs to:
- Orchestration → Command Pattern
- Data Management → CRUD Pattern
- State Management → Registry Pattern
- Protocol → Command Pattern with traits
- Utility → Function-based

### Step 2: Define Request/Response Types

```rust
// For Command Pattern:
#[derive(Debug, Clone)]
pub struct XxxRequest {
    // NO serde derives (pure domain type)
    pub field1: Type1,
}

#[derive(Debug, Clone)]
pub struct XxxResponse {
    // NO serde derives (pure domain type)
    pub field1: Type1,
}
```

### Step 3: Implement Core Function

```rust
pub async fn execute_xxx(
    deps: Dependencies,
    request: XxxRequest,
) -> Result<XxxResponse> {
    // Narration for observability
    Narration::new(ACTOR, ACTION, &request.field1)
        .human("Executing xxx")
        .emit();
    
    // Business logic
    
    // Return structured response
    Ok(XxxResponse { ... })
}
```

### Step 4: Create HTTP Wrapper (if needed)

```rust
// In http.rs:
#[derive(Deserialize)]  // serde ONLY in HTTP layer
pub struct HttpXxxRequest {
    pub field1: Type1,
}

#[derive(Serialize)]  // serde ONLY in HTTP layer
pub struct HttpXxxResponse {
    pub field1: Type1,
}

pub async fn handle_xxx(
    State(deps): State<Dependencies>,
    Json(req): Json<HttpXxxRequest>,
) -> Result<Json<HttpXxxResponse>, (StatusCode, String)> {
    // Convert HTTP → Domain
    let request = crate::XxxRequest {
        field1: req.field1,
    };
    
    // Call pure business logic
    let response = crate::execute_xxx(deps, request).await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    
    // Convert Domain → HTTP
    Ok(Json(HttpXxxResponse {
        field1: response.field1,
    }))
}
```

---

## Migration Plan

### Phase 1: Fix Asymmetric Crates (Immediate)

**Target:** `hive-lifecycle`

**Current:**
```rust
pub async fn ensure_hive_running(...) -> Result<String>
```

**New:**
```rust
#[derive(Debug, Clone)]
pub struct HiveStartResponse {
    pub hive_url: String,
    pub hive_id: String,
    pub port: u16,
}

pub async fn ensure_hive_running(...) -> Result<HiveStartResponse>
```

**Impact:** HTTP wrapper becomes consistent with other wrappers

---

### Phase 2: Standardize Naming (Next)

**Current:** Inconsistent function names
- `ensure_hive_running` (verb + noun)
- `orchestrate_job` (verb + noun)
- `handle_hive_heartbeat` (verb + noun + noun)

**New:** Consistent naming convention
- `execute_hive_start` (execute + noun + action)
- `execute_job_orchestration` (execute + noun + action)
- `execute_heartbeat_processing` (execute + noun + action)

**OR keep current names but document the pattern**

---

### Phase 3: Document All Crates (Ongoing)

Add standard documentation to each crate:

```rust
//! crate-name
//!
//! **Category:** Orchestration / Data Management / State Management / Protocol / Utility
//! **Pattern:** Command / CRUD / Registry / Trait-based / Function-based
//!
//! # Interface
//!
//! ## Request Type
//! ```rust
//! pub struct XxxRequest { ... }
//! ```
//!
//! ## Response Type
//! ```rust
//! pub struct XxxResponse { ... }
//! ```
//!
//! ## Entrypoint
//! ```rust
//! pub async fn execute_xxx(...) -> Result<XxxResponse>
//! ```
```

---

### Phase 4: Create Crate Template (Future)

Create a template for new crates:

```
templates/
├── orchestration-crate/
│   ├── src/
│   │   └── lib.rs (with standard pattern)
│   └── Cargo.toml
├── crud-crate/
├── registry-crate/
└── protocol-crate/
```

---

## Examples

### Example 1: Orchestration Crate (hive-lifecycle)

**BEFORE:**
```rust
pub async fn ensure_hive_running(
    catalog: Arc<HiveCatalog>,
    queen_url: &str,
) -> Result<String> {
    // ...
    Ok(hive_url)
}
```

**AFTER:**
```rust
#[derive(Debug, Clone)]
pub struct HiveStartRequest {
    pub queen_url: String,
}

#[derive(Debug, Clone)]
pub struct HiveStartResponse {
    pub hive_url: String,
    pub hive_id: String,
    pub port: u16,
    pub status: String,
}

pub async fn execute_hive_start(
    catalog: Arc<HiveCatalog>,
    request: HiveStartRequest,
) -> Result<HiveStartResponse> {
    Narration::new(ACTOR, ACTION, "start")
        .human("Starting hive")
        .emit();
    
    // Orchestration logic
    let host = "localhost";
    let port = 8600;
    
    // Add to catalog
    catalog.add_hive(...).await?;
    
    // Spawn process
    spawn_hive(port, &request.queen_url).await?;
    
    Ok(HiveStartResponse {
        hive_url: format!("http://{}:{}", host, port),
        hive_id: host.to_string(),
        port,
        status: "started".to_string(),
    })
}
```

**HTTP Wrapper:**
```rust
#[derive(Deserialize)]
pub struct HttpHiveStartRequest {
    pub queen_url: Option<String>,  // Optional, defaults to localhost:8500
}

#[derive(Serialize)]
pub struct HttpHiveStartResponse {
    pub hive_url: String,
    pub hive_id: String,
    pub port: u16,
    pub status: String,
}

pub async fn handle_hive_start(
    State(catalog): State<Arc<HiveCatalog>>,
    Json(req): Json<HttpHiveStartRequest>,
) -> Result<Json<HttpHiveStartResponse>, (StatusCode, String)> {
    let request = queen_rbee_hive_lifecycle::HiveStartRequest {
        queen_url: req.queen_url.unwrap_or_else(|| "http://localhost:8500".to_string()),
    };
    
    let response = queen_rbee_hive_lifecycle::execute_hive_start(catalog, request).await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    
    Ok(Json(HttpHiveStartResponse {
        hive_url: response.hive_url,
        hive_id: response.hive_id,
        port: response.port,
        status: response.status,
    }))
}
```

---

### Example 2: CRUD Crate (hive-catalog)

**CURRENT (already good):**
```rust
pub struct HiveCatalog { ... }

impl HiveCatalog {
    pub async fn add_hive(&self, hive: HiveRecord) -> Result<()>
    pub async fn get_hive(&self, id: &str) -> Result<Option<HiveRecord>>
    pub async fn list_hives(&self) -> Result<Vec<HiveRecord>>
    pub async fn update_hive(&self, hive: HiveRecord) -> Result<()>
    pub async fn remove_hive(&self, id: &str) -> Result<()>
}
```

**HTTP Wrappers:**
```rust
// POST /hives
pub async fn handle_create_hive(
    State(catalog): State<Arc<HiveCatalog>>,
    Json(hive): Json<HiveRecord>,
) -> Result<StatusCode, (StatusCode, String)> {
    catalog.add_hive(hive).await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(StatusCode::CREATED)
}

// GET /hives/:id
pub async fn handle_get_hive(
    State(catalog): State<Arc<HiveCatalog>>,
    Path(id): Path<String>,
) -> Result<Json<Option<HiveRecord>>, (StatusCode, String)> {
    catalog.get_hive(&id).await
        .map(Json)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
}

// GET /hives
pub async fn handle_list_hives(
    State(catalog): State<Arc<HiveCatalog>>,
) -> Result<Json<Vec<HiveRecord>>, (StatusCode, String)> {
    catalog.list_hives().await
        .map(Json)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
}
```

---

### Example 3: Protocol Crate (heartbeat)

**CURRENT (already good):**
```rust
pub async fn handle_hive_heartbeat<C, D>(
    catalog: Arc<C>,
    payload: HiveHeartbeatPayload,
    device_detector: Arc<D>,
) -> Result<HeartbeatAcknowledgement, HeartbeatError>
where
    C: HiveCatalog,
    D: DeviceDetector,
{
    // Protocol logic
}
```

**HTTP Wrapper:**
```rust
pub async fn handle_heartbeat(
    State(state): State<HeartbeatState>,
    Json(payload): Json<HiveHeartbeatPayload>,
) -> Result<Json<HeartbeatAcknowledgement>, (StatusCode, String)> {
    rbee_heartbeat::handle_hive_heartbeat(
        state.hive_catalog,
        payload,
        state.device_detector,
    ).await
    .map(Json)
    .map_err(|e| match e {
        HeartbeatError::HiveNotFound(id) => {
            (StatusCode::NOT_FOUND, format!("Hive {} not found", id))
        }
        _ => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    })
}
```

---

## Benefits Summary

### For Developers

1. **Predictable** - All crates follow same pattern
2. **Easy to learn** - One pattern to understand
3. **Self-documenting** - Pattern tells you what it does
4. **Maintainable** - Consistent code is easier to change

### For HTTP Layer

1. **Uniform wrappers** - All look the same
2. **Simple conversion** - Domain ↔ HTTP types
3. **Consistent errors** - Same error handling everywhere
4. **Easy to test** - Predictable behavior

### For Crates

1. **Pure business logic** - No HTTP pollution
2. **Reusable** - Can be used in CLI, tests, other binaries
3. **Testable** - No HTTP mocking needed
4. **Flexible** - Can change HTTP layer without touching crates

---

## Checklist for New Crates

- [ ] Classify crate category (Orchestration / CRUD / Registry / Protocol / Utility)
- [ ] Choose appropriate pattern
- [ ] Define Request type (if Command Pattern)
- [ ] Define Response type (structured, not primitive)
- [ ] Implement core function with standard signature
- [ ] Add narration for observability
- [ ] Document interface in crate-level docs
- [ ] Create HTTP wrapper in http.rs (if needed)
- [ ] Add integration test
- [ ] Update this document with example

---

## Conclusion

By following this standard, we ensure:
- ✅ Consistent crate interfaces
- ✅ Predictable HTTP wrappers
- ✅ Easy to understand and maintain
- ✅ Self-documenting architecture

**Next Steps:**
1. Review and approve this standard
2. Fix `hive-lifecycle` to return structured response
3. Document all existing crates with their category/pattern
4. Create crate templates for future development

---

**TEAM-164 OUT** 🎯
