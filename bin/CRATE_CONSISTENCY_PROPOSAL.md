# Crate Consistency Proposal

## Problem: Asymmetric Crate Interfaces

The HTTP wrappers have different shapes because the crates return different types:

```rust
// hive-lifecycle: Returns primitive String
ensure_hive_running(...) -> Result<String>

// scheduler: Returns structured response
orchestrate_job(...) -> Result<JobResponse>

// heartbeat: Returns structured response
handle_hive_heartbeat(...) -> Result<HeartbeatAcknowledgement>
```

This forces HTTP layer to do different things:
- **Hive start**: Wrap string in struct
- **Job create**: Convert between types
- **Heartbeat**: Pass through

---

## Proposed Solution: Consistent Response Pattern

All crates should return **structured response types**:

### Pattern

```rust
// In crate:
pub struct XxxResponse {
    // Domain-specific fields
}

pub async fn do_xxx(...) -> Result<XxxResponse> {
    // Business logic
    Ok(XxxResponse { ... })
}

// In http.rs:
#[derive(Serialize)]  // HTTP-specific serialization
pub struct HttpXxxResponse {
    // Same fields as XxxResponse
}

pub async fn handle_xxx(...) -> Result<Json<HttpXxxResponse>, (StatusCode, String)> {
    let response = crate::do_xxx(...).await?;
    
    // Simple field-by-field copy
    Ok(Json(HttpXxxResponse {
        field1: response.field1,
        field2: response.field2,
    }))
}
```

---

## Refactoring Plan

### 1. hive-lifecycle

**BEFORE:**
```rust
pub async fn ensure_hive_running(
    catalog: Arc<HiveCatalog>,
    queen_url: &str,
) -> Result<String> {  // ← Returns primitive
    // ...
    Ok(hive_url)
}
```

**AFTER:**
```rust
#[derive(Debug, Clone)]
pub struct HiveStartResponse {
    pub hive_url: String,
    pub hive_id: String,
    pub port: u16,
}

pub async fn ensure_hive_running(
    catalog: Arc<HiveCatalog>,
    queen_url: &str,
) -> Result<HiveStartResponse> {  // ← Returns structured response
    // ...
    Ok(HiveStartResponse {
        hive_url,
        hive_id: host,
        port,
    })
}
```

**HTTP wrapper becomes simpler:**
```rust
// In http.rs
#[derive(Serialize)]
pub struct HttpHiveStartResponse {
    pub hive_url: String,
    pub hive_id: String,
    pub port: u16,
}

pub async fn handle_hive_start(...) -> Result<Json<HttpHiveStartResponse>, ...> {
    let response = queen_rbee_hive_lifecycle::ensure_hive_running(...).await?;
    
    // Simple conversion, no wrapping needed
    Ok(Json(HttpHiveStartResponse {
        hive_url: response.hive_url,
        hive_id: response.hive_id,
        port: response.port,
    }))
}
```

---

### 2. scheduler

**CURRENT (already good):**
```rust
pub struct JobResponse {
    pub job_id: String,
    pub sse_url: String,
}

pub async fn orchestrate_job(...) -> Result<JobResponse> {
    // ...
}
```

**HTTP wrapper:**
```rust
#[derive(Serialize)]
pub struct HttpJobResponse {
    pub job_id: String,
    pub sse_url: String,
}

pub async fn handle_create_job(...) -> Result<Json<HttpJobResponse>, ...> {
    let response = queen_rbee_scheduler::orchestrate_job(...).await?;
    
    Ok(Json(HttpJobResponse {
        job_id: response.job_id,
        sse_url: response.sse_url,
    }))
}
```

---

### 3. heartbeat

**CURRENT (already good):**
```rust
pub async fn handle_hive_heartbeat<C, D>(
    catalog: Arc<C>,
    payload: HiveHeartbeatPayload,
    device_detector: Arc<D>,
) -> Result<HeartbeatAcknowledgement, HeartbeatError> {
    // ...
}
```

**HTTP wrapper:**
```rust
#[derive(Serialize)]
pub struct HttpHeartbeatAcknowledgement {
    pub status: String,
    pub message: String,
}

pub async fn handle_heartbeat(...) -> Result<Json<HttpHeartbeatAcknowledgement>, ...> {
    let response = rbee_heartbeat::handle_hive_heartbeat(...).await?;
    
    Ok(Json(HttpHeartbeatAcknowledgement {
        status: response.status,
        message: response.message,
    }))
}
```

---

## Benefits

### 1. **Uniform HTTP Wrapper Pattern**

All wrappers do the same thing:
```rust
pub async fn handle_xxx(...) -> Result<Json<HttpXxxResponse>, (StatusCode, String)> {
    let response = crate::do_xxx(...).await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    
    Ok(Json(HttpXxxResponse {
        field1: response.field1,
        field2: response.field2,
    }))
}
```

### 2. **Crates Return Rich Information**

Instead of primitives, crates return structured data:
- More information available
- Easier to extend (add fields without breaking API)
- Self-documenting

### 3. **Clear Separation**

```
Domain Response (in crate)
    ↓
    No serde, no HTTP
    Pure business logic type
    ↓
HTTP Response (in http.rs)
    ↓
    #[derive(Serialize)]
    HTTP-specific type
```

### 4. **Consistent Error Handling**

All wrappers convert errors the same way:
```rust
.map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
```

---

## Implementation

### Step 1: Fix hive-lifecycle

```rust
// In hive-lifecycle/src/lib.rs
#[derive(Debug, Clone)]
pub struct HiveStartResponse {
    pub hive_url: String,
    pub hive_id: String,
    pub port: u16,
}

pub async fn ensure_hive_running(
    catalog: Arc<HiveCatalog>,
    queen_url: &str,
) -> Result<HiveStartResponse> {
    // ... existing logic ...
    
    Ok(HiveStartResponse {
        hive_url: format!("http://{}:{}", host, port),
        hive_id: host,
        port,
    })
}
```

### Step 2: Update http.rs

```rust
// In http.rs
#[derive(Serialize)]
pub struct HttpHiveStartResponse {
    pub hive_url: String,
    pub hive_id: String,
    pub port: u16,
}

pub async fn handle_hive_start(
    State(catalog): State<HiveStartState>,
) -> Result<(StatusCode, Json<HttpHiveStartResponse>), (StatusCode, String)> {
    let response = queen_rbee_hive_lifecycle::ensure_hive_running(
        Arc::clone(&catalog),
        "http://localhost:8500"
    )
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok((StatusCode::OK, Json(HttpHiveStartResponse {
        hive_url: response.hive_url,
        hive_id: response.hive_id,
        port: response.port,
    })))
}
```

---

## Result: Uniform Pattern

After refactoring, ALL HTTP wrappers look the same:

```rust
pub async fn handle_xxx(
    State(state): State<XxxState>,
    Json(req): Json<HttpXxxRequest>,  // If needed
) -> Result<Json<HttpXxxResponse>, (StatusCode, String)> {
    // 1. Convert HTTP request to domain request (if needed)
    let request = crate::XxxRequest { ... };
    
    // 2. Call pure business logic
    let response = crate::do_xxx(state, request).await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    
    // 3. Convert domain response to HTTP response
    Ok(Json(HttpXxxResponse {
        field1: response.field1,
        field2: response.field2,
    }))
}
```

**Same shape, same pattern, easy to understand!**

---

## Should We Do This?

**Pros:**
- ✅ Consistent crate interfaces
- ✅ Uniform HTTP wrapper pattern
- ✅ Easier to understand and maintain
- ✅ Crates return richer information

**Cons:**
- ⚠️ Requires refactoring hive-lifecycle
- ⚠️ Changes return type (but internal to project)

**Recommendation:** YES, do it. The consistency is worth the small refactor.
