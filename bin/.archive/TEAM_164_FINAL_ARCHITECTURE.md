# TEAM-164: Final Architecture - NO HTTP Features in Crates

**Date:** 2025-10-20  
**Mission:** Clean architecture with HTTP layer separated from business logic

---

## ✅ Problem Solved

**BEFORE:** HTTP features polluting business logic crates  
**AFTER:** All HTTP code in `http.rs`, crates are pure business logic

---

## Architecture Principle

```
┌─────────────────────────────────────────────────────────┐
│ HTTP Layer (http.rs)                                    │
│ - Axum handlers                                         │
│ - JSON serialization                                    │
│ - HTTP status codes                                     │
│ - Request/Response types                                │
└─────────────────────────────────────────────────────────┘
                        ↓ calls
┌─────────────────────────────────────────────────────────┐
│ Business Logic Crates                                   │
│ - Pure Rust functions                                   │
│ - No HTTP dependencies                                  │
│ - No serde (unless needed for domain logic)             │
│ - No axum                                               │
└─────────────────────────────────────────────────────────┘
```

---

## Why HTTP Code Can't Live in Crates

### Example 1: Hive Start Endpoint

**HTTP-specific code:**
```rust
pub async fn handle_hive_start(
    State(catalog): State<HiveStartState>,  // ← axum::extract::State
) -> Result<(StatusCode, Json<HiveStartResponse>), (StatusCode, String)> {
    //        ↑ axum::http::StatusCode
    //                ↑ axum::Json
```

**Dependencies required:**
- `axum` - HTTP framework
- `serde` - JSON serialization

**If we put this in `hive-lifecycle` crate:**
- ❌ Forces axum dependency on pure lifecycle logic
- ❌ Forces serde dependency (not needed for spawning processes)
- ❌ Pollutes business logic with HTTP concerns
- ❌ Can't use crate in non-HTTP contexts (CLI tools, tests, etc.)

**Solution:**
```rust
// In hive-lifecycle crate (pure business logic):
pub async fn ensure_hive_running(
    catalog: Arc<HiveCatalog>,
    queen_url: &str,
) -> Result<String> {
    // No HTTP dependencies!
}

// In http.rs (HTTP wrapper):
pub async fn handle_hive_start(
    State(catalog): State<HiveStartState>,
) -> Result<(StatusCode, Json<HiveStartResponse>), (StatusCode, String)> {
    let hive_url = queen_rbee_hive_lifecycle::ensure_hive_running(
        catalog,
        "http://localhost:8500"
    ).await?;
    
    Ok((StatusCode::OK, Json(HiveStartResponse { hive_url, ... })))
}
```

---

### Example 2: Job Create Endpoint

**HTTP-specific code:**
```rust
#[derive(Deserialize)]  // ← serde for HTTP request parsing
pub struct HttpJobRequest {
    pub model: String,
    // ...
}

pub async fn handle_create_job(
    State(state): State<SchedulerState>,  // ← axum::extract::State
    Json(req): Json<HttpJobRequest>,      // ← axum::Json
) -> Result<Json<HttpJobResponse>, (StatusCode, String)> {
    //        ↑ axum::Json           ↑ axum::http::StatusCode
```

**Dependencies required:**
- `axum` - HTTP framework
- `serde` - JSON serialization with `#[derive(Deserialize, Serialize)]`

**If we put this in `scheduler` crate:**
- ❌ Forces serde on pure domain types (JobRequest doesn't need JSON)
- ❌ Forces axum dependency on scheduling logic
- ❌ Can't use scheduler in non-HTTP contexts

**Solution:**
```rust
// In scheduler crate (pure business logic):
#[derive(Debug, Clone)]  // No serde!
pub struct JobRequest {
    pub model: String,
    pub prompt: String,
    // ...
}

pub async fn orchestrate_job(
    registry: Arc<JobRegistry<String>>,
    catalog: Arc<HiveCatalog>,
    request: JobRequest,
) -> Result<JobResponse> {
    // No HTTP dependencies!
}

// In http.rs (HTTP wrapper):
#[derive(Deserialize)]  // serde ONLY in HTTP layer
pub struct HttpJobRequest {
    pub model: String,
    // ...
}

pub async fn handle_create_job(
    State(state): State<SchedulerState>,
    Json(req): Json<HttpJobRequest>,
) -> Result<Json<HttpJobResponse>, (StatusCode, String)> {
    // Convert HTTP request to domain type
    let request = queen_rbee_scheduler::JobRequest {
        model: req.model,
        // ...
    };
    
    // Call pure business logic
    let response = queen_rbee_scheduler::orchestrate_job(
        state.registry,
        state.hive_catalog,
        request
    ).await?;
    
    // Convert domain response to HTTP response
    Ok(Json(HttpJobResponse { ... }))
}
```

---

### Example 3: HttpDeviceDetector

**HTTP-specific code:**
```rust
pub struct HttpDeviceDetector {
    client: reqwest::Client,  // ← HTTP client library
}

#[async_trait]  // ← async_trait for trait impl
impl DeviceDetector for HttpDeviceDetector {
    async fn detect_devices(&self, hive_url: &str) -> Result<DeviceResponse> {
        self.client.get(&url).send().await?  // ← HTTP request
            .json().await?
    }
}
```

**Dependencies required:**
- `reqwest` - HTTP client library
- `async_trait` - For async trait methods

**Why this can't live in `hive-lifecycle` or `rbee-heartbeat`:**

The `DeviceDetector` trait is abstract and can have multiple implementations:
- `HttpDeviceDetector` - Uses HTTP requests
- `SshDeviceDetector` - Uses SSH commands
- `LocalDeviceDetector` - Uses local system calls
- `MockDeviceDetector` - For testing

**If we put HttpDeviceDetector in hive-lifecycle:**
- ❌ Forces reqwest on everyone, even SSH/local users
- ❌ Pollutes lifecycle logic with HTTP client code

**If we put it in rbee-heartbeat:**
- ❌ Forces reqwest on the heartbeat trait crate
- ❌ Couples abstract trait to one implementation

**Solution:**
```rust
// In rbee-heartbeat crate (pure trait):
pub trait DeviceDetector {
    async fn detect_devices(&self, hive_url: &str) -> Result<DeviceResponse>;
}

// In http.rs (HTTP implementation):
pub struct HttpDeviceDetector {
    client: reqwest::Client,
}

impl DeviceDetector for HttpDeviceDetector {
    async fn detect_devices(&self, hive_url: &str) -> Result<DeviceResponse> {
        self.client.get(&url).send().await?.json().await
    }
}
```

---

## Final File Structure

```
bin/10_queen_rbee/src/
├── main.rs           # Main binary + shutdown endpoint
├── health.rs         # Health endpoint (no dependencies)
└── http.rs           # ALL HTTP code (endpoints + device detector)

bin/15_queen_rbee_crates/
├── hive-lifecycle/   # Pure hive orchestration (no HTTP)
└── scheduler/        # Pure job orchestration (no HTTP)

bin/99_shared_crates/
└── heartbeat/        # Pure heartbeat logic + traits (no HTTP)
```

---

## Dependencies

### Crates (Pure Business Logic)
```toml
[dependencies]
anyhow = "1.0"
tokio = { workspace = true }
observability-narration-core = { ... }
# NO axum, NO serde, NO reqwest
```

### queen-rbee Binary (HTTP Layer)
```toml
[dependencies]
axum = { workspace = true }
serde = { version = "1.0", features = ["derive"] }
reqwest = { version = "0.12", features = ["json"] }
async-trait = "0.1"

# Pure business logic crates (no HTTP features)
queen-rbee-hive-lifecycle = { path = "..." }
queen-rbee-scheduler = { path = "..." }
rbee-heartbeat = { path = "..." }
```

---

## Benefits

1. **Clean separation:** HTTP ≠ Business logic
2. **No feature flags:** Crates are pure, no `#[cfg(feature = "http")]`
3. **Reusable:** Crates can be used in CLI tools, tests, other binaries
4. **Testable:** Test business logic without HTTP mocking
5. **Maintainable:** Each layer has single responsibility

---

## What Lives Where

### http.rs (HTTP Layer)
- ✅ Axum handlers (`handle_*` functions)
- ✅ HTTP request types (`#[derive(Deserialize)]`)
- ✅ HTTP response types (`#[derive(Serialize)]`)
- ✅ HTTP-specific implementations (`HttpDeviceDetector`)
- ✅ StatusCode, Json, State, Path, etc.

### Crates (Business Logic)
- ✅ Pure domain types (no serde unless needed for domain)
- ✅ Business logic functions
- ✅ Traits and abstractions
- ✅ Process spawning, orchestration, scheduling
- ❌ NO axum
- ❌ NO HTTP types
- ❌ NO feature flags

---

## Verification

✅ **Build:** `cargo build --bin queen-rbee` - Success  
✅ **Test:** `cargo xtask e2e:hive` - PASSED  
✅ **Crates:** No HTTP dependencies  
✅ **No feature flags:** Clean Cargo.toml files

---

**TEAM-164 OUT** 🎯

**Architecture is now CLEAN and MAINTAINABLE**
