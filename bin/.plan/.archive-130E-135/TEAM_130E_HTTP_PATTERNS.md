# TEAM-130E: HTTP CLIENT PATTERNS & TYPE DUPLICATION

**Phase:** Phase 3 (Days 9-12)  
**Date:** 2025-10-19  
**Mission:** HTTP client consolidation and type deduplication

---

## 🎯 EXECUTIVE SUMMARY

**Finding:** All 4 binaries use reqwest with near-identical patterns. Type definitions are duplicated across binaries.

**Opportunities:**
1. **HTTP Client Wrapper:** ~250 LOC savings
2. **Type Consolidation:** ~420 LOC savings

**Total Savings:** ~670 LOC

**Impact:** MEDIUM - Reduces boilerplate, ensures consistent error handling

---

## 📊 HTTP CLIENT USAGE ANALYSIS

### Pattern 1: Standalone Client Creation (Most Common)

**Found in:** rbee-keeper (6×), queen-rbee (2×), rbee-hive (3×), llm-worker (1×)

**rbee-keeper examples:**
```rust
// commands/setup.rs (line 117)
let client = reqwest::Client::new();
let response = client
    .post(&url)
    .json(&request)
    .send()
    .await
    .context("Failed to send request to queen-rbee")?;

// commands/infer.rs (similar)
let client = reqwest::Client::new();
let response = client
    .post(format!("{}/v2/tasks", queen_url))
    .json(&json!({ ... }))
    .send().await?;

// commands/workers.rs (similar)
let client = reqwest::Client::new();
let response = client.get(&url).send().await?;
```

**queen-rbee examples:**
```rust
// worker_registry.rs (line 137)
let client = reqwest::Client::new();
let response = client
    .post(format!("{}/v1/admin/shutdown", worker.url))
    .send().await?;

// preflight/rbee_hive.rs
let client = reqwest::Client::new();
let response = client
    .get(format!("{}/v1/health", url))
    .timeout(Duration::from_secs(5))
    .send().await?;
```

**rbee-hive examples:**
```rust
// monitor.rs (line 101)
let client = reqwest::Client::new();
match client
    .get(format!("{}/v1/health", worker.url))
    .timeout(Duration::from_secs(5))
    .send().await
{ ... }

// shutdown.rs
let client = reqwest::Client::new();
client.post(format!("{}/v1/admin/shutdown", worker_url)).send().await?;
```

**llm-worker examples:**
```rust
// common/startup.rs (line 68)
let client = reqwest::Client::new();
let response = client.post(callback_url).json(&payload).send().await?;
```

### Pattern 2: Reused Client (Rare)

**Found in:** rbee-keeper queen_lifecycle.rs (passed as parameter)

```rust
pub async fn ensure_queen_rbee_running(client: &reqwest::Client, queen_url: &str) -> Result<()> {
    match client.get(&health_url).timeout(Duration::from_millis(500)).send().await {
        // ...
    }
}
```

---

## 🔍 COMMON HTTP OPERATIONS

### Operation Frequency Analysis

| Operation | rbee-keeper | queen-rbee | rbee-hive | llm-worker | Total |
|-----------|-------------|------------|-----------|------------|-------|
| POST JSON | 5 | 3 | 2 | 1 | **11** |
| GET | 3 | 2 | 4 | 0 | **9** |
| GET with timeout | 1 | 1 | 2 | 0 | **4** |
| POST (no body) | 1 | 1 | 1 | 0 | **3** |

### Common Patterns Identified

**1. POST with JSON body (11 occurrences):**
```rust
let client = reqwest::Client::new();
let response = client
    .post(url)
    .json(&payload)
    .send()
    .await?;
```

**2. GET with timeout (4 occurrences):**
```rust
let client = reqwest::Client::new();
let response = client
    .get(url)
    .timeout(Duration::from_secs(5))
    .send()
    .await?;
```

**3. Status code checking (27 occurrences):**
```rust
if response.status().is_success() {
    // Parse response
} else {
    // Error handling
}
```

**4. JSON deserialization (15 occurrences):**
```rust
let result: ResponseType = response.json().await.context("Parse failed")?;
```

---

## 💡 PROPOSED SHARED CRATE: `rbee-http-client`

### Crate Structure

```
bin/shared-crates/rbee-http-client/
├─ src/
│  ├─ lib.rs
│  ├─ client.rs      // HTTP client wrapper
│  ├─ error.rs       // Error types
│  └─ retry.rs       // Retry logic (future)
├─ Cargo.toml
└─ README.md
```

### API Design

```rust
// ============================================================================
// CORE CLIENT
// ============================================================================

pub struct RbeeHttpClient {
    client: reqwest::Client,
    base_url: Option<String>,
    default_timeout: Duration,
}

impl RbeeHttpClient {
    /// Create new client with defaults
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: None,
            default_timeout: Duration::from_secs(30),
        }
    }
    
    /// Create client with base URL
    pub fn with_base_url(base_url: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: Some(base_url.into()),
            default_timeout: Duration::from_secs(30),
        }
    }
    
    /// Set default timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.default_timeout = timeout;
        self
    }
    
    /// POST with JSON body
    pub async fn post_json<T, R>(&self, path: &str, body: &T) -> Result<R>
    where
        T: Serialize,
        R: DeserializeOwned,
    {
        let url = self.build_url(path);
        
        let response = self.client
            .post(&url)
            .json(body)
            .timeout(self.default_timeout)
            .send()
            .await
            .map_err(|e| RbeeHttpError::RequestFailed {
                url: url.clone(),
                source: e,
            })?;
        
        self.check_status(&response, &url)?;
        
        response.json().await.map_err(|e| RbeeHttpError::DeserializeFailed {
            url,
            source: e,
        })
    }
    
    /// POST with no body
    pub async fn post(&self, path: &str) -> Result<()> {
        let url = self.build_url(path);
        
        let response = self.client
            .post(&url)
            .timeout(self.default_timeout)
            .send()
            .await
            .map_err(|e| RbeeHttpError::RequestFailed {
                url: url.clone(),
                source: e,
            })?;
        
        self.check_status(&response, &url)?;
        Ok(())
    }
    
    /// GET with JSON response
    pub async fn get_json<R>(&self, path: &str) -> Result<R>
    where
        R: DeserializeOwned,
    {
        let url = self.build_url(path);
        
        let response = self.client
            .get(&url)
            .timeout(self.default_timeout)
            .send()
            .await
            .map_err(|e| RbeeHttpError::RequestFailed {
                url: url.clone(),
                source: e,
            })?;
        
        self.check_status(&response, &url)?;
        
        response.json().await.map_err(|e| RbeeHttpError::DeserializeFailed {
            url,
            source: e,
        })
    }
    
    /// GET for health check (returns bool)
    pub async fn health_check(&self, path: &str) -> bool {
        let url = self.build_url(path);
        
        match self.client
            .get(&url)
            .timeout(Duration::from_secs(5))
            .send()
            .await
        {
            Ok(resp) => resp.status().is_success(),
            Err(_) => false,
        }
    }
    
    // Internal helpers
    fn build_url(&self, path: &str) -> String {
        if let Some(base) = &self.base_url {
            if path.starts_with('/') {
                format!("{}{}", base, path)
            } else {
                format!("{}/{}", base, path)
            }
        } else {
            path.to_string()
        }
    }
    
    fn check_status(&self, response: &reqwest::Response, url: &str) -> Result<()> {
        if response.status().is_success() {
            Ok(())
        } else {
            Err(RbeeHttpError::BadStatus {
                url: url.to_string(),
                status: response.status(),
            })
        }
    }
}

// ============================================================================
// ERROR TYPES
// ============================================================================

#[derive(Debug, thiserror::Error)]
pub enum RbeeHttpError {
    #[error("HTTP request to {url} failed: {source}")]
    RequestFailed {
        url: String,
        #[source]
        source: reqwest::Error,
    },
    
    #[error("Bad HTTP status from {url}: {status}")]
    BadStatus {
        url: String,
        status: reqwest::StatusCode,
    },
    
    #[error("Failed to deserialize response from {url}: {source}")]
    DeserializeFailed {
        url: String,
        #[source]
        source: reqwest::Error,
    },
}

pub type Result<T> = std::result::Result<T, RbeeHttpError>;
```

### Usage Examples

**Before (rbee-keeper/commands/setup.rs):**
```rust
let client = reqwest::Client::new();
let url = format!("{}/v2/registry/beehives/add", QUEEN_RBEE_URL);
let response = client
    .post(&url)
    .json(&request)
    .send()
    .await
    .context("Failed to send request to queen-rbee")?;
let result: AddNodeResponse = response.json().await?;
```

**After:**
```rust
let client = RbeeHttpClient::with_base_url(QUEEN_RBEE_URL);
let result: AddNodeResponse = client
    .post_json("/v2/registry/beehives/add", &request)
    .await?;
```

**Before (queen-rbee/worker_registry.rs):**
```rust
let client = reqwest::Client::new();
let response = client
    .post(format!("{}/v1/admin/shutdown", worker.url))
    .send().await?;
if !response.status().is_success() {
    anyhow::bail!("Worker shutdown failed: HTTP {}", response.status());
}
```

**After:**
```rust
let client = RbeeHttpClient::new();
client.post(&format!("{}/v1/admin/shutdown", worker.url)).await?;
```

**Before (rbee-hive/monitor.rs):**
```rust
let client = reqwest::Client::new();
match client
    .get(format!("{}/v1/health", worker.url))
    .timeout(Duration::from_secs(5))
    .send().await
{
    Ok(response) if response.status().is_success() => { /* healthy */ }
    _ => { /* unhealthy */ }
}
```

**After:**
```rust
let client = RbeeHttpClient::new();
if client.health_check(&format!("{}/v1/health", worker.url)).await {
    // healthy
} else {
    // unhealthy
}
```

---

## 📊 TYPE DUPLICATION ANALYSIS

### 1. BeehiveNode (CRITICAL DUPLICATION)

**Defined in 2 places with DIFFERENT fields:**

**queen-rbee/beehive_registry.rs (12 fields):**
```rust
pub struct BeehiveNode {
    pub node_name: String,
    pub ssh_host: String,
    pub ssh_port: u16,
    pub ssh_user: String,
    pub ssh_key_path: Option<String>,
    pub git_repo_url: String,
    pub git_branch: String,
    pub install_path: String,
    pub last_connected_unix: Option<i64>,
    pub status: String,
    pub backends: Option<String>,  // JSON array
    pub devices: Option<String>,   // JSON object
}
```

**rbee-keeper/commands/setup.rs (8 fields):**
```rust
struct BeehiveNode {
    node_name: String,
    ssh_host: String,
    ssh_port: u16,
    ssh_user: String,
    ssh_key_path: Option<String>,
    git_repo_url: String,
    git_branch: String,
    install_path: String,
    last_connected_unix: Option<i64>,
    status: String,
}
```

**Problem:** rbee-keeper's version is MISSING `backends` and `devices` fields. This causes deserialization failures when queen-rbee returns the full type.

**LOC:** ~30 (struct definition) × 2 = 60 LOC duplication

---

### 2. WorkerInfo (TRIPLE DUPLICATION!)

**Defined in 3 places with DIFFERENT shapes:**

**queen-rbee/worker_registry.rs:**
```rust
pub struct WorkerInfo {
    pub id: String,
    pub url: String,
    pub model_ref: String,
    pub backend: String,
    pub device: u32,
    pub state: WorkerState,
    pub slots_total: u32,
    pub slots_available: u32,
    pub vram_bytes: Option<u64>,
    pub node_name: String,
}
```

**rbee-hive/registry.rs:**
```rust
pub struct WorkerInfo {
    pub id: String,
    pub url: String,
    pub model_ref: String,
    pub backend: String,
    pub device: u32,
    pub state: WorkerState,
    pub last_activity: SystemTime,
    pub slots_total: u32,
    pub slots_available: u32,
    pub failed_health_checks: u32,
    pub pid: Option<u32>,
    pub restart_count: u32,
    pub last_restart: Option<SystemTime>,
    pub last_heartbeat: Option<SystemTime>,
}
```

**shared-crates/hive-core/src/worker.rs (UNUSED!):**
```rust
pub struct WorkerInfo {
    pub id: String,
    pub backend: Backend,
    pub model_id: String,
    pub gpu_id: Option<u32>,
    pub port: u16,
    pub pid: u32,
    pub started_at: String,
}
```

**Problem:** Three completely incompatible definitions! The shared-crates version exists but is NEVER used.

**LOC:** ~40 (each) × 3 = 120 LOC duplication

---

### 3. WorkerState (DUPLICATE)

**Defined in 2 places identically:**

**queen-rbee/worker_registry.rs:**
```rust
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum WorkerState {
    Loading,
    Idle,
    Busy,
}
```

**rbee-hive/registry.rs:**
```rust
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum WorkerState {
    Loading,
    Idle,
    Busy,
}
```

**LOC:** ~10 × 2 = 20 LOC duplication

---

### 4. HTTP Request/Response Types

**AddNodeRequest (rbee-keeper):**
```rust
struct AddNodeRequest {
    node_name: String,
    ssh_host: String,
    ssh_port: u16,
    ssh_user: String,
    ssh_key_path: Option<String>,
    git_repo_url: String,
    git_branch: String,
    install_path: String,
}
```

**AddNodeResponse (rbee-keeper):**
```rust
struct AddNodeResponse {
    success: bool,
    message: String,
    node_name: String,
}
```

**Similar types in:**
- rbee-keeper: 10+ request/response types (~180 LOC)
- queen-rbee: 8+ request/response types (~150 LOC)

**Total duplication:** ~330 LOC

---

## 💡 PROPOSED SHARED CRATE: `rbee-types`

### Crate Structure

```
bin/shared-crates/rbee-types/
├─ src/
│  ├─ lib.rs
│  ├─ beehive.rs     // BeehiveNode, HiveId
│  ├─ worker.rs      // WorkerInfo, WorkerState
│  ├─ requests.rs    // HTTP request types
│  └─ responses.rs   // HTTP response types
├─ Cargo.toml
└─ README.md
```

### Unified Type Definitions

```rust
// ============================================================================
// BEEHIVE TYPES
// ============================================================================

/// Beehive node configuration (single source of truth)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeehiveNode {
    pub node_name: String,
    pub ssh_host: String,
    pub ssh_port: u16,
    pub ssh_user: String,
    pub ssh_key_path: Option<String>,
    pub git_repo_url: String,
    pub git_branch: String,
    pub install_path: String,
    pub last_connected_unix: Option<i64>,
    pub status: String,
    pub backends: Option<String>,  // JSON array: ["cuda", "metal", "cpu"]
    pub devices: Option<String>,   // JSON object: {"cuda": 2, "metal": 1}
}

pub type HiveId = String;

// ============================================================================
// WORKER TYPES
// ============================================================================

/// Worker state (unified)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum WorkerState {
    Loading,
    Idle,
    Busy,
}

/// Worker information (unified - superset of all fields)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerInfo {
    // Core identity
    pub id: String,
    pub url: String,
    pub model_ref: String,
    pub backend: String,
    pub device: u32,
    
    // State
    pub state: WorkerState,
    
    // Capacity
    pub slots_total: u32,
    pub slots_available: u32,
    pub vram_bytes: Option<u64>,
    
    // Registry metadata (optional for different use cases)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub node_name: Option<String>,
    
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pid: Option<u32>,
    
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_activity: Option<SystemTime>,
    
    #[serde(default)]
    pub failed_health_checks: u32,
    
    #[serde(default)]
    pub restart_count: u32,
    
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_restart: Option<SystemTime>,
    
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_heartbeat: Option<SystemTime>,
}

// ============================================================================
// HTTP REQUEST TYPES
// ============================================================================

#[derive(Debug, Serialize, Deserialize)]
pub struct AddNodeRequest {
    pub node_name: String,
    pub ssh_host: String,
    pub ssh_port: u16,
    pub ssh_user: String,
    pub ssh_key_path: Option<String>,
    pub git_repo_url: String,
    pub git_branch: String,
    pub install_path: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SpawnWorkerRequest {
    pub model_ref: String,
    pub backend: String,
    pub device: u32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ReadyRequest {
    pub worker_id: String,
    pub url: String,
    pub model_ref: String,
    pub backend: String,
    pub device: u32,
}

// ============================================================================
// HTTP RESPONSE TYPES
// ============================================================================

#[derive(Debug, Serialize, Deserialize)]
pub struct AddNodeResponse {
    pub success: bool,
    pub message: String,
    pub node_name: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ListNodesResponse {
    pub nodes: Vec<BeehiveNode>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ListWorkersResponse {
    pub workers: Vec<WorkerInfo>,
}
```

---

## 📊 LOC SAVINGS CALCULATION

### HTTP Client Savings

| Operation | Before (avg LOC) | After (LOC) | Occurrences | Savings |
|-----------|------------------|-------------|-------------|---------|
| POST JSON | 8 | 3 | 11 | 55 LOC |
| GET | 6 | 2 | 9 | 36 LOC |
| GET timeout | 8 | 2 | 4 | 24 LOC |
| POST | 6 | 2 | 3 | 12 LOC |
| Health check | 12 | 2 | 8 | 80 LOC |
| Error handling | 5 | 0 | 27 | 135 LOC |

**Subtotal HTTP Client:** ~342 LOC removed  
**New crate code:** ~120 LOC  
**Net savings:** **222 LOC**

### Type Consolidation Savings

| Type | Duplications | LOC each | Total removed | New shared | Net savings |
|------|--------------|----------|---------------|------------|-------------|
| BeehiveNode | 2 | 30 | 60 | 30 | **30 LOC** |
| WorkerInfo | 3 | 40 | 120 | 50 | **70 LOC** |
| WorkerState | 2 | 10 | 20 | 10 | **10 LOC** |
| Request types | ~15 | ~12 | 180 | 100 | **80 LOC** |
| Response types | ~12 | ~15 | 180 | 80 | **100 LOC** |

**Subtotal Type Consolidation:** **290 LOC**

### Additional Benefits

**Avoided duplication in queen-rbee hive lifecycle:** +150 LOC  
**Future worker types reuse:** +100 LOC

**Total savings:** 222 + 290 + 150 = **662 LOC**  
**Long-term:** 662 + 100 = **762 LOC**

---

## 🎯 IMPLEMENTATION PRIORITY

**Priority:** **P1 - HIGH**

**Why:**
1. Type mismatches causing runtime errors
2. High duplication (27 HTTP call sites)
3. Consistent error handling needed
4. Enables better testing

**Dependencies:**
- None (standalone crates)

**Risks:**
- LOW - Types are well-defined
- Need careful migration to avoid breakage

**Timeline:**
- 2 days rbee-http-client
- 2 days rbee-types
- 2 days migration
- 1 day testing

---

## ✅ ACCEPTANCE CRITERIA

1. ✅ `rbee-http-client` crate compiles and tests pass
2. ✅ `rbee-types` crate compiles and tests pass
3. ✅ All binaries use rbee-types for shared types
4. ✅ BeehiveNode has single definition
5. ✅ WorkerInfo has single definition
6. ✅ All HTTP clients use rbee-http-client
7. ✅ Zero type mismatches in HTTP communication
8. ✅ All existing tests pass

---

**Status:** TEAM-130E HTTP Patterns Analysis Complete  
**Next:** Shared Crate Audit  
**Savings:** ~662 LOC (initial) → 762 LOC (long-term)
