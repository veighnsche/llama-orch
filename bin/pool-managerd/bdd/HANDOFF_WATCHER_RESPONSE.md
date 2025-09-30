# 🚨 RESPONSE: Handoff Watcher Architecture - pool-managerd Team

**To**: orchestratord Team  
**From**: pool-managerd Team  
**Date**: 2025-09-30  
**Priority**: HIGH  
**Subject**: RE: Handoff Watcher Ownership - AGREED, We'll Own It

---

## Executive Summary

**WE AGREE 100%** - The handoff watcher **MUST** be owned by pool-managerd. The orchestratord team is absolutely correct that the current implementation is a HOME_PROFILE-only hack that breaks in distributed deployments.

**Status**: We accept ownership and will implement this in Phase 3.

---

## Our Analysis: You're Completely Right

### The Core Issue

The orchestratord team correctly identified that:

1. ✅ **Filesystem co-location**: pool-managerd runs on same machine as engine-provisioner
2. ✅ **Responsibility alignment**: pool-managerd already owns pool state
3. ✅ **Cloud profile requirement**: No shared filesystem in distributed deployments
4. ✅ **Separation of concerns**: Watcher belongs with pool lifecycle management

**This is a textbook example of proper service boundaries.**

### Why We Didn't Catch This Earlier

Looking at our implementation history:
- We implemented `Registry::register_ready_from_handoff()` (OC-POOL-3101)
- We assumed orchestratord would call it via HTTP
- We **never questioned** who should watch the filesystem
- **Classic oversight** - we built the API but not the watcher!

---

## Our Proposed Implementation

### Phase 3 Addition: Handoff Watcher Module

**File**: `bin/pool-managerd/src/watcher/handoff.rs` (NEW)

```rust
//! Handoff file watcher for automatic pool readiness detection
//!
//! Spec: OC-POOL-3101, OC-POOL-3102
//! Watches .runtime/engines/*.json for handoff files written by engine-provisioner
//! Updates registry when pools become ready

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use anyhow::{Context, Result};
use serde::Deserialize;
use tokio::time::interval;

use crate::core::registry::Registry;

/// Handoff file structure (matches engine-provisioner output)
#[derive(Debug, Clone, Deserialize)]
pub struct HandoffFile {
    pub pool_id: String,
    pub replica_id: String,
    pub engine: String,
    pub engine_version: String,
    pub device_mask: Option<String>,
    pub slots: Option<i32>,
    pub http_endpoint: String,
}

/// Handoff watcher configuration
#[derive(Debug, Clone)]
pub struct HandoffWatcherConfig {
    pub runtime_dir: PathBuf,
    pub poll_interval_ms: u64,
    pub auto_delete_processed: bool,
}

impl Default for HandoffWatcherConfig {
    fn default() -> Self {
        Self {
            runtime_dir: PathBuf::from(".runtime/engines"),
            poll_interval_ms: 1000,
            auto_delete_processed: true,
        }
    }
}

impl HandoffWatcherConfig {
    /// Load from environment variables
    pub fn from_env() -> Self {
        Self {
            runtime_dir: std::env::var("POOL_MANAGERD_RUNTIME_DIR")
                .map(PathBuf::from)
                .unwrap_or_else(|_| PathBuf::from(".runtime/engines")),
            poll_interval_ms: std::env::var("POOL_MANAGERD_WATCH_INTERVAL_MS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(1000),
            auto_delete_processed: std::env::var("POOL_MANAGERD_AUTO_DELETE_HANDOFF")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(true),
        }
    }
}

/// Spawn handoff watcher background task
pub fn spawn_handoff_watcher(
    registry: Arc<Mutex<Registry>>,
    config: HandoffWatcherConfig,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut interval = interval(Duration::from_millis(config.poll_interval_ms));
        
        tracing::info!(
            runtime_dir = %config.runtime_dir.display(),
            poll_interval_ms = config.poll_interval_ms,
            "handoff watcher started"
        );
        
        loop {
            interval.tick().await;
            
            if let Err(e) = scan_handoff_files(&registry, &config).await {
                tracing::error!(error = %e, "handoff scan failed");
            }
        }
    })
}

/// Scan runtime directory for handoff files
async fn scan_handoff_files(
    registry: &Arc<Mutex<Registry>>,
    config: &HandoffWatcherConfig,
) -> Result<()> {
    // Ensure directory exists
    if !config.runtime_dir.exists() {
        tokio::fs::create_dir_all(&config.runtime_dir).await
            .context("failed to create runtime directory")?;
        return Ok(());
    }
    
    let mut entries = tokio::fs::read_dir(&config.runtime_dir).await
        .context("failed to read runtime directory")?;
    
    while let Some(entry) = entries.next_entry().await? {
        let path = entry.path();
        
        if path.extension().and_then(|s| s.to_str()) == Some("json") {
            if let Err(e) = process_handoff_file(registry, &path, config).await {
                tracing::warn!(
                    path = %path.display(),
                    error = %e,
                    "failed to process handoff file"
                );
            }
        }
    }
    
    Ok(())
}

/// Process a single handoff file
async fn process_handoff_file(
    registry: &Arc<Mutex<Registry>>,
    path: &Path,
    config: &HandoffWatcherConfig,
) -> Result<()> {
    // Read and parse handoff file
    let content = tokio::fs::read_to_string(path).await
        .context("failed to read handoff file")?;
    
    let handoff: HandoffFile = serde_json::from_str(&content)
        .context("failed to parse handoff file")?;
    
    tracing::info!(
        pool_id = %handoff.pool_id,
        replica_id = %handoff.replica_id,
        engine = %handoff.engine,
        engine_version = %handoff.engine_version,
        path = %path.display(),
        "processing handoff file"
    );
    
    // Update registry
    {
        let mut reg = registry.lock().unwrap();
        
        // Use existing register_ready_from_handoff API (OC-POOL-3101)
        reg.register_ready_from_handoff(
            &handoff.pool_id,
            &handoff.engine_version,
            handoff.device_mask.as_deref(),
            handoff.slots,
        );
        
        tracing::info!(
            pool_id = %handoff.pool_id,
            "pool marked as ready via handoff"
        );
    }
    
    // Optionally delete processed file
    if config.auto_delete_processed {
        tokio::fs::remove_file(path).await
            .context("failed to delete handoff file")?;
        
        tracing::debug!(
            path = %path.display(),
            "deleted processed handoff file"
        );
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    
    #[tokio::test]
    async fn test_handoff_watcher_detects_file() {
        let temp_dir = TempDir::new().unwrap();
        let registry = Arc::new(Mutex::new(Registry::new()));
        
        let config = HandoffWatcherConfig {
            runtime_dir: temp_dir.path().to_path_buf(),
            poll_interval_ms: 100,
            auto_delete_processed: false,
        };
        
        // Write handoff file
        let handoff = HandoffFile {
            pool_id: "test-pool".to_string(),
            replica_id: "r0".to_string(),
            engine: "llamacpp".to_string(),
            engine_version: "b1234".to_string(),
            device_mask: Some("0".to_string()),
            slots: Some(4),
            http_endpoint: "http://localhost:8080".to_string(),
        };
        
        let handoff_path = temp_dir.path().join("test-pool-r0.json");
        tokio::fs::write(&handoff_path, serde_json::to_string(&handoff).unwrap())
            .await
            .unwrap();
        
        // Process file
        process_handoff_file(&registry, &handoff_path, &config)
            .await
            .unwrap();
        
        // Verify registry updated
        let reg = registry.lock().unwrap();
        let health = reg.get_health("test-pool").unwrap();
        assert!(health.ready);
        assert_eq!(reg.get_engine_version("test-pool"), Some("b1234".to_string()));
    }
}
```

### Integration with pool-managerd Main

**File**: `bin/pool-managerd/src/main.rs`

```rust
mod watcher;

#[tokio::main]
async fn main() -> Result<()> {
    // ... existing setup ...
    
    let registry = Arc::new(Mutex::new(Registry::new()));
    
    // Spawn handoff watcher (NEW)
    let watcher_config = watcher::handoff::HandoffWatcherConfig::from_env();
    watcher::handoff::spawn_handoff_watcher(registry.clone(), watcher_config);
    
    tracing::info!("handoff watcher started");
    
    // ... rest of main ...
}
```

---

## HTTP API for orchestratord

### Endpoint: GET /v2/pools/{id}/status

**Already exists!** orchestratord can poll this.

**Response**:
```json
{
  "pool_id": "pool-0",
  "live": true,
  "ready": true,
  "draining": false,
  "slots_total": 4,
  "slots_free": 2,
  "active_leases": 2,
  "engine": "llamacpp",
  "engine_version": "b1234",
  "device_mask": "0",
  "vram_total_bytes": 24000000000,
  "vram_free_bytes": 18000000000
}
```

### Optional: POST /v2/callbacks/ready (Future)

**For efficiency** - pool-managerd can notify orchestratord when ready.

```rust
// In pool-managerd watcher
async fn notify_orchestratord_if_configured(pool_id: &str) {
    if let Ok(callback_url) = std::env::var("ORCHESTRATORD_CALLBACK_URL") {
        let client = reqwest::Client::new();
        let _ = client.post(format!("{}/callbacks/pool-ready", callback_url))
            .json(&json!({ "pool_id": pool_id }))
            .send()
            .await;
    }
}
```

---

## Answers to orchestratord Team's Questions

### 1. **Timeline**: When can you implement the handoff watcher?

**Answer**: **Phase 3 (Week 3 of current sprint)**

- Week 1: ✅ Device discovery & masks (in progress)
- Week 2: ✅ Observability (metrics + logging)
- **Week 3**: ✅ **Handoff watcher** + Configuration
- Week 4: ✅ Security & integration

**ETA**: 7-10 days from now

### 2. **API Design**: Do you prefer polling or callbacks?

**Answer**: **Polling first, callbacks later**

**Phase 1 (v0.2.0)**: HTTP polling
- orchestratord polls `GET /v2/pools/{id}/status` every 5s
- Simple, reliable, no new complexity
- Works for both HOME and CLOUD profiles

**Phase 2 (v1.0.0)**: Add callbacks
- pool-managerd POSTs to orchestratord when ready
- Reduces latency and network overhead
- Requires orchestratord callback endpoint

### 3. **Backwards Compat**: Should we support both modes?

**Answer**: **Yes, with feature flags**

```rust
// pool-managerd config
pub struct WatcherMode {
    pub enabled: bool,              // Enable watcher
    pub callback_url: Option<String>, // If set, POST to orchestratord
}

// orchestratord config
pub struct PoolPollingConfig {
    pub enabled: bool,              // Enable polling
    pub interval_ms: u64,           // Poll interval
    pub accept_callbacks: bool,     // Accept POST /callbacks/pool-ready
}
```

**Migration path**:
- v0.1.0: orchestratord watcher (HOME_PROFILE only)
- v0.2.0: pool-managerd watcher + orchestratord polling
- v1.0.0: Add callbacks, deprecate orchestratord watcher

### 4. **Testing**: How should we test distributed scenarios?

**Answer**: **Multi-level testing strategy**

**Unit Tests** (pool-managerd):
```rust
#[test]
fn test_handoff_file_processing() {
    // Write handoff file
    // Process it
    // Assert registry updated
}
```

**Integration Tests** (pool-managerd):
```rust
#[tokio::test]
async fn test_watcher_detects_new_files() {
    // Spawn watcher
    // Write handoff file
    // Wait for detection
    // Assert pool ready
}
```

**E2E Tests** (orchestratord + pool-managerd):
```rust
#[tokio::test]
async fn test_distributed_handoff() {
    // Start pool-managerd on port 9200
    // Start orchestratord on port 8080
    // Write handoff file
    // Poll orchestratord until pool ready
    // Assert adapter bound
}
```

**BDD Tests** (test-harness):
```gherkin
Scenario: Cloud profile handoff detection
  Given pool-managerd is running on machine-b
  And orchestratord is running on machine-a
  When engine-provisioner writes handoff file on machine-b
  Then pool-managerd detects the handoff within 2 seconds
  And pool-managerd marks pool as ready
  When orchestratord polls pool-managerd
  Then orchestratord sees pool is ready
  And orchestratord binds adapter
```

### 5. **Ownership**: Who owns the adapter binding decision?

**Answer**: **orchestratord owns binding, pool-managerd owns readiness**

**Clear separation**:

| Responsibility | Owner | Reason |
|----------------|-------|--------|
| Watch handoff files | pool-managerd | Filesystem co-location |
| Update pool readiness | pool-managerd | Owns pool state |
| Detect pool ready | orchestratord | Polls or receives callback |
| **Bind adapter** | **orchestratord** | **Owns adapter-host** |
| Route tasks | orchestratord | Owns placement logic |

**Flow**:
```
1. engine-provisioner writes handoff file
2. pool-managerd watcher detects it
3. pool-managerd updates registry (ready=true)
4. orchestratord polls pool-managerd
5. orchestratord sees ready=true
6. orchestratord binds adapter ← ORCHESTRATORD DECISION
7. orchestratord can now route tasks
```

---

## Migration Plan (Detailed)

### Phase 1: v0.1.0 (Current - HOME_PROFILE Only)

**Status**: ✅ Acceptable for initial release

**What stays**:
- orchestratord handoff watcher (HOME_PROFILE only)
- Direct filesystem access
- Single-machine deployment

**What we add**:
- Big warning comments in orchestratord
- Documentation of limitation
- Spec updates noting HOME_PROFILE only

**Timeline**: This week (already done)

### Phase 2: v0.2.0 (CLOUD_PROFILE Support)

**Status**: 🚧 In development (Phase 3)

**pool-managerd changes**:
- ✅ Implement `src/watcher/handoff.rs`
- ✅ Spawn watcher in main.rs
- ✅ Add configuration via env vars
- ✅ Unit + integration tests

**orchestratord changes**:
- ✅ Implement HTTP polling of pool-managerd
- ✅ Remove filesystem watcher (or mark deprecated)
- ✅ Update adapter binding to use polling
- ✅ E2E tests with real pool-managerd

**Timeline**: 2 weeks (Week 3-4 of current sprint)

### Phase 3: v1.0.0 (Production Ready)

**Status**: 📋 Planned

**Optimizations**:
- ✅ Add callback mechanism (pool-managerd → orchestratord)
- ✅ Reduce polling overhead
- ✅ Event-driven architecture
- ✅ Metrics for handoff latency

**Timeline**: Next sprint (4 weeks out)

---

## Spec Updates Required

### 1. Update `.specs/30-pool-managerd.md`

**Add**:
```markdown
## OC-POOL-3105: Handoff Watcher

- [OC-POOL-3105] pool-managerd MUST watch the runtime directory for handoff files
- [OC-POOL-3106] When a handoff file is detected, pool-managerd MUST update the registry
- [OC-POOL-3107] Handoff files MUST be processed within 2 seconds of creation
- [OC-POOL-3108] Processed handoff files MAY be deleted automatically
- [OC-POOL-3109] Watcher MUST be configurable via POOL_MANAGERD_RUNTIME_DIR
```

### 2. Update `.specs/20-orchestratord.md`

**Add**:
```markdown
## OC-CTRL-2070: Pool Readiness Detection (CLOUD_PROFILE)

- [OC-CTRL-2070] orchestratord MUST poll pool-managerd for readiness status
- [OC-CTRL-2071] Polling interval MUST be configurable (default: 5s)
- [OC-CTRL-2072] When a pool becomes ready, orchestratord MUST bind the adapter
- [OC-CTRL-2073] orchestratord MUST NOT assume filesystem access to handoff files
```

### 3. Create `.specs/01_cloud_profile.md`

**New file** (as you requested earlier):
```markdown
# Cloud Profile Specification

## Service Boundaries

### Filesystem Isolation
- Each service MUST only access its local filesystem
- No shared filesystem assumptions
- All inter-service communication via HTTP

### Handoff Flow (Cloud Profile)
1. engine-provisioner writes handoff file (local filesystem)
2. pool-managerd watches local filesystem
3. pool-managerd updates own registry
4. orchestratord polls pool-managerd via HTTP
5. orchestratord binds adapter when ready
```

---

## Risks & Mitigations

### Risk 1: Polling Latency

**Risk**: 5s polling interval means 5s delay before adapter binding

**Mitigation**:
- Start with 1s polling for v0.2.0
- Add callbacks in v1.0.0
- Acceptable tradeoff for cloud compatibility

### Risk 2: Backward Compatibility

**Risk**: Existing tests/deployments rely on orchestratord watcher

**Mitigation**:
- Keep orchestratord watcher for HOME_PROFILE
- Add feature flag: `ORCHESTRATORD_HANDOFF_MODE=local|remote`
- Deprecate gradually over 2 releases

### Risk 3: Race Conditions

**Risk**: Pool becomes ready between polls

**Mitigation**:
- Idempotent adapter binding (already implemented)
- orchestratord tracks bound adapters
- Re-binding same adapter is safe (no-op)

---

## Metrics to Add

### pool-managerd Metrics

```rust
// Handoff watcher metrics
handoff_files_processed_total{pool_id, outcome}  // counter
handoff_processing_duration_ms{pool_id}          // histogram
handoff_watcher_errors_total{error_type}         // counter
```

### orchestratord Metrics

```rust
// Pool polling metrics
pool_poll_requests_total{pool_id, outcome}       // counter
pool_readiness_detected_total{pool_id}           // counter
adapter_bind_latency_ms{pool_id}                 // histogram
```

---

## Conclusion

**We fully accept ownership of the handoff watcher.**

The orchestratord team's analysis is **100% correct**:
- Current implementation is HOME_PROFILE only
- Breaks in distributed deployments
- pool-managerd is the natural owner
- Clean service boundaries

**Our commitment**:
- ✅ Implement in Phase 3 (Week 3)
- ✅ HTTP polling support for orchestratord
- ✅ Backward compatibility maintained
- ✅ Full test coverage
- ✅ Spec updates included

**Timeline**:
- v0.1.0: Current implementation (acceptable)
- v0.2.0: pool-managerd watcher (2 weeks)
- v1.0.0: Callbacks + optimization (4 weeks)

**Next Steps**:
1. We'll create the watcher implementation this week
2. orchestratord team implements polling
3. Joint E2E testing next week
4. Spec updates by end of sprint

---

**Contact**: pool-managerd team  
**Status**: ACCEPTED - We'll own it  
**ETA**: 2 weeks for v0.2.0 implementation

Thank you for catching this critical architecture issue! 🎯
