# orchestratord → pool-managerd Integration Update

**Date**: 2025-09-30  
**Change**: pool-managerd is now a standalone daemon on port 9200  
**Impact**: orchestratord must call HTTP API instead of embedded Registry

---

## Current State (Embedded)

orchestratord embeds `pool_managerd::registry::Registry` as an in-process library:

```rust
// bin/orchestratord/src/state.rs
use pool_managerd::registry::Registry as PoolRegistry;

pub struct AppState {
    pub pool_manager: Arc<Mutex<PoolRegistry>>,  // ← In-process
    // ...
}
```

**Usage**:
- `state.pool_manager.lock().unwrap().get_health(pool_id)`
- `state.pool_manager.lock().unwrap().register_ready_from_handoff()`
- Direct method calls, no network

---

## New State (HTTP Client)

pool-managerd is now a daemon listening on `127.0.0.1:9200`:

**API Endpoints**:
- `GET /health` — daemon health
- `POST /pools/{id}/preload` — spawn engine
- `GET /pools/{id}/status` — get pool status

**orchestratord needs**:
- HTTP client to call pool-managerd
- Replace `Arc<Mutex<Registry>>` with `PoolManagerClient`
- Handle network errors gracefully

---

## Changes Needed

### 1. Create HTTP Client

**File**: `bin/orchestratord/src/clients/pool_manager.rs`

```rust
use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Clone)]
pub struct PoolManagerClient {
    base_url: String,
    client: reqwest::Client,
}

#[derive(Deserialize)]
pub struct PoolStatus {
    pub pool_id: String,
    pub live: bool,
    pub ready: bool,
    pub active_leases: i32,
    pub slots_total: i32,
    pub slots_free: i32,
}

impl PoolManagerClient {
    pub fn new(base_url: String) -> Self {
        Self {
            base_url,
            client: reqwest::Client::new(),
        }
    }

    pub async fn get_pool_status(&self, pool_id: &str) -> Result<PoolStatus> {
        let url = format!("{}/pools/{}/status", self.base_url, pool_id);
        let resp = self.client.get(&url).send().await?;
        let status = resp.json().await?;
        Ok(status)
    }

    pub async fn daemon_health(&self) -> Result<bool> {
        let url = format!("{}/health", self.base_url);
        let resp = self.client.get(&url).send().await?;
        Ok(resp.status().is_success())
    }
}
```

### 2. Update AppState

**File**: `bin/orchestratord/src/state.rs`

```rust
// BEFORE:
use pool_managerd::registry::Registry as PoolRegistry;
pub pool_manager: Arc<Mutex<PoolRegistry>>,

// AFTER:
use crate::clients::pool_manager::PoolManagerClient;
pub pool_manager: PoolManagerClient,
```

### 3. Update Usage Sites

**Files to update**:
- `src/api/control.rs` - `get_pool_health()`
- `src/services/streaming.rs` - `should_dispatch()`
- `src/services/handoff.rs` - `register_ready_from_handoff()` (remove, daemon handles this)

**Example** (`src/api/control.rs`):

```rust
// BEFORE:
let (live, ready, last_error) = {
    let reg = state.pool_manager.lock().expect("pool_manager lock");
    let h = reg.get_health(&id).unwrap_or_default();
    let e = reg.get_last_error(&id);
    (h.live, h.ready, e)
};

// AFTER:
let status = state.pool_manager.get_pool_status(&id).await
    .unwrap_or_else(|_| PoolStatus {
        pool_id: id.clone(),
        live: false,
        ready: false,
        active_leases: 0,
        slots_total: 0,
        slots_free: 0,
    });
let (live, ready) = (status.live, status.ready);
```

### 4. Update Handoff Watcher

**File**: `src/services/handoff.rs`

The handoff watcher should **no longer** update the registry directly. Instead:

**Option A**: Remove registry updates (daemon handles it)
```rust
// REMOVE:
let mut reg = state.pool_manager.lock()?;
reg.register_ready_from_handoff(pool_id, &handoff);

// Daemon reads handoff files itself
```

**Option B**: Call daemon's preload endpoint
```rust
// POST to pool-managerd
let preload_req = PreloadRequest {
    prepared: prepared_engine,
};
state.pool_manager.preload_pool(pool_id, preload_req).await?;
```

### 5. Add Dependency

**File**: `bin/orchestratord/Cargo.toml`

```toml
[dependencies]
reqwest = { version = "0.11", features = ["json"] }
```

### 6. Configuration

**Environment Variable**:
```bash
POOL_MANAGERD_URL=http://127.0.0.1:9200
```

**In code**:
```rust
let pool_manager_url = std::env::var("POOL_MANAGERD_URL")
    .unwrap_or_else(|_| "http://127.0.0.1:9200".to_string());
let pool_manager = PoolManagerClient::new(pool_manager_url);
```

---

## Migration Strategy

### Phase 1: Add HTTP Client (Keep Embedded)
1. Add `PoolManagerClient` alongside existing Registry
2. Test HTTP calls work
3. Keep embedded Registry as fallback

### Phase 2: Switch to HTTP (Remove Embedded)
1. Replace `Arc<Mutex<Registry>>` with `PoolManagerClient`
2. Update all call sites
3. Remove pool-managerd dependency from orchestratord

### Phase 3: Update Tests
1. Mock pool-managerd HTTP responses
2. Or start real pool-managerd daemon in tests
3. Update BDD scenarios

---

## Compatibility Notes

### Home Profile (Current)
- orchestratord embeds Registry (in-process)
- Single binary, no network calls
- Simple deployment

### Cloud Profile (Future)
- orchestratord calls pool-managerd via HTTP
- Separate processes, can scale independently
- pool-managerd as DaemonSet on GPU nodes

### Hybrid Approach
Keep both modes:
```rust
pub enum PoolManager {
    Embedded(Arc<Mutex<Registry>>),
    Remote(PoolManagerClient),
}
```

Configure via feature flag or env var.

---

## Testing Impact

### BDD Tests
- Need to either:
  - Start pool-managerd daemon before tests
  - Mock HTTP responses
  - Keep embedded mode for tests

### E2E Tests
- Start both daemons:
  ```bash
  pool-managerd &
  orchestratord &
  # Run tests
  ```

---

## Timeline

**Estimated**: 2-3 hours

1. **Create HTTP client** (30 min)
2. **Update AppState** (15 min)
3. **Update call sites** (45 min)
4. **Update tests** (60 min)
5. **Documentation** (30 min)

---

## Decision: Immediate or Deferred?

**Option A**: Update now (2-3 hours)
- orchestratord calls pool-managerd via HTTP
- Full daemon architecture
- More complex testing

**Option B**: Defer (keep embedded for now)
- Keep current embedded Registry
- Update later when needed
- Simpler for current development

**Recommendation**: **Option B (Defer)** until BDD suite is complete and stable.

---

## Notes

- pool-managerd daemon is ready and working
- orchestratord can continue using embedded Registry for now
- Migration can happen incrementally
- Both modes can coexist during transition

---

**Status**: Analysis complete, migration path defined, recommend deferring until BDD complete 🎯
