# 🚨 URGENT: Handoff Watcher Architecture Issue

**To**: pool-managerd Team  
**From**: orchestratord Team  
**Date**: 2025-09-30  
**Priority**: HIGH  
**Subject**: Handoff Watcher Ownership - HOME_PROFILE vs CLOUD_PROFILE Architecture Conflict

---

## Executive Summary

We discovered a **critical architecture flaw** during BDD test implementation: the handoff watcher is currently implemented in `orchestratord`, but this **only works for HOME_PROFILE** (single machine). For **CLOUD_PROFILE** (distributed deployment), this design is fundamentally broken because orchestratord cannot access the filesystem where engine-provisioner writes handoff files.

**Action Required**: Move handoff watcher from orchestratord to pool-managerd.

---

## The Problem

### Current Implementation (BROKEN for CLOUD_PROFILE):

```
┌─────────────────────────────────────────────────────────────┐
│ Machine A (Control Plane - No GPU)                          │
│                                                              │
│  orchestratord                                               │
│    ├── services/handoff.rs                                   │
│    │   └── spawn_handoff_autobind_watcher()  ❌ WRONG!      │
│    │       └── watches .runtime/engines/*.json               │
│    │       └── binds adapters                                │
│    └── adapter-host                                          │
└─────────────────────────────────────────────────────────────┘
                           │
                           │ ❌ Cannot access filesystem!
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ Machine B (GPU Node)                                         │
│                                                              │
│  engine-provisioner                                          │
│    └── writes .runtime/engines/pool-0-r0.json               │
│                                                              │
│  pool-managerd                                               │
│    └── (no watcher currently)                                │
└─────────────────────────────────────────────────────────────┘
```

**Why This Fails**:
- orchestratord runs on control plane (no GPU)
- engine-provisioner runs on GPU node
- **No shared filesystem** between machines
- orchestratord **cannot read** handoff files written by engine-provisioner

---

## The Solution

### Correct Architecture (CLOUD_PROFILE Compatible):

```
┌─────────────────────────────────────────────────────────────┐
│ Machine A (Control Plane - No GPU)                          │
│                                                              │
│  orchestratord                                               │
│    ├── clients/pool_manager.rs                               │
│    │   └── get_pool_status() via HTTP ✅                     │
│    └── adapter-host                                          │
│        └── bind() when pool becomes ready                    │
└─────────────────────────────────────────────────────────────┘
                           │
                           │ HTTP polling or callbacks
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ Machine B (GPU Node)                                         │
│                                                              │
│  engine-provisioner                                          │
│    └── writes .runtime/engines/pool-0-r0.json               │
│                           │                                  │
│                           ↓ (same filesystem)                │
│  pool-managerd ✅ CORRECT OWNER                              │
│    ├── spawn_handoff_watcher()                               │
│    │   └── watches .runtime/engines/*.json                   │
│    │   └── updates own registry                              │
│    │   └── marks pools as Ready                              │
│    └── HTTP API                                              │
│        └── GET /pools/{id}/status                            │
│        └── POST /pools/{id}/ready (optional callback)        │
└─────────────────────────────────────────────────────────────┘
```

---

## Why pool-managerd Should Own the Watcher

### ✅ Reasons:

1. **Filesystem Co-location**
   - pool-managerd runs on same machine as engine-provisioner
   - Can read handoff files without network overhead
   - No distributed filesystem required

2. **Responsibility Alignment**
   - pool-managerd already manages pool state
   - Already has registry of pools
   - Natural owner of "pool readiness" state

3. **Cloud Profile Compatibility**
   - Works in distributed deployments
   - No filesystem coupling between services
   - HTTP-only communication

4. **Separation of Concerns**
   - orchestratord: Control plane, routing, admission
   - pool-managerd: Pool lifecycle, GPU management
   - Clean boundaries

5. **Scalability**
   - Multiple GPU nodes, each with own pool-managerd
   - Each watches its own local filesystem
   - orchestratord polls all via HTTP

---

## Proposed Changes

### 1. Move Watcher to pool-managerd

**File**: `bin/pool-managerd/src/watcher.rs` (new)

```rust
pub fn spawn_handoff_watcher(registry: Arc<Mutex<Registry>>) {
    tokio::spawn(async move {
        let runtime_dir = std::env::var("POOL_MANAGERD_RUNTIME_DIR")
            .unwrap_or_else(|_| ".runtime/engines".to_string());
        let interval_ms = std::env::var("POOL_MANAGERD_WATCH_INTERVAL_MS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1000);

        loop {
            tokio::time::sleep(Duration::from_millis(interval_ms)).await;
            
            // Watch for handoff files
            if let Ok(entries) = std::fs::read_dir(&runtime_dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.extension() == Some("json") {
                        if let Err(e) = process_handoff_file(&registry, &path).await {
                            warn!("failed to process handoff: {}", e);
                        }
                    }
                }
            }
        }
    });
}

async fn process_handoff_file(
    registry: &Arc<Mutex<Registry>>, 
    path: &Path
) -> anyhow::Result<()> {
    let content = std::fs::read_to_string(path)?;
    let handoff: HandoffFile = serde_json::from_str(&content)?;
    
    // Update registry
    let mut reg = registry.lock().unwrap();
    reg.register_ready_from_handoff(&handoff.pool_id, &handoff);
    
    Ok(())
}
```

### 2. Remove Watcher from orchestratord

**File**: `bin/orchestratord/src/services/handoff.rs`

**Action**: DELETE or mark as HOME_PROFILE only

### 3. Update orchestratord to Poll pool-managerd

**File**: `bin/orchestratord/src/services/pool_health.rs` (new)

```rust
/// Periodically poll pool-managerd for pool status updates
pub fn spawn_pool_health_poller(state: AppState) {
    tokio::spawn(async move {
        let interval_ms = std::env::var("ORCHD_POOL_POLL_INTERVAL_MS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(5000);

        loop {
            tokio::time::sleep(Duration::from_millis(interval_ms)).await;
            
            // Poll all known pools
            // When a pool becomes ready, bind adapter
            if let Ok(status) = state.pool_manager.get_pool_status("pool-0").await {
                if status.ready {
                    // Bind adapter if not already bound
                    bind_adapter_if_needed(&state, &status.pool_id, &status.replica_id).await;
                }
            }
        }
    });
}
```

### 4. Optional: Add Callback Mechanism

**File**: `bin/pool-managerd/src/api.rs`

```rust
/// POST /callbacks/register
/// Allow orchestratord to register for pool readiness callbacks
pub async fn register_callback(
    Json(req): Json<CallbackRequest>
) -> Result<StatusCode, Error> {
    // When pool becomes ready, POST to orchestratord
    // orchestratord can then bind adapter immediately
    Ok(StatusCode::CREATED)
}
```

---

## Migration Strategy

### Phase 1: HOME_PROFILE (Current)
- Keep watcher in orchestratord
- Add big warning comments
- Document as HOME_PROFILE only
- **Status**: Acceptable for v0.1.0

### Phase 2: CLOUD_PROFILE (Required for v1.0.0)
- Move watcher to pool-managerd
- Implement HTTP polling in orchestratord
- Update specs and contracts
- **Status**: Required before production

### Phase 3: Optimization (Future)
- Add webhook/callback mechanism
- Reduce polling overhead
- Event-driven architecture
- **Status**: Nice to have

---

## Impact Assessment

### Breaking Changes:
- ❌ orchestratord can no longer watch filesystem directly
- ❌ Tests that rely on handoff watcher will break
- ✅ pool-managerd API must expose pool readiness

### Non-Breaking:
- ✅ HOME_PROFILE still works (same machine)
- ✅ Adapter binding logic stays in orchestratord
- ✅ HTTP client already implemented

### Test Impact:
- BDD tests currently call `process_handoff_file()` directly
- Tests will need to mock pool-managerd HTTP responses
- Or run real pool-managerd daemon in tests

---

## Recommendations

### Immediate (This Sprint):
1. **Document** the HOME_PROFILE limitation clearly
2. **Add TODO** comments in handoff.rs
3. **Update specs** to reflect cloud_profile requirements
4. **Keep current implementation** for v0.1.0 (HOME_PROFILE only)

### Short Term (Next Sprint):
1. **Implement watcher** in pool-managerd
2. **Add HTTP endpoints** for pool readiness
3. **Update orchestratord** to poll instead of watch
4. **Update tests** to use HTTP mocks

### Long Term (v1.0.0):
1. **Remove filesystem coupling** entirely
2. **Implement callbacks** for efficiency
3. **Support multi-node** deployments
4. **Cloud-native architecture**

---

## Questions for pool-managerd Team

1. **Timeline**: When can you implement the handoff watcher?
2. **API Design**: Do you prefer polling or callbacks?
3. **Backwards Compat**: Should we support both modes?
4. **Testing**: How should we test distributed scenarios?
5. **Ownership**: Who owns the adapter binding decision?

---

## References

- **Spec**: `.specs/20_pool-managerd.md`
- **Contract**: `contracts/openapi/pool-managerd.yaml` (needs update)
- **Current Code**: `bin/orchestratord/src/services/handoff.rs`
- **Target Code**: `bin/pool-managerd/src/watcher.rs` (doesn't exist yet)
- **Profile Docs**: `.specs/00_home_profile.md`, `.specs/00_cloud_profile.md`

---

## Conclusion

The handoff watcher **must** be owned by pool-managerd for cloud_profile to work. The current implementation in orchestratord is a **HOME_PROFILE-only hack** that will break in distributed deployments.

We recommend:
- **v0.1.0**: Keep current implementation, document limitation
- **v0.2.0**: Move to pool-managerd, implement HTTP polling
- **v1.0.0**: Production-ready cloud architecture

Please advise on timeline and preferred approach.

---

**Contact**: orchestratord team  
**Urgency**: HIGH (blocks cloud_profile)  
**Blocker**: Yes (for distributed deployments)  

cc: @architecture-team @cloud-profile-wg
