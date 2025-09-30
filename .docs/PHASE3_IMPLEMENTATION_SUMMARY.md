# Phase 3 Implementation Summary - pool-managerd Integration

**Date**: 2025-09-30  
**Status**: ✅ COMPLETE  
**Phase**: 3 of 9 (Cloud Profile Migration)

---

## What Was Implemented

### 1. Configuration Module (`src/config.rs` - 245 lines)

Created comprehensive configuration system supporting both profiles:

**Structures**:
```rust
pub struct Config {
    pub bind_addr: String,
    pub cloud_profile: bool,
    pub node_config: Option<NodeConfig>,
    pub handoff_config: HandoffConfig,
}

pub struct NodeConfig {
    pub node_id: String,
    pub machine_id: String,
    pub address: String,
    pub orchestratord_url: String,
    pub pools: Vec<String>,
    pub capabilities: NodeCapabilities,
    pub heartbeat_interval_secs: u64,
    pub api_token: Option<String>,
    pub register_on_startup: bool,
}

pub struct HandoffConfig {
    pub runtime_dir: PathBuf,
    pub poll_interval_ms: u64,
}
```

**Features**:
- Environment-based configuration with sensible defaults
- Automatic hostname detection for node_id
- CPU core detection via `num_cpus`
- Validates required fields for cloud profile
- Backward compatible with HOME_PROFILE

**Unit Tests** (5 tests):
```rust
- test_config_default_home_profile()
- test_config_cloud_profile_validates_required_fields()
- test_config_cloud_profile_complete()
- test_handoff_config_defaults()
- test_handoff_config_custom()
```

### 2. Updated main.rs (129 lines)

Integrated handoff-watcher and node-registration with conditional spawning:

**HOME_PROFILE Flow** (cloud_profile = false):
```rust
1. Load config
2. Create registry
3. Start HTTP server (no watcher, no registration)
```

**CLOUD_PROFILE Flow** (cloud_profile = true):
```rust
1. Load config
2. Create registry
3. Spawn HandoffWatcher
   ├── Watch .runtime/engines/*.json
   ├── Callback updates registry on new handoffs
   └── Runs in background task
4. Register with orchestratord
   ├── POST /v2/nodes/register on startup
   └── Spawn heartbeat task (every 10s)
5. Start HTTP server
```

**Handoff Callback**:
```rust
let callback = Box::new(move |payload: HandoffPayload| {
    let mut reg = registry_clone.lock().unwrap();
    let handoff_json = serde_json::json!({
        "engine_version": payload.engine_version,
        "device_mask": payload.device_mask,
        "slots": payload.slots,
    });
    reg.register_ready_from_handoff(&payload.pool_id, &handoff_json);
    Ok(())
});
```

**Heartbeat Callback**:
```rust
let _heartbeat_handle = registration.spawn_heartbeat(move || {
    let reg = registry_clone.lock().unwrap();
    let snapshots = reg.snapshots();
    
    snapshots.into_iter().map(|snap| {
        HeartbeatPoolStatus {
            pool_id: snap.pool_id,
            ready: snap.health.ready,
            draining: snap.draining,
            slots_free: snap.slots_free.unwrap_or(0) as u32,
            slots_total: snap.slots_total.unwrap_or(0) as u32,
            vram_free_bytes: snap.vram_free_bytes.unwrap_or(0),
            engine: snap.engine_version,
        }
    }).collect()
});
```

### 3. Dependency Updates (`Cargo.toml`)

Added:
```toml
hostname = "0.4"
num_cpus = "1"
pool-registry-types = { path = "../../libs/shared/pool-registry-types" }
handoff-watcher = { path = "../../libs/gpu-node/handoff-watcher" }
node-registration = { path = "../../libs/gpu-node/node-registration" }
service-registry = { path = "../../libs/control-plane/service-registry" }
```

### 4. Library Export (`src/lib.rs`)

Added config module export:
```rust
pub mod config;
```

---

## Architecture Achieved

### HOME_PROFILE (Default - Unchanged)
```
pool-managerd (POOL_MANAGERD_CLOUD_PROFILE=false)
    ├── HTTP API on :9200
    ├── Registry (in-memory)
    └── No watcher, no registration
```

### CLOUD_PROFILE (New)
```
pool-managerd (POOL_MANAGERD_CLOUD_PROFILE=true)
    ├── HTTP API on :9200
    ├── Registry (in-memory)
    ├── HandoffWatcher
    │   ├── Polls .runtime/engines/*.json every 1s
    │   ├── Detects new engine handoffs
    │   └── Updates registry → ready=true
    ├── NodeRegistration
    │   ├── POST /v2/nodes/register on startup
    │   ├── Heartbeat every 10s with pool status
    │   └── Deregister on shutdown (TODO)
    └── orchestratord knows about this node
```

### End-to-End Flow (CLOUD_PROFILE)

```
1. pool-managerd starts
   ↓
2. Loads config (CLOUD_PROFILE=true)
   ↓
3. Spawns HandoffWatcher
   ↓
4. Registers with orchestratord
   ├── POST /v2/nodes/register
   └── Response: 200 OK
   ↓
5. Spawns heartbeat task (every 10s)
   ↓
6. engine-provisioner writes handoff file
   ↓
7. HandoffWatcher detects file
   ↓
8. Callback updates registry
   ├── pool_id: "pool-0"
   ├── ready: true
   └── slots: 4
   ↓
9. Next heartbeat sends pool status
   ├── POST /v2/nodes/{id}/heartbeat
   └── pools: [{ pool_id, ready, slots_free, ... }]
   ↓
10. orchestratord updates placement cache
```

---

## Configuration Reference

### Environment Variables

**Cloud Profile Control**:
```bash
# Enable cloud profile (default: false)
POOL_MANAGERD_CLOUD_PROFILE=true

# Node identity
POOL_MANAGERD_NODE_ID=gpu-node-1              # Default: hostname
POOL_MANAGERD_MACHINE_ID=machine-alpha        # Default: same as node_id

# Network
POOL_MANAGERD_ADDRESS=http://192.168.1.100:9200  # Required for cloud profile
POOL_MANAGERD_ADDR=0.0.0.0:9200                  # Bind address (default: 127.0.0.1:9200)

# orchestratord connection
ORCHESTRATORD_URL=http://192.168.1.1:8080     # Required for cloud profile
LLORCH_API_TOKEN=secret123                    # Optional API token

# Pools
POOL_MANAGERD_POOLS=pool-0,pool-1             # Comma-separated (default: pool-0)

# Heartbeat
POOL_MANAGERD_HEARTBEAT_INTERVAL_SECS=10      # Default: 10
ORCHESTRATORD_REGISTER_ON_STARTUP=true        # Default: true

# Handoff watcher
POOL_MANAGERD_RUNTIME_DIR=.runtime/engines    # Default: .runtime/engines
POOL_MANAGERD_WATCH_INTERVAL_MS=1000          # Default: 1000 (1s)
```

**Backward Compatible** (HOME_PROFILE):
```bash
# All existing environment variables work as before
POOL_MANAGERD_ADDR=127.0.0.1:9200
```

---

## Testing

### Unit Tests Written: 5 Total

**config.rs** (5 tests):
- Default state is HOME_PROFILE
- Cloud profile validates required fields
- Complete cloud profile configuration
- Handoff config defaults
- Handoff config custom values

### To Run Tests

```bash
# Run config tests (single-threaded to avoid env var pollution)
cargo test -p pool-managerd --lib config::tests -- --test-threads=1

# Check compilation
cargo check -p pool-managerd

# All tests pass ✅
```

---

## Integration Points

### With Phase 1 Libraries

**handoff-watcher**:
- ✅ Spawned in main.rs when cloud_profile=true
- ✅ Callback updates registry on handoff detection
- ✅ Runs in background tokio task

**node-registration**:
- ✅ Registers on startup with orchestratord
- ✅ Spawns heartbeat task with pool status
- ✅ Uses registry snapshots for pool data

**service-registry**:
- ✅ HeartbeatPoolStatus struct used for heartbeat payload
- ✅ Integrates with node-registration

### With Phase 2 (orchestratord)

**Registration Flow**:
```
pool-managerd → POST /v2/nodes/register → orchestratord
                                            ↓
                                    ServiceRegistry.register()
                                            ↓
                                    200 OK response
```

**Heartbeat Flow**:
```
pool-managerd → POST /v2/nodes/{id}/heartbeat → orchestratord
                     (every 10s)                     ↓
                                             ServiceRegistry.heartbeat()
                                                     ↓
                                             200 OK + next_interval
```

---

## Files Created/Modified

### Created
- `bin/pool-managerd/src/config.rs` (245 lines with 5 tests)

### Modified
- `bin/pool-managerd/Cargo.toml` (added 6 dependencies)
- `bin/pool-managerd/src/lib.rs` (exported config module)
- `bin/pool-managerd/src/main.rs` (129 lines, integrated watcher + registration)

---

## Example Usage

### HOME_PROFILE (Default)
```bash
# Start pool-managerd (no cloud features)
POOL_MANAGERD_ADDR=127.0.0.1:9200 \
  pool-managerd
```

### CLOUD_PROFILE
```bash
# Start pool-managerd with cloud profile
POOL_MANAGERD_CLOUD_PROFILE=true \
POOL_MANAGERD_NODE_ID=gpu-node-1 \
POOL_MANAGERD_ADDRESS=http://192.168.1.100:9200 \
ORCHESTRATORD_URL=http://192.168.1.1:8080 \
POOL_MANAGERD_POOLS=pool-0,pool-1 \
LLORCH_API_TOKEN=secret123 \
  pool-managerd
```

**Expected Logs**:
```
INFO pool_managerd: Configuration loaded. Cloud profile: true
INFO pool_managerd: Handoff watcher started
INFO pool_managerd: Successfully registered with orchestratord
INFO pool_managerd: Heartbeat task started
INFO pool_managerd: pool-managerd listening on 0.0.0.0:9200
```

**When engine-provisioner writes handoff**:
```
INFO pool_managerd: Handoff detected, updating registry
  pool_id: pool-0
  replica_id: r0
  engine: llamacpp
  url: http://localhost:8081
```

---

## Specifications Addressed

From `.specs/01_cloud_profile_.md`:

- ✅ **CLOUD-3001**: HandoffWatcher spawned on startup
- ✅ **CLOUD-3002**: Handoff callback updates registry
- ✅ **CLOUD-3003**: Registry snapshots exported for heartbeat
- ✅ **CLOUD-3010**: NodeRegistration registers on startup
- ✅ **CLOUD-3011**: Heartbeat spawned with pool status
- ✅ **CLOUD-3012**: Pool status includes ready, draining, slots, vram
- ✅ **CLOUD-12001**: HOME_PROFILE backward compatibility maintained
- ✅ **CLOUD-12002**: Feature flags control cloud profile

---

## Verification

### Compilation
```bash
cargo check -p pool-managerd
# ✅ Finished `dev` profile [unoptimized + debuginfo] target(s) in 5.04s
```

### Unit Tests
```bash
cargo test -p pool-managerd --lib config::tests -- --test-threads=1
# ✅ test result: ok. 5 passed; 0 failed; 0 ignored
```

### Integration Test (Manual)

1. Start orchestratord with cloud profile:
```bash
ORCHESTRATORD_CLOUD_PROFILE=true \
ORCHESTRATORD_ADDR=0.0.0.0:8080 \
  cargo run -p orchestratord --bin orchestratord
```

2. Start pool-managerd with cloud profile:
```bash
POOL_MANAGERD_CLOUD_PROFILE=true \
POOL_MANAGERD_NODE_ID=test-node \
POOL_MANAGERD_ADDRESS=http://localhost:9200 \
ORCHESTRATORD_URL=http://localhost:8080 \
  cargo run -p pool-managerd
```

3. Verify registration:
```bash
curl http://localhost:8080/v2/nodes
# Should show test-node registered
```

4. Create handoff file:
```bash
mkdir -p .runtime/engines
echo '{"pool_id":"pool-0","replica_id":"r0","engine":"llamacpp","url":"http://localhost:8081","engine_version":"v1","slots":4}' \
  > .runtime/engines/pool-0_r0.json
```

5. Verify heartbeat includes pool status:
```bash
# Check orchestratord logs for heartbeat with pool-0 status
```

---

## Next Steps (Phase 4-9)

### Phase 4: Update orchestratord Placement Logic
- Read from ServiceRegistry instead of local adapter_host
- Route tasks to nodes with available slots
- Handle multi-node placement

### Phase 5: GPU Detection
- Implement actual GPU capability detection
- Report VRAM, compute capability
- Update NodeCapabilities in heartbeat

### Phase 6: Graceful Shutdown
- Deregister on SIGTERM
- Drain pools before shutdown
- Wait for in-flight tasks

### Phase 7-9: Testing & Documentation
- End-to-end integration tests
- Performance testing
- Update deployment docs

---

## Summary

**Phase 3 Status**: ✅ **COMPLETE**

- Configuration module with 5 unit tests
- HandoffWatcher integration with registry callback
- NodeRegistration with startup + heartbeat
- Backward compatibility maintained
- All tests pass
- Clean compilation

**Ready For**: Phase 4 - orchestratord placement logic updates

---

## References

- Phase 1: `.docs/MIGRATION_COMPLETE_PHASE1.md`
- Phase 2: `.docs/PHASE2_IMPLEMENTATION_SUMMARY.md`
- Architecture: `.docs/ARCHITECTURE_LIBRARY_ORGANIZATION.md`
- Spec: `.specs/01_cloud_profile_.md`
- Roadmap: `TODO_CLOUD_PROFILE.md`
