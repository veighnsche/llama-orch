# handoff-watcher

Watches for handoff files written by engine-provisioner on GPU nodes.

## Purpose

In CLOUD_PROFILE, handoff detection must happen on the GPU node (same filesystem as engine-provisioner). This watcher runs as part of pool-managerd and notifies when engines are ready.

## Usage

```rust
use handoff_watcher::{HandoffWatcher, HandoffWatcherConfig, HandoffPayload};

// Create callback
let callback = Box::new(|payload: HandoffPayload| {
    println!("Engine ready: {} at {}", payload.pool_id, payload.url);
    // Update local registry
    Ok(())
});

// Configure watcher
let config = HandoffWatcherConfig {
    runtime_dir: PathBuf::from(".runtime/engines"),
    poll_interval_ms: 1000,
};

// Spawn watcher task
let watcher = HandoffWatcher::new(config, callback);
let handle = watcher.spawn();
```

## Architecture

```
engine-provisioner
    │
    └─> writes .runtime/engines/pool-0-r0.json
              │
              ↓
        HandoffWatcher (polls every 1s)
              │
              └─> invokes callback
                      │
                      └─> pool-managerd updates registry
                              │
                              └─> next heartbeat includes ready=true
```

## Specifications

- **CLOUD-3001**: pool-managerd owns handoff watcher
- **CLOUD-3010**: Watch `.runtime/engines/*.json`
- **CLOUD-3011**: Update local registry + heartbeat
- **CLOUD-3013**: Process within 2 seconds

## Migration from orchestratord

Previously, orchestratord watched handoff files. This breaks in CLOUD_PROFILE because orchestratord can't access pool-managerd's filesystem.

**Before (HOME_PROFILE)**:
- orchestratord watches `.runtime/engines/*.json`
- Direct filesystem access (same machine)

**After (CLOUD_PROFILE)**:
- pool-managerd watches `.runtime/engines/*.json` (local)
- orchestratord polls pool-managerd HTTP API
- Works across network boundaries

## References

- `.specs/01_cloud_profile_.md` - Handoff detection requirements
- `.docs/CLOUD_PROFILE_MIGRATION.md` - Phase 1 implementation
