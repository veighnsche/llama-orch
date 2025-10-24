# Package Manager Operations Design

**Date:** Oct 23, 2025  
**Status:** DESIGN  
**Author:** TEAM-276

## Overview

This document defines the new **package manager operations** for declarative lifecycle management in rbee.

---

## Operation Changes Summary

### Operations to ADD

```rust
Operation::PackageSync { .. }        // Sync all to config
Operation::PackageStatus { .. }      // Check drift
Operation::PackageInstall { .. }     // Install all (alias)
Operation::PackageUninstall { .. }   // Uninstall all
Operation::PackageValidate { .. }    // Validate config
Operation::PackageMigrate { .. }     // Generate config from current state
```

### Operations to REMOVE

```rust
Operation::HiveInstall { .. }        // → PackageSync
Operation::HiveUninstall { .. }      // → PackageUninstall
Operation::WorkerDownload { .. }     // → PackageSync
Operation::WorkerBuild { .. }        // → PackageSync
Operation::WorkerBinaryList { .. }   // → PackageStatus
Operation::WorkerBinaryGet { .. }    // → PackageStatus
Operation::WorkerBinaryDelete { .. } // → PackageSync
```

### Operations to KEEP

```rust
// Hive daemon management (runtime)
Operation::HiveStart { .. }
Operation::HiveStop { .. }
Operation::HiveList { .. }
Operation::HiveGet { .. }
Operation::HiveStatus { .. }
Operation::HiveCapabilities { .. }

// Worker process management (runtime)
Operation::WorkerSpawn { .. }
Operation::WorkerProcessList { .. }
Operation::WorkerProcessGet { .. }
Operation::WorkerProcessDelete { .. }

// Model management (on-demand)
Operation::ModelDownload { .. }
Operation::ModelList { .. }
Operation::ModelGet { .. }
Operation::ModelDelete { .. }

// Inference
Operation::Infer { .. }
```

---

## New Operations Detail

### 1. PackageSync

**Purpose:** Sync actual state to match declarative config

**Definition:**
```rust
Operation::PackageSync {
    /// Path to config file (default: ~/.config/rbee/hives.conf)
    config_path: Option<String>,
    
    /// Dry run - show what would change without applying
    dry_run: bool,
    
    /// Remove components not in config
    remove_extra: bool,
    
    /// Force reinstall even if already installed
    force: bool,
}
```

**Behavior:**
1. Read `hives.conf`
2. Query actual state (concurrent)
3. Compute diff (desired vs actual)
4. Apply changes (concurrent):
   - Install missing hives
   - Install missing workers
   - Remove extra components (if `remove_extra`)
   - Start hives (if `auto_start`)
5. Return sync report

**CLI:**
```bash
# Sync all hives
rbee sync

# Dry run (show what would change)
rbee sync --dry-run

# Remove extra components
rbee sync --remove-extra

# Force reinstall
rbee sync --force

# Sync specific hive
rbee sync --hive gpu-1
```

**Response:**
```json
{
  "status": "success",
  "summary": {
    "hives_installed": 2,
    "hives_already_installed": 1,
    "workers_installed": 4,
    "workers_already_installed": 2,
    "duration_seconds": 28.5
  },
  "hives": [
    {
      "alias": "gpu-server-1",
      "status": "installed",
      "workers": ["vllm", "llama-cpp"],
      "duration_seconds": 15.2
    },
    {
      "alias": "gpu-server-2",
      "status": "already_installed",
      "workers": ["vllm", "comfyui"],
      "duration_seconds": 0.5
    },
    {
      "alias": "gpu-server-3",
      "status": "installed",
      "workers": ["llama-cpp"],
      "duration_seconds": 12.8
    }
  ]
}
```

---

### 2. PackageStatus

**Purpose:** Check if actual state matches config (drift detection)

**Definition:**
```rust
Operation::PackageStatus {
    /// Path to config file
    config_path: Option<String>,
    
    /// Show detailed diff
    verbose: bool,
}
```

**Behavior:**
1. Read `hives.conf`
2. Query actual state (concurrent)
3. Compute diff
4. Return status report (no changes applied)

**CLI:**
```bash
# Check status
rbee status

# Verbose output
rbee status --verbose
```

**Response:**
```json
{
  "status": "drift_detected",
  "summary": {
    "total_hives": 3,
    "hives_ok": 2,
    "hives_missing": 0,
    "hives_extra": 1,
    "workers_missing": 1,
    "workers_extra": 0
  },
  "drift": [
    {
      "hive": "gpu-server-2",
      "issue": "missing_worker",
      "details": "Worker 'comfyui' not installed"
    },
    {
      "hive": "gpu-server-4",
      "issue": "extra_hive",
      "details": "Hive not in config"
    }
  ]
}
```

---

### 3. PackageInstall

**Purpose:** Install all components from config (alias for PackageSync)

**Definition:**
```rust
Operation::PackageInstall {
    /// Path to config file
    config_path: Option<String>,
    
    /// Force reinstall
    force: bool,
    
    /// Install only specific hive
    hive_alias: Option<String>,
}
```

**Behavior:**
- Same as `PackageSync` but with `remove_extra: false`
- Only installs, never removes

**CLI:**
```bash
# Install all
rbee install

# Install specific hive
rbee install --hive gpu-1

# Force reinstall
rbee install --force
```

---

### 4. PackageUninstall

**Purpose:** Uninstall components

**Definition:**
```rust
Operation::PackageUninstall {
    /// Uninstall specific hive (None = all)
    hive_alias: Option<String>,
    
    /// Remove config files too
    purge: bool,
}
```

**Behavior:**
1. Stop hives
2. Uninstall hive binaries
3. Uninstall worker binaries
4. (Optional) Remove config files

**CLI:**
```bash
# Uninstall all
rbee uninstall

# Uninstall specific hive
rbee uninstall --hive gpu-1

# Purge (remove config too)
rbee uninstall --purge
```

---

### 5. PackageValidate

**Purpose:** Validate config without applying changes

**Definition:**
```rust
Operation::PackageValidate {
    /// Path to config file
    config_path: Option<String>,
}
```

**Behavior:**
1. Parse config file
2. Validate syntax
3. Check SSH connectivity
4. Check hostname resolution
5. Check worker types exist
6. Return validation report

**CLI:**
```bash
# Validate config
rbee validate

# Validate specific file
rbee validate --config /path/to/hives.conf
```

**Response:**
```json
{
  "status": "valid",
  "warnings": [
    {
      "hive": "gpu-server-1",
      "warning": "SSH key not found, will prompt for password"
    }
  ],
  "errors": []
}
```

---

### 6. PackageMigrate

**Purpose:** Generate config from current state

**Definition:**
```rust
Operation::PackageMigrate {
    /// Output path for generated config
    output_path: String,
}
```

**Behavior:**
1. Query all hives
2. Query all workers on each hive
3. Generate `hives.conf` from actual state
4. Write to output path

**CLI:**
```bash
# Generate config
rbee migrate --output ~/.config/rbee/

# Creates:
# - ~/.config/rbee/hives.conf
```

**Generated Config:**
```toml
# Generated by rbee migrate on 2025-10-23

[[hive]]
alias = "gpu-server-1"
hostname = "192.168.1.100"
ssh_user = "vince"
ssh_port = 22
hive_port = 8600
workers = [
    { type = "vllm", version = "latest" },
    { type = "llama-cpp", version = "latest" },
]
auto_start = true

# ... more hives
```

---

## Operation Routing

### Package Operations (Handled by Queen)

```rust
// job_router.rs

match operation {
    // Package manager operations - handled by queen
    Operation::PackageSync { .. } => {
        package_manager::sync_all_hives(operation, config).await?
    }
    
    Operation::PackageStatus { .. } => {
        package_manager::check_status(operation, config).await?
    }
    
    Operation::PackageInstall { .. } => {
        package_manager::install_all(operation, config).await?
    }
    
    Operation::PackageUninstall { .. } => {
        package_manager::uninstall_all(operation, config).await?
    }
    
    Operation::PackageValidate { .. } => {
        package_manager::validate_config(operation, config).await?
    }
    
    Operation::PackageMigrate { .. } => {
        package_manager::migrate_to_config(operation, config).await?
    }
    
    // ... other operations
}
```

### Runtime Operations (Forwarded to Hive)

```rust
// job_router.rs

match operation {
    // Worker process operations - forwarded to hive
    Operation::WorkerSpawn { .. } => {
        hive_forwarder::forward_to_hive(&job_id, operation, config).await?
    }
    
    Operation::WorkerProcessList { .. } => {
        hive_forwarder::forward_to_hive(&job_id, operation, config).await?
    }
    
    // ... other forwarded operations
}
```

---

## Updated `should_forward_to_hive()`

```rust
impl Operation {
    /// Check if operation should be forwarded to hive
    ///
    /// **Forwarded to Hive:**
    /// - Worker process operations - Spawn/list/kill worker processes
    /// - Model operations - Download/manage models
    ///
    /// **Handled by Queen:**
    /// - Package operations - Declarative lifecycle management
    /// - Active worker operations - Query heartbeat registry
    /// - Infer - Scheduling and routing
    /// - Hive operations - Daemon management
    pub fn should_forward_to_hive(&self) -> bool {
        matches!(
            self,
            // Worker process operations (hive-local)
            Operation::WorkerSpawn { .. }
                | Operation::WorkerProcessList { .. }
                | Operation::WorkerProcessGet { .. }
                | Operation::WorkerProcessDelete { .. }
                // Model operations (hive-local)
                | Operation::ModelDownload { .. }
                | Operation::ModelList { .. }
                | Operation::ModelGet { .. }
                | Operation::ModelDelete { .. }
        )
    }
}
```

**Note:** Worker binary operations removed (no longer forwarded)

---

## CLI Command Mapping

### New Commands

| Command | Operation | Description |
|---------|-----------|-------------|
| `rbee sync` | `PackageSync` | Sync all hives to config |
| `rbee sync --dry-run` | `PackageSync { dry_run: true }` | Show what would change |
| `rbee status` | `PackageStatus` | Check drift |
| `rbee install` | `PackageInstall` | Install all |
| `rbee install --hive X` | `PackageInstall { hive_alias: Some("X") }` | Install specific hive |
| `rbee uninstall` | `PackageUninstall` | Uninstall all |
| `rbee validate` | `PackageValidate` | Validate config |
| `rbee migrate` | `PackageMigrate` | Generate config |

### Deprecated Commands (Removed)

| Old Command | New Command | Status |
|-------------|-------------|--------|
| `rbee install-hive` | `rbee sync` | ❌ Removed |
| `rbee uninstall-hive` | `rbee uninstall` | ❌ Removed |
| `rbee install-worker` | `rbee sync` | ❌ Removed |
| `rbee list-workers` | `rbee status` | ❌ Removed |

### Kept Commands (Runtime Management)

| Command | Operation | Description |
|---------|-----------|-------------|
| `rbee start-hive` | `HiveStart` | Start hive daemon |
| `rbee stop-hive` | `HiveStop` | Stop hive daemon |
| `rbee list-hives` | `HiveList` | List hives |
| `rbee spawn-worker` | `WorkerSpawn` | Spawn worker process |
| `rbee list-workers` | `WorkerProcessList` | List worker processes |
| `rbee infer` | `Infer` | Run inference |

---

## Example Workflows

### Workflow 1: Initial Setup

```bash
# 1. Create config
vim ~/.config/rbee/hives.conf

# 2. Validate config
rbee validate
# ✅ Config valid

# 3. Install everything
rbee install
# ✅ Installed 3 hives, 6 workers (30s)

# 4. Check status
rbee status
# ✅ All components installed
```

### Workflow 2: Add New Hive

```bash
# 1. Edit config
vim ~/.config/rbee/hives.conf
# Add new hive

# 2. Sync (installs only new hive)
rbee sync
# ✅ Installed gpu-server-4

# 3. Verify
rbee status
# ✅ All components match config
```

### Workflow 3: Drift Detection

```bash
# 1. Someone manually installs worker
ssh gpu-server-1 'curl -L ... -o ~/.local/share/rbee/workers/rbee-worker-comfyui'

# 2. Detect drift
rbee status
# ⚠️  gpu-server-1: Extra worker 'comfyui'

# 3. Fix drift (remove extra)
rbee sync --remove-extra
# ✅ Removed comfyui from gpu-server-1

# OR adopt the change
vim ~/.config/rbee/hives.conf
# Add comfyui to gpu-server-1 workers
rbee sync
# ✅ Config updated, no changes needed
```

### Workflow 4: Migration from Imperative

```bash
# 1. Generate config from current state
rbee migrate --output ~/.config/rbee/
# ✅ Generated hives.conf

# 2. Review generated config
cat ~/.config/rbee/hives.conf

# 3. From now on, use declarative
rbee sync
```

---

## Implementation Checklist

### Phase 1: Add Operations
- [ ] Add `PackageSync` to Operation enum
- [ ] Add `PackageStatus` to Operation enum
- [ ] Add `PackageInstall` to Operation enum
- [ ] Add `PackageUninstall` to Operation enum
- [ ] Add `PackageValidate` to Operation enum
- [ ] Add `PackageMigrate` to Operation enum
- [ ] Update `should_forward_to_hive()` (remove worker binary ops)
- [ ] Add serialization tests

### Phase 2: Implement in Queen
- [ ] Create `queen-rbee/src/package_manager/` module
- [ ] Implement `sync.rs` (concurrent sync logic)
- [ ] Implement `diff.rs` (compute desired vs actual)
- [ ] Implement `install.rs` (install hives and workers via SSH)
- [ ] Implement `status.rs` (query actual state)
- [ ] Implement `validate.rs` (validate config)
- [ ] Implement `migrate.rs` (generate config)
- [ ] Add to job_router.rs

### Phase 3: Update CLI
- [ ] Add `rbee sync` command
- [ ] Add `rbee status` command
- [ ] Add `rbee install` command
- [ ] Add `rbee uninstall` command
- [ ] Add `rbee validate` command
- [ ] Add `rbee migrate` command
- [ ] Remove old commands

### Phase 4: Remove Old Operations
- [ ] Remove `HiveInstall` from Operation enum
- [ ] Remove `HiveUninstall` from Operation enum
- [ ] Remove `WorkerDownload` from Operation enum
- [ ] Remove `WorkerBuild` from Operation enum
- [ ] Remove `WorkerBinaryList` from Operation enum
- [ ] Remove `WorkerBinaryGet` from Operation enum
- [ ] Remove `WorkerBinaryDelete` from Operation enum
- [ ] Update job_router.rs to reject removed operations

---

## Conclusion

**Package manager operations provide:**

✅ **Declarative lifecycle** - Config declares desired state  
✅ **Concurrent installation** - 3-10x faster  
✅ **Drift detection** - Know when actual ≠ desired  
✅ **Simpler architecture** - Queen manages everything  
✅ **Better for scheduler** - Queen knows all workers  

**Next Steps:**
1. Implement Phase 1 (add operations)
2. Implement Phase 2 (package manager in queen)
3. Implement Phase 3 (update CLI)
4. Implement Phase 4 (remove old operations)

---

**Created by:** TEAM-276  
**Status:** Design document  
**Next:** Implement operations in rbee-operations crate
